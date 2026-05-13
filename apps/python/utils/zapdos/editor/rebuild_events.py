from __future__ import annotations

import asyncio
import json
import queue
import threading
from concurrent.futures import Future as ConcurrentFuture
from typing import Any

from fastapi import HTTPException

from utils.zapdos.editor.rebuild_job import SceneRebuildJob

SCENE_REBUILD_POLL_SEC = 0.1


def ensure_scene_rebuild_state(session: Any) -> None:
    if not hasattr(session, "overlay_completions"):
        session.overlay_completions = queue.Queue()
    if not hasattr(session, "scene_rebuild_job_counter"):
        session.scene_rebuild_job_counter = 0
    if not hasattr(session, "scene_rebuild_jobs"):
        session.scene_rebuild_jobs = {}
    if not hasattr(session, "scene_rebuild_jobs_lock"):
        session.scene_rebuild_jobs_lock = threading.Lock()


def create_scene_rebuild_job(session: Any, op_id: str, success_payload: dict[str, object]) -> None:
    ensure_scene_rebuild_state(session)
    with session.scene_rebuild_jobs_lock:
        session.scene_rebuild_jobs[op_id] = SceneRebuildJob(
            future=ConcurrentFuture(),
            success_payload=success_payload,
            events=queue.Queue(),
        )


def next_scene_rebuild_job_id(session: Any) -> str:
    ensure_scene_rebuild_state(session)
    session.scene_rebuild_job_counter += 1
    return f"op-{session.scene_rebuild_job_counter}"


def scene_rebuild_future(session: Any, op_id: str) -> ConcurrentFuture:
    job = lookup_scene_rebuild_job(session, op_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Scene rebuild job not found: {op_id}")
    return job.future


def discard_scene_rebuild_job(session: Any, op_id: str) -> None:
    ensure_scene_rebuild_state(session)
    with session.scene_rebuild_jobs_lock:
        session.scene_rebuild_jobs.pop(op_id, None)


async def stream_scene_rebuild_job(session: Any, op_id: str):
    delivered = False
    try:
        future = asyncio.wrap_future(session.scene_rebuild_future(op_id))
        yield _sse_event("started", {"op_id": op_id})
        while True:
            for name, payload in drain_scene_rebuild_events(session, op_id):
                yield _sse_event(name, payload)
            done, _ = await asyncio.wait({future}, timeout=SCENE_REBUILD_POLL_SEC)
            for name, payload in drain_scene_rebuild_events(session, op_id):
                yield _sse_event(name, payload)
            if not done:
                continue
            delivered = True
            yield _sse_event("done", await future)
            break
    except HTTPException:
        raise
    except Exception as exc:
        delivered = True
        payload = {"detail": getattr(exc, "detail", None) or str(exc)}
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            payload["status_code"] = status_code
        yield _sse_event("failed", payload)
    finally:
        if delivered:
            session.discard_scene_rebuild_job(op_id)


def emit_scene_rebuild_progress(session: Any, op_id: str | None, stage: str, **payload: object) -> None:
    if op_id:
        queue_scene_rebuild_event(session, op_id, "progress", {"stage": stage, **payload})


def fail_scene_rebuild_job(session: Any, op_id: str, error: Exception) -> None:
    session.rebuilding_scene = False
    job = lookup_scene_rebuild_job(session, op_id)
    if job is not None and not job.future.done():
        job.future.set_exception(error)


def lookup_scene_rebuild_job(session: Any, op_id: str) -> SceneRebuildJob | None:
    ensure_scene_rebuild_state(session)
    with session.scene_rebuild_jobs_lock:
        return session.scene_rebuild_jobs.get(op_id)


def drain_scene_rebuild_events(session: Any, op_id: str) -> list[tuple[str, dict[str, object]]]:
    job = lookup_scene_rebuild_job(session, op_id)
    if job is None:
        return []
    events: list[tuple[str, dict[str, object]]] = []
    while True:
        try:
            events.append(job.events.get_nowait())
        except queue.Empty:
            return events


def queue_scene_rebuild_event(session: Any, op_id: str, name: str, payload: dict[str, object]) -> None:
    job = lookup_scene_rebuild_job(session, op_id)
    if job is not None:
        job.events.put_nowait((name, payload))


def _sse_event(name: str, payload: object) -> str:
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"
