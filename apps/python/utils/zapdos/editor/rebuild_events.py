from __future__ import annotations

import asyncio
import queue
import threading
from concurrent.futures import Future as ConcurrentFuture
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException

from utils.sse import sse_json_event
from utils.zapdos.editor.rebuild_job import SceneRebuildJob

SCENE_REBUILD_POLL_SEC = 0.1


@dataclass(slots=True)
class SceneRebuildState:
    job_counter: int = 0
    jobs: dict[str, SceneRebuildJob] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    tasks: dict[str, ConcurrentFuture[Any] | asyncio.Future[Any]] = field(default_factory=dict)
    candidate_generation: int = 0
    candidate_generations: dict[str, int] = field(default_factory=dict)
    latest_candidate_op_id: str | None = None
    latest_candidate_overlay: dict[str, object] | None = None
    applying_op_id: str | None = None


def ensure_scene_rebuild_state(session: Any) -> SceneRebuildState:
    state = getattr(session, "scene_rebuild_state", None)
    if state is None:
        raise AttributeError("scene_rebuild_state must be initialized explicitly")
    return state


def create_scene_rebuild_job(session: Any, op_id: str, success_payload: dict[str, object]) -> None:
    state = ensure_scene_rebuild_state(session)
    with state.lock:
        state.jobs[op_id] = SceneRebuildJob(
            future=ConcurrentFuture(),
            success_payload=success_payload,
            events=queue.Queue(),
        )


def next_scene_rebuild_job_id(session: Any) -> str:
    state = ensure_scene_rebuild_state(session)
    state.job_counter += 1
    return f"op-{state.job_counter}"


def scene_rebuild_future(session: Any, op_id: str) -> ConcurrentFuture:
    job = lookup_scene_rebuild_job(session, op_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Scene rebuild job not found: {op_id}")
    return job.future


def discard_scene_rebuild_job(session: Any, op_id: str) -> None:
    state = ensure_scene_rebuild_state(session)
    with state.lock:
        state.jobs.pop(op_id, None)
        state.tasks.pop(op_id, None)
        _clear_candidate_locked(state, op_id)


def cancel_scene_rebuild_job(session: Any, op_id: str) -> None:
    state = ensure_scene_rebuild_state(session)
    with state.lock:
        job = state.jobs.pop(op_id, None)
        task = state.tasks.pop(op_id, None)
        _clear_candidate_locked(state, op_id)
    if task is not None and not task.done():
        task.cancel()
    if job is not None and not job.future.done():
        job.future.cancel()


async def stream_scene_rebuild_events(
    session: Any,
    op_id: str,
    *,
    started_payload: dict[str, object] | None = None,
    cancel_on_disconnect: bool = False,
):
    delivered = False
    future = asyncio.wrap_future(session.scene_rebuild_future(op_id))
    try:
        if started_payload is not None:
            yield "started", started_payload
        while True:
            for name, payload in drain_scene_rebuild_events(session, op_id):
                yield name, payload
            done, _ = await asyncio.wait({future}, timeout=SCENE_REBUILD_POLL_SEC)
            for name, payload in drain_scene_rebuild_events(session, op_id):
                yield name, payload
            if not done:
                continue
            delivered = True
            yield "done", await future
            break
    except Exception as exc:
        delivered = True
        payload = {"detail": getattr(exc, "detail", None) or str(exc)}
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            payload["status_code"] = status_code
        yield "failed", payload
    finally:
        if delivered:
            session.discard_scene_rebuild_job(op_id)
        elif cancel_on_disconnect:
            cancel_scene_rebuild_job(session, op_id)


async def stream_scene_rebuild_job(session: Any, op_id: str):
    async for name, payload in stream_scene_rebuild_events(
        session,
        op_id,
        started_payload={"op_id": op_id},
    ):
        yield sse_json_event(name, payload)


def emit_scene_rebuild_progress(session: Any, op_id: str | None, stage: str, **payload: object) -> None:
    if op_id:
        queue_scene_rebuild_event(session, op_id, "progress", {"stage": stage, **payload})


def fail_scene_rebuild_job(session: Any, op_id: str, error: Exception) -> None:
    finish_scene_rebuild_apply(session, op_id)
    clear_scene_rebuild_candidate(session, op_id)
    job = lookup_scene_rebuild_job(session, op_id)
    if job is not None and not job.future.done():
        job.future.set_exception(error)


def lookup_scene_rebuild_job(session: Any, op_id: str) -> SceneRebuildJob | None:
    state = ensure_scene_rebuild_state(session)
    with state.lock:
        return state.jobs.get(op_id)


def register_scene_rebuild_task(session: Any, op_id: str, task: ConcurrentFuture[Any] | asyncio.Future[Any]) -> None:
    state = ensure_scene_rebuild_state(session)
    with state.lock:
        state.tasks[op_id] = task


def discard_scene_rebuild_task(session: Any, op_id: str) -> None:
    state = ensure_scene_rebuild_state(session)
    with state.lock:
        state.tasks.pop(op_id, None)


def register_scene_rebuild_candidate(session: Any, op_id: str, overlay: dict[str, object]) -> None:
    state = ensure_scene_rebuild_state(session)
    with state.lock:
        state.candidate_generation += 1
        state.candidate_generations[op_id] = state.candidate_generation
        state.latest_candidate_op_id = op_id
        state.latest_candidate_overlay = deepcopy(overlay)


def latest_scene_rebuild_candidate_overlay(session: Any) -> dict[str, object] | None:
    state = ensure_scene_rebuild_state(session)
    with state.lock:
        if state.latest_candidate_overlay is None:
            return None
        return deepcopy(state.latest_candidate_overlay)


def is_latest_scene_rebuild_candidate(session: Any, op_id: str) -> bool:
    state = ensure_scene_rebuild_state(session)
    with state.lock:
        if op_id not in state.candidate_generations:
            return True
        return state.latest_candidate_op_id == op_id


def scene_rebuild_superseded_error(session: Any, op_id: str) -> RuntimeError:
    state = ensure_scene_rebuild_state(session)
    with state.lock:
        latest = state.latest_candidate_op_id
    if latest:
        return RuntimeError(f"Scene rebuild {op_id} superseded by {latest}")
    return RuntimeError(f"Scene rebuild {op_id} superseded")


def begin_scene_rebuild_apply(session: Any, op_id: str) -> bool:
    state = ensure_scene_rebuild_state(session)
    with state.lock:
        if op_id in state.candidate_generations and state.latest_candidate_op_id != op_id:
            return False
        state.applying_op_id = op_id
    session.rebuilding_scene = True
    return True


def finish_scene_rebuild_apply(session: Any, op_id: str | None) -> None:
    if op_id is None:
        session.rebuilding_scene = False
        return
    state = ensure_scene_rebuild_state(session)
    with state.lock:
        if state.applying_op_id == op_id:
            state.applying_op_id = None
            session.rebuilding_scene = False
        elif state.applying_op_id is None:
            session.rebuilding_scene = False


def clear_scene_rebuild_candidate(session: Any, op_id: str) -> None:
    state = ensure_scene_rebuild_state(session)
    with state.lock:
        _clear_candidate_locked(state, op_id)


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


def _clear_candidate_locked(state: SceneRebuildState, op_id: str) -> None:
    state.candidate_generations.pop(op_id, None)
    if state.latest_candidate_op_id == op_id:
        state.latest_candidate_op_id = None
        state.latest_candidate_overlay = None


def _sse_event(name: str, payload: object) -> str:
    return sse_json_event(name, payload)
