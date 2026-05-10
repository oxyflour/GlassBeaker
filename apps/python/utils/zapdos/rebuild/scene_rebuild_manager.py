from __future__ import annotations

import asyncio
from concurrent.futures import Future as ConcurrentFuture
from copy import deepcopy
from dataclasses import dataclass
import json
import queue
import threading
from typing import Any, Callable

from fastapi import HTTPException

from utils.zapdos.bundle import RenderBundle
from utils.zapdos.overlay.overlay_repository import save_overlay_state
from utils.zapdos.overlay.overlay_state import default_overlay_state
from utils.zapdos.rebuild.scene_rebuild_job import SceneRebuildJob

SCENE_REBUILD_POLL_SEC = 0.1


@dataclass(frozen=True)
class PreparedOverlayRebuild:
    bundle: RenderBundle
    next_overlay: dict[str, object]
    previous_overlay: dict[str, object]
    previous_revision: str
    next_revision: str


@dataclass(frozen=True)
class OverlayRebuildCompletion:
    op_id: str
    prepared: PreparedOverlayRebuild | None = None
    error: Exception | None = None


def ensure_scene_rebuild_state(session: Any) -> None:
    if not hasattr(session, "overlay_completions"):
        session.overlay_completions = queue.Queue()
    if not hasattr(session, "scene_rebuild_job_counter"):
        session.scene_rebuild_job_counter = 0
    if not hasattr(session, "scene_rebuild_jobs"):
        session.scene_rebuild_jobs = {}
    if not hasattr(session, "scene_rebuild_jobs_lock"):
        session.scene_rebuild_jobs_lock = threading.Lock()


def start_overlay_operation(
    session: Any,
    next_overlay: dict[str, object],
    success_payload: dict[str, object],
) -> dict[str, object]:
    ensure_scene_rebuild_state(session)
    previous_overlay = deepcopy(session.overlay_state)
    previous_revision = session.scene_revision
    support_infos = session._build_support_infos()
    op_id = _next_scene_rebuild_job_id(session)
    with session.scene_rebuild_jobs_lock:
        session.scene_rebuild_jobs[op_id] = SceneRebuildJob(
            future=ConcurrentFuture(),
            success_payload=deepcopy(success_payload),
            events=queue.Queue(),
        )
    session.rebuilding_scene = True
    try:
        session.overlay_executor.submit(
            session._run_overlay_rebuild_background,
            op_id,
            next_overlay,
            support_infos,
            previous_overlay,
            previous_revision,
        )
    except Exception:
        session.rebuilding_scene = False
        discard_scene_rebuild_job(session, op_id)
        raise
    return {"ok": True, "op_id": op_id}


def prepare_overlay_rebuild(
    session: Any,
    next_overlay: dict[str, object],
    support_infos: dict[str, dict[str, float]],
    previous_overlay: dict[str, object],
    previous_revision: str,
    op_id: str | None = None,
) -> PreparedOverlayRebuild:
    request_payload = {
        "robot_usd": str(session.robot_usd),
        "base_scene_usd": str(session.base_scene_usd),
        "composed_scene_usd": str(session.composed_scene_usd),
        "next_overlay": next_overlay,
        "support_infos": support_infos,
    }
    stage_logger = lambda stage: emit_scene_rebuild_progress(
        session,
        op_id,
        f"prepare_overlay_rebuild.{stage}",
    )
    emit_scene_rebuild_progress(session, op_id, "prepare_overlay_rebuild.inline.started")
    payload = _run_overlay_rebuild_inline(request_payload, stage_logger)
    emit_scene_rebuild_progress(session, op_id, "prepare_overlay_rebuild.inline.done")
    bundle_payload = payload.get("bundle")
    next_revision = payload.get("next_revision")
    if not isinstance(bundle_payload, dict) or not isinstance(next_revision, str):
        raise RuntimeError("Overlay rebuild script returned invalid payload")
    return PreparedOverlayRebuild(
        bundle=RenderBundle.from_json(bundle_payload),
        next_overlay=next_overlay,
        previous_overlay=previous_overlay,
        previous_revision=previous_revision,
        next_revision=next_revision,
    )


def run_overlay_rebuild_background(
    session: Any,
    op_id: str,
    next_overlay: dict[str, object],
    support_infos: dict[str, dict[str, float]],
    previous_overlay: dict[str, object],
    previous_revision: str,
) -> None:
    ensure_scene_rebuild_state(session)
    try:
        emit_scene_rebuild_progress(session, op_id, "prepare_overlay_rebuild.started")
        prepared = session._prepare_overlay_rebuild(
            next_overlay,
            support_infos,
            previous_overlay,
            previous_revision,
            op_id=op_id,
        )
        emit_scene_rebuild_progress(session, op_id, "prepare_overlay_rebuild.done")
        session.overlay_completions.put_nowait(OverlayRebuildCompletion(op_id=op_id, prepared=prepared))
    except Exception as err:
        _fail_scene_rebuild_job(session, op_id, err)


def apply_prepared_overlay_rebuild(
    session: Any,
    prepared: PreparedOverlayRebuild,
    op_id: str | None = None,
) -> str:
    try:
        session._swap_runtime_bundle(prepared.bundle, prepared.next_overlay, op_id)
        save_overlay_state(session.overlay_path, prepared.next_overlay)
        session.overlay_state = prepared.next_overlay
        session.scene_revision = prepared.next_revision
        if not session.msgs.full():
            session.msgs.put_nowait({"scene_revision": session.scene_revision})
        return session.scene_revision
    except Exception:
        session.overlay_state = prepared.previous_overlay
        session.scene_revision = prepared.previous_revision
        save_overlay_state(session.overlay_path, prepared.previous_overlay)
        raise
    finally:
        session.rebuilding_scene = False


def drain_overlay_completions(session: Any) -> None:
    ensure_scene_rebuild_state(session)
    while not session.overlay_completions.empty():
        try:
            completion = session.overlay_completions.get_nowait()
        except queue.Empty:
            break
        _complete_overlay_rebuild(session, completion)


def scene_rebuild_future(session: Any, op_id: str) -> ConcurrentFuture:
    job = _lookup_scene_rebuild_job(session, op_id)
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
            for name, payload in _drain_scene_rebuild_events(session, op_id):
                yield _sse_event(name, payload)
            done, _ = await asyncio.wait({future}, timeout=SCENE_REBUILD_POLL_SEC)
            for name, payload in _drain_scene_rebuild_events(session, op_id):
                yield _sse_event(name, payload)
            if not done:
                continue
            payload = await future
            delivered = True
            yield _sse_event("done", payload)
            break
    except HTTPException:
        raise
    except Exception as exc:
        delivered = True
        detail = getattr(exc, "detail", None)
        status_code = getattr(exc, "status_code", None)
        payload = {"detail": detail or str(exc)}
        if status_code is not None:
            payload["status_code"] = status_code
        yield _sse_event("failed", payload)
    finally:
        if delivered:
            session.discard_scene_rebuild_job(op_id)


def emit_scene_rebuild_progress(
    session: Any,
    op_id: str | None,
    stage: str,
    **payload: object,
) -> None:
    if not op_id:
        return
    _queue_scene_rebuild_event(session, op_id, "progress", {"stage": stage, **payload})


def _complete_overlay_rebuild(session: Any, completion: OverlayRebuildCompletion) -> None:
    job = _lookup_scene_rebuild_job(session, completion.op_id)
    if completion.error is not None:
        _fail_scene_rebuild_job(session, completion.op_id, completion.error)
        return
    if completion.prepared is None:
        session.rebuilding_scene = False
        err = RuntimeError(f"Scene rebuild job {completion.op_id} completed without payload")
        if job is not None and not job.future.done():
            job.future.set_exception(err)
        return
    try:
        emit_scene_rebuild_progress(session, completion.op_id, "apply_overlay_rebuild.started")
        revision = session._apply_prepared_overlay_rebuild(completion.prepared, completion.op_id)
        emit_scene_rebuild_progress(
            session,
            completion.op_id,
            "apply_overlay_rebuild.done",
            scene_revision=revision,
        )
    except Exception as err:
        if job is not None and not job.future.done():
            job.future.set_exception(err)
        return
    if job is not None and not job.future.done():
        job.future.set_result({**job.success_payload, "scene_revision": revision})


def _next_scene_rebuild_job_id(session: Any) -> str:
    ensure_scene_rebuild_state(session)
    session.scene_rebuild_job_counter += 1
    return f"op-{session.scene_rebuild_job_counter}"


def _lookup_scene_rebuild_job(session: Any, op_id: str) -> SceneRebuildJob | None:
    ensure_scene_rebuild_state(session)
    with session.scene_rebuild_jobs_lock:
        return session.scene_rebuild_jobs.get(op_id)


def _fail_scene_rebuild_job(session: Any, op_id: str, error: Exception) -> None:
    session.rebuilding_scene = False
    job = _lookup_scene_rebuild_job(session, op_id)
    if job is not None and not job.future.done():
        job.future.set_exception(error)


def _sse_event(name: str, payload: object) -> str:
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def _run_overlay_rebuild_inline(
    request_payload: dict[str, object],
    stage_logger: Callable[[str], None] | None = None,
) -> dict[str, object]:
    from utils.zapdos.rebuild.overlay_rebuild_runner import prepare_overlay_rebuild_request

    return prepare_overlay_rebuild_request(request_payload, stage_logger)


def _drain_scene_rebuild_events(
    session: Any,
    op_id: str,
) -> list[tuple[str, dict[str, object]]]:
    job = _lookup_scene_rebuild_job(session, op_id)
    if job is None:
        return []
    events: list[tuple[str, dict[str, object]]] = []
    while True:
        try:
            events.append(job.events.get_nowait())
        except queue.Empty:
            return events


def _queue_scene_rebuild_event(
    session: Any,
    op_id: str,
    name: str,
    payload: dict[str, object],
) -> None:
    job = _lookup_scene_rebuild_job(session, op_id)
    if job is None:
        return
    job.events.put_nowait((name, payload))
