from __future__ import annotations

import queue
from copy import deepcopy
from typing import Any, Callable

from utils.zapdos.bundle import RenderBundle
from utils.zapdos.editor.rebuild_events import (
    create_scene_rebuild_job,
    emit_scene_rebuild_progress,
    ensure_scene_rebuild_state,
    fail_scene_rebuild_job,
    lookup_scene_rebuild_job,
    next_scene_rebuild_job_id,
)
from utils.zapdos.editor.rebuild_types import OverlayRebuildCompletion, PreparedOverlayRebuild
from utils.zapdos.editor.repository import save_overlay_state


def start_overlay_operation(
    session: Any,
    next_overlay: dict[str, object],
    success_payload: dict[str, object],
) -> dict[str, object]:
    ensure_scene_rebuild_state(session)
    previous_overlay = deepcopy(session.overlay_state)
    previous_revision = session.scene_revision
    op_id = next_scene_rebuild_job_id(session)
    create_scene_rebuild_job(session, op_id, deepcopy(success_payload))
    session.rebuilding_scene = True
    try:
        session.overlay_executor.submit(
            session._run_overlay_rebuild_background,
            op_id,
            next_overlay,
            session._build_support_infos(),
            previous_overlay,
            previous_revision,
        )
    except Exception:
        session.rebuilding_scene = False
        session.discard_scene_rebuild_job(op_id)
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
    stage_logger = lambda stage: emit_scene_rebuild_progress(session, op_id, f"prepare_overlay_rebuild.{stage}")
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
        fail_scene_rebuild_job(session, op_id, err)


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


def _run_overlay_rebuild_inline(
    request_payload: dict[str, object],
    stage_logger: Callable[[str], None] | None = None,
) -> dict[str, object]:
    from utils.zapdos.editor.rebuild_runner import prepare_overlay_rebuild_request

    return prepare_overlay_rebuild_request(request_payload, stage_logger)


def _complete_overlay_rebuild(session: Any, completion: OverlayRebuildCompletion) -> None:
    job = lookup_scene_rebuild_job(session, completion.op_id)
    if completion.error is not None:
        fail_scene_rebuild_job(session, completion.op_id, completion.error)
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
