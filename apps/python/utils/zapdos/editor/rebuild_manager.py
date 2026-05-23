from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any, Callable

from utils.zapdos.bundle import RenderBundle
from utils.zapdos.editor.rebuild_events import (
    begin_scene_rebuild_apply,
    clear_scene_rebuild_candidate,
    create_scene_rebuild_job,
    discard_scene_rebuild_task,
    emit_scene_rebuild_progress,
    ensure_scene_rebuild_state,
    fail_scene_rebuild_job,
    finish_scene_rebuild_apply,
    is_latest_scene_rebuild_candidate,
    lookup_scene_rebuild_job,
    next_scene_rebuild_job_id,
    register_scene_rebuild_candidate,
    register_scene_rebuild_task,
    scene_rebuild_superseded_error,
)
from utils.zapdos.editor.rebuild_types import PreparedOverlayRebuild
from utils.zapdos.editor.repository import save_overlay_state
from utils.zapdos.editor.support_infos import resolve_support_infos


def start_overlay_operation(
    session: Any,
    next_overlay: dict[str, object],
    success_payload: dict[str, object],
) -> dict[str, object]:
    ensure_scene_rebuild_state(session)
    owner_session = getattr(session, "session", session)
    previous_overlay = deepcopy(session.overlay_state)
    previous_revision = session.scene_revision
    op_id = next_scene_rebuild_job_id(session)
    create_scene_rebuild_job(session, op_id, deepcopy(success_payload))
    register_scene_rebuild_candidate(session, op_id, next_overlay)
    try:
        task = owner_session.schedule_on_owner_loop(session._run_overlay_rebuild(
            op_id,
            next_overlay,
            previous_overlay,
            previous_revision,
        ))
        register_scene_rebuild_task(session, op_id, task)
        task.add_done_callback(lambda _task, current_op_id=op_id: discard_scene_rebuild_task(session, current_op_id))
    except Exception:
        session.discard_scene_rebuild_job(op_id)
        raise
    return {
        **deepcopy(success_payload),
        "ok": True,
        "op_id": op_id,
        "status": "started",
    }


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
        "composed_scene_usd": str(_candidate_scene_usd(session, op_id)),
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


async def run_overlay_rebuild(
    session: Any,
    op_id: str,
    next_overlay: dict[str, object],
    previous_overlay: dict[str, object],
    previous_revision: str,
) -> None:
    ensure_scene_rebuild_state(session)
    owner_session = getattr(session, "session", session)
    try:
        support_snapshot = await owner_session.run_sync(
            lambda current: getattr(current, "editor", current)._capture_support_info_inputs(),
        )
        support_infos = await asyncio.to_thread(resolve_support_infos, support_snapshot)
        emit_scene_rebuild_progress(session, op_id, "prepare_overlay_rebuild.started")
        prepared = await asyncio.to_thread(
            session._prepare_overlay_rebuild,
            next_overlay,
            support_infos,
            previous_overlay,
            previous_revision,
            op_id=op_id,
        )
        emit_scene_rebuild_progress(session, op_id, "prepare_overlay_rebuild.done")
        if not is_latest_scene_rebuild_candidate(session, op_id):
            fail_scene_rebuild_job(session, op_id, scene_rebuild_superseded_error(session, op_id))
            return
        async with owner_session.reserve_world() as world_token:
            if not begin_scene_rebuild_apply(session, op_id):
                fail_scene_rebuild_job(session, op_id, scene_rebuild_superseded_error(session, op_id))
                return
            emit_scene_rebuild_progress(session, op_id, "apply_overlay_rebuild.started")
            revision = await owner_session.run_sync(
                lambda current: getattr(current, "editor", current)._apply_prepared_overlay_rebuild(prepared, op_id),
                world_token=world_token,
            )
        emit_scene_rebuild_progress(
            session,
            op_id,
            "apply_overlay_rebuild.done",
            scene_revision=revision,
        )
    except asyncio.CancelledError as err:
        fail_scene_rebuild_job(session, op_id, RuntimeError("Scene rebuild cancelled"))
        raise err
    except Exception as err:
        fail_scene_rebuild_job(session, op_id, err)
        return
    job = lookup_scene_rebuild_job(session, op_id)
    if job is not None and not job.future.done():
        job.future.set_result({**job.success_payload, "scene_revision": revision})


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
        if op_id is not None:
            clear_scene_rebuild_candidate(session, op_id)
        finish_scene_rebuild_apply(session, op_id)


def _run_overlay_rebuild_inline(
    request_payload: dict[str, object],
    stage_logger: Callable[[str], None] | None = None,
) -> dict[str, object]:
    from utils.zapdos.editor.rebuild_runner import prepare_overlay_rebuild_request

    return prepare_overlay_rebuild_request(request_payload, stage_logger)


def _candidate_scene_usd(session: Any, op_id: str | None):
    path = session.composed_scene_usd
    if op_id is None:
        return path
    return path.with_name(f"{path.stem}-{op_id}{path.suffix}")
