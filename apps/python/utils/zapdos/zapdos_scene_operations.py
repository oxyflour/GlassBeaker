from __future__ import annotations

import asyncio
from concurrent.futures import Future as ConcurrentFuture
from copy import deepcopy
from dataclasses import dataclass
import json
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import HTTPException

from utils.genie_sim_runtime import resolve_assets_root
from utils.zapdos.rl_bundle import RenderBundle
from utils.zapdos.zapdos_asset_library import resolve_asset_record
from utils.zapdos.zapdos_overlay import (
    OverlayState,
    default_overlay_state,
    overlay_body_name,
    save_overlay_state,
)
from utils.zapdos.zapdos_overlay_scene import normalize_placement

REPO_ROOT = Path(__file__).resolve().parents[4]
OVERLAY_REBUILD_SCRIPT = REPO_ROOT / "apps" / "python" / "scripts" / "prepare_zapdos_overlay_rebuild.py"


@dataclass(frozen=True)
class PreparedOverlayRebuild:
    bundle: RenderBundle
    next_overlay: OverlayState
    previous_overlay: OverlayState
    previous_revision: str
    next_revision: str


@dataclass(frozen=True)
class OverlayRebuildCompletion:
    op_id: str
    prepared: PreparedOverlayRebuild | None = None
    error: Exception | None = None


@dataclass
class SceneOperation:
    future: ConcurrentFuture
    success_payload: dict[str, object]


def build_set_scene_assets_overlay(
    session: Any,
    assets: list[dict[str, object]],
) -> tuple[OverlayState, list[dict[str, str]]]:
    if session.rebuilding_scene:
        raise HTTPException(status_code=409, detail="Scene rebuild already in progress")
    if not isinstance(assets, list) or not assets:
        raise HTTPException(status_code=400, detail="assets must be a non-empty list")
    assets_root = resolve_assets_root(session.overlay_state.get("assets_root"))
    counts: dict[str, int] = {}
    next_instances = []
    result_items = []
    for item in assets:
        asset_id = item.get("asset_id")
        motion = item.get("motion")
        placement = item.get("placement")
        if not isinstance(asset_id, str) or not asset_id.strip():
            raise HTTPException(status_code=400, detail="asset_id must be a non-empty string")
        if motion not in {"static", "dynamic"}:
            raise HTTPException(status_code=400, detail=f"Unsupported motion: {motion}")
        try:
            normalized_placement = normalize_placement(placement)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        asset = resolve_asset_record(asset_id, assets_root)
        counts[asset_id] = counts.get(asset_id, 0) + 1
        instance_id = f"{asset_id}_{counts[asset_id]:02d}"
        next_instances.append({
            "id": instance_id,
            "asset_id": asset["asset_id"],
            "url": asset["url"],
            "motion": motion,
            "placement": normalized_placement,
        })
        result_items.append({
            "asset_id": asset["asset_id"],
            "instance_id": instance_id,
            "body": overlay_body_name(instance_id),
        })
    next_overlay = default_overlay_state(str(assets_root))
    next_overlay["instances"] = next_instances
    next_overlay["pose_overrides"] = {}
    return next_overlay, result_items


def build_remove_asset_overlay(session: Any, instance_id: str) -> tuple[OverlayState, str]:
    if session.rebuilding_scene:
        raise HTTPException(status_code=409, detail="Scene rebuild already in progress")
    if not any(item["id"] == instance_id for item in session.overlay_state["instances"]):
        raise HTTPException(status_code=404, detail=f"Overlay instance not found: {instance_id}")
    body = overlay_body_name(instance_id)
    next_overlay = deepcopy(session.overlay_state)
    next_overlay["instances"] = [item for item in next_overlay["instances"] if item["id"] != instance_id]
    next_overlay["pose_overrides"].pop(body, None)
    return next_overlay, body


def ensure_scene_operation_state(session: Any) -> None:
    if not hasattr(session, "overlay_completions"):
        session.overlay_completions = queue.Queue()
    if not hasattr(session, "scene_operation_counter"):
        session.scene_operation_counter = 0
    if not hasattr(session, "scene_operations"):
        session.scene_operations = {}
    if not hasattr(session, "scene_operations_lock"):
        session.scene_operations_lock = threading.Lock()


def start_overlay_operation(
    session: Any,
    next_overlay: OverlayState,
    success_payload: dict[str, object],
) -> dict[str, object]:
    ensure_scene_operation_state(session)
    previous_overlay = deepcopy(session.overlay_state)
    previous_revision = session.scene_revision
    support_infos = session._build_support_infos()
    op_id = _next_scene_operation_id(session)
    with session.scene_operations_lock:
        session.scene_operations[op_id] = SceneOperation(
            future=ConcurrentFuture(),
            success_payload=deepcopy(success_payload),
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
        discard_scene_operation(session, op_id)
        raise
    return {"ok": True, "op_id": op_id}


def prepare_overlay_rebuild(
    session: Any,
    next_overlay: OverlayState,
    support_infos: dict[str, dict[str, float]],
    previous_overlay: OverlayState,
    previous_revision: str,
) -> PreparedOverlayRebuild:
    request_payload = {
        "robot_usd": str(session.robot_usd),
        "base_scene_usd": str(session.base_scene_usd),
        "composed_scene_usd": str(session.composed_scene_usd),
        "next_overlay": next_overlay,
        "support_infos": support_infos,
    }
    token = uuid4().hex
    request_path = session.session_dir / f"overlay-rebuild-{token}.request.json"
    response_path = session.session_dir / f"overlay-rebuild-{token}.response.json"
    request_path.write_text(json.dumps(request_payload), encoding="utf-8")
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(OVERLAY_REBUILD_SCRIPT),
                str(request_path),
                str(response_path),
            ],
            cwd=str(REPO_ROOT / "apps" / "python"),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(detail or f"Overlay rebuild script failed with exit code {result.returncode}")
        payload = json.loads(response_path.read_text(encoding="utf-8"))
    finally:
        request_path.unlink(missing_ok=True)
        response_path.unlink(missing_ok=True)
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
    next_overlay: OverlayState,
    support_infos: dict[str, dict[str, float]],
    previous_overlay: OverlayState,
    previous_revision: str,
) -> None:
    ensure_scene_operation_state(session)
    try:
        prepared = session._prepare_overlay_rebuild(
            next_overlay,
            support_infos,
            previous_overlay,
            previous_revision,
        )
        session.overlay_completions.put_nowait(OverlayRebuildCompletion(op_id=op_id, prepared=prepared))
    except Exception as err:
        session.overlay_completions.put_nowait(OverlayRebuildCompletion(op_id=op_id, error=err))


def apply_prepared_overlay_rebuild(session: Any, prepared: PreparedOverlayRebuild) -> str:
    try:
        session._swap_runtime_bundle(prepared.bundle, prepared.next_overlay)
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
    ensure_scene_operation_state(session)
    while not session.overlay_completions.empty():
        try:
            completion = session.overlay_completions.get_nowait()
        except queue.Empty:
            break
        _complete_overlay_rebuild(session, completion)


def scene_operation_future(session: Any, op_id: str) -> ConcurrentFuture:
    operation = _lookup_scene_operation(session, op_id)
    if operation is None:
        raise HTTPException(status_code=404, detail=f"Scene operation not found: {op_id}")
    return operation.future


def discard_scene_operation(session: Any, op_id: str) -> None:
    ensure_scene_operation_state(session)
    with session.scene_operations_lock:
        session.scene_operations.pop(op_id, None)


async def stream_scene_operation(session: Any, op_id: str):
    delivered = False
    try:
        payload = await asyncio.wrap_future(session.scene_operation_future(op_id))
        delivered = True
        yield _sse_event("done", payload)
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
            session.discard_scene_operation(op_id)


def _next_scene_operation_id(session: Any) -> str:
    ensure_scene_operation_state(session)
    session.scene_operation_counter += 1
    return f"op-{session.scene_operation_counter}"


def _lookup_scene_operation(session: Any, op_id: str) -> SceneOperation | None:
    ensure_scene_operation_state(session)
    with session.scene_operations_lock:
        return session.scene_operations.get(op_id)


def _complete_overlay_rebuild(session: Any, completion: OverlayRebuildCompletion) -> None:
    operation = _lookup_scene_operation(session, completion.op_id)
    if completion.error is not None:
        session.rebuilding_scene = False
        if operation is not None and not operation.future.done():
            operation.future.set_exception(completion.error)
        return
    if completion.prepared is None:
        session.rebuilding_scene = False
        err = RuntimeError(f"Scene operation {completion.op_id} completed without payload")
        if operation is not None and not operation.future.done():
            operation.future.set_exception(err)
        return
    try:
        revision = session._apply_prepared_overlay_rebuild(completion.prepared)
    except Exception as err:
        if operation is not None and not operation.future.done():
            operation.future.set_exception(err)
        return
    if operation is not None and not operation.future.done():
        operation.future.set_result({
            **operation.success_payload,
            "scene_revision": revision,
        })


def _sse_event(name: str, payload: object) -> str:
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"
