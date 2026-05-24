from __future__ import annotations

from copy import deepcopy
from typing import Any

from fastapi import HTTPException

from utils.genie_sim import resolve_assets_root
from utils.zapdos.editor.rebuild_events import latest_scene_rebuild_candidate_overlay
from utils.zapdos.editor.placement import normalize_placement
from utils.zapdos.editor.state import default_overlay_state, overlay_body_name
from utils.zapdos.zapdos_asset_library import resolve_asset_record


def build_set_scene_assets_overlay(
    session: Any,
    assets: list[dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, str]]]:
    assets_root = resolve_assets_root(session.overlay_state.get("assets_root"))
    next_overlay = default_overlay_state(str(assets_root))
    result_items = _append_scene_asset_instances(next_overlay["instances"], assets, assets_root)
    next_overlay["pose_overrides"] = {}
    return next_overlay, result_items


def build_add_scene_assets_overlay(
    session: Any,
    assets: list[dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, str]]]:
    base_overlay = latest_scene_rebuild_candidate_overlay(session) or session.overlay_state
    assets_root = resolve_assets_root(base_overlay.get("assets_root") or session.overlay_state.get("assets_root"))
    next_overlay = deepcopy(base_overlay)
    next_overlay["assets_root"] = str(assets_root)
    instances = next_overlay.get("instances")
    if not isinstance(instances, list):
        instances = []
        next_overlay["instances"] = instances
    result_items = _append_scene_asset_instances(instances, assets, assets_root)
    return next_overlay, result_items


def build_remove_asset_overlay(session: Any, instance_id: str) -> tuple[dict[str, object], str]:
    next_overlay, removed_ids = build_remove_assets_overlay(session, [instance_id])
    return next_overlay, overlay_body_name(removed_ids[0])


def build_remove_assets_overlay(session: Any, instance_ids: list[str]) -> tuple[dict[str, object], list[str]]:
    if not isinstance(instance_ids, list) or not instance_ids:
        raise HTTPException(status_code=400, detail="instance_ids must be a non-empty list")
    normalized_ids = []
    seen_ids = set()
    for raw_instance_id in instance_ids:
        if not isinstance(raw_instance_id, str) or not raw_instance_id.strip():
            raise HTTPException(status_code=400, detail="instance_id must be a non-empty string")
        instance_id = raw_instance_id.strip()
        if instance_id in seen_ids:
            raise HTTPException(status_code=400, detail=f"Duplicate instance_id: {instance_id}")
        seen_ids.add(instance_id)
        normalized_ids.append(instance_id)
    base_overlay = latest_scene_rebuild_candidate_overlay(session) or session.overlay_state
    existing_ids = {item["id"] for item in base_overlay["instances"]}
    missing_ids = [instance_id for instance_id in normalized_ids if instance_id not in existing_ids]
    if missing_ids:
        raise HTTPException(status_code=404, detail=f"Overlay instances not found: {', '.join(missing_ids)}")
    remove_ids = set(normalized_ids)
    bodies = [overlay_body_name(instance_id) for instance_id in normalized_ids]
    next_overlay = deepcopy(base_overlay)
    next_overlay["instances"] = [item for item in next_overlay["instances"] if item["id"] not in remove_ids]
    for body in bodies:
        next_overlay["pose_overrides"].pop(body, None)
    return next_overlay, normalized_ids


def _append_scene_asset_instances(
    instances: list[dict[str, object]],
    assets: list[dict[str, object]],
    assets_root,
) -> list[dict[str, str]]:
    if not isinstance(assets, list) or not assets:
        raise HTTPException(status_code=400, detail="assets must be a non-empty list")
    counts: dict[str, int] = {}
    existing_ids = set()
    for instance in instances:
        instance_id = instance.get("id")
        asset_id = instance.get("asset_id")
        if isinstance(instance_id, str):
            existing_ids.add(instance_id)
        if isinstance(asset_id, str):
            counts[asset_id] = counts.get(asset_id, 0) + 1
    result_items = []
    for item in assets:
        asset_id = item.get("asset_id")
        motion = item.get("motion")
        if not isinstance(asset_id, str) or not asset_id.strip():
            raise HTTPException(status_code=400, detail="asset_id must be a non-empty string")
        if motion not in {"static", "dynamic"}:
            raise HTTPException(status_code=400, detail=f"Unsupported motion: {motion}")
        try:
            placement = normalize_placement(item.get("placement"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        asset = _require_asset_record(asset_id, assets_root)
        counts[asset_id] = counts.get(asset_id, 0) + 1
        instance_id = f"{asset_id}_{counts[asset_id]:02d}"
        while instance_id in existing_ids:
            counts[asset_id] += 1
            instance_id = f"{asset_id}_{counts[asset_id]:02d}"
        existing_ids.add(instance_id)
        instances.append(
            {
                "id": instance_id,
                "asset_id": asset["asset_id"],
                "url": asset["url"],
                "motion": motion,
                "placement": placement,
            }
        )
        result_items.append(
            {"asset_id": asset["asset_id"], "instance_id": instance_id, "body": overlay_body_name(instance_id)}
        )
    return result_items


def _require_asset_record(asset_id: str, assets_root) -> dict[str, object]:
    try:
        return resolve_asset_record(asset_id, assets_root)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Asset not found: {asset_id}") from exc
    except (FileNotFoundError, ImportError, OSError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


__all__ = [
    "build_add_scene_assets_overlay",
    "build_remove_asset_overlay",
    "build_remove_assets_overlay",
    "build_set_scene_assets_overlay",
]
