from __future__ import annotations

from copy import deepcopy
from typing import Any

from fastapi import HTTPException

from utils.genie_sim import resolve_assets_root
from utils.zapdos.overlay.overlay_placement import normalize_placement
from utils.zapdos.overlay.overlay_state import default_overlay_state, overlay_body_name
from utils.zapdos.zapdos_asset_library import resolve_asset_record


def build_set_scene_assets_overlay(
    session: Any,
    assets: list[dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, str]]]:
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


def build_remove_asset_overlay(session: Any, instance_id: str) -> tuple[dict[str, object], str]:
    if session.rebuilding_scene:
        raise HTTPException(status_code=409, detail="Scene rebuild already in progress")
    if not any(item["id"] == instance_id for item in session.overlay_state["instances"]):
        raise HTTPException(status_code=404, detail=f"Overlay instance not found: {instance_id}")
    body = overlay_body_name(instance_id)
    next_overlay = deepcopy(session.overlay_state)
    next_overlay["instances"] = [item for item in next_overlay["instances"] if item["id"] != instance_id]
    next_overlay["pose_overrides"].pop(body, None)
    return next_overlay, body


__all__ = ["build_remove_asset_overlay", "build_set_scene_assets_overlay"]
