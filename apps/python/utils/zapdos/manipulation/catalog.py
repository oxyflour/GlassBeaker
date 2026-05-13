from __future__ import annotations

from pathlib import Path

from utils.genie_sim import resolve_assets_root
from utils.zapdos.editor.state import overlay_body_name
from utils.zapdos.manipulation.types import SceneObject
from utils.zapdos.zapdos_asset_library import asset_local_bounds, resolve_asset_record


def build_scene_object_catalog(
    scene_bodies: dict[str, object],
    overlay_state: dict[str, object],
) -> list[SceneObject]:
    overlay_by_body = {
        overlay_body_name(instance["id"]): instance
        for instance in overlay_state.get("instances", [])
        if isinstance(instance, dict) and isinstance(instance.get("id"), str)
    }
    assets_root = overlay_state.get("assets_root")
    catalog: list[SceneObject] = []
    for raw_item in scene_bodies.get("items", []):
        if not isinstance(raw_item, dict) or not isinstance(raw_item.get("body"), str):
            continue
        body = raw_item["body"]
        label = str(raw_item.get("label") or body)
        matrix = _matrix(raw_item.get("matrix"))
        overlay = overlay_by_body.get(body)
        tags = _tags_for_body(body, label)
        asset_id = None
        motion = None
        support_body = None
        bounds_min = None
        bounds_max = None
        if overlay is not None:
            asset_id = _string_or_none(overlay.get("asset_id"))
            motion = _string_or_none(overlay.get("motion"))
            support_body = _support_body(overlay.get("placement"))
            if asset_id:
                record = resolve_asset_record(asset_id, assets_root)
                tags.extend(_tags_from_description(record.get("description")))
                tags.extend(_text_forms(asset_id))
                bounds_min, bounds_max = _bounds(assets_root, str(record["url"]))
        catalog.append(
            {
                "body": body,
                "label": label,
                "asset_id": asset_id,
                "motion": motion,
                "tags": _dedupe(tags),
                "support_body": support_body,
                "position": None if matrix is None else [matrix[12], matrix[13], matrix[14]],
                "matrix": matrix,
                "top_z": _top_z(raw_item.get("support")),
                "bounds_min": bounds_min,
                "bounds_max": bounds_max,
                "world_aabb": _world_aabb(raw_item.get("world_aabb")),
            }
        )
    return catalog


def _matrix(value: object) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 16:
        return None
    return [float(item) for item in value]


def _top_z(value: object) -> float | None:
    if not isinstance(value, dict) or "top_z" not in value:
        return None
    return float(value["top_z"])


def _support_body(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    body = value.get("body")
    return body if value.get("kind") == "on_top_of_body" and isinstance(body, str) else None


def _bounds(assets_root: object, asset_url: str) -> tuple[list[float] | None, list[float] | None]:
    path = Path(resolve_assets_root(assets_root)) / asset_url
    bounds = asset_local_bounds(path)
    return (
        [float(value) for value in bounds["min"]],
        [float(value) for value in bounds["max"]],
    )


def _world_aabb(value: object) -> dict[str, list[float]] | None:
    if not isinstance(value, dict):
        return None
    min_values = _vector3(value.get("min"))
    max_values = _vector3(value.get("max"))
    if min_values is None or max_values is None:
        return None
    return {"min": min_values, "max": max_values}


def _tags_for_body(body: str, label: str) -> list[str]:
    return _text_forms(label) + _text_forms(body)


def _tags_from_description(value: object) -> list[str]:
    if not isinstance(value, dict):
        return []
    tags: list[str] = []
    for key in ("semantic_name", "full_description"):
        raw = value.get(key)
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str):
                    tags.extend(_text_forms(item))
    return tags


def _text_forms(value: str) -> list[str]:
    normalized = " ".join(value.replace("_", " ").replace("-", " ").split()).strip().lower()
    if not normalized:
        return []
    parts = [token for token in normalized.split(" ") if token]
    return [normalized] + parts


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _vector3(value: object) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    return [float(item) for item in value]


__all__ = ["build_scene_object_catalog"]
