from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypedDict

from utils.zapdos.usd_to_mjcf import sanitize_name


class OverlayPoseOverride(TypedDict):
    pos: list[float]
    quat: list[float]


class OverlayPlacement(TypedDict, total=False):
    kind: str
    xy: list[float]
    z_offset: float
    yaw: float
    body: str
    gap: float
    pos: list[float]
    quat: list[float]
    payload_quat: list[float]


class OverlayInstance(TypedDict):
    id: str
    asset_id: str
    url: str
    motion: str
    placement: OverlayPlacement


class OverlayState(TypedDict):
    version: int
    assets_root: str | None
    instances: list[OverlayInstance]
    pose_overrides: dict[str, OverlayPoseOverride]


def default_overlay_state(assets_root: str | None = None) -> OverlayState:
    return {"version": 1, "assets_root": assets_root, "instances": [], "pose_overrides": {}}


def overlay_body_name(instance_id: str) -> str:
    return f"Scene_{sanitize_name(instance_id)}"


def scene_revision(scene_usd: Path, state: OverlayState) -> str:
    return _digest(["scene", _fingerprint(scene_usd), _normalized_instances(state)])


def bundle_revision(robot_usd: Path, scene_usd: Path, state: OverlayState) -> str:
    return _digest(["bundle", _fingerprint(robot_usd), _fingerprint(scene_usd), _normalized_instances(state)])


def load_overlay_state(path: Path) -> OverlayState:
    from utils.zapdos.editor.repository import load_overlay_state as _load_overlay_state

    return _load_overlay_state(path)


def save_overlay_state(path: Path, state: OverlayState) -> None:
    from utils.zapdos.editor.repository import save_overlay_state as _save_overlay_state

    _save_overlay_state(path, state)


def _normalized_instances(state: OverlayState) -> list[dict[str, object]]:
    return sorted(
        [
            {
                "id": item["id"],
                "asset_id": item["asset_id"],
                "url": item["url"],
                "motion": item["motion"],
                "placement": item["placement"],
            }
            for item in state["instances"]
        ],
        key=lambda item: item["id"],
    )


def _fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path.resolve()), "mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


def _digest(parts: list[object]) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "OverlayInstance",
    "OverlayPlacement",
    "OverlayPoseOverride",
    "OverlayState",
    "bundle_revision",
    "default_overlay_state",
    "load_overlay_state",
    "overlay_body_name",
    "save_overlay_state",
    "scene_revision",
]
