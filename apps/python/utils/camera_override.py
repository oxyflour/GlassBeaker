from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from utils.zapdos.bundle.camera_specs import RenderCamera
from utils.user_config import read_user_config, write_user_config


@dataclass(frozen=True)
class CameraOverride:
    parent_prim: str
    name: str
    pos: list[float]
    quat: list[float]
    fovy: float
    horizontal_aperture: float
    vertical_aperture: float
    clipping_range: list[float]


def load_camera_overrides(payload: dict[str, Any] | None = None) -> dict[tuple[str, str], CameraOverride]:
    data = read_user_config() if payload is None else payload
    override = data.get("override", {})
    if not isinstance(override, dict):
        raise RuntimeError("override must be a JSON object.")
    camera = override.get("camera", {})
    if not isinstance(camera, dict):
        raise RuntimeError("override.camera must be a JSON object.")
    loaded: dict[tuple[str, str], CameraOverride] = {}
    for parent_prim, entries in camera.items():
        if not isinstance(entries, dict):
            raise RuntimeError(f"override.camera.{parent_prim} must be a JSON object.")
        for name, spec in entries.items():
            if not isinstance(spec, dict):
                raise RuntimeError(f"override.camera.{parent_prim}.{name} must be a JSON object.")
            loaded[(str(parent_prim), str(name))] = CameraOverride(
                parent_prim=str(parent_prim),
                name=str(name),
                pos=[float(value) for value in spec["pos"]],
                quat=[float(value) for value in spec["quat"]],
                fovy=float(spec["fovy"]),
                horizontal_aperture=float(spec["horizontal_aperture"]),
                vertical_aperture=float(spec["vertical_aperture"]),
                clipping_range=[float(value) for value in spec["clipping_range"]],
            )
    return loaded


def apply_camera_overrides(
    cameras: list[RenderCamera],
    overrides: dict[tuple[str, str], CameraOverride] | None = None,
) -> list[RenderCamera]:
    resolved = load_camera_overrides() if overrides is None else overrides
    updated: list[RenderCamera] = []
    for camera in cameras:
        key = (PurePosixPath(camera.prim).parent.as_posix(), camera.name)
        spec = resolved.get(key)
        if spec is None:
            updated.append(camera)
            continue
        updated.append(RenderCamera(
            name=camera.name,
            prim=camera.prim,
            topic=camera.topic,
            frame_id=camera.frame_id,
            body=camera.body,
            pos=list(spec.pos),
            quat=list(spec.quat),
            fovy=spec.fovy,
            horizontal_aperture=spec.horizontal_aperture,
            vertical_aperture=spec.vertical_aperture,
            clipping_range=list(spec.clipping_range),
        ))
    return updated


def save_camera_overrides(snapshot: list[dict[str, Any]]) -> tuple[Path, int]:
    payload = read_user_config()
    override = payload.setdefault("override", {})
    if not isinstance(override, dict):
        raise RuntimeError("override must be a JSON object.")
    camera = override.setdefault("camera", {})
    if not isinstance(camera, dict):
        raise RuntimeError("override.camera must be a JSON object.")
    saved = 0
    for item in snapshot:
        parent_prim = str(item["parent_prim"])
        name = str(item["name"])
        group = camera.setdefault(parent_prim, {})
        if not isinstance(group, dict):
            raise RuntimeError(f"override.camera.{parent_prim} must be a JSON object.")
        group[name] = {
            "pos": [float(value) for value in item["pos"]],
            "quat": [float(value) for value in item["quat"]],
            "fovy": float(item["fovy"]),
            "horizontal_aperture": float(item["horizontal_aperture"]),
            "vertical_aperture": float(item["vertical_aperture"]),
            "clipping_range": [float(value) for value in item["clipping_range"]],
        }
        saved += 1
    return write_user_config(payload), saved

