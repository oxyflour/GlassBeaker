from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def build_kokoro_scene_dict(
    *,
    checkpoint_path: Path,
    hdr_path: Path,
    width: int = 512,
    height: int = 384,
    width_m: float = 0.10,
    depth_m: float = 0.10,
    spp: int = 64,
    env_scale: float = 5.0,
    fov: float = 65.0,
) -> dict[str, Any]:
    scene: dict[str, Any] = {
        "type": "scene",
        "integrator": {"type": "path"},
        "environment": {"type": "envmap", "filename": str(hdr_path), "scale": float(env_scale)},
        "inspection_light": {
            "type": "rectangle",
            "to_world_matrix": [
                [0.16, 0.0, 0.0, 0.0],
                [0.0, -0.16, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.14],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "emitter": {"type": "area", "radiance": {"type": "rgb", "value": [5.0, 5.2, 5.6]}},
        },
        "surface": {
            "type": "rectangle",
            "to_world_matrix": [
                [float(width_m) * 0.5, 0.0, 0.0, 0.0],
                [0.0, float(depth_m) * 0.5, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "bsdf": {
                "type": "kokoro_neural_reflector",
                "checkpoint": str(checkpoint_path),
                "reflectance": [0.86, 0.88, 0.92],
                "lobe_kappa": 96.0,
            },
        },
        "sensor": {
            "type": "perspective",
            "fov": float(fov),
            "to_world_look_at": {
                "origin": [0.09, -0.15, 0.10],
                "target": [0.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
            },
            "sampler": {"type": "independent", "sample_count": int(spp)},
            "film": {"type": "hdrfilm", "width": int(width), "height": int(height), "rfilter": {"type": "box"}},
        },
    }
    return scene


def prepare_mitsuba_scene_dict(scene: dict[str, Any], mi) -> dict[str, Any]:
    converted = dict(scene)
    for key, value in list(converted.items()):
        if not isinstance(value, dict):
            continue
        entry = dict(value)
        look_at = entry.pop("to_world_look_at", None)
        if look_at is not None:
            entry["to_world"] = mi.ScalarTransform4f.look_at(
                origin=look_at["origin"],
                target=look_at["target"],
                up=look_at["up"],
            )
        matrix = entry.pop("to_world_matrix", None)
        if matrix is not None:
            entry["to_world"] = mi.ScalarTransform4f(matrix)
        converted[key] = entry
    return converted


def orbit_scene_dicts(
    scene: dict[str, Any],
    *,
    frame_count: int,
    radius_m: float,
    height_m: float,
) -> list[dict[str, Any]]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    frames: list[dict[str, Any]] = []
    for index in range(frame_count):
        angle = 2.0 * math.pi * index / frame_count
        frame = dict(scene)
        sensor = dict(scene["sensor"])
        sensor["to_world_look_at"] = {
            "origin": [
                round(math.sin(angle) * radius_m, 10),
                round(-math.cos(angle) * radius_m, 10),
                float(height_m),
            ],
            "target": [0.0, 0.0, 0.0],
            "up": [0.0, 0.0, 1.0],
        }
        frame["sensor"] = sensor
        frames.append(frame)
    return frames
