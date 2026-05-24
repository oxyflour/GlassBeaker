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
    light_source: str = "point",
    env_scale: float = 1.0,
    inspection_light_scale: float = 0.0,
    fov: float = 65.0,
    top_light_height_m: float = 0.06,
    top_light_intensity: float = 6.0,
    lobe_kappa: float = 96.0,
    sampler_type: str = "ldsampler",
    reconstruction_filter: str = "box",
) -> dict[str, Any]:
    scene: dict[str, Any] = {
        "type": "scene",
        "integrator": {"type": "path"},
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
                "lobe_kappa": float(lobe_kappa),
                "ring_lobe_count": 4,
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
            "sampler": {"type": sampler_type, "sample_count": int(spp)},
            "film": {"type": "hdrfilm", "width": int(width), "height": int(height), "rfilter": {"type": reconstruction_filter}},
        },
    }
    if light_source == "point":
        scene["top_point_light"] = {
            "type": "point",
            "position": [0.0, 0.0, float(top_light_height_m)],
            "intensity": {"type": "rgb", "value": [float(top_light_intensity)] * 3},
        }
    elif light_source == "hdr":
        scene["environment"] = {"type": "envmap", "filename": str(hdr_path), "scale": float(env_scale)}
    else:
        raise ValueError("light_source must be 'point' or 'hdr'")
    if inspection_light_scale > 0.0:
        scale = float(inspection_light_scale)
        scene["inspection_light"] = {
            "type": "rectangle",
            "to_world_matrix": [
                [0.16, 0.0, 0.0, 0.0],
                [0.0, -0.16, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.14],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "emitter": {"type": "area", "radiance": {"type": "rgb", "value": [5.0 * scale, 5.2 * scale, 5.6 * scale]}},
        }
    return scene


def build_kokoro_ring_diagnostic_scene_dict(
    *,
    checkpoint_path: Path,
    width: int = 512,
    height: int = 512,
    width_m: float = 0.10,
    depth_m: float = 0.10,
    spp: int = 512,
    fov: float = 55.0,
    light_height_m: float = 0.04,
    camera_height_m: float = 0.12,
    light_intensity: float = 6.0,
    sampler_type: str = "ldsampler",
    reconstruction_filter: str = "box",
) -> dict[str, Any]:
    return {
        "type": "scene",
        "integrator": {"type": "path"},
        "ring_point_light": {
            "type": "point",
            "position": [0.0, 0.0, float(light_height_m)],
            "intensity": {"type": "rgb", "value": [float(light_intensity)] * 3},
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
                "lobe_kappa": 128.0,
                "ring_lobe_count": 4,
            },
        },
        "sensor": {
            "type": "perspective",
            "fov": float(fov),
            "to_world_look_at": {
                "origin": [0.0, 0.0, float(camera_height_m)],
                "target": [0.0, 0.0, 0.0],
                "up": [0.0, 1.0, 0.0],
            },
            "sampler": {"type": sampler_type, "sample_count": int(spp)},
            "film": {"type": "hdrfilm", "width": int(width), "height": int(height), "rfilter": {"type": reconstruction_filter}},
        },
    }


def build_height_field_reference_scene_dict(
    *,
    height_source: str,
    width: int = 512,
    height: int = 384,
    width_m: float = 0.10,
    depth_m: float = 0.10,
    spp: int = 64,
    hdr_path: Path | None = None,
    light_source: str = "point",
    env_scale: float = 1.0,
    inspection_light_scale: float = 0.0,
    fov: float = 65.0,
    normal_step_m: float = 25e-6,
    lobe_kappa: float = 4096.0,
    sampler_type: str = "ldsampler",
    reconstruction_filter: str = "box",
) -> dict[str, Any]:
    scene = build_kokoro_scene_dict(
        checkpoint_path=Path("unused.npz"),
        hdr_path=hdr_path if hdr_path is not None else Path("unused.hdr"),
        width=width,
        height=height,
        width_m=width_m,
        depth_m=depth_m,
        spp=spp,
        light_source=light_source,
        env_scale=env_scale,
        inspection_light_scale=inspection_light_scale,
        fov=fov,
        sampler_type=sampler_type,
        reconstruction_filter=reconstruction_filter,
    )
    scene["surface"]["bsdf"] = {
        "type": "kokoro_height_field_reflector",
        "height_source": height_source,
        "width_m": float(width_m),
        "depth_m": float(depth_m),
        "normal_step_m": float(normal_step_m),
        "reflectance": [0.86, 0.88, 0.92],
        "lobe_kappa": float(lobe_kappa),
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
