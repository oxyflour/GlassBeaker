from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from .direction_metrics import (
    DirectionHoldoutConfig,
    angular_error_degrees,
    build_direction_holdout_dataset,
)
from .height_field import HeightProgram
from .mitsuba_scene import build_kokoro_scene_dict


@dataclass(frozen=True)
class ValidationArtifacts:
    root: Path

    @property
    def flat_neural(self) -> Path:
        return self.root / "flat_neural.png"

    @property
    def flat_mirror(self) -> Path:
        return self.root / "flat_mirror.png"

    @property
    def flat_diff(self) -> Path:
        return self.root / "flat_absdiff.png"

    @property
    def pyramid_neural(self) -> Path:
        return self.root / "pyramid_neural.png"

    @property
    def pyramid_ply(self) -> Path:
        return self.root / "pyramid_ply.png"

    @property
    def pyramid_diff(self) -> Path:
        return self.root / "pyramid_absdiff.png"

    @property
    def pyramid_ply_mesh(self) -> Path:
        return self.root / "pyramid_validation_surface.ply"

    @property
    def metrics(self) -> Path:
        return self.root / "validation_metrics.json"


def image_metrics(
    reference: Path,
    candidate: Path,
    diff_path: Path,
    *,
    attempts: int = 20,
    delay_s: float = 0.05,
) -> dict[str, float]:
    ref = np.asarray(_open_rgb_with_retry(reference, attempts=attempts, delay_s=delay_s), dtype=np.float32)
    cand = np.asarray(_open_rgb_with_retry(candidate, attempts=attempts, delay_s=delay_s), dtype=np.float32)
    diff = np.abs(cand - ref)
    Image.fromarray(np.asarray(np.clip(diff * 4.0, 0, 255), dtype=np.uint8), mode="RGB").save(diff_path)
    return {
        "mean_abs_255": float(diff.mean()),
        "rmse_255": float(np.sqrt(np.mean((cand - ref) ** 2))),
        "max_abs_255": float(diff.max()),
    }


def mirror_plane_scene(scene: dict[str, Any]) -> dict[str, Any]:
    converted = _clone_scene(scene)
    converted["surface"]["bsdf"] = {"type": "conductor", "material": "Ag"}
    return converted


def neural_plane_scene(
    checkpoint: Path,
    hdr_path: Path,
    *,
    width: int,
    height: int,
    spp: int,
    lobe_kappa: float,
) -> dict[str, Any]:
    scene = build_kokoro_scene_dict(checkpoint_path=checkpoint, hdr_path=hdr_path, width=width, height=height, spp=spp)
    scene["surface"]["bsdf"]["lobe_kappa"] = float(lobe_kappa)
    return scene


def pyramid_ply_scene(
    program: HeightProgram,
    hdr_path: Path,
    mesh_path: Path,
    *,
    width: int,
    height: int,
    grid_size: int,
    width_m: float = 0.10,
    depth_m: float = 0.10,
    spp: int = 64,
) -> dict[str, Any]:
    write_height_ply(program, mesh_path, grid_size=grid_size, width_m=width_m, depth_m=depth_m)
    scene = build_kokoro_scene_dict(checkpoint_path=Path("unused.npz"), hdr_path=hdr_path, width=width, height=height, spp=spp)
    scene["surface"] = {
        "type": "ply",
        "filename": str(mesh_path),
        "bsdf": {"type": "conductor", "material": "Ag"},
    }
    return scene


def write_height_ply(
    program: HeightProgram,
    path: Path,
    *,
    grid_size: int,
    width_m: float,
    depth_m: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = torch.linspace(-width_m * 0.5, width_m * 0.5, grid_size)
    ys = torch.linspace(-depth_m * 0.5, depth_m * 0.5, grid_size)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    zz = program.evaluate(xx.reshape(-1), yy.reshape(-1)).reshape(grid_size, grid_size)
    vertices = [
        (float(xx[row, col]), float(yy[row, col]), float(zz[row, col]))
        for row in range(grid_size)
        for col in range(grid_size)
    ]
    faces = []
    for row in range(grid_size - 1):
        for col in range(grid_size - 1):
            a = row * grid_size + col
            b = a + 1
            c = a + grid_size
            d = c + 1
            faces.extend([(a, b, c), (b, d, c)])
    _write_ply(path, vertices, faces)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_ply(path: Path, vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]]) -> None:
    lines = [
        "ply", "format ascii 1.0", f"element vertex {len(vertices)}",
        "property float x", "property float y", "property float z",
        f"element face {len(faces)}", "property list uchar int vertex_indices", "end_header",
    ]
    lines.extend(f"{x:.9g} {y:.9g} {z:.9g}" for x, y, z in vertices)
    lines.extend(f"3 {a} {b} {c}" for a, b, c in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _clone_scene(scene: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(scene))


def _open_rgb_with_retry(path: Path, *, attempts: int, delay_s: float) -> Image.Image:
    last_error: Exception | None = None
    for attempt in range(max(1, int(attempts))):
        try:
            with Image.open(path) as image:
                return image.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(float(delay_s))
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(path)
