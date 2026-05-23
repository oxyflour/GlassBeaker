from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch


@dataclass(frozen=True)
class HeightProgram:
    source: str
    height: Callable[[torch.Tensor, torch.Tensor], Any]

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = self.height(x, y)
        z_tensor = torch.as_tensor(z, dtype=x.dtype, device=x.device)
        if z_tensor.shape == ():
            return torch.zeros_like(x) + z_tensor
        return torch.broadcast_to(z_tensor, x.shape)


@dataclass(frozen=True)
class HeightSamples:
    positions: torch.Tensor
    normals: torch.Tensor


def compile_height_program(source: str) -> HeightProgram:
    namespace: dict[str, Any] = {
        "math": math,
        "np": np,
        "numpy": np,
        "pyramid_height": pyramid_height,
        "torch": torch,
    }
    exec(compile(source, "<kokoro-height>", "exec"), namespace)
    height = namespace.get("height")
    if not callable(height):
        raise ValueError("height source must define a callable height(x, y)")
    return HeightProgram(source=source, height=height)


def pyramid_height(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    period_m: float = 500e-6,
    amplitude_m: float = 150e-6,
) -> torch.Tensor:
    if period_m <= 0:
        raise ValueError("period_m must be positive")
    u = torch.remainder(x, period_m) / period_m
    v = torch.remainder(y, period_m) / period_m
    edge_distance = torch.maximum(torch.abs(u - 0.5), torch.abs(v - 0.5))
    return float(amplitude_m) * torch.clamp(1.0 - 2.0 * edge_distance, min=0.0)


def sample_height_field(
    program: HeightProgram,
    *,
    sample_count: int,
    width_m: float,
    depth_m: float,
    seed: int = 0,
    normal_step_m: float = 25e-6,
) -> HeightSamples:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    x = (torch.rand(sample_count, generator=gen) - 0.5) * float(width_m)
    y = (torch.rand(sample_count, generator=gen) - 0.5) * float(depth_m)
    z = program.evaluate(x, y)
    normals = surface_normals(program, x, y, step_m=normal_step_m)
    return HeightSamples(positions=torch.stack([x, y, z], dim=1), normals=normals)


def surface_normals(
    program: HeightProgram,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    step_m: float = 25e-6,
) -> torch.Tensor:
    step = torch.as_tensor(float(step_m), dtype=x.dtype, device=x.device)
    dzdx = (program.evaluate(x + step, y) - program.evaluate(x - step, y)) / (2.0 * step)
    dzdy = (program.evaluate(x, y + step) - program.evaluate(x, y - step)) / (2.0 * step)
    normals = torch.stack([-dzdx, -dzdy, torch.ones_like(dzdx)], dim=1)
    return torch.nn.functional.normalize(normals, dim=1)
