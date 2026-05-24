from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .brdf import (
    BrdfDataset,
    KokoroBrdfNet,
    angles_to_direction,
    make_features,
    reflect,
)
from .height_field import HeightProgram, surface_normals


@dataclass(frozen=True)
class DirectionHoldoutConfig:
    x_count: int = 32
    y_count: int = 32
    theta_count: int = 5
    phi_count: int = 8
    min_cos_theta: float = 0.15
    max_cos_theta: float = 0.95


def angular_error_degrees(model: KokoroBrdfNet, features: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        predicted = torch.nn.functional.normalize(model(features)[:, :3], dim=1)
        target = torch.nn.functional.normalize(targets[:, :3], dim=1)
    cosine = torch.clamp(torch.sum(predicted * target, dim=1), -1.0, 1.0)
    error = torch.acos(cosine) * (180.0 / torch.pi)
    return {
        "mean_deg": float(error.mean()),
        "p95_deg": float(torch.quantile(error, 0.95)),
        "p99_deg": float(torch.quantile(error, 0.99)),
        "max_deg": float(error.max()),
    }


def build_direction_holdout_dataset(
    program: HeightProgram,
    config: DirectionHoldoutConfig,
    *,
    width_m: float,
    depth_m: float,
    feature_period_m: float | None = None,
    local_feature_period_m: float | None = None,
    dft_phase_vectors: tuple[tuple[float, float], ...] = (),
    position_frequency_count: int = 0,
    include_position_features: bool = True,
    include_incident_features: bool = True,
    radial_cell_feature_period_m: float | None = None,
    radial_cell_feature_max_rotation_rad: float = math.pi / 2.0,
    radial_cell_feature_radial_power: float = 1.0,
    radial_cell_facet_features: bool = False,
    normal_step_m: float = 25e-6,
    target_mode: str = "reflection",
) -> BrdfDataset:
    for name, value in (
        ("x_count", config.x_count),
        ("y_count", config.y_count),
        ("theta_count", config.theta_count),
        ("phi_count", config.phi_count),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 < config.min_cos_theta < config.max_cos_theta <= 1.0:
        raise ValueError("cosine theta bounds must satisfy 0 < min < max <= 1")

    xs = torch.linspace(-float(width_m) * 0.5, float(width_m) * 0.5, int(config.x_count))
    ys = torch.linspace(-float(depth_m) * 0.5, float(depth_m) * 0.5, int(config.y_count))
    cos_theta = torch.linspace(float(config.min_cos_theta), float(config.max_cos_theta), int(config.theta_count))
    theta_values = torch.acos(cos_theta)
    phi_values = torch.linspace(-torch.pi, torch.pi, int(config.phi_count) + 1)[:-1]
    yy, xx, theta, phi = torch.meshgrid(ys, xs, theta_values, phi_values, indexing="ij")
    x = xx.reshape(-1)
    y = yy.reshape(-1)
    theta = theta.reshape(-1)
    phi = phi.reshape(-1)
    wi = angles_to_direction(theta, phi)
    normals = surface_normals(program, x, y, step_m=normal_step_m)
    targets = normals if target_mode == "normal" else reflect(wi, normals)
    features = make_features(
        x,
        y,
        theta,
        phi,
        width_m,
        depth_m,
        feature_period_m=feature_period_m,
        local_feature_period_m=local_feature_period_m,
        dft_phase_vectors=dft_phase_vectors,
        position_frequency_count=position_frequency_count,
        include_position_features=include_position_features,
        include_incident_features=include_incident_features,
        radial_cell_feature_period_m=radial_cell_feature_period_m,
        radial_cell_feature_max_rotation_rad=radial_cell_feature_max_rotation_rad,
        radial_cell_feature_radial_power=radial_cell_feature_radial_power,
        radial_cell_facet_features=radial_cell_facet_features,
    )
    return BrdfDataset(
        features=features,
        targets=targets,
        width_m=width_m,
        depth_m=depth_m,
        feature_period_m=feature_period_m,
        local_feature_period_m=local_feature_period_m,
        dft_phase_vectors=dft_phase_vectors,
        position_frequency_count=position_frequency_count,
        include_position_features=include_position_features,
        include_incident_features=include_incident_features,
        radial_cell_feature_period_m=radial_cell_feature_period_m,
        radial_cell_feature_max_rotation_rad=radial_cell_feature_max_rotation_rad,
        radial_cell_feature_radial_power=radial_cell_feature_radial_power,
        radial_cell_facet_features=radial_cell_facet_features,
        normal_step_m=float(normal_step_m),
        target_mode=target_mode,
    )
