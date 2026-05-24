from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .height_field import HeightProgram, sample_height_field, surface_normals


@dataclass(frozen=True)
class BrdfDataset:
    features: torch.Tensor
    targets: torch.Tensor
    width_m: float
    depth_m: float
    feature_period_m: float | None = None
    local_feature_period_m: float | None = None
    position_frequency_count: int = 0
    include_position_features: bool = True
    include_incident_features: bool = True
    radial_cell_feature_period_m: float | None = None
    radial_cell_feature_max_rotation_rad: float = math.pi / 2.0
    radial_cell_feature_radial_power: float = 1.0
    radial_cell_facet_features: bool = False
    average_patch_radius_m: float = 0.0
    average_patch_sample_count: int = 1
    target_mode: str = "reflection"


@dataclass(frozen=True)
class BrdfTrainingConfig:
    hidden_dim: int = 32
    hidden_layer_count: int = 2
    epochs: int = 80
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 0
    activation: str = "tanh"
    omega_0: float = 4.0


@dataclass(frozen=True)
class BrdfTrainingResult:
    model: "KokoroBrdfNet"
    loss_history: list[float]


class KokoroBrdfNet(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int = 32,
        *,
        input_dim: int = 5,
        output_dim: int = 3,
        activation: str = "tanh",
        omega_0: float = 4.0,
        hidden_layer_count: int = 2,
    ) -> None:
        super().__init__()
        if activation not in {"sine", "tanh"}:
            raise ValueError("activation must be 'sine' or 'tanh'")
        if output_dim < 3:
            raise ValueError("output_dim must be at least 3")
        if not 1 <= int(hidden_layer_count) <= 5:
            raise ValueError("hidden_layer_count must be between 1 and 5")
        self.activation = activation
        self.omega_0 = float(omega_0)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_layer_count = int(hidden_layer_count)
        layers = [torch.nn.Linear(self.input_dim, hidden_dim)]
        for _ in range(self.hidden_layer_count - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        layers.append(torch.nn.Linear(hidden_dim, self.output_dim))
        self.layers = torch.nn.ModuleList(layers)
        if activation == "sine":
            self._init_siren_layers()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features
        for layer in self.layers[:-1]:
            x = self._activate(layer(x))
        raw = self.layers[-1](x)
        direction = torch.nn.functional.normalize(raw[:, :3], dim=1)
        if self.output_dim == 3:
            return direction
        cone_cos = torch.sigmoid(raw[:, 3:4])
        if self.output_dim < 6:
            return torch.cat([direction, cone_cos, torch.sigmoid(raw[:, 4:])], dim=1)
        phase = torch.nn.functional.normalize(raw[:, 4:6], dim=1)
        return torch.cat([direction, cone_cos, phase, torch.sigmoid(raw[:, 6:])], dim=1)

    def _activate(self, value: torch.Tensor) -> torch.Tensor:
        if self.activation == "sine":
            return torch.sin(self.omega_0 * value)
        return torch.tanh(value)

    def _init_siren_layers(self) -> None:
        with torch.no_grad():
            first = self.layers[0]
            first.weight.uniform_(-1.0 / first.in_features, 1.0 / first.in_features)
            for layer in self.layers[1:]:
                bound = np.sqrt(6.0 / layer.in_features) / self.omega_0
                layer.weight.uniform_(-bound, bound)


class NpzSurrogate:
    def __init__(self, weights: list[np.ndarray], biases: list[np.ndarray], metadata: dict[str, object]) -> None:
        self.weights = weights
        self.biases = biases
        self.metadata = metadata

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        activation = self.metadata.get("activation", "tanh")
        omega_0 = float(self.metadata.get("omega_0", 12.0))
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = x @ weight.T + bias
            if index < len(self.weights) - 1:
                x = np.sin(omega_0 * x) if activation == "sine" else np.tanh(x)
        direction = x[:, :3] / np.linalg.norm(x[:, :3], axis=1, keepdims=True).clip(min=1e-8)
        if x.shape[1] == 3:
            return direction
        cone_cos = 1.0 / (1.0 + np.exp(-x[:, 3:4]))
        if x.shape[1] < 6:
            extras = 1.0 / (1.0 + np.exp(-x[:, 4:]))
            return np.concatenate([direction, cone_cos, extras], axis=1)
        phase = x[:, 4:6] / np.linalg.norm(x[:, 4:6], axis=1, keepdims=True).clip(min=1e-8)
        extras = 1.0 / (1.0 + np.exp(-x[:, 6:]))
        return np.concatenate([direction, cone_cos, phase, extras], axis=1)


def build_brdf_dataset(
    program: HeightProgram,
    *,
    sample_count: int,
    width_m: float,
    depth_m: float,
    seed: int = 0,
    feature_period_m: float | None = None,
    local_feature_period_m: float | None = None,
    position_frequency_count: int = 0,
    include_position_features: bool = True,
    include_incident_features: bool | None = None,
    radial_cell_feature_period_m: float | None = None,
    radial_cell_feature_max_rotation_rad: float = math.pi / 2.0,
    radial_cell_feature_radial_power: float = 1.0,
    radial_cell_facet_features: bool = False,
    average_patch_radius_m: float = 0.0,
    average_patch_sample_count: int = 1,
    target_mode: str = "reflection",
) -> BrdfDataset:
    if target_mode not in {"reflection", "normal"}:
        raise ValueError("target_mode must be 'reflection' or 'normal'")
    surface = sample_height_field(program, sample_count=sample_count, width_m=width_m, depth_m=depth_m, seed=seed)
    gen = torch.Generator()
    gen.manual_seed(int(seed) + 31)
    cos_theta = 0.15 + 0.8 * torch.rand(sample_count, generator=gen)
    theta = torch.acos(cos_theta)
    phi = -torch.pi + 2.0 * torch.pi * torch.rand(sample_count, generator=gen)
    wi = angles_to_direction(theta, phi)
    uses_incident = target_mode == "reflection" if include_incident_features is None else bool(include_incident_features)
    if target_mode == "normal":
        wo = surface.normals
    elif average_patch_sample_count > 1 or average_patch_radius_m > 0.0:
        wo = patch_reflection_moments(
            program,
            surface.positions[:, 0],
            surface.positions[:, 1],
            wi,
            patch_radius_m=average_patch_radius_m,
            patch_sample_count=average_patch_sample_count,
            seed=int(seed) + 53,
        )
    else:
        wo = reflect(wi, surface.normals)
    features = make_features(
        surface.positions[:, 0],
        surface.positions[:, 1],
        theta,
        phi,
        width_m,
        depth_m,
        feature_period_m=feature_period_m,
        local_feature_period_m=local_feature_period_m,
        position_frequency_count=position_frequency_count,
        include_position_features=include_position_features,
        include_incident_features=uses_incident,
        radial_cell_feature_period_m=radial_cell_feature_period_m,
        radial_cell_feature_max_rotation_rad=radial_cell_feature_max_rotation_rad,
        radial_cell_feature_radial_power=radial_cell_feature_radial_power,
        radial_cell_facet_features=radial_cell_facet_features,
    )
    return BrdfDataset(
        features=features,
        targets=wo,
        width_m=width_m,
        depth_m=depth_m,
        feature_period_m=feature_period_m,
        local_feature_period_m=local_feature_period_m,
        position_frequency_count=position_frequency_count,
        include_position_features=include_position_features,
        include_incident_features=uses_incident,
        radial_cell_feature_period_m=radial_cell_feature_period_m,
        radial_cell_feature_max_rotation_rad=radial_cell_feature_max_rotation_rad,
        radial_cell_feature_radial_power=radial_cell_feature_radial_power,
        radial_cell_facet_features=radial_cell_facet_features,
        average_patch_radius_m=average_patch_radius_m,
        average_patch_sample_count=average_patch_sample_count,
        target_mode=target_mode,
    )


def train_brdf_surrogate(dataset: BrdfDataset, config: BrdfTrainingConfig) -> BrdfTrainingResult:
    torch.manual_seed(int(config.seed))
    model = KokoroBrdfNet(
        hidden_dim=config.hidden_dim,
        hidden_layer_count=config.hidden_layer_count,
        input_dim=dataset.features.shape[1],
        output_dim=dataset.targets.shape[1],
        activation=config.activation,
        omega_0=config.omega_0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(config.lr))
    losses: list[float] = []
    count = dataset.features.shape[0]
    gen = torch.Generator()
    gen.manual_seed(int(config.seed) + 97)
    for _ in range(int(config.epochs)):
        perm = torch.randperm(count, generator=gen)
        epoch_loss = 0.0
        for start in range(0, count, int(config.batch_size)):
            idx = perm[start:start + int(config.batch_size)]
            pred = model(dataset.features[idx])
            loss = torch.nn.functional.mse_loss(pred, dataset.targets[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        losses.append(epoch_loss / count)
    return BrdfTrainingResult(model=model.eval(), loss_history=losses)


def predict_outgoing_angles(
    model: KokoroBrdfNet,
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
    width_m: float,
    depth_m: float,
    feature_period_m: float | None = None,
    local_feature_period_m: float | None = None,
    position_frequency_count: int = 0,
    include_position_features: bool = True,
    include_incident_features: bool = True,
    radial_cell_feature_period_m: float | None = None,
    radial_cell_feature_max_rotation_rad: float = math.pi / 2.0,
    radial_cell_feature_radial_power: float = 1.0,
    radial_cell_facet_features: bool = False,
) -> torch.Tensor:
    features = make_features(
        x,
        y,
        theta,
        phi,
        width_m,
        depth_m,
        feature_period_m=feature_period_m,
        local_feature_period_m=local_feature_period_m,
        position_frequency_count=position_frequency_count,
        include_position_features=include_position_features,
        include_incident_features=include_incident_features,
        radial_cell_feature_period_m=radial_cell_feature_period_m,
        radial_cell_feature_max_rotation_rad=radial_cell_feature_max_rotation_rad,
        radial_cell_feature_radial_power=radial_cell_feature_radial_power,
        radial_cell_facet_features=radial_cell_facet_features,
    )
    with torch.no_grad():
        return directions_to_angles(model(features)[:, :3])


def export_surrogate_npz(
    model: KokoroBrdfNet,
    path: Path,
    *,
    width_m: float,
    depth_m: float,
    feature_period_m: float | None = None,
    local_feature_period_m: float | None = None,
    position_frequency_count: int = 0,
    include_position_features: bool | None = None,
    include_incident_features: bool = True,
    radial_cell_feature_period_m: float | None = None,
    radial_cell_feature_max_rotation_rad: float = math.pi / 2.0,
    radial_cell_feature_radial_power: float = 1.0,
    radial_cell_facet_features: bool = False,
    average_patch_radius_m: float = 0.0,
    average_patch_sample_count: int = 1,
    target_mode: str = "reflection",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if target_mode not in {"reflection", "normal"}:
        raise ValueError("target_mode must be 'reflection' or 'normal'")
    incident_dim = 3 if include_incident_features else 0
    uses_position = model.input_dim > incident_dim if include_position_features is None else bool(include_position_features)
    metadata = {
        "width_m": float(width_m),
        "depth_m": float(depth_m),
        "activation": model.activation,
        "omega_0": float(model.omega_0),
        "position_frequency_count": int(position_frequency_count),
        "hidden_layer_count": int(model.hidden_layer_count),
        "include_position_features": uses_position,
        "include_incident_features": bool(include_incident_features),
        "input_dim": int(model.input_dim),
        "output_dim": int(model.output_dim),
        "average_patch_radius_m": float(average_patch_radius_m),
        "average_patch_sample_count": int(average_patch_sample_count),
        "target_mode": target_mode,
    }
    if feature_period_m is not None:
        metadata["feature_period_m"] = float(feature_period_m)
    if local_feature_period_m is not None:
        metadata["local_feature_period_m"] = float(local_feature_period_m)
    if radial_cell_feature_period_m is not None:
        metadata["radial_cell_feature_period_m"] = float(radial_cell_feature_period_m)
        metadata["radial_cell_feature_max_rotation_rad"] = float(radial_cell_feature_max_rotation_rad)
        metadata["radial_cell_feature_radial_power"] = float(radial_cell_feature_radial_power)
        metadata["radial_cell_facet_features"] = bool(radial_cell_facet_features)
    arrays: dict[str, object] = {
        "metadata": json.dumps(metadata),
        "layer_count": np.asarray([len(model.layers)], dtype=np.int32),
    }
    for index, layer in enumerate(model.layers):
        arrays[f"weight_{index}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"bias_{index}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    np.savez(path, **arrays)


def load_npz_surrogate(path: Path) -> NpzSurrogate:
    with np.load(path, allow_pickle=False) as data:
        layer_count = int(data["layer_count"][0])
        weights = [data[f"weight_{index}"].astype(np.float32) for index in range(layer_count)]
        biases = [data[f"bias_{index}"].astype(np.float32) for index in range(layer_count)]
        metadata = json.loads(str(data["metadata"]))
    return NpzSurrogate(weights=weights, biases=biases, metadata=metadata)


def make_features(
    x: torch.Tensor,
    y: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
    width_m: float,
    depth_m: float,
    feature_period_m: float | None = None,
    local_feature_period_m: float | None = None,
    position_frequency_count: int = 0,
    include_position_features: bool = True,
    include_incident_features: bool = True,
    radial_cell_feature_period_m: float | None = None,
    radial_cell_feature_max_rotation_rad: float = math.pi / 2.0,
    radial_cell_feature_radial_power: float = 1.0,
    radial_cell_facet_features: bool = False,
) -> torch.Tensor:
    sin_theta = torch.sin(theta)
    incident = [torch.cos(theta), sin_theta * torch.cos(phi), sin_theta * torch.sin(phi)]
    if not include_position_features:
        if not include_incident_features:
            raise ValueError("at least one feature group must be enabled")
        return torch.stack(incident, dim=1).to(dtype=torch.float32)
    if feature_period_m is None:
        x_feature = x / (float(width_m) * 0.5)
        y_feature = y / (float(depth_m) * 0.5)
    else:
        if feature_period_m <= 0:
            raise ValueError("feature_period_m must be positive")
        period = float(feature_period_m)
        x_feature = torch.remainder(x, period) / (period * 0.5) - 1.0
        y_feature = torch.remainder(y, period) / (period * 0.5) - 1.0
    encoded = [x_feature, y_feature]
    if local_feature_period_m is not None:
        if local_feature_period_m <= 0:
            raise ValueError("local_feature_period_m must be positive")
        local_period = float(local_feature_period_m)
        encoded.extend([
            torch.remainder(x, local_period) / (local_period * 0.5) - 1.0,
            torch.remainder(y, local_period) / (local_period * 0.5) - 1.0,
        ])
    if radial_cell_feature_period_m is not None:
        encoded.extend(_radial_cell_features(
            x,
            y,
            width_m=width_m,
            depth_m=depth_m,
            period_m=radial_cell_feature_period_m,
            max_rotation_rad=radial_cell_feature_max_rotation_rad,
            radial_power=radial_cell_feature_radial_power,
            include_facet_features=radial_cell_facet_features,
        ))
    for index in range(max(0, int(position_frequency_count))):
        frequency = float(2 ** index) * torch.pi
        encoded.extend([
            torch.sin(frequency * x_feature),
            torch.cos(frequency * x_feature),
            torch.sin(frequency * y_feature),
            torch.cos(frequency * y_feature),
        ])
    values = [*encoded, *incident] if include_incident_features else encoded
    return torch.stack(values, dim=1).to(dtype=torch.float32)


def _radial_cell_features(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    width_m: float,
    depth_m: float,
    period_m: float,
    max_rotation_rad: float,
    radial_power: float,
    include_facet_features: bool = False,
) -> list[torch.Tensor]:
    if period_m <= 0:
        raise ValueError("radial_cell_feature_period_m must be positive")
    if radial_power <= 0:
        raise ValueError("radial_cell_feature_radial_power must be positive")
    period = float(period_m)
    center_x = torch.floor(x / period + 0.5) * period
    center_y = torch.floor(y / period + 0.5) * period
    local_x = x - center_x
    local_y = y - center_y
    radius = torch.sqrt(center_x * center_x + center_y * center_y)
    max_radius = math.sqrt((float(width_m) * 0.5) ** 2 + (float(depth_m) * 0.5) ** 2)
    angle = float(max_rotation_rad) * radius / max_radius ** float(radial_power)
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    rotated_x = (cos_angle * local_x + sin_angle * local_y) / (period * 0.5)
    rotated_y = (-sin_angle * local_x + cos_angle * local_y) / (period * 0.5)
    features = [rotated_x, rotated_y, sin_angle, cos_angle]
    if include_facet_features:
        sign_x = torch.where(rotated_x >= 0.0, torch.ones_like(rotated_x), -torch.ones_like(rotated_x))
        sign_y = torch.where(rotated_y >= 0.0, torch.ones_like(rotated_y), -torch.ones_like(rotated_y))
        x_dominant = torch.abs(rotated_x) >= torch.abs(rotated_y)
        facet_slope_x = torch.where(x_dominant, -sign_x * cos_angle, sign_y * sin_angle)
        facet_slope_y = torch.where(x_dominant, -sign_x * sin_angle, -sign_y * cos_angle)
        features.extend([facet_slope_x, facet_slope_y])
    return features


def average_patch_reflections(
    program: HeightProgram,
    x: torch.Tensor,
    y: torch.Tensor,
    wi: torch.Tensor,
    *,
    patch_radius_m: float,
    patch_sample_count: int,
    seed: int = 0,
) -> torch.Tensor:
    return patch_reflection_moments(
        program,
        x,
        y,
        wi,
        patch_radius_m=patch_radius_m,
        patch_sample_count=patch_sample_count,
        seed=seed,
    )[:, :3]


def patch_reflection_moments(
    program: HeightProgram,
    x: torch.Tensor,
    y: torch.Tensor,
    wi: torch.Tensor,
    *,
    patch_radius_m: float,
    patch_sample_count: int,
    seed: int = 0,
) -> torch.Tensor:
    if patch_sample_count <= 0:
        raise ValueError("patch_sample_count must be positive")
    if patch_radius_m < 0:
        raise ValueError("patch_radius_m must be non-negative")
    count = x.shape[0]
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))
    if patch_radius_m == 0.0:
        dx = torch.zeros((count, int(patch_sample_count)), dtype=x.dtype, device=x.device)
        dy = torch.zeros_like(dx)
    else:
        radius = float(patch_radius_m)
        dx = (torch.rand((count, int(patch_sample_count)), generator=gen, dtype=x.dtype, device=x.device) * 2.0 - 1.0) * radius
        dy = (torch.rand((count, int(patch_sample_count)), generator=gen, dtype=x.dtype, device=x.device) * 2.0 - 1.0) * radius
    sample_x = (x[:, None] + dx).reshape(-1)
    sample_y = (y[:, None] + dy).reshape(-1)
    normals = surface_normals(program, sample_x, sample_y)
    repeated_wi = wi[:, None, :].expand(count, int(patch_sample_count), 3).reshape(-1, 3)
    reflected = reflect(repeated_wi, normals).reshape(count, int(patch_sample_count), 3)
    mean = reflected.mean(dim=1)
    length = torch.linalg.vector_norm(mean, dim=1, keepdim=True).clamp(min=1e-6, max=1.0)
    axis = mean / length
    tangent = _stable_tangent(axis)
    bitangent = torch.linalg.cross(axis, tangent, dim=1)
    u = torch.sum(reflected * tangent[:, None, :], dim=2)
    v = torch.sum(reflected * bitangent[:, None, :], dim=2)
    phase = 4.0 * torch.atan2(v, u)
    phase_vector = torch.stack([torch.cos(phase).mean(dim=1), torch.sin(phase).mean(dim=1)], dim=1)
    phase_norm = torch.linalg.vector_norm(phase_vector, dim=1, keepdim=True)
    fallback = torch.tensor([1.0, 0.0], dtype=phase_vector.dtype, device=phase_vector.device).expand_as(phase_vector)
    phase_vector = torch.where(phase_norm > 1e-6, phase_vector / phase_norm.clamp(min=1e-6), fallback)
    return torch.cat([axis, length, phase_vector], dim=1)


def _stable_tangent(axis: torch.Tensor) -> torch.Tensor:
    z_cross = torch.stack([-axis[:, 1], axis[:, 0], torch.zeros_like(axis[:, 0])], dim=1)
    z_cross = torch.nn.functional.normalize(z_cross, dim=1)
    x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=axis.dtype, device=axis.device).expand_as(axis)
    use_x = torch.abs(axis[:, 2:3]) > 0.9
    return torch.where(use_x, x_axis, z_cross)


def angles_to_direction(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    sin_theta = torch.sin(theta)
    return torch.stack([sin_theta * torch.cos(phi), sin_theta * torch.sin(phi), torch.cos(theta)], dim=1)


def directions_to_angles(direction: torch.Tensor) -> torch.Tensor:
    unit = torch.nn.functional.normalize(direction, dim=1)
    theta = torch.acos(torch.clamp(unit[:, 2], -1.0, 1.0))
    phi = torch.atan2(unit[:, 1], unit[:, 0])
    return torch.stack([theta, phi], dim=1)


def reflect(wi: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(wi * normal, dim=1, keepdim=True)
    reflected = 2.0 * dot * normal - wi
    reflected = torch.stack([reflected[:, 0], reflected[:, 1], torch.abs(reflected[:, 2])], dim=1)
    return torch.nn.functional.normalize(reflected, dim=1)
