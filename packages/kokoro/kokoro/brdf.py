from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .height_field import HeightProgram, sample_height_field


@dataclass(frozen=True)
class BrdfDataset:
    features: torch.Tensor
    targets: torch.Tensor
    width_m: float
    depth_m: float
    feature_period_m: float | None = None


@dataclass(frozen=True)
class BrdfTrainingConfig:
    hidden_dim: int = 32
    epochs: int = 80
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 0


@dataclass(frozen=True)
class BrdfTrainingResult:
    model: "KokoroBrdfNet"
    loss_history: list[float]


class KokoroBrdfNet(torch.nn.Module):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(5, hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, 3),
        ])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.layers[0](features))
        x = torch.tanh(self.layers[1](x))
        return torch.nn.functional.normalize(self.layers[2](x), dim=1)


class NpzSurrogate:
    def __init__(self, weights: list[np.ndarray], biases: list[np.ndarray], metadata: dict[str, float]) -> None:
        self.weights = weights
        self.biases = biases
        self.metadata = metadata

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = x @ weight.T + bias
            if index < len(self.weights) - 1:
                x = np.tanh(x)
        norm = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)
        return x / norm


def build_brdf_dataset(
    program: HeightProgram,
    *,
    sample_count: int,
    width_m: float,
    depth_m: float,
    seed: int = 0,
    feature_period_m: float | None = None,
) -> BrdfDataset:
    surface = sample_height_field(program, sample_count=sample_count, width_m=width_m, depth_m=depth_m, seed=seed)
    gen = torch.Generator()
    gen.manual_seed(int(seed) + 31)
    cos_theta = 0.15 + 0.8 * torch.rand(sample_count, generator=gen)
    theta = torch.acos(cos_theta)
    phi = -torch.pi + 2.0 * torch.pi * torch.rand(sample_count, generator=gen)
    wi = angles_to_direction(theta, phi)
    wo = reflect(wi, surface.normals)
    features = make_features(
        surface.positions[:, 0],
        surface.positions[:, 1],
        theta,
        phi,
        width_m,
        depth_m,
        feature_period_m=feature_period_m,
    )
    return BrdfDataset(features=features, targets=wo, width_m=width_m, depth_m=depth_m, feature_period_m=feature_period_m)


def train_brdf_surrogate(dataset: BrdfDataset, config: BrdfTrainingConfig) -> BrdfTrainingResult:
    torch.manual_seed(int(config.seed))
    model = KokoroBrdfNet(hidden_dim=config.hidden_dim)
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
) -> torch.Tensor:
    features = make_features(x, y, theta, phi, width_m, depth_m, feature_period_m=feature_period_m)
    with torch.no_grad():
        return directions_to_angles(model(features))


def export_surrogate_npz(
    model: KokoroBrdfNet,
    path: Path,
    *,
    width_m: float,
    depth_m: float,
    feature_period_m: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"width_m": float(width_m), "depth_m": float(depth_m)}
    if feature_period_m is not None:
        metadata["feature_period_m"] = float(feature_period_m)
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
) -> torch.Tensor:
    sin_theta = torch.sin(theta)
    if feature_period_m is None:
        x_feature = x / (float(width_m) * 0.5)
        y_feature = y / (float(depth_m) * 0.5)
    else:
        if feature_period_m <= 0:
            raise ValueError("feature_period_m must be positive")
        period = float(feature_period_m)
        x_feature = torch.remainder(x, period) / (period * 0.5) - 1.0
        y_feature = torch.remainder(y, period) / (period * 0.5) - 1.0
    return torch.stack([
        x_feature,
        y_feature,
        torch.cos(theta),
        sin_theta * torch.cos(phi),
        sin_theta * torch.sin(phi),
    ], dim=1).to(dtype=torch.float32)


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
    return torch.nn.functional.normalize(2.0 * dot * normal - wi, dim=1)
