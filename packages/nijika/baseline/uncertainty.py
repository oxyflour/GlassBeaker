from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from baseline.training_utils import forward_model


def _enable_dropout(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_predict(
    model: nn.Module,
    *,
    points: torch.Tensor,
    ports: torch.Tensor,
    geom: torch.Tensor,
    frame: torch.Tensor,
    cuts: torch.Tensor,
    nibs: torch.Tensor,
    device: torch.device,
    n_samples: int = 20,
    graph_tensors: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MC dropout inference returning (mean, std) per output element."""
    _enable_dropout(model)
    samples = []
    for _ in range(n_samples):
        pred = forward_model(
            model,
            points=points, ports=ports, geom=geom,
            frame=frame, cuts=cuts, nibs=nibs,
            device=device, graph_tensors=graph_tensors,
        )
        samples.append(pred)
    stacked = torch.stack(samples, dim=0)  # (n, batch, freq, ports*ports*2)
    return stacked.mean(dim=0), stacked.std(dim=0)


@torch.no_grad()
def mc_predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MC dropout over a DataLoader. Returns (mean, std, all_samples)."""
    _enable_dropout(model)
    all_means = []
    all_stds = []
    for batch in loader:
        points, ports, geom, frame, cuts, nibs, target, *graph_extra = batch
        graph_tensors = None
        if graph_extra:
            from baseline.training_utils import GRAPH_KEYS
            graph_tensors = {key: tensor for key, tensor in zip(GRAPH_KEYS, graph_extra, strict=False)}
        mean, std = mc_predict(
            model,
            points=points, ports=ports, geom=geom,
            frame=frame, cuts=cuts, nibs=nibs,
            device=device, n_samples=n_samples, graph_tensors=graph_tensors,
        )
        all_means.append(mean.cpu())
        all_stds.append(std.cpu())
    return torch.cat(all_means, dim=0), torch.cat(all_stds, dim=0), target


def uncertainty_summary(
    mean: torch.Tensor,
    std: torch.Tensor,
    port_count: int,
) -> dict[str, float]:
    """Aggregate uncertainty metrics from MC dropout results."""
    pair_std = std.view(std.size(0), std.size(1), port_count, port_count, 2)
    total_std = torch.linalg.vector_norm(pair_std, dim=-1)  # std of magnitude
    mean_mag = torch.linalg.vector_norm(mean.view(mean.size(0), mean.size(1), port_count, port_count, 2), dim=-1) + 1e-6
    rel_std = total_std / mean_mag
    return {
        "mean_abs_std": float(total_std.mean().item()),
        "mean_rel_std": float(rel_std.mean().item()),
        "max_rel_std": float(rel_std.max().item()),
        "mean_std_db": float((20.0 * torch.log10(total_std.clamp_min(1e-6) + 1e-6)).mean().item()),
    }
