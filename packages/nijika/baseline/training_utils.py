from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from baseline.metrics import summarize_prediction_metrics


def _weighted_mean(error: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    expanded = weight.expand_as(error)
    return (error * expanded).sum() / expanded.sum().clamp_min(1e-6)


def _pair_weight(port_count: int, coupling_weight: float, ref: torch.Tensor) -> torch.Tensor:
    weight = torch.ones((1, 1, port_count, port_count, 1), dtype=ref.dtype, device=ref.device)
    if coupling_weight != 1.0:
        eye = torch.eye(port_count, dtype=torch.bool, device=ref.device).view(1, 1, port_count, port_count, 1)
        weight = torch.where(eye, weight, weight * coupling_weight)
    return weight


def _magnitude_db(values: torch.Tensor) -> torch.Tensor:
    mag = torch.linalg.vector_norm(values, dim=-1).clamp_min(1e-6)
    return 20.0 * torch.log10(mag)


def reciprocity_loss(pred: torch.Tensor, port_count: int) -> torch.Tensor:
    """Penalize asymmetry in S-matrix: S_ij should equal S_ji."""
    # pred shape: (batch, freq, port_count, port_count, 2)
    pred_pair = pred.view(pred.size(0), pred.size(1), port_count, port_count, 2)
    # Compute S_ij - S_ji for all i, j
    diff = pred_pair - pred_pair.transpose(2, 3)
    return diff.pow(2).mean()


def passivity_loss(pred: torch.Tensor, port_count: int) -> torch.Tensor:
    """Penalize S-parameters with magnitude > 1 (not passive)."""
    # pred shape: (batch, freq, port_count, port_count, 2)
    pred_pair = pred.view(pred.size(0), pred.size(1), port_count, port_count, 2)
    mag = torch.linalg.vector_norm(pred_pair, dim=-1)
    # Penalty for |S| > 1
    violation = (mag - 1.0).clamp_min(0.0)
    return violation.pow(2).mean()


def composite_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    port_count: int,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    loss_config: dict[str, float],
) -> torch.Tensor:
    pred_pair = pred.view(pred.size(0), pred.size(1), port_count, port_count, 2)
    target_pair = target.view(target.size(0), target.size(1), port_count, port_count, 2)
    pair_weight = _pair_weight(port_count, float(loss_config["coupling_weight"]), pred_pair)
    ri_loss = _weighted_mean((pred_pair - target_pair).pow(2), pair_weight)
    pred_mag = torch.linalg.vector_norm(pred_pair, dim=-1)
    target_mag = torch.linalg.vector_norm(target_pair, dim=-1)
    mag_loss = _weighted_mean((pred_mag - target_mag).abs(), pair_weight.squeeze(-1))
    slope_error = (pred_pair[:, 1:] - pred_pair[:, :-1] - (target_pair[:, 1:] - target_pair[:, :-1])).pow(2)
    slope_loss = _weighted_mean(slope_error, pair_weight)
    db_loss = pred.new_tensor(0.0)
    if loss_config["db_weight"] > 0.0:
        shape = (1, 1, port_count, port_count, 2)
        pred_real = pred_pair * target_std.view(shape) + target_mean.view(shape)
        target_real = target_pair * target_std.view(shape) + target_mean.view(shape)
        pred_db = _magnitude_db(pred_real)
        target_db = _magnitude_db(target_real)
        notch_scale = 1.0 + loss_config["notch_weight"] * (
            (loss_config["notch_threshold_db"] - target_db).clamp_min(0.0) / 20.0
        ).clamp_max(1.0)
        db_loss = _weighted_mean((pred_db - target_db).abs(), pair_weight.squeeze(-1) * notch_scale)
    # Physics-informed losses
    phys_loss = pred.new_tensor(0.0)
    if loss_config.get("reciprocity_weight", 0.0) > 0.0:
        phys_loss = phys_loss + loss_config["reciprocity_weight"] * reciprocity_loss(pred, port_count)
    if loss_config.get("passivity_weight", 0.0) > 0.0:
        phys_loss = phys_loss + loss_config["passivity_weight"] * passivity_loss(pred, port_count)

    return (
        ri_loss
        + loss_config["mag_weight"] * mag_loss
        + loss_config["smooth_weight"] * slope_loss
        + loss_config["db_weight"] * db_loss
        + phys_loss
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    port_count: int,
    loss_config: dict[str, float],
) -> tuple[dict[str, float | list[float]], torch.Tensor, torch.Tensor]:
    model.eval()
    losses = []
    preds = []
    truths = []
    mean_device = target_mean.to(device)
    std_device = target_std.to(device)
    with torch.no_grad():
        for points, ports, geom, frame, cuts, nibs, target in loader:
            pred = model(
                points.to(device),
                ports.to(device),
                geom.to(device),
                frame.to(device),
                cuts.to(device),
                nibs.to(device),
            )
            target_device = target.to(device)
            losses.append(
                composite_loss(
                    pred,
                    target_device,
                    port_count=port_count,
                    target_mean=mean_device,
                    target_std=std_device,
                    loss_config=loss_config,
                ).item()
            )
            preds.append((pred.cpu() * target_std) + target_mean)
            truths.append((target.cpu() * target_std) + target_mean)
    pred_all = torch.cat(preds, dim=0)
    truth_all = torch.cat(truths, dim=0)
    metrics = summarize_prediction_metrics(pred_all, truth_all, port_count=port_count)
    metrics["loss"] = float(np.mean(losses))
    return metrics, pred_all, truth_all
