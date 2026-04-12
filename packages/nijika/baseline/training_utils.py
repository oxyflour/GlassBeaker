from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from baseline.metrics import summarize_prediction_metrics


def composite_loss(pred: torch.Tensor, target: torch.Tensor, mag_weight: float, smooth_weight: float) -> torch.Tensor:
    ri_loss = nn.functional.mse_loss(pred, target)
    pred_pair = pred.view(pred.size(0), pred.size(1), -1, 2)
    target_pair = target.view(target.size(0), target.size(1), -1, 2)
    mag_loss = nn.functional.l1_loss(torch.linalg.vector_norm(pred_pair, dim=-1), torch.linalg.vector_norm(target_pair, dim=-1))
    slope_loss = nn.functional.mse_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])
    return ri_loss + mag_weight * mag_loss + smooth_weight * slope_loss


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    port_count: int,
    mag_weight: float,
    smooth_weight: float,
) -> tuple[dict[str, float | list[float]], torch.Tensor, torch.Tensor]:
    model.eval()
    losses = []
    preds = []
    truths = []
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
            losses.append(composite_loss(pred, target.to(device), mag_weight, smooth_weight).item())
            preds.append((pred.cpu() * target_std) + target_mean)
            truths.append((target.cpu() * target_std) + target_mean)
    pred_all = torch.cat(preds, dim=0)
    truth_all = torch.cat(truths, dim=0)
    metrics = summarize_prediction_metrics(pred_all, truth_all, port_count=port_count)
    metrics["loss"] = float(np.mean(losses))
    return metrics, pred_all, truth_all
