from __future__ import annotations

from typing import Any

import numpy as np
import torch

from baseline.antenna_features import MAX_CUTS, MAX_NIBS, POSITION_ORDER
from baseline.models.graph import build_graph_features_np


def _position_one_hot(position: str, *, device: torch.device) -> torch.Tensor:
    return torch.tensor([1.0 if position == item else 0.0 for item in POSITION_ORDER], dtype=torch.float32, device=device)


def _cross_size(position: str, geom: torch.Tensor) -> torch.Tensor:
    return geom[4] if position in {"left", "right"} else geom[3]


def _frame_tensor(config: dict[str, Any], geom: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    antenna = config["antennaConfig"]
    width = geom[3].clamp_min(1e-6)
    height = geom[4].clamp_min(1e-6)
    return torch.tensor(
        [
            float(antenna["frameWidth"]) / float(width),
            float(antenna["frameWidth"]) / float(height),
            float(antenna["gap"]) / float(width),
            float(antenna["gap"]) / float(height),
            len(antenna.get("cuts", [])) / MAX_CUTS,
            len(antenna.get("nibs", [])) / MAX_NIBS,
        ],
        dtype=torch.float32,
        device=device,
    )


def _cut_tensor(config: dict[str, Any], geom: torch.Tensor, cut_distances: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    values = torch.zeros((MAX_CUTS, 7), dtype=torch.float32, device=device)
    for index, item in enumerate(config["antennaConfig"].get("cuts", [])[:MAX_CUTS]):
        cross = _cross_size(str(item["position"]), geom).clamp_min(1e-6)
        values[index] = torch.cat(
            [
                torch.tensor([1.0], dtype=torch.float32, device=device),
                _position_one_hot(str(item["position"]), device=device),
                cut_distances[index : index + 1] / (cross * 0.5),
                torch.tensor([float(item["width"]) / float(cross)], dtype=torch.float32, device=device),
            ]
        )
    return values


def _nib_tensor(config: dict[str, Any], geom: torch.Tensor, nib_distances: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    depth = geom[5].clamp_min(1e-6)
    values = torch.zeros((MAX_NIBS, 8), dtype=torch.float32, device=device)
    for index, item in enumerate(config["antennaConfig"].get("nibs", [])[:MAX_NIBS]):
        cross = _cross_size(str(item["position"]), geom).clamp_min(1e-6)
        values[index] = torch.cat(
            [
                torch.tensor([1.0], dtype=torch.float32, device=device),
                _position_one_hot(str(item["position"]), device=device),
                nib_distances[index : index + 1] / (cross * 0.5),
                torch.tensor(
                    [
                        float(item["width"]) / float(cross),
                        float(item.get("thickness", 0.0)) / float(depth),
                    ],
                    dtype=torch.float32,
                    device=device,
                ),
            ]
        )
    return values


def _port_tensor(config: dict[str, Any], geom: torch.Tensor, nib_distances: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    antenna = config["antennaConfig"]
    frame_width = torch.tensor(float(antenna["frameWidth"]), dtype=torch.float32, device=device)
    gap = torch.tensor(float(antenna["gap"]), dtype=torch.float32, device=device)
    size_x, size_y, size_z = geom[3], geom[4], geom[5]
    inner_x = size_x - frame_width * 2.0
    inner_y = size_y - frame_width * 2.0
    z_top = geom[2] + size_z * 0.5
    values = []
    for index, item in enumerate(antenna.get("nibs", [])):
        position = str(item["position"])
        width = torch.tensor(float(item["width"]), dtype=torch.float32, device=device)
        axis = "x" if position in {"left", "right"} else "y"
        direction = -1.0 if position in {"left", "bottom"} else 1.0
        axis_size = inner_x if axis == "x" else inner_y
        cross_size = inner_y if axis == "x" else inner_x
        cross_limit = ((cross_size - width) * 0.5).clamp_min(0.0)
        body_edge = direction * ((axis_size - gap) * 0.5).clamp_min(0.0)
        frame_edge = direction * (axis_size * 0.5)
        port_gap = torch.clamp(frame_width * 0.5, min=(frame_edge - body_edge).abs() * 0.2, max=(frame_edge - body_edge).abs() * 0.8)
        nib_edge = frame_edge - direction * port_gap
        cross_axis = torch.clamp(nib_distances[index], min=-cross_limit, max=cross_limit)
        conductor_inset = torch.minimum(frame_width * 0.15, port_gap * 0.25)
        if axis == "x":
            values.append(torch.stack([nib_edge - direction * conductor_inset, cross_axis, z_top, frame_edge + direction * conductor_inset, cross_axis, z_top]))
        else:
            values.append(torch.stack([cross_axis, nib_edge - direction * conductor_inset, z_top, cross_axis, frame_edge + direction * conductor_inset, z_top]))
    return torch.stack(values, dim=0)


def build_optimizer_inputs(
    config: dict[str, Any],
    *,
    points: np.ndarray,
    geom: np.ndarray,
    cut_distances: torch.Tensor,
    nib_distances: torch.Tensor,
    device: torch.device,
    include_graph: bool,
) -> dict[str, Any]:
    geom_tensor = torch.tensor(geom, dtype=torch.float32, device=device)
    frame_tensor = _frame_tensor(config, geom_tensor, device=device)
    cuts_tensor = _cut_tensor(config, geom_tensor, cut_distances, device=device)
    nibs_tensor = _nib_tensor(config, geom_tensor, nib_distances, device=device)
    ports_tensor = _port_tensor(config, geom_tensor, nib_distances, device=device)
    result: dict[str, Any] = {
        "points": torch.tensor(points, dtype=torch.float32, device=device).unsqueeze(0),
        "ports": ports_tensor.unsqueeze(0),
        "geom": geom_tensor.unsqueeze(0),
        "frame": frame_tensor.unsqueeze(0),
        "cuts": cuts_tensor.unsqueeze(0),
        "nibs": nibs_tensor.unsqueeze(0),
        "graph": None,
    }
    if include_graph:
        graph = build_graph_features_np(
            frame=frame_tensor.detach().cpu().numpy(),
            cuts=cuts_tensor.detach().cpu().numpy(),
            nibs=nibs_tensor.detach().cpu().numpy(),
            ports=ports_tensor.detach().cpu().numpy(),
            geom=geom_tensor.detach().cpu().numpy(),
            port_count=ports_tensor.shape[0],
        )
        result["graph"] = {key: torch.tensor(value, dtype=torch.float32, device=device).unsqueeze(0) for key, value in graph.items()}
    return result
