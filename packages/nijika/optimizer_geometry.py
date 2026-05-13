from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def mesh_geom(config: dict[str, Any]) -> np.ndarray:
    verts = np.asarray(config["mesh"]["verts"], dtype=np.float32)
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    center = (mins + maxs) * 0.5
    size = maxs - mins
    return np.concatenate([center, size]).astype(np.float32)


def distance_limit(cross_size: float, span_width: float) -> float:
    return max((float(cross_size) - float(span_width)) * 0.5, 0.0)


def bound_distance(raw: torch.Tensor, *, cross_size: float, span_width: float) -> torch.Tensor:
    limit = distance_limit(cross_size, span_width)
    return torch.tanh(raw) * limit


def get_feed_placement(
    inner_size: dict[str, float],
    gap: float,
    frame_width: float,
    nib: dict[str, Any],
) -> dict[str, float | str]:
    position = str(nib["position"])
    axis = "x" if position in {"left", "right"} else "y"
    direction = -1 if position in {"left", "bottom"} else 1
    axis_size = inner_size["x"] if axis == "x" else inner_size["y"]
    cross_size = inner_size["y"] if axis == "x" else inner_size["x"]
    cross_limit = max((cross_size - float(nib["width"])) * 0.5, 0.0)
    body_edge = direction * max((axis_size - gap) * 0.5, 0.0)
    frame_edge = direction * (axis_size * 0.5)
    available_gap = abs(frame_edge - body_edge)
    port_gap = clamp(frame_width * 0.5, available_gap * 0.2, available_gap * 0.8)
    nib_edge = frame_edge - direction * port_gap
    span_center = (body_edge + nib_edge) * 0.5
    span_length = max(abs(nib_edge - body_edge), 1e-4)
    return {
        "axis": axis,
        "direction": direction,
        "cross_axis": clamp(float(nib["distance"]), -cross_limit, cross_limit),
        "body_edge": body_edge,
        "nib_edge": nib_edge,
        "frame_edge": frame_edge,
        "span_center": span_center,
        "span_length": span_length,
    }


def generate_ports(
    nibs: list[dict[str, Any]],
    *,
    phone_size: np.ndarray,
    frame_width: float,
    gap: float,
    z_top: float,
) -> list[dict[str, Any]]:
    inner_size = {
        "x": float(phone_size[0]) - frame_width * 2.0,
        "y": float(phone_size[1]) - frame_width * 2.0,
        "z": max(float(phone_size[2]) * 0.24, 0.12),
    }
    ports: list[dict[str, Any]] = []
    for index, nib in enumerate(nibs):
        placement = get_feed_placement(inner_size, gap, frame_width, nib)
        port_gap = abs(float(placement["frame_edge"]) - float(placement["nib_edge"]))
        conductor_inset = min(frame_width * 0.15, port_gap * 0.25)
        if placement["axis"] == "x":
            start = {
                "x": float(placement["nib_edge"]) - float(placement["direction"]) * conductor_inset,
                "y": float(placement["cross_axis"]),
                "z": z_top,
            }
            end = {
                "x": float(placement["frame_edge"]) + float(placement["direction"]) * conductor_inset,
                "y": float(placement["cross_axis"]),
                "z": z_top,
            }
        else:
            start = {
                "x": float(placement["cross_axis"]),
                "y": float(placement["nib_edge"]) - float(placement["direction"]) * conductor_inset,
                "z": z_top,
            }
            end = {
                "x": float(placement["cross_axis"]),
                "y": float(placement["frame_edge"]) + float(placement["direction"]) * conductor_inset,
                "z": z_top,
            }
        ports.append(
            {
                "num": index + 1,
                "label": f"port{index + 1}",
                "impedance": 50.0,
                "positions": [{"from": start, "to": end}],
            }
        )
    return ports


def regenerate_ports(config: dict[str, Any]) -> list[dict[str, Any]]:
    geom = mesh_geom(config)
    antenna = config["antennaConfig"]
    return generate_ports(
        antenna.get("nibs", []),
        phone_size=geom[3:],
        frame_width=float(antenna["frameWidth"]),
        gap=float(antenna["gap"]),
        z_top=float(np.asarray(config["mesh"]["verts"], dtype=np.float32)[:, 2].max()),
    )


def rebuild_config_with_distances(
    config: dict[str, Any],
    *,
    cut_distances: list[float] | None = None,
    nib_distances: list[float] | None = None,
) -> dict[str, Any]:
    updated = deepcopy(config)
    antenna = updated["antennaConfig"]
    if cut_distances is not None:
        for item, distance in zip(antenna.get("cuts", []), cut_distances, strict=False):
            item["distance"] = float(distance)
    if nib_distances is not None:
        for item, distance in zip(antenna.get("nibs", []), nib_distances, strict=False):
            item["distance"] = float(distance)
    updated["ports"] = regenerate_ports(updated)
    return updated
