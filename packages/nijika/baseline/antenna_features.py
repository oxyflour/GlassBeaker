from __future__ import annotations

from typing import Any

import numpy as np


MAX_CUTS = 4
MAX_NIBS = 4
POSITION_ORDER = ("left", "right", "top", "bottom")


def _position_one_hot(position: str) -> np.ndarray:
    return np.asarray([float(position == item) for item in POSITION_ORDER], dtype=np.float32)


def _cross_size(position: str, geom: np.ndarray) -> float:
    return float(geom[4] if position in {"left", "right"} else geom[3])


def extract_antenna_features(config: dict[str, Any], geom: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    antenna = config.get("antennaConfig") or {}
    frame_width = float(antenna.get("frameWidth", 0.0))
    gap = float(antenna.get("gap", 0.0))
    num_cuts = float(len(antenna.get("cuts", [])))
    num_nibs = float(len(antenna.get("nibs", [])))
    width = float(geom[3]) if geom[3] > 0 else 1.0
    height = float(geom[4]) if geom[4] > 0 else 1.0
    depth = float(geom[5]) if geom[5] > 0 else 1.0
    frame = np.asarray(
        [
            frame_width / width,
            frame_width / height,
            gap / width,
            gap / height,
            num_cuts / MAX_CUTS,
            num_nibs / MAX_NIBS,
        ],
        dtype=np.float32,
    )

    cuts = np.zeros((MAX_CUTS, 7), dtype=np.float32)
    for idx, cut in enumerate((antenna.get("cuts") or [])[:MAX_CUTS]):
        cross = max(_cross_size(str(cut["position"]), geom), 1e-6)
        cuts[idx] = np.asarray(
            [
                1.0,
                *_position_one_hot(str(cut["position"])),
                float(cut["distance"]) / (cross * 0.5),
                float(cut["width"]) / cross,
            ],
            dtype=np.float32,
        )

    nibs = np.zeros((MAX_NIBS, 8), dtype=np.float32)
    for idx, nib in enumerate((antenna.get("nibs") or [])[:MAX_NIBS]):
        cross = max(_cross_size(str(nib["position"]), geom), 1e-6)
        nibs[idx] = np.asarray(
            [
                1.0,
                *_position_one_hot(str(nib["position"])),
                float(nib["distance"]) / (cross * 0.5),
                float(nib["width"]) / cross,
                float(nib.get("thickness", 0.0)) / depth,
            ],
            dtype=np.float32,
        )

    return frame, cuts, nibs
