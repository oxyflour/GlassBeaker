from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from baseline.antenna_features import extract_antenna_features
from baseline.models.graph import build_graph_features_np


@dataclass
class SampleRecord:
    name: str
    points: np.ndarray
    ports: np.ndarray
    geom: np.ndarray
    frame: np.ndarray
    cuts: np.ndarray
    nibs: np.ndarray
    graph: dict[str, np.ndarray] | None
    target: np.ndarray
    temporal: np.ndarray | None = None


@dataclass
class DatasetBundle:
    records: list[SampleRecord]
    freq_grid: np.ndarray
    port_count: int


def _read_complex_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float, float]] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            continue
    data = np.asarray(rows, dtype=np.float32)
    freqs = data[:, 0]
    values = data[:, 1] + 1j * data[:, 2]
    uniq, inverse = np.unique(freqs, return_inverse=True)
    if len(uniq) == len(freqs):
        return freqs, values
    reduced = np.zeros(len(uniq), dtype=np.complex64)
    counts = np.zeros(len(uniq), dtype=np.float32)
    for idx, value in zip(inverse, values, strict=False):
        reduced[idx] += value
        counts[idx] += 1
    return uniq, reduced / counts


def _interpolate_curve(path: Path, grid: np.ndarray) -> np.ndarray:
    freqs, values = _read_complex_curve(path)
    real = np.interp(grid, freqs, values.real)
    imag = np.interp(grid, freqs, values.imag)
    return real.astype(np.float32) + 1j * imag.astype(np.float32)


def _sample_points(vertices: np.ndarray, n_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    replace = len(vertices) < n_points
    indices = rng.choice(len(vertices), size=n_points, replace=replace)
    return vertices[indices].astype(np.float32)


def _sample_surface_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    ports: np.ndarray,
    n_points: int,
    seed: int,
) -> np.ndarray:
    """Sample unique points on triangle mesh surface, biased toward ports and edges."""
    rng = np.random.default_rng(seed)
    tri_verts = vertices[faces]
    v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
    tri_centers = (v0 + v1 + v2) / 3.0
    tri_areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) / 2.0

    port_centers = (ports[:, :3] + ports[:, 3:]) / 2.0
    dist_to_ports = np.min(
        [np.linalg.norm(tri_centers - pc, axis=1) for pc in port_centers], axis=0
    )
    port_weight = np.exp(-dist_to_ports / 0.02)

    bbox_min = tri_centers.min(axis=0)
    bbox_max = tri_centers.max(axis=0)
    dist_to_edges = np.min(
        [
            np.abs(tri_centers[:, 0] - bbox_min[0]),
            np.abs(tri_centers[:, 0] - bbox_max[0]),
            np.abs(tri_centers[:, 1] - bbox_min[1]),
            np.abs(tri_centers[:, 1] - bbox_max[1]),
        ],
        axis=0,
    )
    edge_weight = np.exp(-dist_to_edges / 0.01)

    importance = tri_areas * (1.0 + 4.0 * port_weight + 2.0 * edge_weight)
    probs = importance / importance.sum()

    n_tris = len(faces)
    replace = n_tris < n_points
    tri_idx = rng.choice(n_tris, size=n_points, replace=replace, p=probs)

    r1 = rng.random(n_points)
    r2 = rng.random(n_points)
    swap = r1 + r2 > 1.0
    r1[swap] = 1.0 - r1[swap]
    r2[swap] = 1.0 - r2[swap]

    sv = vertices[faces[tri_idx]]
    points = (
        sv[:, 0]
        + r1[:, None] * (sv[:, 1] - sv[:, 0])
        + r2[:, None] * (sv[:, 2] - sv[:, 0])
    )
    return points.astype(np.float32)


def _stable_seed(name: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(name)) % (2**32)


def _load_ports(config_ports: list[dict[str, object]]) -> np.ndarray:
    rows = []
    for port in config_ports:
        position = port["positions"][0]
        start = position["from"]
        end = position["to"]
        rows.append(
            [
                start["x"],
                start["y"],
                start["z"],
                end["x"],
                end["y"],
                end["z"],
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _build_input_sample(
    config_path: Path,
    n_points: int,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    config = json.loads(config_path.read_text())
    vertices = np.asarray(config["mesh"]["verts"], dtype=np.float32)
    if len(vertices) == 0:
        raise ValueError(f"No mesh vertices in {config_path}")
    ports = _load_ports(config["ports"])
    faces = np.asarray(config["mesh"].get("faces", []), dtype=np.int32)
    if len(faces) > 0 and len(ports) > 0:
        points = _sample_surface_points(
            vertices, faces, ports, n_points=n_points, seed=_stable_seed(config_path.stem)
        )
    else:
        points = _sample_points(vertices, n_points=n_points, seed=_stable_seed(config_path.stem))
    center = vertices.mean(axis=0)
    size = vertices.max(axis=0) - vertices.min(axis=0)
    geom = np.concatenate([center, size]).astype(np.float32)
    frame, cuts, nibs = extract_antenna_features(config, geom)
    return config_path.stem, points, ports, geom, frame, cuts, nibs


def _read_temporal_signal(path: Path, max_steps: int) -> np.ndarray | None:
    """Read a Port X [Y].txt file, returning the signal column up to max_steps."""
    try:
        data = np.loadtxt(path, dtype=np.float32)
    except (OSError, ValueError):
        return None
    signal = data[:max_steps, 1].copy()
    if len(signal) < max_steps:
        signal = np.pad(signal, (0, max_steps - len(signal)))
    return signal


def _load_temporal_signals(
    sample_dir: Path, port_count: int, max_steps: int
) -> np.ndarray | None:
    """Load all port-pair temporal signals into (port_count*port_count, max_steps)."""
    rows = []
    for row in range(1, port_count + 1):
        for col in range(1, port_count + 1):
            signal = _read_temporal_signal(
                sample_dir / f"Port {row} [{col}].txt", max_steps
            )
            if signal is None:
                return None
            rows.append(signal)
    return np.stack(rows, axis=0).astype(np.float32)


def _build_frequency_grid(sample_dirs: list[Path], freq_bins: int) -> np.ndarray:
    mins: list[float] = []
    maxs: list[float] = []
    for sample_dir in sample_dirs:
        freqs, _ = _read_complex_curve(sample_dir / "S1,1.cst.txt")
        mins.append(float(freqs.min()))
        maxs.append(float(freqs.max()))
    return np.linspace(max(mins), min(maxs), freq_bins, dtype=np.float32)


def load_dataset(root: Path, n_points: int = 128, freq_bins: int = 201, max_temporal_steps: int = 0) -> DatasetBundle:
    sample_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not sample_dirs:
        raise FileNotFoundError(f"No sample directories found in {root}")
    valid_sample_dirs: list[Path] = []
    for sample_dir in sample_dirs:
        config_path = root / f"{sample_dir.name}.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        port_total = len(config["ports"])
        needed = [
            sample_dir / f"S{row},{col}.cst.txt"
            for row in range(1, port_total + 1)
            for col in range(1, port_total + 1)
        ]
        if all(path.exists() for path in needed):
            valid_sample_dirs.append(sample_dir)
    if not valid_sample_dirs:
        raise RuntimeError(f"No complete samples found in {root}")
    freq_grid = _build_frequency_grid(valid_sample_dirs, freq_bins)
    records: list[SampleRecord] = []
    port_count = 0
    for sample_dir in valid_sample_dirs:
        config_path = root / f"{sample_dir.name}.json"
        _, points, ports, geom, frame, cuts, nibs = _build_input_sample(config_path, n_points=n_points)
        port_count = max(port_count, len(ports))
        graph = build_graph_features_np(frame=frame, cuts=cuts, nibs=nibs, ports=ports, geom=geom, port_count=len(ports))
        curves = []
        for row in range(1, len(ports) + 1):
            for col in range(1, len(ports) + 1):
                curve = _interpolate_curve(sample_dir / f"S{row},{col}.cst.txt", freq_grid)
                curves.append(np.stack([curve.real, curve.imag], axis=-1))
        target = np.concatenate(curves, axis=-1).astype(np.float32)
        temporal = None
        if max_temporal_steps > 0:
            temporal = _load_temporal_signals(sample_dir, len(ports), max_temporal_steps)
        records.append(
            SampleRecord(
                name=sample_dir.name,
                points=points,
                ports=ports,
                geom=geom,
                frame=frame,
                cuts=cuts,
                nibs=nibs,
                graph=graph,
                target=target,
                temporal=temporal,
            )
        )
    if not records:
        raise RuntimeError(f"No complete samples found in {root}")
    return DatasetBundle(records=records, freq_grid=freq_grid, port_count=port_count)


def load_inference_input(config_path: Path, n_points: int) -> dict[str, np.ndarray | str]:
    name, points, ports, geom, frame, cuts, nibs = _build_input_sample(config_path, n_points=n_points)
    graph = build_graph_features_np(frame=frame, cuts=cuts, nibs=nibs, ports=ports, geom=geom, port_count=len(ports))
    return {"name": name, "points": points, "ports": ports, "geom": geom, "frame": frame, "cuts": cuts, "nibs": nibs, "graph": graph}


def load_truth_target(sample_dir: Path, port_count: int, freq_grid: np.ndarray) -> np.ndarray | None:
    needed = [
        sample_dir / f"S{row},{col}.cst.txt"
        for row in range(1, port_count + 1)
        for col in range(1, port_count + 1)
    ]
    if not all(path.exists() for path in needed):
        return None
    curves = []
    for row in range(1, port_count + 1):
        for col in range(1, port_count + 1):
            curve = _interpolate_curve(sample_dir / f"S{row},{col}.cst.txt", freq_grid)
            curves.append(np.stack([curve.real, curve.imag], axis=-1))
    return np.concatenate(curves, axis=-1).astype(np.float32)


def split_records(records: list[SampleRecord], seed: int, val_ratio: float = 0.2) -> tuple[list[SampleRecord], list[SampleRecord]]:
    indices = np.arange(len(records))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_count = max(1, int(round(len(records) * val_ratio)))
    val_ids = set(indices[:val_count].tolist())
    train = [record for idx, record in enumerate(records) if idx not in val_ids]
    val = [record for idx, record in enumerate(records) if idx in val_ids]
    return train, val


def stack_records(records: list[SampleRecord]) -> dict[str, torch.Tensor]:
    stacked = {
        "points": torch.tensor(np.stack([record.points for record in records]), dtype=torch.float32),
        "ports": torch.tensor(np.stack([record.ports for record in records]), dtype=torch.float32),
        "geom": torch.tensor(np.stack([record.geom for record in records]), dtype=torch.float32),
        "frame": torch.tensor(np.stack([record.frame for record in records]), dtype=torch.float32),
        "cuts": torch.tensor(np.stack([record.cuts for record in records]), dtype=torch.float32),
        "nibs": torch.tensor(np.stack([record.nibs for record in records]), dtype=torch.float32),
        "target": torch.tensor(np.stack([record.target for record in records]), dtype=torch.float32),
    }
    if records and records[0].graph is not None:
        for key in ("graph_inner", "graph_segment", "graph_port", "graph_mask", "graph_adj", "graph_edge_attr", "pair_topology"):
            stacked[key] = torch.tensor(np.stack([record.graph[key] for record in records]), dtype=torch.float32)
    if records and records[0].temporal is not None:
        stacked["temporal"] = torch.tensor(np.stack([record.temporal for record in records]), dtype=torch.float32)
    return stacked
