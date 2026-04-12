from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from baseline.antenna_features import extract_antenna_features


@dataclass
class SampleRecord:
    name: str
    points: np.ndarray
    ports: np.ndarray
    geom: np.ndarray
    frame: np.ndarray
    cuts: np.ndarray
    nibs: np.ndarray
    target: np.ndarray


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
    points = _sample_points(vertices, n_points=n_points, seed=_stable_seed(config_path.stem))
    center = vertices.mean(axis=0)
    size = vertices.max(axis=0) - vertices.min(axis=0)
    geom = np.concatenate([center, size]).astype(np.float32)
    frame, cuts, nibs = extract_antenna_features(config, geom)
    return config_path.stem, points, ports, geom, frame, cuts, nibs


def _build_frequency_grid(sample_dirs: list[Path], freq_bins: int) -> np.ndarray:
    mins: list[float] = []
    maxs: list[float] = []
    for sample_dir in sample_dirs:
        freqs, _ = _read_complex_curve(sample_dir / "S1,1.cst.txt")
        mins.append(float(freqs.min()))
        maxs.append(float(freqs.max()))
    return np.linspace(max(mins), min(maxs), freq_bins, dtype=np.float32)


def load_dataset(root: Path, n_points: int = 128, freq_bins: int = 201) -> DatasetBundle:
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
        curves = []
        for row in range(1, len(ports) + 1):
            for col in range(1, len(ports) + 1):
                curve = _interpolate_curve(sample_dir / f"S{row},{col}.cst.txt", freq_grid)
                curves.append(np.stack([curve.real, curve.imag], axis=-1))
        target = np.concatenate(curves, axis=-1).astype(np.float32)
        records.append(
            SampleRecord(
                name=sample_dir.name,
                points=points,
                ports=ports,
                geom=geom,
                frame=frame,
                cuts=cuts,
                nibs=nibs,
                target=target,
            )
        )
    if not records:
        raise RuntimeError(f"No complete samples found in {root}")
    return DatasetBundle(records=records, freq_grid=freq_grid, port_count=port_count)


def load_inference_input(config_path: Path, n_points: int) -> dict[str, np.ndarray | str]:
    name, points, ports, geom, frame, cuts, nibs = _build_input_sample(config_path, n_points=n_points)
    return {"name": name, "points": points, "ports": ports, "geom": geom, "frame": frame, "cuts": cuts, "nibs": nibs}


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
    return {
        "points": torch.tensor(np.stack([record.points for record in records]), dtype=torch.float32),
        "ports": torch.tensor(np.stack([record.ports for record in records]), dtype=torch.float32),
        "geom": torch.tensor(np.stack([record.geom for record in records]), dtype=torch.float32),
        "frame": torch.tensor(np.stack([record.frame for record in records]), dtype=torch.float32),
        "cuts": torch.tensor(np.stack([record.cuts for record in records]), dtype=torch.float32),
        "nibs": torch.tensor(np.stack([record.nibs for record in records]), dtype=torch.float32),
        "target": torch.tensor(np.stack([record.target for record in records]), dtype=torch.float32),
    }
