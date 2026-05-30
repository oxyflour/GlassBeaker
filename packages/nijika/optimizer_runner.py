from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from baseline.antenna_features import extract_antenna_features
from baseline.models.graph import build_graph_features_np
from baseline.training_utils import forward_model, uses_graph_features
from optimizer_geometry import bound_distance, mesh_geom, rebuild_config_with_distances
from optimizer_inputs import build_optimizer_inputs
from optimizer_objective import (
    antenna_efficiency,
    efficiency_objective,
    enumerate_role_assignments,
    loaded_input_admittance,
    reflection_from_admittance,
    s_to_y,
)


def _cross_size(position: str, geom: np.ndarray) -> float:
    return float(geom[4] if position in {"left", "right"} else geom[3])


def _sample_points(config: dict[str, Any], sample_points: int) -> np.ndarray:
    verts = np.asarray(config["mesh"]["verts"], dtype=np.float32)
    indices = np.arange(sample_points, dtype=np.int64) % len(verts)
    return verts[indices]


def _ports_array(config: dict[str, Any]) -> np.ndarray:
    rows = []
    for port in config["ports"]:
        pos = port["positions"][0]
        rows.append([pos["from"]["x"], pos["from"]["y"], pos["from"]["z"], pos["to"]["x"], pos["to"]["y"], pos["to"]["z"]])
    return np.asarray(rows, dtype=np.float32)


def _build_tensors(config: dict[str, Any], *, points: np.ndarray, geom: np.ndarray) -> dict[str, Any]:
    ports = _ports_array(config)
    frame, cuts, nibs = extract_antenna_features(config, geom)
    graph = build_graph_features_np(frame=frame, cuts=cuts, nibs=nibs, ports=ports, geom=geom, port_count=len(ports))
    return {"points": points, "ports": ports, "geom": geom, "frame": frame, "cuts": cuts, "nibs": nibs, "graph": graph}


def _raw_from_distance(distance: float, limit: float) -> float:
    if limit <= 1e-6:
        return 0.0
    clipped = float(np.clip(distance / limit, -0.999999, 0.999999))
    return float(np.arctanh(clipped))


def _bounded_distances(items: list[dict[str, Any]], raw: torch.Tensor, geom: np.ndarray) -> list[float]:
    values = []
    for item, raw_value in zip(items, raw, strict=False):
        values.append(float(bound_distance(raw_value, cross_size=_cross_size(str(item["position"]), geom), span_width=float(item["width"])).item()))
    return values


def _predict_s_matrix(model: torch.nn.Module, tensors: dict[str, Any], device: torch.device) -> torch.Tensor:
    graph = None
    if uses_graph_features(model):
        graph = {key: torch.tensor(value, dtype=torch.float32).unsqueeze(0) for key, value in tensors["graph"].items()}
    pred = forward_model(
        model,
        points=torch.tensor(tensors["points"], dtype=torch.float32).unsqueeze(0),
        ports=torch.tensor(tensors["ports"], dtype=torch.float32).unsqueeze(0),
        geom=torch.tensor(tensors["geom"], dtype=torch.float32).unsqueeze(0),
        frame=torch.tensor(tensors["frame"], dtype=torch.float32).unsqueeze(0),
        cuts=torch.tensor(tensors["cuts"], dtype=torch.float32).unsqueeze(0),
        nibs=torch.tensor(tensors["nibs"], dtype=torch.float32).unsqueeze(0),
        device=device,
        graph_tensors=graph,
    )[0]
    port_count = tensors["ports"].shape[0]
    freq_bins = pred.shape[0]
    pair = pred.view(freq_bins, port_count, port_count, 2)
    return torch.complex(pair[..., 0], pair[..., 1])


def _enable_dropout(model: torch.nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def _predict_s_matrix_from_inputs(model: torch.nn.Module, tensors: dict[str, Any], device: torch.device) -> torch.Tensor:
    pred = forward_model(
        model,
        points=tensors["points"],
        ports=tensors["ports"],
        geom=tensors["geom"],
        frame=tensors["frame"],
        cuts=tensors["cuts"],
        nibs=tensors["nibs"],
        device=device,
        graph_tensors=tensors["graph"],
    )[0]
    port_count = tensors["ports"].shape[1]
    freq_bins = pred.shape[0]
    pair = pred.view(freq_bins, port_count, port_count, 2)
    return torch.complex(pair[..., 0], pair[..., 1])


def _mc_predict_s_matrix(
    model: torch.nn.Module,
    tensors: dict[str, Any],
    device: torch.device,
    n_samples: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MC dropout S-matrix prediction with gradients.

    Returns (mean_S, std_mag) where mean_S is the complex mean S-matrix
    and std_mag is the standard deviation of |S| across samples.
    """
    _enable_dropout(model)
    samples = []
    for _ in range(n_samples):
        pred = forward_model(
            model,
            points=tensors["points"],
            ports=tensors["ports"],
            geom=tensors["geom"],
            frame=tensors["frame"],
            cuts=tensors["cuts"],
            nibs=tensors["nibs"],
            device=device,
            graph_tensors=tensors["graph"],
        )[0]
        port_count = tensors["ports"].shape[1]
        freq_bins = pred.shape[0]
        pair = pred.view(freq_bins, port_count, port_count, 2)
        samples.append(torch.complex(pair[..., 0], pair[..., 1]))
    stacked = torch.stack(samples, dim=0)  # (N, F, P, P)
    mean_s = stacked.mean(dim=0)
    std_mag = torch.abs(stacked).std(dim=0)  # (F, P, P) std of |S|
    return mean_s, std_mag


def _candidate_loss(s_matrix: torch.Tensor, feed_index: int, terminations: list[str], *, z0: float = 50.0) -> float:
    term_dict = {i: terminations[i] for i in range(len(terminations))}
    eta = antenna_efficiency(s_matrix, feed_index, terminations=term_dict, z0=z0)
    return float((-eta.mean()).item())


def optimize_model(
    *,
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
    steps: int = 200,
    lr: float = 5e-2,
    top_k: int = 3,
    band_min: float | None = None,
    band_max: float | None = None,
    match_weight: float = 5.0,
    isolation_weight: float = 3.0,
    bandwidth_weight: float = 2.0,
    match_threshold_db: float = -10.0,
    ground_admittance: float = 1e6,
    open_admittance: float = 1e-6,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    geom = mesh_geom(config)
    points = _sample_points(config, int(checkpoint.get("sample_points", 16)))
    antenna = config["antennaConfig"]
    cuts = antenna.get("cuts", [])
    nibs = antenna.get("nibs", [])
    cut_raw = torch.nn.Parameter(
        torch.tensor(
            [_raw_from_distance(float(item["distance"]), max((_cross_size(str(item["position"]), geom) - float(item["width"])) * 0.5, 0.0)) for item in cuts],
            dtype=torch.float32,
            device=device,
        )
    )
    nib_raw = torch.nn.Parameter(
        torch.tensor(
            [_raw_from_distance(float(item["distance"]), max((_cross_size(str(item["position"]), geom) - float(item["width"])) * 0.5, 0.0)) for item in nibs],
            dtype=torch.float32,
            device=device,
        )
    )
    feed_logits = torch.nn.Parameter(torch.zeros(len(nibs), dtype=torch.float32, device=device))
    term_logits = torch.nn.Parameter(torch.zeros(len(nibs), dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([cut_raw, nib_raw, feed_logits, term_logits], lr=lr)
    freq_grid = torch.tensor(checkpoint["freq_grid"], dtype=torch.float32)
    mask = torch.ones(len(freq_grid), dtype=torch.bool)
    if band_min is not None:
        mask &= freq_grid >= band_min
    if band_max is not None:
        mask &= freq_grid <= band_max
    trace = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        cut_distances = torch.stack(
            [bound_distance(raw_value, cross_size=_cross_size(str(item["position"]), geom), span_width=float(item["width"])) for item, raw_value in zip(cuts, cut_raw, strict=False)],
            dim=0,
        ) if cuts else torch.zeros((0,), dtype=torch.float32, device=device)
        nib_distances = torch.stack(
            [bound_distance(raw_value, cross_size=_cross_size(str(item["position"]), geom), span_width=float(item["width"])) for item, raw_value in zip(nibs, nib_raw, strict=False)],
            dim=0,
        ) if nibs else torch.zeros((0,), dtype=torch.float32, device=device)
        model_inputs = build_optimizer_inputs(
            config,
            points=points,
            geom=geom,
            cut_distances=cut_distances,
            nib_distances=nib_distances,
            device=device,
            include_graph=uses_graph_features(model),
        )
        # MC dropout: mean S-matrix + uncertainty
        s_mean, s_std = _mc_predict_s_matrix(model, model_inputs, device, n_samples=8)
        s_mean = s_mean[mask]
        s_std = s_std[mask]

        # Efficiency loss from mean prediction
        eff_loss, detail = efficiency_objective(
            s_mean,
            feed_logits=feed_logits,
            termination_logits=term_logits,
            z0=50.0,
        )

        # Uncertainty penalty: discourage optimizer from exploiting uncertain regions
        unc_penalty = s_std.mean() * 0.5

        loss = eff_loss + unc_penalty
        loss.backward()
        optimizer.step()
        trace.append(
            {
                "step": step,
                "loss": float(loss.item()),
                "eff_loss": float(eff_loss.item()),
                "unc_penalty": float(unc_penalty.item()),
                "feed_probs": detail["feed_probs"].detach().cpu().tolist(),
                "termination_probs": detail["termination_probs"].detach().cpu().tolist(),
                "cut_distances": cut_distances.detach().cpu().tolist(),
                "nib_distances": nib_distances.detach().cpu().tolist(),
            }
        )
    final_config = rebuild_config_with_distances(config, cut_distances=trace[-1]["cut_distances"], nib_distances=trace[-1]["nib_distances"])
    final_s = _predict_s_matrix(model, _build_tensors(final_config, points=points, geom=geom), device)[mask]
    ranked = []
    for candidate in enumerate_role_assignments(port_count=len(nibs)):
        terms = [role["termination"] for role in candidate["roles"] if not role["feed"]]
        score = _candidate_loss(final_s, candidate["feed_index"], terms, z0=50.0)
        ranked.append({"score": score, **candidate})
    ranked.sort(key=lambda item: item["score"])
    soft = {key: trace[-1][key] for key in ("cut_distances", "nib_distances", "feed_probs", "termination_probs")}
    (output_dir / "optimization_trace.json").write_text(json.dumps(trace, indent=2))
    (output_dir / "optimized_soft_solution.json").write_text(json.dumps(soft, indent=2))
    (output_dir / "candidate_ranking.json").write_text(json.dumps(ranked, indent=2))
    for index, candidate in enumerate(ranked[:top_k], start=1):
        candidate_config = json.loads(json.dumps(final_config))
        candidate_config["optimizedRoleAssignment"] = candidate
        (output_dir / f"candidate_{index:02d}.json").write_text(json.dumps(candidate_config, indent=2))
    return {"trace": trace, "soft_solution": soft, "ranking": ranked}
