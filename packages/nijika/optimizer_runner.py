from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from baseline.antenna_features import extract_antenna_features
from baseline.ffs_codec import TorchFfsCodec, codec_state_from_payload
from baseline.models.graph import build_graph_features_np
from baseline.training_utils import forward_model, uses_graph_features
from optimizer_geometry import bound_distance, mesh_geom, rebuild_config_with_distances
from optimizer_inputs import build_optimizer_inputs
from optimizer_objective import (
    antenna_efficiency,
    efficiency_objective,
    enumerate_role_assignments,
)
from optimizer_torch_farfield import (
    combine_farfield_basis,
    decoded_ffs_to_basis,
    derive_currents_and_weights,
    integrate_farfield_efficiency,
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


def _predict_aux(model: torch.nn.Module, tensors: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    graph = None
    if uses_graph_features(model):
        graph = {key: torch.tensor(value, dtype=torch.float32).unsqueeze(0) for key, value in tensors["graph"].items()}
    aux = forward_model(
        model,
        points=torch.tensor(tensors["points"], dtype=torch.float32).unsqueeze(0),
        ports=torch.tensor(tensors["ports"], dtype=torch.float32).unsqueeze(0),
        geom=torch.tensor(tensors["geom"], dtype=torch.float32).unsqueeze(0),
        frame=torch.tensor(tensors["frame"], dtype=torch.float32).unsqueeze(0),
        cuts=torch.tensor(tensors["cuts"], dtype=torch.float32).unsqueeze(0),
        nibs=torch.tensor(tensors["nibs"], dtype=torch.float32).unsqueeze(0),
        device=device,
        graph_tensors=graph,
        return_aux=True,
    )
    assert isinstance(aux, dict)
    return aux


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


def _predict_aux_from_inputs(model: torch.nn.Module, tensors: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    aux = forward_model(
        model,
        points=tensors["points"],
        ports=tensors["ports"],
        geom=tensors["geom"],
        frame=tensors["frame"],
        cuts=tensors["cuts"],
        nibs=tensors["nibs"],
        device=device,
        graph_tensors=tensors["graph"],
        return_aux=True,
    )
    assert isinstance(aux, dict)
    return aux


def _complex_s_from_flat_output(pred: torch.Tensor, port_count: int) -> torch.Tensor:
    pair = pred.view(pred.shape[0], port_count, port_count, 2)
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


def _candidate_loss(s_matrix: torch.Tensor, feed_index: int, terminations: dict[int, str], *, z0: float = 50.0) -> float:
    eta = antenna_efficiency(s_matrix, feed_index, terminations=terminations, z0=z0)
    return float((-eta.mean()).item())


def _decode_ffs_coefficients(coeff_pred: torch.Tensor, checkpoint: dict[str, Any]) -> torch.Tensor:
    payload = checkpoint.get("ffs_codec")
    if payload is None:
        raise ValueError("Checkpoint does not contain FFS codec metadata")
    codec = TorchFfsCodec.from_state(
        codec_state_from_payload(payload),
        dtype=coeff_pred.dtype,
        device=coeff_pred.device,
    )
    return codec.decode(coeff_pred)


def _match_farfield_indices(checkpoint: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    metadata = checkpoint.get("ffs_metadata")
    if metadata is None:
        raise ValueError("Checkpoint does not contain FFS metadata")
    freq_grid = np.asarray(checkpoint["freq_grid"], dtype=np.float64)
    farfield_freqs = np.asarray(metadata["frequencies_hz"], dtype=np.float64)
    if freq_grid.max() < 1e6 and farfield_freqs.max() > 1e6:
        farfield_freqs = farfield_freqs / 1e9
    elif freq_grid.max() > 1e6 and farfield_freqs.max() < 1e6:
        farfield_freqs = farfield_freqs * 1e9
    lower_margin = max(1.0, abs(freq_grid[1] - freq_grid[0]) if len(freq_grid) > 1 else 1.0)
    upper_margin = max(1.0, abs(freq_grid[-1] - freq_grid[-2]) if len(freq_grid) > 1 else 1.0)
    lower_indices = []
    upper_indices = []
    alphas = []
    for freq in farfield_freqs:
        tolerance = max(1.0, abs(freq) * 1e-6)
        if freq < freq_grid[0] - lower_margin - tolerance or freq > freq_grid[-1] + upper_margin + tolerance:
            raise ValueError(f"Farfield frequency {freq} lies outside checkpoint freq_grid support")
        upper = int(np.searchsorted(freq_grid, freq, side="left"))
        if upper <= 0:
            lower = upper = 0
            alpha = 0.0
        elif upper >= len(freq_grid):
            lower = upper = len(freq_grid) - 1
            alpha = 0.0
        elif abs(freq_grid[upper] - freq) <= tolerance:
            lower = upper
            alpha = 0.0
        elif abs(freq_grid[upper - 1] - freq) <= tolerance:
            lower = upper = upper - 1
            alpha = 0.0
        else:
            lower = upper - 1
            alpha = (freq - freq_grid[lower]) / max(freq_grid[upper] - freq_grid[lower], 1e-12)
        lower_indices.append(lower)
        upper_indices.append(upper)
        alphas.append(alpha)
    return (
        torch.tensor(lower_indices, dtype=torch.long, device=device),
        torch.tensor(upper_indices, dtype=torch.long, device=device),
        torch.tensor(alphas, dtype=torch.float32, device=device),
        torch.tensor(farfield_freqs, dtype=torch.float32, device=device),
    )


def _interpolate_frequency_axis(
    values: torch.Tensor,
    lower_indices: torch.Tensor,
    upper_indices: torch.Tensor,
    alphas: torch.Tensor,
) -> torch.Tensor:
    lower = values.index_select(0, lower_indices)
    upper = values.index_select(0, upper_indices)
    shape = (alphas.shape[0],) + (1,) * (values.ndim - 1)
    weight = alphas.to(dtype=values.real.dtype if values.is_complex() else values.dtype, device=values.device).view(shape)
    return lower + (upper - lower) * weight


def _farfield_grid(metadata: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, bool]:
    angles = torch.tensor(metadata["angles_deg"], dtype=torch.float64, device=device)
    phi_count = int(metadata["phi_count"])
    theta_count = int(metadata["theta_count"])
    grid = angles.view(phi_count, theta_count, 2)
    phi = torch.deg2rad(grid[:, 0, 0])
    theta = torch.deg2rad(grid[0, :, 1])
    has_phi_closure = bool(phi_count > 1 and torch.isclose(phi[-1], phi[0] + 2.0 * torch.pi, atol=1e-9, rtol=0.0))
    if has_phi_closure:
        phi = phi[:-1]
    return phi, theta, has_phi_closure


def _decoded_ffs_basis(decoded: torch.Tensor, metadata: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phi, theta, has_phi_closure = _farfield_grid(metadata, decoded.device)
    phi_count = int(metadata["phi_count"])
    theta_count = int(metadata["theta_count"])
    complex_basis = decoded_ffs_to_basis(
        decoded,
        phi_count=phi_count,
        theta_count=theta_count,
        has_phi_closure=has_phi_closure,
    )
    return complex_basis, phi, theta


def _solve_loaded_currents(
    s_matrix: torch.Tensor,
    *,
    feed_index: int,
    termination_probs: torch.Tensor | None = None,
    terminations: dict[int, str] | None = None,
    z0: float = 50.0,
    ground_admittance: float = 1e6,
    open_admittance: float = 1e-6,
) -> torch.Tensor:
    ports = s_matrix.size(-1)
    eye = torch.eye(ports, dtype=s_matrix.dtype, device=s_matrix.device)
    z_matrix = z0 * (eye + s_matrix) @ torch.linalg.inv(eye - s_matrix)
    loaded = z_matrix.clone()
    if termination_probs is not None:
        for port in range(ports):
            if port == feed_index:
                continue
            y_load = termination_probs[port] * ground_admittance + (1.0 - termination_probs[port]) * open_admittance
            loaded[..., port, port] = loaded[..., port, port] + 1.0 / (y_load + 1e-12)
    elif terminations is not None:
        for port, term in terminations.items():
            loaded[..., port, port] = loaded[..., port, port] + (0.0 if term == "ground" else 1e6)
    loaded[..., feed_index, feed_index] = loaded[..., feed_index, feed_index] + z0
    voltage = torch.zeros_like(loaded[..., 0])
    voltage[..., feed_index] = 1.0
    return torch.linalg.solve(loaded, voltage)


def _farfield_efficiency(
    s_matrix: torch.Tensor,
    *,
    ffs_basis: torch.Tensor,
    phi: torch.Tensor,
    theta: torch.Tensor,
    feed_index: int,
    termination_probs: torch.Tensor | None = None,
    terminations: dict[int, str] | None = None,
    z0: float = 50.0,
    ground_admittance: float = 1e6,
    open_admittance: float = 1e-6,
) -> torch.Tensor:
    currents = _solve_loaded_currents(
        s_matrix,
        feed_index=feed_index,
        termination_probs=termination_probs,
        terminations=terminations,
        z0=z0,
        ground_admittance=ground_admittance,
        open_admittance=open_admittance,
    )
    _, weights = derive_currents_and_weights(s_matrix, currents, stim_power=0.5, z0=z0)
    fields = combine_farfield_basis(weights, ffs_basis)
    return (integrate_farfield_efficiency(fields, phi, theta) / 0.5).clamp(0.0, 1.0)


def _farfield_objective(
    s_matrix: torch.Tensor,
    *,
    ffs_basis: torch.Tensor,
    phi: torch.Tensor,
    theta: torch.Tensor,
    feed_logits: torch.Tensor,
    termination_logits: torch.Tensor,
    z0: float = 50.0,
    ground_admittance: float = 1e6,
    open_admittance: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    feed_probs = torch.softmax(feed_logits, dim=-1)
    term_probs = torch.sigmoid(termination_logits)
    losses = []
    for feed_idx in range(s_matrix.size(-1)):
        eta = _farfield_efficiency(
            s_matrix,
            ffs_basis=ffs_basis,
            phi=phi,
            theta=theta,
            feed_index=feed_idx,
            termination_probs=term_probs,
            z0=z0,
            ground_admittance=ground_admittance,
            open_admittance=open_admittance,
        )
        losses.append(-eta.mean())
    per_feed_loss = torch.stack(losses, dim=-1)
    total_loss = (per_feed_loss * feed_probs).sum(dim=-1)
    return total_loss, {
        "feed_probs": feed_probs,
        "termination_probs": term_probs,
        "per_feed_loss": per_feed_loss,
    }


def _candidate_loss_farfield(
    s_matrix: torch.Tensor,
    feed_index: int,
    terminations: dict[int, str],
    *,
    ffs_basis: torch.Tensor,
    phi: torch.Tensor,
    theta: torch.Tensor,
    z0: float = 50.0,
) -> float:
    eta = _farfield_efficiency(
        s_matrix,
        ffs_basis=ffs_basis,
        phi=phi,
        theta=theta,
        feed_index=feed_index,
        terminations=terminations,
        z0=z0,
    )
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
    efficiency_mode: str = "rez",
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
    farfield_indices = None
    farfield_mask = None
    farfield_phi = None
    farfield_theta = None
    if efficiency_mode == "farfield":
        farfield_lower, farfield_upper, farfield_alpha, farfield_freqs = _match_farfield_indices(checkpoint, device)
        farfield_mask = torch.ones(len(farfield_freqs), dtype=torch.bool, device=device)
        if band_min is not None:
            farfield_mask &= farfield_freqs >= band_min
        if band_max is not None:
            farfield_mask &= farfield_freqs <= band_max
        if not bool(farfield_mask.any().item()):
            raise ValueError("No farfield frequencies fall inside the requested optimization band")
        farfield_phi, farfield_theta, _ = _farfield_grid(checkpoint["ffs_metadata"], device)
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
        if efficiency_mode == "farfield":
            aux = _predict_aux_from_inputs(model, model_inputs, device)
            if "ffs_coeff_pred" not in aux or "s_pred" not in aux:
                raise ValueError("Farfield mode requires aux S-parameter and FFS coefficient outputs")
            decoded = _decode_ffs_coefficients(aux["ffs_coeff_pred"], checkpoint)
            ffs_basis, _, _ = _decoded_ffs_basis(decoded, checkpoint["ffs_metadata"])
            port_count = int(checkpoint["port_count"])
            s_mean = _interpolate_frequency_axis(
                _complex_s_from_flat_output(aux["s_pred"][0], port_count),
                farfield_lower,
                farfield_upper,
                farfield_alpha,
            )[farfield_mask]
            s_std = _interpolate_frequency_axis(s_std, farfield_lower, farfield_upper, farfield_alpha)[farfield_mask]
            basis_view = ffs_basis[0][:, farfield_mask]
            eff_loss, detail = _farfield_objective(
                s_mean,
                ffs_basis=basis_view,
                phi=farfield_phi,
                theta=farfield_theta,
                feed_logits=feed_logits,
                termination_logits=term_logits,
                z0=50.0,
                ground_admittance=ground_admittance,
                open_admittance=open_admittance,
            )
        else:
            s_mean = s_mean[mask]
            s_std = s_std[mask]
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
    final_tensors = _build_tensors(final_config, points=points, geom=geom)
    if efficiency_mode == "farfield":
        final_aux = _predict_aux(model, final_tensors, device)
        if "ffs_coeff_pred" not in final_aux or "s_pred" not in final_aux:
            raise ValueError("Farfield mode requires aux S-parameter and FFS coefficient outputs")
        decoded = _decode_ffs_coefficients(final_aux["ffs_coeff_pred"], checkpoint)
        final_basis, _, _ = _decoded_ffs_basis(decoded, checkpoint["ffs_metadata"])
        port_count = int(checkpoint["port_count"])
        final_pair = final_aux["s_pred"][0].view(-1, port_count, port_count, 2)
        final_s = torch.complex(
            _interpolate_frequency_axis(final_pair[..., 0], farfield_lower, farfield_upper, farfield_alpha)[farfield_mask],
            _interpolate_frequency_axis(final_pair[..., 1], farfield_lower, farfield_upper, farfield_alpha)[farfield_mask],
        )
        ranking_basis = final_basis[0][:, farfield_mask]
    else:
        final_s = _predict_s_matrix(model, final_tensors, device)[mask]
    ranked = []
    for candidate in enumerate_role_assignments(port_count=len(nibs)):
        terms = {int(role["port"]): str(role["termination"]) for role in candidate["roles"] if not role["feed"]}
        if efficiency_mode == "farfield":
            score = _candidate_loss_farfield(
                final_s,
                candidate["feed_index"],
                terms,
                ffs_basis=ranking_basis,
                phi=farfield_phi,
                theta=farfield_theta,
                z0=50.0,
            )
        else:
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
