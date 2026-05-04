from __future__ import annotations

from itertools import product
from typing import Any

import torch


def feed_probabilities(feed_logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(feed_logits, dim=-1)


def termination_probabilities(termination_logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(termination_logits)


def enumerate_role_assignments(port_count: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for feed_index in range(port_count):
        others = [idx for idx in range(port_count) if idx != feed_index]
        for states in product(("open", "ground"), repeat=len(others)):
            roles = []
            state_map = dict(zip(others, states, strict=False))
            for index in range(port_count):
                if index == feed_index:
                    roles.append({"port": index, "feed": True, "termination": None})
                else:
                    roles.append({"port": index, "feed": False, "termination": state_map[index]})
            candidates.append({"feed_index": feed_index, "roles": roles})
    return candidates


def s_to_y(s_matrix: torch.Tensor, z0: float = 50.0) -> torch.Tensor:
    ports = s_matrix.size(-1)
    eye = torch.eye(ports, dtype=s_matrix.dtype, device=s_matrix.device)
    while eye.ndim < s_matrix.ndim:
        eye = eye.unsqueeze(0)
    return ((eye - s_matrix) @ torch.linalg.inv(eye + s_matrix)) / z0


def loaded_input_admittance(
    y_matrix: torch.Tensor,
    *,
    feed_index: int,
    other_load_admittances: torch.Tensor,
) -> torch.Tensor:
    ports = y_matrix.size(-1)
    other_indices = [idx for idx in range(ports) if idx != feed_index]
    y_ff = y_matrix[..., feed_index, feed_index]
    if not other_indices:
        return y_ff
    y_fo = y_matrix[..., feed_index, other_indices]
    y_of = y_matrix[..., other_indices, feed_index]
    y_oo = y_matrix[..., other_indices, :][..., :, other_indices]
    loads = other_load_admittances.to(y_matrix.dtype).to(y_matrix.device)
    while loads.ndim < y_matrix.ndim - 1:
        loads = loads.unsqueeze(0)
    loaded_block = y_oo + torch.diag_embed(loads.expand(*y_oo.shape[:-2], len(other_indices)))
    reduction = (y_fo.unsqueeze(-2) @ torch.linalg.inv(loaded_block) @ y_of.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return y_ff - reduction


def reflection_from_admittance(y_in: torch.Tensor, z0: float = 50.0) -> torch.Tensor:
    z0y = y_in * z0
    return (1.0 - z0y) / (1.0 + z0y + torch.as_tensor(1e-9, dtype=z0y.dtype, device=z0y.device))


def soft_role_objective(
    s_matrix: torch.Tensor,
    *,
    feed_logits: torch.Tensor,
    termination_logits: torch.Tensor,
    match_weight: float = 5.0,
    isolation_weight: float = 3.0,
    bandwidth_weight: float = 2.0,
    match_threshold_db: float = -10.0,
    bandwidth_sharpness: float = 8.0,
    open_admittance: float = 1e-6,
    ground_admittance: float = 1e6,
    z0: float = 50.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    y_matrix = s_to_y(s_matrix, z0=z0)
    feed_probs = feed_probabilities(feed_logits)
    term_probs = termination_probabilities(termination_logits)
    losses = []
    per_feed_gamma = []
    threshold_mag = 10.0 ** (match_threshold_db / 20.0)
    for feed_index in range(s_matrix.size(-1)):
        other_indices = [idx for idx in range(s_matrix.size(-1)) if idx != feed_index]
        load_mix = open_admittance + (ground_admittance - open_admittance) * term_probs[other_indices]
        y_in = loaded_input_admittance(y_matrix, feed_index=feed_index, other_load_admittances=load_mix)
        gamma = reflection_from_admittance(y_in, z0=z0)
        per_feed_gamma.append(gamma)
        match_mag = torch.abs(gamma)
        couplings = torch.abs(s_matrix[..., feed_index, other_indices]).mean(dim=-1)
        bandwidth = torch.sigmoid((threshold_mag - match_mag) * bandwidth_sharpness).mean(dim=-1)
        losses.append(match_weight * match_mag + isolation_weight * couplings - bandwidth_weight * bandwidth)
    per_feed_loss = torch.stack(losses, dim=-1)
    total_loss = (per_feed_loss * feed_probs).sum(dim=-1).mean()
    return total_loss, {
        "feed_probs": feed_probs,
        "termination_probs": term_probs,
        "per_feed_loss": per_feed_loss,
        "per_feed_gamma": torch.stack(per_feed_gamma, dim=-1),
    }
