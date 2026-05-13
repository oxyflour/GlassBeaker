from __future__ import annotations

import math

import torch

ETA0 = 120.0 * math.pi
TORCH_TRAPEZOID = torch.trapezoid if hasattr(torch, "trapezoid") else torch.trapz


def stimulated_port_currents_from_s(
    s_matrix: torch.Tensor,
    stim_power: float = 0.5,
    z0: float = 50.0,
) -> torch.Tensor:
    port_count = s_matrix.shape[-1]
    scale = torch.sqrt(
        torch.as_tensor(2.0 * stim_power / z0, dtype=s_matrix.dtype, device=s_matrix.device)
    )
    drive = torch.eye(port_count, dtype=s_matrix.dtype, device=s_matrix.device).unsqueeze(0)
    drive = drive.expand(s_matrix.shape[0], -1, -1) * scale
    reflected = torch.einsum("fij,fjk->fik", s_matrix, drive)
    return (drive - reflected) / torch.sqrt(
        torch.as_tensor(z0, dtype=s_matrix.dtype, device=s_matrix.device)
    )


def derive_currents_and_weights(
    s_matrix: torch.Tensor,
    port_currents: torch.Tensor,
    stim_power: float = 0.5,
    z0: float = 50.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    stimulated_currents = stimulated_port_currents_from_s(s_matrix, stim_power=stim_power, z0=z0)
    weights = torch.linalg.solve(stimulated_currents, port_currents.unsqueeze(-1)).squeeze(-1)
    return stimulated_currents, weights


def combine_farfield_basis(weights: torch.Tensor, port_basis: torch.Tensor) -> torch.Tensor:
    return torch.einsum("fp,pfcxy->fcxy", weights, port_basis)


def decoded_ffs_to_basis(
    decoded: torch.Tensor,
    *,
    phi_count: int,
    theta_count: int,
    has_phi_closure: bool,
) -> torch.Tensor:
    batch, port_count, freq_count, angle_count, channel_count = decoded.shape
    if channel_count != 4 or angle_count != phi_count * theta_count:
        raise ValueError("Decoded FFS tensor shape does not match the requested phi/theta layout")
    grid = decoded.view(batch, port_count, freq_count, phi_count, theta_count, 4)
    basis = torch.stack(
        [
            torch.complex(grid[..., 0], grid[..., 1]),
            torch.complex(grid[..., 2], grid[..., 3]),
        ],
        dim=3,
    )
    if has_phi_closure:
        basis = basis[..., :-1, :]
    return basis


def integrate_decoded_ffs_power(
    decoded: torch.Tensor,
    *,
    phi: torch.Tensor,
    theta: torch.Tensor,
    phi_count: int,
    theta_count: int,
    has_phi_closure: bool,
) -> torch.Tensor:
    basis = decoded_ffs_to_basis(
        decoded,
        phi_count=phi_count,
        theta_count=theta_count,
        has_phi_closure=has_phi_closure,
    )
    if basis.shape[-2] != phi.numel() or basis.shape[-1] != theta.numel():
        raise ValueError("Decoded FFS tensor shape does not match the provided phi/theta vectors")
    batch, port_count, freq_count = basis.shape[:3]
    fields = basis.reshape(batch * port_count * freq_count, 2, basis.shape[-2], basis.shape[-1])
    power = integrate_farfield_efficiency(fields, phi, theta)
    return power.view(batch, port_count, freq_count)


def integrate_farfield_efficiency(
    fields: torch.Tensor,
    phi: torch.Tensor,
    theta: torch.Tensor,
    eta0: float = ETA0,
) -> torch.Tensor:
    sin_theta = torch.sin(theta).to(dtype=fields.real.dtype, device=fields.device)
    density = (torch.abs(fields[:, 0]) ** 2 + torch.abs(fields[:, 1]) ** 2) * sin_theta.view(1, 1, -1)
    return TORCH_TRAPEZOID(TORCH_TRAPEZOID(density, theta, dim=2), phi, dim=1) / (2.0 * eta0)
