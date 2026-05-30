"""Bridge surrogate model S-parameters → chinatsu MNA circuit → efficiency.

Two modes:
- Re(Z): lossless PEC, exact from impedance matrix, no .ffs needed
- Farfield: full farfield integration via .ffs patterns (requires nijika export)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from skrf import Frequency, Network

_chinatsu_root = Path(__file__).resolve().parent.parent.parent / "chinatsu"
if str(_chinatsu_root) not in sys.path:
    sys.path.insert(0, str(_chinatsu_root))

from farfield import Farfield as _Farfield  # noqa: E402
from mna import MnaCircuit, TensorGammaZ0, TensorYNetwork  # noqa: E402


def _s_to_z(s_matrix: torch.Tensor, z0: float = 50.0) -> torch.Tensor:
    P = s_matrix.size(-1)
    I_mat = torch.eye(P, dtype=s_matrix.dtype, device=s_matrix.device)
    return z0 * (I_mat + s_matrix) @ torch.linalg.inv(I_mat - s_matrix)


def antenna_efficiency_rez(
    s_matrix: torch.Tensor,
    feed_index: int,
    terminations: dict[int, str],
    source_impedance: float = 50.0,
) -> torch.Tensor:
    """Efficiency from Re(Z) — exact for lossless PEC antenna."""
    P = s_matrix.size(-1)
    Z = _s_to_z(s_matrix, source_impedance)
    z0 = source_impedance

    Z_mod = Z.clone()
    for port, term in terminations.items():
        Z_mod[..., port, port] = Z_mod[..., port, port] + (0.0 if term == "ground" else 1e6)
    Z_mod[..., feed_index, feed_index] = Z_mod[..., feed_index, feed_index] + z0

    V = torch.zeros_like(Z_mod[..., 0])
    V[..., feed_index] = 1.0
    I = torch.linalg.solve(Z_mod, V)

    p_in = 0.5 * torch.real(V[..., feed_index] * I[..., feed_index].conj())
    p_loads = torch.zeros_like(p_in)
    for port, term in terminations.items():
        if term == "ground":
            p_loads = p_loads + 0.5 * torch.abs(I[..., port]) ** 2 * 0.0
        else:
            p_loads = p_loads + 0.5 * torch.abs(I[..., port]) ** 2 * 1e6

    eta = 1.0 - p_loads / (p_in + 1e-9)
    return eta.clamp(0.0, 1.0)


def _s_tensor_to_network(
    s_tensor: torch.Tensor, freq_grid: np.ndarray, z0: float = 50.0
) -> Network:
    """Convert surrogate S-matrix tensor to scikit-rf Network.

    s_tensor: (F, P, P) complex
    """
    s_np = s_tensor.detach().cpu().numpy()
    freq = Frequency.from_f(freq_grid, unit="Hz")
    ntw = Network(frequency=freq, s=s_np, z0=z0)
    ntw.name = "snp"
    return ntw


class MnaEfficiencyBridge:
    """Differentiable efficiency calculation via chinatsu MNA circuit solver.

    Wraps surrogate S-matrix prediction into the MNA circuit framework,
    supporting both Re(Z) and Farfield (.ffs) efficiency computation.
    """

    def __init__(
        self,
        freq_grid: np.ndarray,
        port_count: int,
        ffs_files: list[str] | None = None,
        device: str = "cuda",
    ):
        self.freq_grid = freq_grid
        self.port_count = port_count
        self.ffs_files = ffs_files
        self.device = device
        self._mna_cache: dict[tuple, Any] = {}

    def build_mna(
        self,
        s_tensor: torch.Tensor,
        feed_index: int,
        terminations: dict[int, str],
    ) -> MnaCircuit:
        """Build an MNA circuit with the given S-matrix and terminations."""
        freq = Frequency.from_f(self.freq_grid, unit="Hz")
        z = TensorGammaZ0(freq)
        s_ntw = _s_tensor_to_network(s_tensor, self.freq_grid)

        # Wrap Y-parameters as callable tensor so gradient flows
        def y_callable():
            y_np = s_ntw.y
            return torch.tensor(y_np, device=self.device, dtype=torch.complex128)

        snp_y = TensorYNetwork(s_ntw, y_callable)

        src = MnaCircuit.Port(freq, name="src-0")
        r = z.resistor(np.array(50.0), name="r-feed")

        conns = [[(src, 0), (r, 0)], [(r, 1), (snp_y, feed_index)]]
        for port, term in terminations.items():
            if term == "ground":
                conns.append([(snp_y, port), (MnaCircuit.Ground(freq, f"gnd-{port}"), 0)])
            else:
                conns.append([(snp_y, port), (MnaCircuit.Open(freq, f"open-{port}"), 0)])

        return MnaCircuit(conns, device=self.device)

    def efficiency_via_mna(
        self,
        s_tensor: torch.Tensor,
        feed_index: int,
        terminations: dict[int, str],
    ) -> torch.Tensor:
        """Compute efficiency using MNA circuit solver (no .ffs needed).

        Uses Re(Z) radiation resistance — exact for lossless PEC antenna.
        """
        mna = self.build_mna(s_tensor, feed_index, terminations)
        mna.update_tensor()

        power = np.array([0.5])  # single source
        phase = np.array([0.0])

        voltages = mna.node_voltages(power, phase)
        P = self.port_count

        # Convert S to Z for radiation power calculation
        s_complex = s_tensor if s_tensor.is_complex() else s_tensor[..., 0] + 1j * s_tensor[..., 1]
        Z = _s_to_z(s_complex)
        R_rad = torch.real(Z)

        # Get port currents from MNA
        s_np = _s_tensor_to_network(s_tensor, self.freq_grid)
        currents = mna.component_currents(voltages, s_np)

        # Build current vector and compute radiated power
        P_rad = torch.zeros(len(self.freq_grid), device=self.device, dtype=torch.float64)
        for f in range(len(self.freq_grid)):
            I_vec = currents[f, :P].to(dtype=torch.complex128)
            R_f = R_rad[f].to(dtype=torch.complex128)
            P_rad[f] = 0.5 * torch.real(I_vec.conj() @ R_f @ I_vec)

        P_stim = 0.5  # per frequency
        P_loads = torch.zeros_like(P_rad)
        for port, term in terminations.items():
            I_p = currents[:, port].to(dtype=torch.complex128)
            if term == "ground":
                pass  # R=0, no dissipation
            else:
                P_loads = P_loads + 0.5 * torch.abs(I_p) ** 2 * 1e6

        eta = (P_rad - P_loads) / (P_stim + 1e-9)
        return eta.clamp(0.0, 1.0)

    def efficiency_via_farfield(
        self,
        s_tensor: torch.Tensor,
        feed_index: int,
        terminations: dict[int, str],
    ) -> torch.Tensor:
        """Compute efficiency using full farfield integration (.ffs files required)."""
        if self.ffs_files is None:
            raise ValueError("ffs_files required for farfield efficiency")

        freq = Frequency.from_f(self.freq_grid, unit="Hz")
        z = TensorGammaZ0(freq)
        s_np = _s_tensor_to_network(s_tensor, self.freq_grid)

        src = MnaCircuit.Port(freq, name="src-0")
        r = z.resistor(np.array(50.0), name="r-feed")

        conns: list[list[tuple]] = [[(src, 0), (r, 0)], [(r, 1), (s_np, feed_index)]]
        for port, term in terminations.items():
            if term == "ground":
                conns.append([(s_np, port), (MnaCircuit.Ground(freq, f"gnd-{port}"), 0)])
            else:
                conns.append([(s_np, port), (MnaCircuit.Open(freq, f"open-{port}"), 0)])

        ff = _Farfield(s_np, self.ffs_files, conns, device=self.device)
        power = np.array([0.5])
        phase = np.array([0.0])
        p_rad = ff.compute(power, phase)  # (F,) total radiated power
        return p_rad / 0.5  # efficiency = P_rad / P_stim
