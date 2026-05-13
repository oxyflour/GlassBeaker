from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from optimizer_torch_farfield import (  # noqa: E402
    ETA0,
    combine_farfield_basis,
    derive_currents_and_weights,
    integrate_decoded_ffs_power,
    integrate_farfield_efficiency,
)

TORCH_TRAPEZOID = torch.trapezoid if hasattr(torch, "trapezoid") else torch.trapz


def reference_stimulated_currents(
    s_matrix: torch.Tensor,
    stim_power: float = 0.5,
    z0: float = 50.0,
) -> torch.Tensor:
    port_count = s_matrix.shape[-1]
    scale = (2.0 * stim_power / z0) ** 0.5
    a = torch.eye(port_count, dtype=s_matrix.dtype, device=s_matrix.device).unsqueeze(0)
    a = a.expand(s_matrix.shape[0], -1, -1) * scale
    b = torch.einsum("fij,fjk->fik", s_matrix, a)
    return (a - b) / (z0**0.5)


class OptimizerTorchFarfieldTest(unittest.TestCase):
    def test_derive_currents_and_weights_recovers_linear_mixture(self):
        s_matrix = torch.tensor(
            [
                [
                    [0.10 + 0.05j, 0.02 - 0.01j],
                    [0.03 + 0.04j, -0.08 + 0.02j],
                ]
            ],
            dtype=torch.complex128,
        )
        expected_currents = reference_stimulated_currents(s_matrix)
        expected_weights = torch.tensor([[1.25 - 0.50j, -0.75 + 0.20j]], dtype=torch.complex128)
        port_currents = torch.einsum("fij,fj->fi", expected_currents, expected_weights)

        currents, weights = derive_currents_and_weights(s_matrix, port_currents)

        torch.testing.assert_close(currents, expected_currents)
        torch.testing.assert_close(weights, expected_weights, rtol=1e-6, atol=1e-6)

    def test_combine_farfield_basis_matches_manual_sum(self):
        weights = torch.tensor([[2.0 + 0.0j, -0.5 + 1.0j]], dtype=torch.complex128)
        basis = torch.tensor(
            [
                [[[[1.0 + 0.0j]], [[0.5 + 0.0j]]]],
                [[[[0.0 + 2.0j]], [[1.0 - 1.0j]]]],
            ],
            dtype=torch.complex128,
        )

        combined = combine_farfield_basis(weights, basis)
        expected = weights[0, 0] * basis[0, 0] + weights[0, 1] * basis[1, 0]

        torch.testing.assert_close(combined[0], expected)

    def test_integrate_farfield_efficiency_matches_manual_trapezoid(self):
        phi = torch.tensor([0.0, torch.pi / 2.0, torch.pi], dtype=torch.float64)
        theta = torch.tensor([0.0, torch.pi / 2.0, torch.pi], dtype=torch.float64)
        fields = torch.zeros((1, 2, phi.numel(), theta.numel()), dtype=torch.complex128)
        fields[:, 0] = 2.0 + 0.0j
        fields[:, 1] = 1.0 + 0.0j

        density = (torch.abs(fields[:, 0]) ** 2 + torch.abs(fields[:, 1]) ** 2) * torch.sin(theta)
        expected = TORCH_TRAPEZOID(
            TORCH_TRAPEZOID(density, theta, dim=2),
            phi,
            dim=1,
        ) / (2.0 * ETA0)

        efficiency = integrate_farfield_efficiency(fields, phi, theta)

        torch.testing.assert_close(efficiency, expected)

    def test_efficiency_pipeline_backpropagates_to_s(self):
        s_matrix = torch.tensor(
            [
                [
                    [0.12 + 0.03j, -0.02 + 0.01j],
                    [0.04 - 0.05j, -0.06 + 0.02j],
                ]
            ],
            dtype=torch.complex128,
            requires_grad=True,
        )
        target_weights = torch.tensor([[0.80 - 0.10j, -0.30 + 0.60j]], dtype=torch.complex128)
        port_currents = torch.einsum(
            "fij,fj->fi",
            reference_stimulated_currents(s_matrix.detach()),
            target_weights,
        )
        basis = torch.tensor(
            [
                [
                    [
                        [[1.0 + 0.0j, 0.2 + 0.1j], [0.4 - 0.1j, -0.3 + 0.2j]],
                        [[0.1 - 0.3j, 0.5 + 0.0j], [0.2 + 0.2j, 0.0 + 0.1j]],
                    ]
                ],
                [
                    [
                        [[0.0 + 0.4j, 0.3 - 0.2j], [0.6 + 0.0j, -0.1 + 0.5j]],
                        [[0.7 + 0.1j, -0.2 + 0.2j], [0.3 - 0.4j, 0.2 + 0.0j]],
                    ]
                ],
            ],
            dtype=torch.complex128,
        )
        phi = torch.tensor([0.0, torch.pi / 2.0], dtype=torch.float64)
        theta = torch.tensor([0.0, torch.pi / 2.0], dtype=torch.float64)

        _, weights = derive_currents_and_weights(s_matrix, port_currents)
        fields = combine_farfield_basis(weights, basis)
        efficiency = integrate_farfield_efficiency(fields, phi, theta)
        efficiency.sum().backward()

        self.assertIsNotNone(s_matrix.grad)
        self.assertTrue(torch.isfinite(s_matrix.grad).all().item())
        self.assertGreater(float(s_matrix.grad.abs().max()), 1e-12)

    def test_integrate_decoded_ffs_power_matches_manual_reference_path(self):
        decoded = torch.tensor(
            [
                [
                    [
                        [
                            [1.0, 0.1, 0.4, -0.2],
                            [0.8, -0.3, 0.2, 0.0],
                            [1.2, 0.4, -0.1, 0.5],
                            [0.5, 0.2, 0.7, -0.4],
                            [1.0, 0.1, 0.4, -0.2],
                            [0.8, -0.3, 0.2, 0.0],
                        ],
                        [
                            [0.3, -0.4, 0.9, 0.2],
                            [0.6, 0.1, -0.2, 0.3],
                            [0.5, 0.2, 0.4, -0.1],
                            [0.7, -0.2, 0.1, 0.6],
                            [0.3, -0.4, 0.9, 0.2],
                            [0.6, 0.1, -0.2, 0.3],
                        ],
                    ],
                    [
                        [
                            [0.4, 0.0, 0.2, 0.1],
                            [0.1, 0.3, 0.5, -0.2],
                            [0.6, -0.1, 0.3, 0.4],
                            [0.2, 0.5, -0.4, 0.7],
                            [0.4, 0.0, 0.2, 0.1],
                            [0.1, 0.3, 0.5, -0.2],
                        ],
                        [
                            [0.9, 0.2, -0.1, 0.3],
                            [0.7, -0.5, 0.2, 0.4],
                            [0.8, 0.1, 0.6, -0.3],
                            [0.3, 0.4, 0.5, 0.2],
                            [0.9, 0.2, -0.1, 0.3],
                            [0.7, -0.5, 0.2, 0.4],
                        ],
                    ],
                ]
            ],
            dtype=torch.float64,
        )
        phi = torch.tensor([0.0, torch.pi], dtype=torch.float64)
        theta = torch.tensor([0.0, torch.pi / 2.0], dtype=torch.float64)
        complex_grid = decoded.view(1, 2, 2, 3, 2, 4)
        manual_fields = torch.stack(
            [
                torch.complex(complex_grid[..., 0], complex_grid[..., 1]),
                torch.complex(complex_grid[..., 2], complex_grid[..., 3]),
            ],
            dim=3,
        )[..., :-1, :]
        density = (torch.abs(manual_fields[..., 0, :, :]) ** 2 + torch.abs(manual_fields[..., 1, :, :]) ** 2) * torch.sin(
            theta
        ).view(1, 1, 1, 1, theta.numel())
        expected = TORCH_TRAPEZOID(
            TORCH_TRAPEZOID(density, theta, dim=4),
            phi,
            dim=3,
        ) / (2.0 * ETA0)

        power = integrate_decoded_ffs_power(
            decoded,
            phi=phi,
            theta=theta,
            phi_count=3,
            theta_count=2,
            has_phi_closure=True,
        )

        self.assertEqual(tuple(power.shape), (1, 2, 2))
        torch.testing.assert_close(power, expected)


if __name__ == "__main__":
    unittest.main()
