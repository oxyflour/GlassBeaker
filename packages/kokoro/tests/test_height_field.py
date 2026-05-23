from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from kokoro.height_field import compile_height_program, pyramid_height, sample_height_field


class HeightFieldTest(unittest.TestCase):
    def test_python_height_program_samples_surface_in_meters(self) -> None:
        program = compile_height_program(
            """
import torch

def height(x, y):
    return 0.001 * (x + y)
"""
        )

        samples = sample_height_field(program, sample_count=16, width_m=0.10, depth_m=0.10, seed=11)

        self.assertEqual(samples.positions.shape, (16, 3))
        self.assertEqual(samples.normals.shape, (16, 3))
        self.assertTrue(torch.all(samples.positions[:, 0] >= -0.05))
        self.assertTrue(torch.all(samples.positions[:, 0] <= 0.05))
        self.assertTrue(torch.all(samples.positions[:, 1] >= -0.05))
        self.assertTrue(torch.all(samples.positions[:, 1] <= 0.05))
        self.assertTrue(torch.allclose(samples.positions[:, 2], 0.001 * samples.positions[:, :2].sum(dim=1)))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(samples.normals, dim=1), torch.ones(16), atol=1e-5))


    def test_pyramid_height_repeats_with_configured_period(self) -> None:
        period = 500e-6
        amplitude = 150e-6
        x = torch.tensor([0.0, period, period * 0.5, period * 1.5], dtype=torch.float32)
        y = torch.tensor([0.0, 0.0, period * 0.5, period * 0.5], dtype=torch.float32)

        z = pyramid_height(x, y, period_m=period, amplitude_m=amplitude)

        self.assertTrue(torch.allclose(z[0], z[1]))
        self.assertTrue(torch.allclose(z[2], z[3]))
        self.assertGreater(z[2].item(), z[0].item())
        self.assertTrue(torch.isclose(z[2], torch.tensor(amplitude, dtype=z.dtype), atol=1e-8))


if __name__ == "__main__":
    unittest.main()
