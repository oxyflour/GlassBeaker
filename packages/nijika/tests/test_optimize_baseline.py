from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from optimize_baseline import optimize_model  # noqa: E402


class _ToySurrogate(torch.nn.Module):
    def forward(
        self,
        points: torch.Tensor,
        ports: torch.Tensor,
        geom: torch.Tensor,
        frame: torch.Tensor,
        cuts: torch.Tensor,
        nibs: torch.Tensor,
    ) -> torch.Tensor:
        del points, geom, frame, ports
        batch = cuts.size(0)
        freq_bins = 3
        port_count = 3
        nib_penalty = nibs[:, :port_count, 5].abs() * 0.25
        cut_penalty = cuts[:, 0, 5].abs().unsqueeze(-1) * 0.15
        diag = 0.15 + nib_penalty + cut_penalty
        offdiag = 0.02 + cut_penalty * 0.5
        matrix = torch.zeros((batch, freq_bins, port_count, port_count, 2), dtype=cuts.dtype, device=cuts.device)
        for idx in range(port_count):
            matrix[:, :, idx, idx, 0] = diag[:, idx].unsqueeze(-1).expand(-1, freq_bins)
        for row in range(port_count):
            for col in range(port_count):
                if row != col:
                    matrix[:, :, row, col, 0] = offdiag[:, 0].unsqueeze(-1).expand(-1, freq_bins)
        return matrix.view(batch, freq_bins, port_count * port_count * 2)


def _toy_config() -> dict[str, object]:
    return {
        "mesh": {
            "verts": [
                [-40.0, -80.0, -4.0],
                [40.0, 80.0, 4.0],
            ]
        },
        "ports": [],
        "antennaConfig": {
            "frameWidth": 4.0,
            "gap": 10.0,
            "cuts": [{"position": "top", "distance": 18.0, "width": 20.0}],
            "nibs": [
                {"position": "top", "distance": 15.0, "width": 20.0, "thickness": 8.0},
                {"position": "left", "distance": -18.0, "width": 24.0, "thickness": 8.0},
                {"position": "bottom", "distance": 12.0, "width": 16.0, "thickness": 8.0},
            ],
        },
    }


class OptimizeBaselineTest(unittest.TestCase):
    def test_optimizer_emits_artifacts_and_keeps_distances_bounded(self):
        checkpoint = {
            "freq_grid": [1.0e9, 1.5e9, 2.0e9],
            "port_count": 3,
            "sample_points": 8,
        }
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            result = optimize_model(
                model=_ToySurrogate(),
                checkpoint=checkpoint,
                config=_toy_config(),
                output_dir=output_dir,
                steps=12,
                lr=0.2,
                top_k=2,
            )

            self.assertGreater(result["trace"][0]["loss"], result["trace"][-1]["loss"])
            self.assertTrue((output_dir / "optimization_trace.json").exists())
            self.assertTrue((output_dir / "optimized_soft_solution.json").exists())
            self.assertTrue((output_dir / "candidate_ranking.json").exists())
            self.assertTrue((output_dir / "candidate_01.json").exists())
            self.assertTrue((output_dir / "candidate_02.json").exists())

            soft = json.loads((output_dir / "optimized_soft_solution.json").read_text())
            self.assertLessEqual(abs(soft["cut_distances"][0]), 30.0)
            self.assertEqual(len(soft["nib_distances"]), 3)
            self.assertNotAlmostEqual(soft["cut_distances"][0], 18.0, places=3)
            self.assertTrue(any(abs(value - start) > 1e-3 for value, start in zip(soft["nib_distances"], [15.0, -18.0, 12.0], strict=False)))


if __name__ == "__main__":
    unittest.main()
