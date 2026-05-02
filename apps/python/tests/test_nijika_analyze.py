from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "packages" / "nijika"))

from baseline.analyze import _predict  # noqa: E402


class _GraphAwareDummyModel(torch.nn.Module):
    uses_graph_features = True
    graph_feature_keys = ("pair_topology",)

    def forward(
        self,
        points: torch.Tensor,
        ports: torch.Tensor,
        geom: torch.Tensor,
        frame: torch.Tensor,
        cuts: torch.Tensor,
        nibs: torch.Tensor,
        pair_topology: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del points, ports, geom, frame, cuts, nibs
        if pair_topology is None:
            raise ValueError("pair_topology missing")
        return pair_topology.mean(dim=(1, 2), keepdim=True).unsqueeze(1)


class NijikaAnalyzeTest(unittest.TestCase):
    def test_predict_passes_pair_topology_for_graph_aware_models(self):
        checkpoint = {"target_mean": [0.0], "target_std": [1.0]}
        tensors = {
            "points": torch.zeros((2, 1, 3), dtype=torch.float32),
            "ports": torch.zeros((2, 3, 6), dtype=torch.float32),
            "geom": torch.zeros((2, 6), dtype=torch.float32),
            "frame": torch.zeros((2, 6), dtype=torch.float32),
            "cuts": torch.zeros((2, 4, 7), dtype=torch.float32),
            "nibs": torch.zeros((2, 4, 8), dtype=torch.float32),
            "pair_topology": torch.tensor(
                [
                    [[1.0, 3.0], [5.0, 7.0]],
                    [[2.0, 4.0], [6.0, 8.0]],
                ],
                dtype=torch.float32,
            ),
        }

        pred = _predict(
            _GraphAwareDummyModel(),
            checkpoint,
            tensors,
            torch.device("cpu"),
            batch_size=1,
        )

        np.testing.assert_allclose(pred.reshape(-1), np.array([4.0, 5.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
