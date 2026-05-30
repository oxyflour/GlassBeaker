from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import skrf as rf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from optimizer_objective import (  # noqa: E402
    enumerate_role_assignments,
    feed_probabilities,
    loaded_input_admittance,
    s_to_y,
    termination_probabilities,
)


class OptimizerObjectiveTest(unittest.TestCase):
    def test_role_relaxation_has_one_feed_distribution(self):
        probs = feed_probabilities(torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=6)
        self.assertTrue(torch.all(probs > 0.0))

        terms = termination_probabilities(torch.tensor([-8.0, 0.0, 8.0], dtype=torch.float32))
        self.assertTrue(torch.all(terms >= 0.0))
        self.assertTrue(torch.all(terms <= 1.0))

    def test_enumerate_hard_assignments_returns_twelve_candidates(self):
        candidates = enumerate_role_assignments(port_count=3)
        self.assertEqual(len(candidates), 12)
        self.assertEqual(sum(item["feed"] for item in candidates[0]["roles"]), 1)

    def test_loaded_input_admittance_matches_reference_network(self):
        s = torch.tensor(
            [
                [
                    [0.10 + 0.02j, 0.03 - 0.01j, 0.02 + 0.04j],
                    [0.03 - 0.01j, 0.12 + 0.03j, 0.01 + 0.02j],
                    [0.02 + 0.04j, 0.01 + 0.02j, 0.08 - 0.02j],
                ]
            ],
            dtype=torch.complex64,
        )
        y = s_to_y(s, z0=50.0)
        load = torch.tensor([0.0 + 0.0j, 1e6 + 0.0j], dtype=torch.complex64)
        loaded = loaded_input_admittance(y, feed_index=0, other_load_admittances=load)

        freq = rf.Frequency.from_f([1.0], unit="hz")
        network = rf.Network(s=s.numpy(), frequency=freq, z0=50.0)
        reference_y = network.y[0]
        reduced = reference_y[[0], :][:, [0]] - reference_y[[0], :][:, [1, 2]] @ np.linalg.inv(
            reference_y[[1, 2], :][:, [1, 2]] + np.diag(load.numpy())
        ) @ reference_y[[1, 2], :][:, [0]]
        np.testing.assert_allclose(
            np.asarray([loaded.item()]),
            reduced.reshape(-1),
            rtol=1e-5,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
