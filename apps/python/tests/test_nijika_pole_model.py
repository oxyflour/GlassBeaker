from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "packages" / "nijika"))

from baseline.model import create_model  # noqa: E402
from baseline.train import parse_args  # noqa: E402


class NijikaPoleModelTest(unittest.TestCase):
    def test_pair_pole_offset_model_starts_from_shared_poles(self):
        model = create_model(
            freq_grid=np.linspace(1.0, 2.0, 5, dtype=np.float32),
            port_count=3,
            model_kind="structured_pair_pole_offset_residue_head",
            model_config={"hidden_dim": 8, "num_poles": 4},
        )
        torch.manual_seed(7)
        global_latent = torch.randn(2, 8)
        pair_latent = torch.randn(2, len(model.pairs), 8)

        shared_poles = model._build_shared_poles(global_latent)
        pair_poles = model._build_pair_poles(shared_poles, pair_latent)

        self.assertEqual(pair_poles.shape, (2, len(model.pairs), model.num_poles))
        self.assertTrue(torch.allclose(pair_poles, shared_poles.unsqueeze(1).expand_as(pair_poles)))

    def test_pair_pole_offset_model_can_shift_one_pair_without_moving_others(self):
        model = create_model(
            freq_grid=np.linspace(1.0, 2.0, 5, dtype=np.float32),
            port_count=3,
            model_kind="structured_pair_pole_offset_residue_head",
            model_config={"hidden_dim": 8, "num_poles": 4},
        )
        with torch.no_grad():
            model.pair_pole_offset_head.weight.zero_()
            model.pair_pole_offset_head.bias.zero_()
            model.pair_pole_offset_head.weight[0, 0] = 1.0
            model.pair_pole_offset_head.weight[1, 1] = -1.0
        shared_poles = torch.complex(-torch.full((1, 4), 0.5), torch.full((1, 4), 0.25))
        pair_latent = torch.zeros(1, len(model.pairs), 8)
        pair_latent[0, 0, 0] = 2.0
        pair_latent[0, 1, 1] = 3.0

        pair_poles = model._build_pair_poles(shared_poles, pair_latent)

        self.assertFalse(torch.allclose(pair_poles[:, 0], pair_poles[:, 1]))
        expected = shared_poles.unsqueeze(1).expand_as(pair_poles)
        self.assertTrue(torch.allclose(pair_poles[:, 2:], expected[:, 2:]))

    def test_train_parser_accepts_pair_pole_offset_model_kind(self):
        old_argv = sys.argv
        sys.argv = ["train.py", "--model-kind", "structured_pair_pole_offset_residue_head"]
        try:
            args = parse_args()
        finally:
            sys.argv = old_argv
        self.assertEqual(args.model_kind, "structured_pair_pole_offset_residue_head")


if __name__ == "__main__":
    unittest.main()
