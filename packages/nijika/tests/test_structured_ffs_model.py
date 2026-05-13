from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baseline.model import create_model
from baseline.training_utils import forward_model


def _inputs(batch_size: int = 2, freq_bins: int = 5, port_count: int = 3) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    gen = torch.Generator().manual_seed(123)

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=gen, dtype=torch.float32)

    tensors = {
        "points": rand(batch_size, 16, 3),
        "ports": rand(batch_size, port_count, 6),
        "geom": rand(batch_size, 6),
        "frame": rand(batch_size, 6),
        "cuts": rand(batch_size, 4, 7),
        "nibs": rand(batch_size, 4, 8),
    }
    freq_grid = torch.linspace(1.0, 2.0, freq_bins)
    return tensors, freq_grid


class StructuredFfsModelContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.port_count = 3
        self.ffs_coeff_dim = 7
        self.tensors, self.freq_grid = _inputs(port_count=self.port_count)

    def test_create_model_exposes_s_only_forward_and_aux_path_for_ffs_head(self) -> None:
        model = create_model(
            freq_grid=self.freq_grid,
            port_count=self.port_count,
            model_kind="structured_pair_spectral_ffs_head",
            model_config={"hidden_dim": 16, "dropout": 0.0, "ffs_coeff_dim": self.ffs_coeff_dim},
        )

        s_pred = model(
            self.tensors["points"],
            self.tensors["ports"],
            self.tensors["geom"],
            self.tensors["frame"],
            self.tensors["cuts"],
            self.tensors["nibs"],
        )
        aux = model.forward_with_aux(
            self.tensors["points"],
            self.tensors["ports"],
            self.tensors["geom"],
            self.tensors["frame"],
            self.tensors["cuts"],
            self.tensors["nibs"],
        )

        self.assertEqual(s_pred.shape, (2, len(self.freq_grid), self.port_count * self.port_count * 2))
        self.assertTrue(torch.equal(s_pred, aux["s_pred"]))
        self.assertEqual(aux["ffs_coeff_pred"].shape, (2, self.ffs_coeff_dim))

    def test_forward_model_can_return_aux_outputs_for_new_ffs_head(self) -> None:
        model = create_model(
            freq_grid=self.freq_grid,
            port_count=self.port_count,
            model_kind="structured_pair_spectral_ffs_head",
            model_config={"hidden_dim": 16, "dropout": 0.0, "ffs_coeff_dim": self.ffs_coeff_dim},
        )

        aux = forward_model(
            model,
            points=self.tensors["points"],
            ports=self.tensors["ports"],
            geom=self.tensors["geom"],
            frame=self.tensors["frame"],
            cuts=self.tensors["cuts"],
            nibs=self.tensors["nibs"],
            device=self.device,
            return_aux=True,
        )

        self.assertEqual(aux["s_pred"].shape, (2, len(self.freq_grid), self.port_count * self.port_count * 2))
        self.assertEqual(aux["ffs_coeff_pred"].shape, (2, self.ffs_coeff_dim))

    def test_forward_model_return_aux_stays_backward_compatible_for_existing_head(self) -> None:
        model = create_model(
            freq_grid=self.freq_grid,
            port_count=self.port_count,
            model_kind="structured_pair_spectral_head",
            model_config={"hidden_dim": 16, "dropout": 0.0},
        )

        aux = forward_model(
            model,
            points=self.tensors["points"],
            ports=self.tensors["ports"],
            geom=self.tensors["geom"],
            frame=self.tensors["frame"],
            cuts=self.tensors["cuts"],
            nibs=self.tensors["nibs"],
            device=self.device,
            return_aux=True,
        )

        self.assertEqual(set(aux.keys()), {"s_pred"})
        self.assertEqual(aux["s_pred"].shape, (2, len(self.freq_grid), self.port_count * self.port_count * 2))


if __name__ == "__main__":
    unittest.main()
