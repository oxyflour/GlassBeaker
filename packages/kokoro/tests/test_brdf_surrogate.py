from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from kokoro.brdf import (
    BrdfTrainingConfig,
    KokoroBrdfNet,
    build_brdf_dataset,
    export_surrogate_npz,
    load_npz_surrogate,
    make_features,
    predict_outgoing_angles,
    train_brdf_surrogate,
)
from kokoro.height_field import compile_height_program


class BrdfSurrogateTest(unittest.TestCase):
    def test_training_surrogate_reduces_reflection_direction_loss(self) -> None:
        torch.manual_seed(3)
        program = compile_height_program(
            """
def height(x, y):
    return 0.006 * x - 0.003 * y
"""
        )
        dataset = build_brdf_dataset(program, sample_count=96, width_m=0.10, depth_m=0.10, seed=5)

        result = train_brdf_surrogate(
            dataset,
            BrdfTrainingConfig(hidden_dim=24, epochs=35, batch_size=32, lr=0.03, seed=7),
        )

        self.assertLess(result.loss_history[-1], result.loss_history[0] * 0.45)
        angles = predict_outgoing_angles(
            result.model,
            x=torch.tensor([0.0, 0.01]),
            y=torch.tensor([0.0, -0.02]),
            theta=torch.tensor([0.4, 0.7]),
            phi=torch.tensor([0.1, -0.2]),
            width_m=0.10,
            depth_m=0.10,
        )
        self.assertEqual(angles.shape, (2, 2))
        self.assertTrue(torch.all(angles[:, 0] >= 0.0))
        self.assertTrue(torch.all(angles[:, 0] <= torch.pi))


    def test_npz_export_round_trips_surrogate_weights(self) -> None:
        program = compile_height_program(
            """
def height(x, y):
    return 0.002 * x
"""
        )
        dataset = build_brdf_dataset(program, sample_count=64, width_m=0.10, depth_m=0.10, seed=9)
        result = train_brdf_surrogate(
            dataset,
            BrdfTrainingConfig(hidden_dim=16, epochs=20, batch_size=32, lr=0.025, seed=2),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "kokoro_brdf.npz"
            export_surrogate_npz(result.model, checkpoint, width_m=0.10, depth_m=0.10)
            loaded = load_npz_surrogate(checkpoint)

        with torch.no_grad():
            torch_pred = result.model(dataset.features[:8]).numpy()
        np_pred = loaded.predict(dataset.features[:8].numpy())

        self.assertTrue(np.allclose(np_pred, torch_pred, atol=1e-6))
        self.assertEqual(loaded.metadata["width_m"], 0.10)
        self.assertEqual(loaded.metadata["depth_m"], 0.10)

    def test_periodic_feature_encoding_repeats_across_cells(self) -> None:
        theta = torch.tensor([0.4, 0.4])
        phi = torch.tensor([0.2, 0.2])

        features = make_features(
            torch.tensor([0.001, 0.006]),
            torch.tensor([0.002, 0.007]),
            theta,
            phi,
            width_m=0.10,
            depth_m=0.10,
            feature_period_m=0.005,
        )

        self.assertTrue(torch.allclose(features[0], features[1], atol=1e-6))

    def test_npz_export_records_periodic_feature_metadata(self) -> None:
        model = KokoroBrdfNet(hidden_dim=4)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "periodic.npz"

            export_surrogate_npz(model, checkpoint, width_m=0.10, depth_m=0.10, feature_period_m=0.005)
            loaded = load_npz_surrogate(checkpoint)

        self.assertEqual(loaded.metadata["feature_period_m"], 0.005)


if __name__ == "__main__":
    unittest.main()
