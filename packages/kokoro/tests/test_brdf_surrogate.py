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
    average_patch_reflections,
    build_brdf_dataset,
    angles_to_direction,
    export_surrogate_npz,
    load_npz_surrogate,
    make_features,
    patch_reflection_moments,
    predict_outgoing_angles,
    train_brdf_surrogate,
    reflect,
)
from kokoro.height_field import compile_height_program


class BrdfSurrogateTest(unittest.TestCase):
    def test_mlp_supports_five_hidden_layers(self) -> None:
        model = KokoroBrdfNet(hidden_dim=128, hidden_layer_count=5, input_dim=41)

        self.assertEqual(model.hidden_layer_count, 5)
        self.assertEqual(len(model.layers), 6)
        self.assertEqual(model.layers[0].in_features, 41)
        self.assertEqual(model.layers[0].out_features, 128)
        self.assertEqual(model.layers[4].out_features, 128)
        self.assertEqual(model.layers[-1].out_features, 3)

    def test_mlp_rejects_hidden_layer_counts_above_demo_limit(self) -> None:
        with self.assertRaises(ValueError):
            KokoroBrdfNet(hidden_layer_count=6)

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
            BrdfTrainingConfig(hidden_dim=24, epochs=35, batch_size=32, lr=0.01, seed=7),
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

    def test_default_features_include_macro_surface_position(self) -> None:
        features = make_features(
            torch.tensor([0.001, 0.006]),
            torch.tensor([0.002, 0.007]),
            torch.tensor([0.4, 0.4]),
            torch.tensor([0.2, 0.2]),
            width_m=0.10,
            depth_m=0.10,
        )

        self.assertEqual(features.shape, (2, 5))
        self.assertFalse(torch.allclose(features[0], features[1], atol=1e-6))

    def test_patch_average_reflection_differs_from_center_microfacet(self) -> None:
        program = compile_height_program(
            """
def height(x, y):
    return 0.0004 * torch.sin(600.0 * x)
"""
        )
        x = torch.tensor([0.0])
        y = torch.tensor([0.0])
        wi = angles_to_direction(torch.tensor([0.6]), torch.tensor([0.0]))

        center = average_patch_reflections(program, x, y, wi, patch_radius_m=0.0, patch_sample_count=1, seed=3)
        averaged = average_patch_reflections(program, x, y, wi, patch_radius_m=0.01, patch_sample_count=256, seed=3)

        self.assertGreater(torch.linalg.norm(center - averaged).item(), 0.05)

    def test_reflect_keeps_outgoing_lobes_above_surface_for_mitsuba_bsdf(self) -> None:
        wi = torch.tensor([[0.8, 0.0, 0.6]], dtype=torch.float32)
        normal = torch.nn.functional.normalize(torch.tensor([[-0.8, 0.0, 0.6]], dtype=torch.float32), dim=1)

        reflected = reflect(wi, normal)

        self.assertGreaterEqual(reflected[0, 2].item(), 0.0)
        self.assertAlmostEqual(torch.linalg.vector_norm(reflected[0]).item(), 1.0, places=6)

    def test_patch_reflection_moments_preserve_ring_cone_angle(self) -> None:
        program = compile_height_program(
            """
def height(x, y):
    return radial_rotated_pyramid_height(
        x,
        y,
        period_m=500e-6,
        amplitude_m=150e-6,
        max_rotation_rad=torch.pi / 4.0,
    )
"""
        )
        x = torch.tensor([0.03])
        y = torch.tensor([0.0])
        wi = angles_to_direction(torch.tensor([0.0]), torch.tensor([0.0]))

        moments = patch_reflection_moments(program, x, y, wi, patch_radius_m=0.003, patch_sample_count=4096, seed=4)

        self.assertEqual(moments.shape, (1, 6))
        self.assertGreater(moments[0, 2].item(), 0.95)
        self.assertLess(moments[0, 3].item(), 0.8)
        self.assertAlmostEqual(torch.linalg.vector_norm(moments[0, 4:6]).item(), 1.0, places=5)

    def test_patch_average_dataset_targets_include_cone_cosine(self) -> None:
        program = compile_height_program(
            """
def height(x, y):
    return radial_rotated_pyramid_height(x, y, period_m=500e-6, amplitude_m=150e-6)
"""
        )

        dataset = build_brdf_dataset(
            program,
            sample_count=12,
            width_m=0.10,
            depth_m=0.10,
            seed=8,
            average_patch_radius_m=0.001,
            average_patch_sample_count=32,
        )

        self.assertEqual(dataset.targets.shape, (12, 6))
        self.assertTrue(torch.all(dataset.targets[:, 3] >= 0.0))
        self.assertTrue(torch.all(dataset.targets[:, 3] <= 1.0))

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

    def test_npz_export_round_trips_five_hidden_layer_surrogate_weights(self) -> None:
        model = KokoroBrdfNet(hidden_dim=8, hidden_layer_count=5, input_dim=5, activation="sine", omega_0=4.0)
        features = torch.tensor([
            [0.2, -0.3, 0.8, 0.4, -0.1],
            [-0.7, 0.1, 0.6, -0.2, 0.3],
        ], dtype=torch.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "deep.npz"

            export_surrogate_npz(model, checkpoint, width_m=0.10, depth_m=0.10)
            loaded = load_npz_surrogate(checkpoint)

        with torch.no_grad():
            torch_pred = model(features).numpy()
        np_pred = loaded.predict(features.numpy())

        self.assertTrue(np.allclose(np_pred, torch_pred, atol=1e-6))

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
            include_position_features=True,
        )

        self.assertTrue(torch.allclose(features[0], features[1], atol=1e-6))

    def test_multiscale_position_features_expand_without_known_period(self) -> None:
        features = make_features(
            torch.tensor([0.001, 0.006]),
            torch.tensor([0.002, 0.007]),
            torch.tensor([0.4, 0.4]),
            torch.tensor([0.2, 0.2]),
            width_m=0.10,
            depth_m=0.10,
            position_frequency_count=4,
            include_position_features=True,
        )

        self.assertEqual(features.shape, (2, 21))
        self.assertFalse(torch.allclose(features[0], features[1]))

    def test_local_feature_period_appends_cell_phase_without_dropping_macro_position(self) -> None:
        theta = torch.tensor([0.4, 0.4])
        phi = torch.tensor([0.2, 0.2])

        features = make_features(
            torch.tensor([0.001, 0.006]),
            torch.tensor([0.002, 0.007]),
            theta,
            phi,
            width_m=0.10,
            depth_m=0.10,
            local_feature_period_m=0.005,
            include_position_features=True,
        )

        self.assertEqual(features.shape, (2, 7))
        self.assertFalse(torch.allclose(features[0, :2], features[1, :2], atol=1e-6))
        self.assertTrue(torch.allclose(features[0, 2:4], features[1, 2:4], atol=1e-6))

    def test_npz_export_records_periodic_feature_metadata(self) -> None:
        model = KokoroBrdfNet(hidden_dim=4)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "periodic.npz"

            export_surrogate_npz(
                model,
                checkpoint,
                width_m=0.10,
                depth_m=0.10,
                feature_period_m=0.005,
                position_frequency_count=4,
                include_position_features=True,
            )
            loaded = load_npz_surrogate(checkpoint)

        self.assertEqual(loaded.metadata["feature_period_m"], 0.005)
        self.assertEqual(loaded.metadata["position_frequency_count"], 4)
        self.assertTrue(loaded.metadata["include_position_features"])

    def test_npz_export_records_local_feature_period_metadata(self) -> None:
        model = KokoroBrdfNet(hidden_dim=4, input_dim=7)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "local_period.npz"

            export_surrogate_npz(
                model,
                checkpoint,
                width_m=0.10,
                depth_m=0.10,
                local_feature_period_m=0.005,
                include_position_features=True,
            )
            loaded = load_npz_surrogate(checkpoint)

        self.assertEqual(loaded.metadata["local_feature_period_m"], 0.005)

    def test_npz_export_records_averaged_brdf_metadata_by_default(self) -> None:
        model = KokoroBrdfNet(hidden_dim=4)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "averaged.npz"

            export_surrogate_npz(model, checkpoint, width_m=0.10, depth_m=0.10)
            loaded = load_npz_surrogate(checkpoint)

        self.assertTrue(loaded.metadata["include_position_features"])
        self.assertEqual(loaded.metadata["input_dim"], 5)

    def test_npz_export_records_patch_average_metadata(self) -> None:
        model = KokoroBrdfNet(hidden_dim=4)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "patch.npz"

            export_surrogate_npz(
                model,
                checkpoint,
                width_m=0.10,
                depth_m=0.10,
                average_patch_radius_m=0.002,
                average_patch_sample_count=16,
            )
            loaded = load_npz_surrogate(checkpoint)

        self.assertEqual(loaded.metadata["average_patch_radius_m"], 0.002)
        self.assertEqual(loaded.metadata["average_patch_sample_count"], 16)

    def test_npz_export_round_trips_sine_activation_metadata(self) -> None:
        model = KokoroBrdfNet(hidden_dim=8, activation="sine", omega_0=12.0)
        features = torch.tensor([[0.2, -0.3, 0.8, 0.4, -0.1]], dtype=torch.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "siren.npz"

            export_surrogate_npz(model, checkpoint, width_m=0.10, depth_m=0.10)
            loaded = load_npz_surrogate(checkpoint)

        with torch.no_grad():
            torch_pred = model(features).numpy()
        self.assertEqual(loaded.metadata["activation"], "sine")
        self.assertEqual(loaded.metadata["omega_0"], 12.0)
        self.assertTrue(np.allclose(loaded.predict(features.numpy()), torch_pred, atol=1e-6))

    def test_npz_export_records_hidden_layer_count_metadata(self) -> None:
        model = KokoroBrdfNet(hidden_dim=8, hidden_layer_count=5)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "layers.npz"

            export_surrogate_npz(model, checkpoint, width_m=0.10, depth_m=0.10)
            loaded = load_npz_surrogate(checkpoint)

        self.assertEqual(loaded.metadata["hidden_layer_count"], 5)


if __name__ == "__main__":
    unittest.main()
