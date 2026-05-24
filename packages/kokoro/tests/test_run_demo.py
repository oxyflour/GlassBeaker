from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from kokoro.height_field import compile_height_program
from run_demo import (
    DEFAULT_ACTIVATION,
    DEFAULT_AVERAGE_PATCH_RADIUS_M,
    DEFAULT_AVERAGE_PATCH_SAMPLES,
    DEFAULT_FEATURE_PERIOD_M,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_FILM_HEIGHT,
    DEFAULT_FILM_WIDTH,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_HIDDEN_LAYERS,
    DEFAULT_HOLDOUT_GRID_SIZE,
    DEFAULT_HOLDOUT_PHI_COUNT,
    DEFAULT_HOLDOUT_THETA_COUNT,
    DEFAULT_HDR_PATH,
    DEFAULT_HEIGHT_SOURCE,
    DEFAULT_INSPECTION_LIGHT_SCALE,
    DEFAULT_LIGHT_SOURCE,
    DEFAULT_LOCAL_FEATURE_PERIOD_M,
    DEFAULT_LOBE_KAPPA,
    DEFAULT_LR,
    DEFAULT_OMEGA_0,
    DEFAULT_POSITION_FREQUENCY_COUNT,
    DEFAULT_RADIAL_CELL_FEATURE_MAX_ROTATION_RAD,
    DEFAULT_RADIAL_CELL_FACET_FEATURES,
    DEFAULT_RADIAL_CELL_FEATURE_PERIOD_M,
    DEFAULT_RECONSTRUCTION_FILTER,
    DEFAULT_RENDER_SAMPLER,
    DEFAULT_SAMPLES,
    DEFAULT_SPP,
    DEFAULT_TARGET_MODE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_FRAMES,
    render_output_path,
    reference_render_output_path,
    ring_diagnostic_output_path,
    video_reference_output_path,
    video_output_path,
)


class RunDemoTest(unittest.TestCase):
    def test_default_output_dir_stays_inside_kokoro_package(self) -> None:
        self.assertEqual(DEFAULT_OUTPUT_DIR, Path("packages/kokoro/tmp"))

    def test_default_feature_period_does_not_assume_periodic_height_fields(self) -> None:
        self.assertIsNone(DEFAULT_FEATURE_PERIOD_M)

    def test_default_height_source_uses_radial_rotated_pyramid_baseline(self) -> None:
        self.assertIn("radial_rotated_pyramid_height", DEFAULT_HEIGHT_SOURCE)
        self.assertIn("max_rotation_rad=", DEFAULT_HEIGHT_SOURCE)

    def test_default_height_source_is_not_zero_rotation(self) -> None:
        zero_rotation = compile_height_program(
            """
def height(x, y):
    return radial_rotated_pyramid_height(
        x,
        y,
        period_m=500e-6,
        amplitude_m=150e-6,
        max_rotation_rad=0.0,
    )
"""
        )
        default = compile_height_program(DEFAULT_HEIGHT_SOURCE)
        x = torch.tensor([0.0402, 0.0302, 0.0202])
        y = torch.tensor([0.00017, 0.01017, 0.02017])

        self.assertFalse(torch.allclose(default.evaluate(x, y), zero_rotation.evaluate(x, y), atol=1e-8))

    def test_default_uses_normal_target_with_explicit_facet_features(self) -> None:
        self.assertEqual(DEFAULT_TARGET_MODE, "normal")
        self.assertEqual(DEFAULT_POSITION_FREQUENCY_COUNT, 0)
        self.assertIsNone(DEFAULT_LOCAL_FEATURE_PERIOD_M)
        self.assertEqual(DEFAULT_AVERAGE_PATCH_RADIUS_M, 0.0)
        self.assertEqual(DEFAULT_AVERAGE_PATCH_SAMPLES, 1)
        self.assertEqual(DEFAULT_LOBE_KAPPA, 2048.0)
        self.assertEqual(DEFAULT_RADIAL_CELL_FEATURE_PERIOD_M, 500e-6)
        self.assertTrue(DEFAULT_RADIAL_CELL_FACET_FEATURES)
        self.assertAlmostEqual(DEFAULT_RADIAL_CELL_FEATURE_MAX_ROTATION_RAD, torch.pi / 2.0)

    def test_default_neural_lobe_width_targets_spp1024_band_visibility(self) -> None:
        self.assertEqual(DEFAULT_LOBE_KAPPA, 2048.0)

    def test_default_sampler_reduces_render_noise_without_reconstruction_blur(self) -> None:
        self.assertEqual(DEFAULT_RENDER_SAMPLER, "ldsampler")
        self.assertEqual(DEFAULT_RECONSTRUCTION_FILTER, "box")
        self.assertEqual(DEFAULT_SPP, 1024)
        self.assertEqual(DEFAULT_FILM_WIDTH, 384)
        self.assertEqual(DEFAULT_FILM_HEIGHT, 288)

    def test_default_activation_is_stable_for_cell_phase_brdf(self) -> None:
        self.assertEqual(DEFAULT_ACTIVATION, "tanh")
        self.assertEqual(DEFAULT_OMEGA_0, 4.0)

    def test_default_training_budget_matches_multiscale_mlp(self) -> None:
        self.assertEqual(DEFAULT_SAMPLES, 16384)
        self.assertEqual(DEFAULT_EPOCHS, 240)
        self.assertEqual(DEFAULT_HIDDEN_DIM, 128)
        self.assertEqual(DEFAULT_HIDDEN_LAYERS, 3)
        self.assertEqual(DEFAULT_BATCH_SIZE, 256)
        self.assertEqual(DEFAULT_LR, 2e-3)

    def test_default_holdout_grid_is_dense_enough_for_direction_diagnostics(self) -> None:
        self.assertEqual(DEFAULT_HOLDOUT_GRID_SIZE, 32)
        self.assertEqual(DEFAULT_HOLDOUT_THETA_COUNT, 5)
        self.assertEqual(DEFAULT_HOLDOUT_PHI_COUNT, 8)

    def test_default_inspection_light_is_disabled(self) -> None:
        self.assertEqual(DEFAULT_INSPECTION_LIGHT_SCALE, 0.0)

    def test_default_light_source_preserves_point_light_but_has_hdr_asset_ready(self) -> None:
        self.assertEqual(DEFAULT_LIGHT_SOURCE, "point")
        self.assertEqual(DEFAULT_HDR_PATH, Path("apps/web/public/studio_small_03_1k.hdr"))

    def test_default_video_duration_is_five_seconds_per_orbit(self) -> None:
        self.assertEqual(DEFAULT_VIDEO_FRAMES / DEFAULT_VIDEO_FPS, 5.0)

    def test_render_output_path_is_png(self) -> None:
        self.assertEqual(render_output_path(Path("tmp/kokoro")), Path("tmp/kokoro/kokoro_render.png"))

    def test_reference_render_paths_are_png(self) -> None:
        self.assertEqual(
            reference_render_output_path(Path("tmp/kokoro")),
            Path("tmp/kokoro/kokoro_height_reference.png"),
        )

    def test_ring_diagnostic_output_path_is_png(self) -> None:
        self.assertEqual(
            ring_diagnostic_output_path(Path("tmp/kokoro")),
            Path("tmp/kokoro/kokoro_ring_diagnostic.png"),
        )

    def test_video_output_path_is_mp4(self) -> None:
        self.assertEqual(video_output_path(Path("tmp/kokoro")), Path("tmp/kokoro/kokoro_orbit.mp4"))

    def test_video_reference_output_path_is_mp4(self) -> None:
        self.assertEqual(
            video_reference_output_path(Path("tmp/kokoro")),
            Path("tmp/kokoro/kokoro_orbit_reference.mp4"),
        )


if __name__ == "__main__":
    unittest.main()
