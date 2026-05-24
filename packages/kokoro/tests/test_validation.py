from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from kokoro.height_field import compile_height_program
from kokoro import validation
from kokoro.validation import ValidationArtifacts, angular_error_degrees, image_metrics, write_height_ply
import validate_material


class FixedDirectionModel(torch.nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output


class ValidationTest(unittest.TestCase):
    def test_validation_artifact_names_are_stable(self) -> None:
        artifacts = ValidationArtifacts(Path("packages/kokoro/tmp/validation"))

        self.assertEqual(artifacts.flat_neural.name, "flat_neural.png")
        self.assertEqual(artifacts.flat_mirror.name, "flat_mirror.png")
        self.assertEqual(artifacts.pyramid_neural.name, "pyramid_neural.png")
        self.assertEqual(artifacts.pyramid_ply.name, "pyramid_ply.png")
        self.assertEqual(artifacts.metrics.name, "validation_metrics.json")

    def test_validation_defaults_are_full_comparison_quality(self) -> None:
        self.assertEqual(validate_material.DEFAULT_SAMPLES, 4096)
        self.assertEqual(validate_material.DEFAULT_EPOCHS, 200)
        self.assertEqual(validate_material.DEFAULT_HIDDEN_DIM, 96)
        self.assertEqual(validate_material.DEFAULT_LOBE_KAPPA, 4096.0)
        self.assertGreater(validate_material.DEFAULT_AVERAGE_PATCH_RADIUS_M, 0.0)
        self.assertGreater(validate_material.DEFAULT_AVERAGE_PATCH_SAMPLES, 1)

    def test_validation_ply_is_local_artifact_for_large_period_reference(self) -> None:
        program = compile_height_program("def height(x, y):\n    return x * 0.0\n")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "reference.ply"

            write_height_ply(program, path, grid_size=4, width_m=0.1, depth_m=0.1)

            text = path.read_text(encoding="utf-8")
            self.assertIn("element vertex 16", text)
            self.assertIn("element face 18", text)

    def test_validation_ply_faces_are_wound_upward_for_flat_surface(self) -> None:
        program = compile_height_program("def height(x, y):\n    return x * 0.0\n")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "reference.ply"

            write_height_ply(program, path, grid_size=3, width_m=0.1, depth_m=0.1)

            lines = path.read_text(encoding="utf-8").splitlines()
            first_face = lines[lines.index("end_header") + 10]
            self.assertEqual(first_face, "3 0 1 3")

    def test_image_metrics_retries_until_rendered_png_is_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "reference.png"
            candidate = root / "candidate.png"
            diff = root / "diff.png"
            Image.new("RGB", (2, 2), (12, 16, 20)).save(reference)
            Image.new("RGB", (2, 2), (10, 18, 22)).save(candidate)
            original_open = validation.Image.open
            calls = {"count": 0}

            def flaky_open(path):
                if Path(path) == reference and calls["count"] == 0:
                    calls["count"] += 1
                    raise UnidentifiedImageError("not readable yet")
                return original_open(path)

            validation.Image.open = flaky_open
            try:
                metrics = image_metrics(reference, candidate, diff, attempts=2, delay_s=0.0)
            finally:
                validation.Image.open = original_open

            self.assertEqual(calls["count"], 1)
            self.assertGreater(metrics["mean_abs_255"], 0.0)
            self.assertTrue(diff.exists())

    def test_angular_error_uses_vector_angle_near_surface_normal(self) -> None:
        z = (1.0 - 0.001 * 0.001) ** 0.5
        predicted = torch.tensor([[0.001, 0.0, z]], dtype=torch.float32)
        target = torch.tensor([[-0.001, 0.0, z]], dtype=torch.float32)

        metrics = angular_error_degrees(FixedDirectionModel(predicted), torch.zeros((1, 5)), target)

        self.assertLess(metrics["mean_deg"], 0.2)


if __name__ == "__main__":
    unittest.main()
