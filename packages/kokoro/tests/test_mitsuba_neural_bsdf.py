from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from kokoro.brdf import KokoroBrdfNet, export_surrogate_npz
from kokoro.mitsuba_height_field_bsdf import register_height_field_bsdf
from kokoro.mitsuba_neural_bsdf import register_kokoro_bsdf


class MitsubaNeuralBsdfTest(unittest.TestCase):
    def test_eval_pdf_and_sample_use_same_finite_lobe(self) -> None:
        import mitsuba as mi

        mi.set_variant("scalar_rgb")
        register_kokoro_bsdf(mi)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "upward.npz"
            model = KokoroBrdfNet(hidden_dim=4)
            with torch.no_grad():
                for layer in model.layers:
                    layer.weight.zero_()
                    layer.bias.zero_()
                model.layers[-1].bias[:] = torch.tensor([0.0, 0.0, 1.0])
            export_surrogate_npz(model, checkpoint, width_m=0.1, depth_m=0.1)

            scene = mi.load_dict({
                "type": "scene",
                "shape": {
                    "type": "rectangle",
                    "bsdf": {
                        "type": "kokoro_neural_reflector",
                        "checkpoint": str(checkpoint),
                        "reflectance": [0.5, 0.6, 0.7],
                        "lobe_kappa": 32.0,
                    },
                },
            })
            bsdf = scene.shapes()[0].bsdf()
            si = mi.SurfaceInteraction3f()
            si.p = mi.Point3f(0.0, 0.0, 0.0)
            si.wi = mi.Vector3f(0.0, 0.0, 1.0)
            ctx = mi.BSDFContext()
            wo = mi.Vector3f(0.0, 0.0, 1.0)

            pdf = float(bsdf.pdf(ctx, si, wo, True))
            value = bsdf.eval(ctx, si, wo, True)
            sample, weight = bsdf.sample(ctx, si, 0.25, mi.Point2f(0.4, 0.7), True)

            self.assertGreater(pdf, 0.0)
            self.assertGreater(float(value[0]), 0.0)
            self.assertGreater(float(sample.pdf), 0.0)
            sampled_value = bsdf.eval(ctx, si, sample.wo, True)
            self.assertAlmostEqual(float(weight[0]), float(sampled_value[0]) / float(sample.pdf), places=5)
            self.assertAlmostEqual(float(weight[1]), float(sampled_value[1]) / float(sample.pdf), places=5)
            self.assertAlmostEqual(float(weight[2]), float(sampled_value[2]) / float(sample.pdf), places=5)

    def test_periodic_checkpoint_features_repeat_across_cells(self) -> None:
        import mitsuba as mi

        mi.set_variant("scalar_rgb")
        register_kokoro_bsdf(mi)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "periodic.npz"
            model = KokoroBrdfNet(hidden_dim=5, input_dim=5)
            with torch.no_grad():
                for layer in model.layers:
                    layer.weight.zero_()
                    layer.bias.zero_()
                model.layers[0].weight[0, 0] = 3.0
                model.layers[1].weight[0, 0] = 3.0
                model.layers[2].weight[0, 0] = 1.0
                model.layers[2].bias[:] = torch.tensor([0.0, 0.0, 1.0])
            export_surrogate_npz(
                model,
                checkpoint,
                width_m=0.1,
                depth_m=0.1,
                feature_period_m=0.005,
                include_position_features=True,
            )

            scene = mi.load_dict({
                "type": "scene",
                "shape": {
                    "type": "rectangle",
                    "bsdf": {"type": "kokoro_neural_reflector", "checkpoint": str(checkpoint), "lobe_kappa": 16.0},
                },
            })
            bsdf = scene.shapes()[0].bsdf()
            ctx = mi.BSDFContext()
            wo = mi.Vector3f(0.5, 0.0, 0.8660254)
            first = mi.SurfaceInteraction3f()
            first.p = mi.Point3f(0.001, 0.0, 0.0)
            first.wi = mi.Vector3f(0.0, 0.0, 1.0)
            second = mi.SurfaceInteraction3f()
            second.p = mi.Point3f(0.006, 0.0, 0.0)
            second.wi = mi.Vector3f(0.0, 0.0, 1.0)

            self.assertAlmostEqual(float(bsdf.pdf(ctx, first, wo, True)), float(bsdf.pdf(ctx, second, wo, True)), places=6)

    def test_averaged_checkpoint_ignores_surface_position(self) -> None:
        import mitsuba as mi

        mi.set_variant("scalar_rgb")
        register_kokoro_bsdf(mi)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "averaged.npz"
            model = KokoroBrdfNet(hidden_dim=5, input_dim=3)
            with torch.no_grad():
                for layer in model.layers:
                    layer.weight.zero_()
                    layer.bias.zero_()
                model.layers[0].weight[0, 1] = 3.0
                model.layers[1].weight[0, 0] = 3.0
                model.layers[2].weight[0, 0] = 1.0
                model.layers[2].bias[:] = torch.tensor([0.0, 0.0, 1.0])
            export_surrogate_npz(model, checkpoint, width_m=0.1, depth_m=0.1)

            scene = mi.load_dict({
                "type": "scene",
                "shape": {
                    "type": "rectangle",
                    "bsdf": {"type": "kokoro_neural_reflector", "checkpoint": str(checkpoint), "lobe_kappa": 16.0},
                },
            })
            bsdf = scene.shapes()[0].bsdf()
            ctx = mi.BSDFContext()
            wo = mi.Vector3f(0.5, 0.0, 0.8660254)
            first = mi.SurfaceInteraction3f()
            first.p = mi.Point3f(-0.04, -0.03, 0.0)
            first.wi = mi.Vector3f(0.0, 0.0, 1.0)
            second = mi.SurfaceInteraction3f()
            second.p = mi.Point3f(0.04, 0.03, 0.0)
            second.wi = mi.Vector3f(0.0, 0.0, 1.0)

            self.assertAlmostEqual(float(bsdf.pdf(ctx, first, wo, True)), float(bsdf.pdf(ctx, second, wo, True)), places=6)

    def test_sine_activation_checkpoint_loads_in_mitsuba_bsdf(self) -> None:
        import mitsuba as mi

        mi.set_variant("scalar_rgb")
        register_kokoro_bsdf(mi)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "sine.npz"
            model = KokoroBrdfNet(hidden_dim=6, activation="sine", omega_0=10.0)
            export_surrogate_npz(model, checkpoint, width_m=0.1, depth_m=0.1)

            scene = mi.load_dict({
                "type": "scene",
                "shape": {
                    "type": "rectangle",
                    "bsdf": {"type": "kokoro_neural_reflector", "checkpoint": str(checkpoint), "lobe_kappa": 16.0},
                },
            })
            bsdf = scene.shapes()[0].bsdf()
            si = mi.SurfaceInteraction3f()
            si.p = mi.Point3f(0.01, -0.02, 0.0)
            si.wi = mi.Vector3f(0.0, 0.0, 1.0)

            pdf = float(bsdf.pdf(mi.BSDFContext(), si, mi.Vector3f(0.0, 0.0, 1.0), True))
            self.assertGreaterEqual(pdf, 0.0)

    def test_cone_checkpoint_prefers_ring_direction_over_axis(self) -> None:
        import mitsuba as mi

        mi.set_variant("scalar_rgb")
        register_kokoro_bsdf(mi)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "cone.npz"
            model = KokoroBrdfNet(hidden_dim=5, output_dim=4)
            with torch.no_grad():
                for layer in model.layers:
                    layer.weight.zero_()
                    layer.bias.zero_()
                model.layers[-1].bias[:] = torch.tensor([0.0, 0.0, 1.0, 0.0])
            export_surrogate_npz(model, checkpoint, width_m=0.1, depth_m=0.1)

            scene = mi.load_dict({
                "type": "scene",
                "shape": {
                    "type": "rectangle",
                    "bsdf": {
                        "type": "kokoro_neural_reflector",
                        "checkpoint": str(checkpoint),
                        "lobe_kappa": 128.0,
                        "ring_lobe_count": 16,
                    },
                },
            })
            bsdf = scene.shapes()[0].bsdf()
            si = mi.SurfaceInteraction3f()
            si.p = mi.Point3f(0.0, 0.0, 0.0)
            si.wi = mi.Vector3f(0.0, 0.0, 1.0)
            ctx = mi.BSDFContext()
            axis = mi.Vector3f(0.0, 0.0, 1.0)
            on_ring = mi.Vector3f(0.8660254, 0.0, 0.5)

            self.assertGreater(float(bsdf.pdf(ctx, si, on_ring, True)), float(bsdf.pdf(ctx, si, axis, True)) * 5.0)

    def test_normal_target_checkpoint_reflects_runtime_incident_direction(self) -> None:
        import mitsuba as mi

        mi.set_variant("scalar_rgb")
        register_kokoro_bsdf(mi)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "normal_target.npz"
            model = KokoroBrdfNet(hidden_dim=4, input_dim=2)
            normal = torch.nn.functional.normalize(torch.tensor([0.6, 0.0, 0.8]), dim=0)
            with torch.no_grad():
                for layer in model.layers:
                    layer.weight.zero_()
                    layer.bias.zero_()
                model.layers[-1].bias[:] = normal
            export_surrogate_npz(
                model,
                checkpoint,
                width_m=0.1,
                depth_m=0.1,
                include_incident_features=False,
                target_mode="normal",
            )

            scene = mi.load_dict({
                "type": "scene",
                "shape": {
                    "type": "rectangle",
                    "bsdf": {
                        "type": "kokoro_neural_reflector",
                        "checkpoint": str(checkpoint),
                        "lobe_kappa": 512.0,
                    },
                },
            })
            bsdf = scene.shapes()[0].bsdf()
            si = mi.SurfaceInteraction3f()
            si.p = mi.Point3f(0.0, 0.0, 0.0)
            si.wi = mi.Vector3f(0.0, 0.0, 1.0)
            ctx = mi.BSDFContext()
            reflected = mi.Vector3f(0.96, 0.0, 0.28)
            upward = mi.Vector3f(0.0, 0.0, 1.0)

            self.assertGreater(float(bsdf.pdf(ctx, si, reflected, True)), float(bsdf.pdf(ctx, si, upward, True)) * 100.0)

    def test_oriented_cone_checkpoint_preserves_fourfold_phase(self) -> None:
        import mitsuba as mi

        mi.set_variant("scalar_rgb")
        register_kokoro_bsdf(mi)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "oriented_cone.npz"
            model = KokoroBrdfNet(hidden_dim=5, output_dim=6)
            with torch.no_grad():
                for layer in model.layers:
                    layer.weight.zero_()
                    layer.bias.zero_()
                model.layers[-1].bias[:] = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
            export_surrogate_npz(model, checkpoint, width_m=0.1, depth_m=0.1)

            scene = mi.load_dict({
                "type": "scene",
                "shape": {
                    "type": "rectangle",
                    "bsdf": {
                        "type": "kokoro_neural_reflector",
                        "checkpoint": str(checkpoint),
                        "lobe_kappa": 256.0,
                    },
                },
            })
            bsdf = scene.shapes()[0].bsdf()
            si = mi.SurfaceInteraction3f()
            si.p = mi.Point3f(0.0, 0.0, 0.0)
            si.wi = mi.Vector3f(0.0, 0.0, 1.0)
            ctx = mi.BSDFContext()
            on_lobe = mi.Vector3f(0.8660254, 0.0, 0.5)
            between_lobes = mi.Vector3f(0.6123724, 0.6123724, 0.5)

            self.assertGreater(float(bsdf.pdf(ctx, si, on_lobe, True)), float(bsdf.pdf(ctx, si, between_lobes, True)) * 10.0)

    def test_multiscale_position_checkpoint_loads_in_mitsuba_bsdf(self) -> None:
        import mitsuba as mi

        mi.set_variant("scalar_rgb")
        register_kokoro_bsdf(mi)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "fourier.npz"
            model = KokoroBrdfNet(hidden_dim=6, activation="sine", omega_0=8.0, input_dim=13)
            export_surrogate_npz(
                model,
                checkpoint,
                width_m=0.1,
                depth_m=0.1,
                position_frequency_count=2,
                include_position_features=True,
            )

            scene = mi.load_dict({
                "type": "scene",
                "shape": {
                    "type": "rectangle",
                    "bsdf": {"type": "kokoro_neural_reflector", "checkpoint": str(checkpoint), "lobe_kappa": 16.0},
                },
            })
            bsdf = scene.shapes()[0].bsdf()
            si = mi.SurfaceInteraction3f()
            si.p = mi.Point3f(0.01, 0.02, 0.0)
            si.wi = mi.Vector3f(0.0, 0.0, 1.0)

            pdf = float(bsdf.pdf(mi.BSDFContext(), si, mi.Vector3f(0.0, 0.0, 1.0), True))
            self.assertGreaterEqual(pdf, 0.0)

    def test_local_feature_period_checkpoint_uses_macro_and_cell_phase_features(self) -> None:
        import mitsuba as mi

        mi.set_variant("scalar_rgb")
        register_kokoro_bsdf(mi)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "local_period.npz"
            model = KokoroBrdfNet(hidden_dim=5, input_dim=7)
            with torch.no_grad():
                for layer in model.layers:
                    layer.weight.zero_()
                    layer.bias.zero_()
                model.layers[0].weight[0, 0] = 2.0
                model.layers[0].weight[1, 2] = 2.0
                model.layers[1].weight[0, 0] = 2.0
                model.layers[1].weight[0, 1] = 2.0
                model.layers[2].weight[0, 0] = 1.0
                model.layers[2].bias[:] = torch.tensor([0.0, 0.0, 1.0])
            export_surrogate_npz(
                model,
                checkpoint,
                width_m=0.1,
                depth_m=0.1,
                local_feature_period_m=0.005,
                include_position_features=True,
            )

            scene = mi.load_dict({
                "type": "scene",
                "shape": {
                    "type": "rectangle",
                    "bsdf": {"type": "kokoro_neural_reflector", "checkpoint": str(checkpoint), "lobe_kappa": 16.0},
                },
            })
            bsdf = scene.shapes()[0].bsdf()
            ctx = mi.BSDFContext()
            wo = mi.Vector3f(0.5, 0.0, 0.8660254)
            first = mi.SurfaceInteraction3f()
            first.p = mi.Point3f(0.001, 0.0, 0.0)
            first.wi = mi.Vector3f(0.0, 0.0, 1.0)
            second = mi.SurfaceInteraction3f()
            second.p = mi.Point3f(0.006, 0.0, 0.0)
            second.wi = mi.Vector3f(0.0, 0.0, 1.0)
            third = mi.SurfaceInteraction3f()
            third.p = mi.Point3f(0.002, 0.0, 0.0)
            third.wi = mi.Vector3f(0.0, 0.0, 1.0)

            first_pdf = float(bsdf.pdf(ctx, first, wo, True))
            second_pdf = float(bsdf.pdf(ctx, second, wo, True))
            third_pdf = float(bsdf.pdf(ctx, third, wo, True))
            self.assertGreater(abs(first_pdf - second_pdf), 1e-6)
            self.assertGreater(abs(first_pdf - third_pdf), 1e-6)

    def test_radial_cell_feature_checkpoint_uses_rotated_local_coordinates(self) -> None:
        import mitsuba as mi

        mi.set_variant("scalar_rgb")
        register_kokoro_bsdf(mi)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "radial_cell.npz"
            model = KokoroBrdfNet(hidden_dim=5, input_dim=9)
            with torch.no_grad():
                for layer in model.layers:
                    layer.weight.zero_()
                    layer.bias.zero_()
                model.layers[0].weight[0, 2] = 3.0
                model.layers[1].weight[0, 0] = 3.0
                model.layers[2].weight[0, 0] = 1.0
                model.layers[2].bias[:] = torch.tensor([0.0, 0.0, 1.0])
            export_surrogate_npz(
                model,
                checkpoint,
                width_m=0.1,
                depth_m=0.1,
                radial_cell_feature_period_m=500e-6,
                radial_cell_feature_max_rotation_rad=1.57079632679,
                include_position_features=True,
            )

            scene = mi.load_dict({
                "type": "scene",
                "shape": {
                    "type": "rectangle",
                    "bsdf": {"type": "kokoro_neural_reflector", "checkpoint": str(checkpoint), "lobe_kappa": 16.0},
                },
            })
            bsdf = scene.shapes()[0].bsdf()
            ctx = mi.BSDFContext()
            wo = mi.Vector3f(0.5, 0.0, 0.8660254)
            first = mi.SurfaceInteraction3f()
            first.p = mi.Point3f(55e-6, 115e-6, 0.0)
            first.wi = mi.Vector3f(0.0, 0.0, 1.0)
            second = mi.SurfaceInteraction3f()
            second.p = mi.Point3f(225e-6, 115e-6, 0.0)
            second.wi = mi.Vector3f(0.0, 0.0, 1.0)

            self.assertGreater(abs(float(bsdf.pdf(ctx, first, wo, True)) - float(bsdf.pdf(ctx, second, wo, True))), 1e-6)

    def test_radial_cell_facet_feature_checkpoint_uses_dominant_slope_hint(self) -> None:
        import mitsuba as mi

        mi.set_variant("scalar_rgb")
        register_kokoro_bsdf(mi)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "radial_cell_facet.npz"
            model = KokoroBrdfNet(hidden_dim=5, input_dim=11)
            with torch.no_grad():
                for layer in model.layers:
                    layer.weight.zero_()
                    layer.bias.zero_()
                model.layers[0].weight[0, 6] = 3.0
                model.layers[1].weight[0, 0] = 3.0
                model.layers[2].weight[0, 0] = 1.0
                model.layers[2].bias[:] = torch.tensor([0.0, 0.0, 1.0])
            export_surrogate_npz(
                model,
                checkpoint,
                width_m=0.1,
                depth_m=0.1,
                radial_cell_feature_period_m=500e-6,
                radial_cell_facet_features=True,
                include_position_features=True,
            )

            scene = mi.load_dict({
                "type": "scene",
                "shape": {
                    "type": "rectangle",
                    "bsdf": {"type": "kokoro_neural_reflector", "checkpoint": str(checkpoint), "lobe_kappa": 16.0},
                },
            })
            bsdf = scene.shapes()[0].bsdf()
            ctx = mi.BSDFContext()
            wo = mi.Vector3f(0.5, 0.0, 0.8660254)
            x_facet = mi.SurfaceInteraction3f()
            x_facet.p = mi.Point3f(225e-6, 115e-6, 0.0)
            x_facet.wi = mi.Vector3f(0.0, 0.0, 1.0)
            y_facet = mi.SurfaceInteraction3f()
            y_facet.p = mi.Point3f(55e-6, 225e-6, 0.0)
            y_facet.wi = mi.Vector3f(0.0, 0.0, 1.0)

            self.assertGreater(abs(float(bsdf.pdf(ctx, x_facet, wo, True)) - float(bsdf.pdf(ctx, y_facet, wo, True))), 1e-6)

    def test_height_field_reference_bsdf_evaluates_embedded_source(self) -> None:
        import mitsuba as mi

        mi.set_variant("scalar_rgb")
        register_height_field_bsdf(mi)

        def load_pdf(source: str, wo) -> float:
            scene = mi.load_dict({
                "type": "scene",
                "shape": {
                    "type": "rectangle",
                    "bsdf": {
                        "type": "kokoro_height_field_reflector",
                        "height_source": source,
                        "normal_step_m": 1e-4,
                        "lobe_kappa": 64.0,
                    },
                },
            })
            si = mi.SurfaceInteraction3f()
            si.p = mi.Point3f(0.0, 0.0, 0.0)
            si.wi = mi.Vector3f(0.0, 0.0, 1.0)
            return float(scene.shapes()[0].bsdf().pdf(mi.BSDFContext(), si, wo, True))

        positive_x_lobe = mi.Vector3f(0.8, 0.0, 0.6)
        positive_x_source = "def height(x, y):\n    return -0.5 * x\n"
        negative_x_source = "def height(x, y):\n    return 0.5 * x\n"

        self.assertGreater(
            load_pdf(positive_x_source, positive_x_lobe),
            load_pdf(negative_x_source, positive_x_lobe) * 100.0,
        )


if __name__ == "__main__":
    unittest.main()
