from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from kokoro.brdf import KokoroBrdfNet, export_surrogate_npz
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
            model = KokoroBrdfNet(hidden_dim=5)
            with torch.no_grad():
                for layer in model.layers:
                    layer.weight.zero_()
                    layer.bias.zero_()
                model.layers[0].weight[0, 0] = 3.0
                model.layers[1].weight[0, 0] = 3.0
                model.layers[2].weight[0, 0] = 1.0
                model.layers[2].bias[:] = torch.tensor([0.0, 0.0, 1.0])
            export_surrogate_npz(model, checkpoint, width_m=0.1, depth_m=0.1, feature_period_m=0.005)

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


if __name__ == "__main__":
    unittest.main()
