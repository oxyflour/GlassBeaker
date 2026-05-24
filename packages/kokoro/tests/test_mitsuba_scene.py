from __future__ import annotations

import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from kokoro.mitsuba_scene import (
    build_height_field_reference_scene_dict,
    build_kokoro_ring_diagnostic_scene_dict,
    build_kokoro_scene_dict,
    orbit_scene_dicts,
    prepare_mitsuba_scene_dict,
)


class MitsubaSceneTest(unittest.TestCase):
    def test_scene_dict_uses_equivalent_plane_and_references_neural_bsdf(self) -> None:
        hdr_path = REPO_ROOT / "apps" / "web" / "public" / "studio_small_03_1k.hdr"
        checkpoint = Path("packages/kokoro/tmp/kokoro_brdf.npz")

        scene = build_kokoro_scene_dict(
            checkpoint_path=checkpoint,
            hdr_path=hdr_path,
            width=64,
            height=48,
            width_m=0.10,
            depth_m=0.10,
        )

        self.assertNotIn("environment", scene)
        self.assertEqual(scene["top_point_light"]["type"], "point")
        self.assertEqual(scene["top_point_light"]["position"], [0.0, 0.0, 0.06])
        self.assertEqual(scene["top_point_light"]["intensity"]["value"], [6.0, 6.0, 6.0])
        self.assertNotIn("inspection_light", scene)
        self.assertEqual(scene["surface"]["type"], "rectangle")
        self.assertNotIn("filename", scene["surface"])
        self.assertEqual(scene["surface"]["to_world_matrix"][0][0], 0.05)
        self.assertEqual(scene["surface"]["to_world_matrix"][1][1], 0.05)
        self.assertEqual(scene["surface"]["bsdf"]["type"], "kokoro_neural_reflector")
        self.assertEqual(scene["surface"]["bsdf"]["checkpoint"], str(checkpoint))
        self.assertIn("to_world_look_at", scene["sensor"])
        self.assertEqual(scene["sensor"]["fov"], 65.0)
        self.assertEqual(scene["sensor"]["film"]["width"], 64)
        self.assertEqual(scene["sensor"]["film"]["height"], 48)

    def test_scene_dict_can_enable_optional_inspection_light(self) -> None:
        scene = build_kokoro_scene_dict(
            checkpoint_path=Path("packages/kokoro/tmp/kokoro_brdf.npz"),
            hdr_path=REPO_ROOT / "apps" / "web" / "public" / "studio_small_03_1k.hdr",
            inspection_light_scale=1.0,
        )

        self.assertEqual(scene["inspection_light"]["type"], "rectangle")
        self.assertEqual(scene["inspection_light"]["emitter"]["radiance"]["value"], [5.0, 5.2, 5.6])

    def test_ring_diagnostic_scene_uses_top_down_camera_and_point_light(self) -> None:
        checkpoint = Path("packages/kokoro/tmp/kokoro_brdf.npz")

        scene = build_kokoro_ring_diagnostic_scene_dict(
            checkpoint_path=checkpoint,
            width=128,
            height=128,
            width_m=0.10,
            depth_m=0.10,
            spp=256,
        )

        self.assertNotIn("environment", scene)
        self.assertEqual(scene["ring_point_light"]["type"], "point")
        self.assertEqual(scene["ring_point_light"]["position"], [0.0, 0.0, 0.04])
        self.assertEqual(scene["sensor"]["to_world_look_at"]["origin"], [0.0, 0.0, 0.12])
        self.assertEqual(scene["sensor"]["to_world_look_at"]["up"], [0.0, 1.0, 0.0])
        self.assertEqual(scene["sensor"]["film"]["width"], 128)
        self.assertEqual(scene["surface"]["bsdf"]["checkpoint"], str(checkpoint))

    def test_height_field_reference_scene_embeds_height_source_in_mitsuba_bsdf(self) -> None:
        source = "def height(x, y):\n    return 0.5 * x\n"

        scene = build_height_field_reference_scene_dict(
            height_source=source,
            width=64,
            height=48,
            width_m=0.10,
            depth_m=0.10,
            spp=32,
        )

        self.assertEqual(scene["surface"]["type"], "rectangle")
        self.assertEqual(scene["surface"]["bsdf"]["type"], "kokoro_height_field_reflector")
        self.assertEqual(scene["surface"]["bsdf"]["height_source"], source)
        self.assertEqual(scene["surface"]["bsdf"]["width_m"], 0.10)
        self.assertEqual(scene["surface"]["bsdf"]["depth_m"], 0.10)
        self.assertEqual(scene["sensor"]["film"]["width"], 64)

    def test_prepare_scene_converts_serialized_look_at_transform(self) -> None:
        class FakeTransform:
            @staticmethod
            def look_at(origin, target, up):
                return {"origin": origin, "target": target, "up": up}

        class FakeMitsuba:
            ScalarTransform4f = FakeTransform

        scene = {
            "type": "scene",
            "sensor": {
                "type": "perspective",
                "to_world_look_at": {"origin": [0, -1, 1], "target": [0, 0, 0], "up": [0, 0, 1]},
            },
        }

        prepared = prepare_mitsuba_scene_dict(scene, FakeMitsuba)

        self.assertNotIn("to_world_look_at", prepared["sensor"])
        self.assertEqual(
            prepared["sensor"]["to_world"],
            {"origin": [0, -1, 1], "target": [0, 0, 0], "up": [0, 0, 1]},
        )

    def test_orbit_scene_dicts_rotate_camera_without_mutating_source(self) -> None:
        scene = {
            "type": "scene",
            "sensor": {"type": "perspective", "to_world_look_at": {"origin": [0, -1, 1], "target": [0, 0, 0], "up": [0, 0, 1]}},
        }

        frames = orbit_scene_dicts(scene, frame_count=4, radius_m=0.2, height_m=0.12)
        origins = [frame["sensor"]["to_world_look_at"]["origin"] for frame in frames]

        self.assertEqual(len(frames), 4)
        self.assertEqual(scene["sensor"]["to_world_look_at"]["origin"], [0, -1, 1])
        self.assertEqual(origins[0], [0.0, -0.2, 0.12])
        self.assertNotEqual(origins[0], origins[1])
        self.assertEqual(frames[0]["sensor"]["to_world_look_at"]["target"], [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
