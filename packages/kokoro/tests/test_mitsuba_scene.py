from __future__ import annotations

import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from kokoro.mitsuba_scene import build_kokoro_scene_dict, orbit_scene_dicts, prepare_mitsuba_scene_dict


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

        self.assertEqual(scene["environment"]["type"], "envmap")
        self.assertEqual(scene["environment"]["filename"], str(hdr_path))
        self.assertGreater(scene["environment"]["scale"], 1.0)
        self.assertEqual(scene["inspection_light"]["type"], "rectangle")
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
