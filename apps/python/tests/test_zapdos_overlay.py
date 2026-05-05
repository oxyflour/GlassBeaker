from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils import zapdos_overlay


class ZapdosOverlayTest(unittest.TestCase):
    def test_scene_revision_ignores_pose_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene.usda"
            scene.write_text("#usda 1.0\n", encoding="utf-8")
            base = zapdos_overlay.default_overlay_state("C:/assets")
            base["instances"] = [{
                "id": "table_000_01",
                "asset_id": "table_000",
                "url": "objects/table_000/Aligned.usda",
                "motion": "static",
                "placement": {"kind": "floor_at_xy", "xy": [0.0, 0.0], "z_offset": 0.0, "yaw": 0.0},
            }]
            edited = json.loads(json.dumps(base))
            edited["pose_overrides"]["Scene_table_000_01"] = {
                "pos": [1.0, 2.0, 3.0],
                "quat": [1.0, 0.0, 0.0, 0.0],
            }

            self.assertEqual(
                zapdos_overlay.scene_revision(scene, base),
                zapdos_overlay.scene_revision(scene, edited),
            )

    def test_bundle_revision_changes_when_robot_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            robot_a = Path(tmp) / "robot-a.usda"
            robot_b = Path(tmp) / "robot-b.usda"
            scene = Path(tmp) / "scene.usda"
            for path in (robot_a, robot_b, scene):
                path.write_text("#usda 1.0\n", encoding="utf-8")

            overlay = zapdos_overlay.default_overlay_state("C:/assets")
            self.assertNotEqual(
                zapdos_overlay.bundle_revision(robot_a, scene, overlay),
                zapdos_overlay.bundle_revision(robot_b, scene, overlay),
            )

    def test_save_and_load_overlay_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "overlay.json"
            overlay = zapdos_overlay.default_overlay_state("C:/assets")
            overlay["instances"].append({
                "id": "crate_001_01",
                "asset_id": "crate_001",
                "url": "objects/crate_001/Aligned.usda",
                "motion": "dynamic",
                "placement": {
                    "kind": "world_pose",
                    "pos": [1.0, 2.0, 0.5],
                    "quat": [1.0, 0.0, 0.0, 0.0],
                },
            })

            zapdos_overlay.save_overlay_state(path, overlay)

            self.assertEqual(zapdos_overlay.load_overlay_state(path), overlay)
            self.assertEqual(zapdos_overlay.overlay_body_name("crate_001_01"), "Scene_crate_001_01")


if __name__ == "__main__":
    unittest.main()
