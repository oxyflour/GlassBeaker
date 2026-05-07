from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils.camera_override import apply_camera_overrides, load_camera_overrides, save_camera_overrides
from utils.zapdos.rl_cameras import RenderCamera
from utils.user_config import read_user_config


class CameraOverrideTest(unittest.TestCase):
    def test_read_user_config_rejects_non_object_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / ".glass-beaker" / "config.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("[]", encoding="utf-8")

            with mock.patch.dict(os.environ, {"USERPROFILE": tmp}, clear=False):
                with self.assertRaises(RuntimeError) as err:
                    read_user_config()

        self.assertIn("must contain a JSON object", str(err.exception))

    def test_save_camera_overrides_merges_existing_config(self):
        snapshot = [{
            "name": "head_camera",
            "parent_prim": "/MyRobot/zed_link",
            "pos": [0.1, 0.2, 0.3],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "fovy": 60.0,
            "horizontal_aperture": 30.0,
            "vertical_aperture": 20.0,
            "clipping_range": [0.2, 80.0],
        }]

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / ".glass-beaker" / "config.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps({"keep": {"value": 1}}, indent=2), encoding="utf-8")

            with mock.patch.dict(os.environ, {"USERPROFILE": tmp}, clear=False):
                written_path, saved = save_camera_overrides(snapshot)

            payload = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(written_path, config_path)
        self.assertEqual(saved, 1)
        self.assertEqual(payload["keep"]["value"], 1)
        self.assertEqual(payload["override"]["camera"]["/MyRobot/zed_link"]["head_camera"]["fovy"], 60.0)

    def test_apply_camera_overrides_updates_only_matching_camera(self):
        cameras = [
            RenderCamera(
                name="head_camera",
                prim="/MyRobot/zed_link/head_camera",
                topic="/env_0/head_camera/image_raw",
                frame_id="head_camera",
                body="Root_r1_pro_with_gripper_zed_link",
                pos=[0.0, 0.0, 0.0],
                quat=[1.0, 0.0, 0.0, 0.0],
                fovy=45.0,
            ),
            RenderCamera(
                name="left_wrist_camera",
                prim="/MyRobot/left_link/left_wrist_camera",
                topic="/env_0/left_wrist_camera/image_raw",
                frame_id="left_wrist_camera",
                body="Root_r1_pro_with_gripper_left_realsense_link",
                pos=[0.0, 0.0, 0.0],
                quat=[1.0, 0.0, 0.0, 0.0],
                fovy=45.0,
            ),
        ]
        overrides = load_camera_overrides({
            "override": {
                "camera": {
                    "/MyRobot/zed_link": {
                        "head_camera": {
                            "pos": [0.4, 0.5, 0.6],
                            "quat": [0.0, 0.0, 1.0, 0.0],
                            "fovy": 55.0,
                            "horizontal_aperture": 31.0,
                            "vertical_aperture": 21.0,
                            "clipping_range": [0.3, 90.0],
                        }
                    }
                }
            }
        })

        updated = apply_camera_overrides(cameras, overrides)

        self.assertEqual(updated[0].pos, [0.4, 0.5, 0.6])
        self.assertEqual(updated[0].fovy, 55.0)
        self.assertEqual(updated[0].horizontal_aperture, 31.0)
        self.assertEqual(updated[1].pos, [0.0, 0.0, 0.0])
        self.assertEqual(updated[1].fovy, 45.0)


if __name__ == "__main__":
    unittest.main()

