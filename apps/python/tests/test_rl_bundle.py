from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mujoco  # type: ignore
from pxr import Usd, UsdGeom

import utils.rl_bundle as MODULE
from utils.rl_bundle import DEFAULT_SCENE_USD, ensure_render_bundle
from utils.rl_cameras import build_render_cameras

REPO_ROOT = Path(__file__).resolve().parents[3]
ROBOT_USD = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"


class RLBundleTest(unittest.TestCase):
    def test_default_scene_contains_ground_mesh(self):
        stage = Usd.Stage.Open(str(DEFAULT_SCENE_USD))
        self.assertIsNotNone(stage)
        ground = stage.GetPrimAtPath("/World/Ground")
        self.assertTrue(ground.IsValid(), str(DEFAULT_SCENE_USD))
        self.assertTrue(ground.IsA(UsdGeom.Mesh))

    def test_ensure_render_bundle_creates_expected_outputs(self):
        bundle = ensure_render_bundle(ROBOT_USD, DEFAULT_SCENE_USD)
        for path in bundle.outputs():
            self.assertTrue(path.exists(), str(path))

        model = mujoco.MjModel.from_xml_path(str(bundle.mjcf))  # type: ignore
        robot_bodies = []
        for body_id in range(1, model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)  # type: ignore
            if name and name.startswith("Root_r1_pro_with_gripper_"):
                robot_bodies.append(name)

        body_map = json.loads(bundle.body_map_json.read_text(encoding="utf-8"))
        body_map_jsona = json.loads(bundle.body_map_jsona.read_text(encoding="utf-8"))
        self.assertEqual(sorted(robot_bodies), sorted(body_map.keys()))
        self.assertEqual(body_map, body_map_jsona)
        wrapper_stage = Usd.Stage.Open(str(bundle.robot_wrapper_usda))
        render_stage = Usd.Stage.Open(str(bundle.render_scene_usda))
        self.assertEqual(str(UsdGeom.GetStageUpAxis(wrapper_stage)), "Z")
        self.assertEqual(str(UsdGeom.GetStageUpAxis(render_stage)), "Z")

        cameras = build_render_cameras(model, body_map)
        self.assertEqual([camera.name for camera in cameras], bundle.camera_names())
        self.assertEqual(
            [camera.name for camera in cameras],
            [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)  # type: ignore
                for cam_id in range(model.ncam)
            ],
        )
        self.assertEqual(
            [camera.topic for camera in cameras],
            [f"/env_0/{camera.name}/image_raw" for camera in cameras],
        )

    def test_bundle_key_changes_when_camera_override_config_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / ".glass-beaker" / "config.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps({"override": {"camera": {}}}, indent=2), encoding="utf-8")
            with mock.patch.dict(os.environ, {"USERPROFILE": tmp}, clear=False):
                before = MODULE._bundle_key(ROBOT_USD.resolve(), DEFAULT_SCENE_USD.resolve())
                config_path.write_text(json.dumps({
                    "override": {
                        "camera": {
                            "/MyRobot/Root_r1_pro_with_gripper_zed_link": {
                                "head_camera": {
                                    "pos": [0.1, 0.2, 0.3],
                                    "quat": [1.0, 0.0, 0.0, 0.0],
                                    "fovy": 60.0,
                                    "horizontal_aperture": 30.0,
                                    "vertical_aperture": 20.0,
                                    "clipping_range": [0.2, 80.0],
                                }
                            }
                        }
                    }
                }, indent=2), encoding="utf-8")
                after = MODULE._bundle_key(ROBOT_USD.resolve(), DEFAULT_SCENE_USD.resolve())

        self.assertNotEqual(before, after)

    def test_bundle_key_changes_when_converter_dependency_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            dependency = Path(tmp) / "usd_to_mjcf.py"
            dependency.write_text("alpha", encoding="utf-8")
            with mock.patch.object(MODULE, "BUNDLE_DEPENDENCY_FILES", (dependency,)):
                before = MODULE._bundle_key(ROBOT_USD.resolve(), DEFAULT_SCENE_USD.resolve())
                dependency.write_text("beta gamma", encoding="utf-8")
                after = MODULE._bundle_key(ROBOT_USD.resolve(), DEFAULT_SCENE_USD.resolve())

        self.assertNotEqual(before, after)

    def test_ensure_render_bundle_writes_overridden_camera_values_to_usd(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / ".glass-beaker" / "config.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps({
                "override": {
                    "camera": {
                        "/MyRobot/Root_r1_pro_with_gripper_zed_link": {
                            "head_camera": {
                                "pos": [0.1, 0.2, 0.3],
                                "quat": [1.0, 0.0, 0.0, 0.0],
                                "fovy": 60.0,
                                "horizontal_aperture": 30.0,
                                "vertical_aperture": 20.0,
                                "clipping_range": [0.2, 80.0],
                            }
                        }
                    }
                }
            }, indent=2), encoding="utf-8")

            with mock.patch.dict(os.environ, {"USERPROFILE": tmp}, clear=False):
                bundle = MODULE.ensure_render_bundle(ROBOT_USD, DEFAULT_SCENE_USD)

        stage = Usd.Stage.Open(str(bundle.robot_wrapper_usda))
        camera_prim = stage.GetPrimAtPath("/MyRobot/Root_r1_pro_with_gripper_zed_link/head_camera")
        camera = UsdGeom.Camera(camera_prim)
        self.assertEqual(float(camera.GetHorizontalApertureAttr().Get()), 30.0)
        self.assertEqual(float(camera.GetVerticalApertureAttr().Get()), 20.0)
        clipping = tuple(camera.GetClippingRangeAttr().Get())
        self.assertAlmostEqual(clipping[0], 0.2)
        self.assertAlmostEqual(clipping[1], 80.0)
        self.assertEqual(tuple(camera_prim.GetAttribute("xformOp:translate").Get()), (0.1, 0.2, 0.3))


if __name__ == "__main__":
    unittest.main()
