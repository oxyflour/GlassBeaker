from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mujoco  # type: ignore
from pxr import Usd, UsdGeom

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

import utils.zapdos.bundle.bundle_builder as MODULE
from utils.zapdos.bundle.bundle_builder import ensure_render_bundle
from utils.zapdos.bundle.camera_specs import build_render_cameras
from utils.zapdos.bundle.render_bundle import DEFAULT_SCENE_USD
from utils.zapdos.bundle.usd_to_mjcf_adapter import sanitize_name

ROBOT_USD = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"


class RLBundleTest(unittest.TestCase):
    def test_ensure_render_bundle_promotes_scene_object_into_body_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_path = Path(tmpdir) / "scene_object.usda"
            stage = Usd.Stage.CreateNew(str(scene_path))
            stage.SetMetadata("metersPerUnit", 1.0)
            UsdGeom.SetStageUpAxis(stage, "Z")
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            UsdGeom.Cube.Define(stage, "/World/Ground").CreateSizeAttr(10.0)
            crate = UsdGeom.Xform.Define(stage, "/World/Crate")
            UsdGeom.Cube.Define(stage, "/World/Crate/Visual").CreateSizeAttr(0.5)
            UsdGeom.Camera.Define(stage, "/World/EditorCamera")
            stage.GetRootLayer().Save()

            bundle = ensure_render_bundle(ROBOT_USD, scene_path)

        body_map = json.loads(bundle.body_map_json.read_text(encoding="utf-8"))
        scene_body_name = sanitize_name("/Scene/Crate")
        self.assertEqual(body_map.get(scene_body_name), "Crate")
        self.assertNotIn(sanitize_name("/Scene/Ground"), body_map)
        render_stage = Usd.Stage.Open(str(bundle.render_scene_usda))
        self.assertTrue(render_stage.GetPrimAtPath("/RenderScene/Crate").IsValid())

    def test_ensure_render_bundle_promotes_objects_container_children_individually(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_path = Path(tmpdir) / "scene_objects_container.usda"
            stage = Usd.Stage.CreateNew(str(scene_path))
            stage.SetMetadata("metersPerUnit", 1.0)
            UsdGeom.SetStageUpAxis(stage, "Z")
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            objects = UsdGeom.Xform.Define(stage, "/World/Objects")
            crate = UsdGeom.Xform.Define(stage, "/World/Objects/Crate")
            UsdGeom.Cube.Define(stage, "/World/Objects/Crate/Visual").CreateSizeAttr(0.5)
            bowl = UsdGeom.Xform.Define(stage, "/World/Objects/Bowl")
            UsdGeom.Sphere.Define(stage, "/World/Objects/Bowl/Visual").CreateRadiusAttr(0.25)
            stage.GetRootLayer().Save()

            bundle = ensure_render_bundle(ROBOT_USD, scene_path)

        body_map = json.loads(bundle.body_map_json.read_text(encoding="utf-8"))
        self.assertEqual(body_map.get(sanitize_name("/Scene/Objects/Crate")), "Objects/Crate")
        self.assertEqual(body_map.get(sanitize_name("/Scene/Objects/Bowl")), "Objects/Bowl")
        self.assertNotIn(sanitize_name("/Scene/Objects"), body_map)

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

        cameras = build_render_cameras(model, {body: f"MyRobot/{body}" for body in robot_bodies})
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

    def test_bundle_key_ignores_scene_mtime_when_scene_content_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_path = Path(tmp) / "scene.usda"
            scene_path.write_text("scene", encoding="utf-8")
            original_stat = scene_path.stat()

            before = MODULE._bundle_key(ROBOT_USD.resolve(), scene_path.resolve())

            os.utime(
                scene_path,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000_000),
            )
            after = MODULE._bundle_key(ROBOT_USD.resolve(), scene_path.resolve())

        self.assertEqual(before, after)

    def test_ensure_render_bundle_opens_robot_stage_once_per_build(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir) / "bundles"
            scene_path = Path(tmpdir) / "scene.usda"
            stage = Usd.Stage.CreateNew(str(scene_path))
            stage.SetMetadata("metersPerUnit", 1.0)
            UsdGeom.SetStageUpAxis(stage, "Z")
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            UsdGeom.Cube.Define(stage, "/World/Ground").CreateSizeAttr(10.0)
            stage.GetRootLayer().Save()

            with mock.patch.object(MODULE, "TMP_ROOT", tmp_root):
                with mock.patch("pxr.Usd.Stage.Open", wraps=Usd.Stage.Open) as stage_open:
                    bundle = MODULE.ensure_render_bundle(ROBOT_USD, scene_path)
                    self.assertTrue(bundle.mjcf.exists())

        robot_opens = [
            call
            for call in stage_open.call_args_list
            if call.args and Path(call.args[0]).resolve() == ROBOT_USD.resolve()
        ]
        self.assertEqual(len(robot_opens), 1)

    def test_ensure_render_bundle_reuses_cached_robot_stage_across_distinct_builds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir) / "bundles"
            scene_a = Path(tmpdir) / "scene-a.usda"
            scene_b = Path(tmpdir) / "scene-b.usda"
            for scene_path in (scene_a, scene_b):
                stage = Usd.Stage.CreateNew(str(scene_path))
                stage.SetMetadata("metersPerUnit", 1.0)
                UsdGeom.SetStageUpAxis(stage, "Z")
                world = UsdGeom.Xform.Define(stage, "/World")
                stage.SetDefaultPrim(world.GetPrim())
                UsdGeom.Cube.Define(stage, "/World/Ground").CreateSizeAttr(10.0)
                stage.GetRootLayer().Save()

            stage_cache = getattr(MODULE, "_STAGE_CACHE", None)
            if stage_cache is not None:
                stage_cache.clear()

            with mock.patch.object(MODULE, "TMP_ROOT", tmp_root):
                with mock.patch("pxr.Usd.Stage.Open", wraps=Usd.Stage.Open) as stage_open:
                    first = MODULE.ensure_render_bundle(ROBOT_USD, scene_a)
                    second = MODULE.ensure_render_bundle(ROBOT_USD, scene_b)
                    self.assertTrue(first.mjcf.exists())
                    self.assertTrue(second.mjcf.exists())

            if stage_cache is not None:
                stage_cache.clear()

        robot_opens = [
            call
            for call in stage_open.call_args_list
            if call.args and Path(call.args[0]).resolve() == ROBOT_USD.resolve()
        ]
        self.assertEqual(len(robot_opens), 1)

    def test_ensure_render_bundle_reuses_written_sim_stage_without_reopen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir) / "bundles"
            scene_path = Path(tmpdir) / "scene.usda"
            stage = Usd.Stage.CreateNew(str(scene_path))
            stage.SetMetadata("metersPerUnit", 1.0)
            UsdGeom.SetStageUpAxis(stage, "Z")
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            UsdGeom.Cube.Define(stage, "/World/Ground").CreateSizeAttr(10.0)
            stage.GetRootLayer().Save()

            with mock.patch.object(MODULE, "TMP_ROOT", tmp_root):
                with mock.patch("pxr.Usd.Stage.Open", wraps=Usd.Stage.Open) as stage_open:
                    bundle = MODULE.ensure_render_bundle(ROBOT_USD, scene_path)
                    self.assertTrue(bundle.sim_scene_usda.exists())
                    self.assertTrue(bundle.mjcf.exists())

        sim_scene_opens = [
            call
            for call in stage_open.call_args_list
            if call.args and Path(call.args[0]).resolve() == bundle.sim_scene_usda.resolve()
        ]
        self.assertEqual(len(sim_scene_opens), 0)

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

    def test_r1pro_bundle_keeps_original_camera_names(self):
        bundle = ensure_render_bundle(ROBOT_USD, DEFAULT_SCENE_USD)
        self.assertIn("head_camera", bundle.camera_names())
        self.assertNotIn("main", bundle.camera_names())


if __name__ == "__main__":
    unittest.main()
