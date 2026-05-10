from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import unittest
from functools import lru_cache
from pathlib import Path

import mujoco  # type: ignore
from pxr import Usd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.zapdos.bundle.bundle_builder import ensure_render_bundle
from utils.zapdos.bundle.camera_specs import build_render_cameras
from utils.zapdos.bundle.render_bundle import DEFAULT_SCENE_USD

R1PRO_USD = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"
MOZ1_USDA = REPO_ROOT / "deps" / "spirit01_model" / "USD" / "Moz1_robot_only.usda"


@lru_cache(maxsize=1)
def moz1_bundle():
    return ensure_render_bundle(MOZ1_USDA, DEFAULT_SCENE_USD)


@lru_cache(maxsize=1)
def compiled_moz1_body_names() -> list[str]:
    model = mujoco.MjModel.from_xml_path(str(moz1_bundle().mjcf))  # type: ignore
    names: list[str] = []
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)  # type: ignore
        if name:
            names.append(name)
    return names


class RLBundleUrdfTest(unittest.TestCase):
    def load_robot_assets(self):
        importlib.invalidate_caches()
        spec = importlib.util.find_spec("utils.zapdos.bundle.robot_assets")
        self.assertIsNotNone(spec, "robot_assets module is missing")
        return importlib.import_module("utils.zapdos.bundle.robot_assets")

    def test_resolve_robot_assets_supports_r1pro_and_moz1_usda_inputs(self):
        robot_assets = self.load_robot_assets()
        probe_dir = REPO_ROOT / "apps" / "python" / "tmp" / "rl_bundles" / "_probe" / "moz1_usda_test"
        r1pro = robot_assets.resolve_robot_assets(R1PRO_USD, probe_dir)
        moz1 = robot_assets.resolve_robot_assets(MOZ1_USDA, probe_dir)

        self.assertEqual(r1pro.robot_input, R1PRO_USD.resolve())
        self.assertEqual(r1pro.physics_input, R1PRO_USD.resolve())
        self.assertEqual(r1pro.visual_usd, R1PRO_USD.resolve())
        self.assertEqual(r1pro.visual_root, "")
        self.assertEqual(r1pro.attachments_by_body, {})
        self.assertEqual(r1pro.static_visual_paths, [])
        self.assertEqual(r1pro.dependency_paths, [R1PRO_USD.resolve()])

        self.assertEqual(moz1.robot_input, MOZ1_USDA.resolve())
        self.assertEqual(moz1.physics_input, MOZ1_USDA.resolve())
        self.assertEqual(moz1.visual_usd, MOZ1_USDA.resolve())
        self.assertEqual(moz1.visual_root, "")
        self.assertEqual(moz1.attachments_by_body, {})
        self.assertEqual(moz1.static_visual_paths, [])
        self.assertEqual(moz1.dependency_paths, [MOZ1_USDA.resolve()])

    def test_ensure_render_bundle_creates_expected_outputs_for_moz1_usda(self):
        bundle = moz1_bundle()
        self.assertEqual(bundle.robot_usd, MOZ1_USDA.resolve())
        self.assertEqual(bundle.mjcf.name, "sim_scene.xml")
        self.assertEqual(bundle.robot_wrapper_usda.name, "robot_wrapper.usda")
        self.assertEqual(bundle.render_scene_usda.name, "render_scene.usda")
        self.assertTrue(bundle.mjcf.exists())
        self.assertTrue(bundle.robot_wrapper_usda.exists())
        self.assertTrue(bundle.render_scene_usda.exists())
        manifest = json.loads((bundle.bundle_dir / "manifest-v1.json").read_text(encoding="utf-8"))
        self.assertEqual(Path(manifest["robot_usd"]).resolve(), MOZ1_USDA.resolve())

    def test_build_render_cameras_synthesizes_main_for_camera_less_moz1_usda(self):
        bundle = moz1_bundle()
        model = mujoco.MjModel.from_xml_path(str(bundle.mjcf))  # type: ignore
        body_paths = {body_name: f"MyRobot/{body_name}" for body_name in compiled_moz1_body_names()}
        cameras = build_render_cameras(model, body_paths)

        self.assertEqual([camera.name for camera in cameras], ["main"])
        self.assertEqual(cameras[0].prim, "/SceneRender/main")
        self.assertIsNone(cameras[0].body)

    def test_ensure_render_bundle_exposes_main_camera_for_moz1_usda(self):
        self.assertEqual(moz1_bundle().camera_names(), ["main"])

    def test_ensure_render_bundle_wraps_moz1_body_prims_in_robot_wrapper(self):
        bundle = moz1_bundle()
        stage = Usd.Stage.Open(str(bundle.robot_wrapper_usda))
        body_map = json.loads(bundle.body_map_json.read_text(encoding="utf-8"))

        self.assertTrue(stage.GetPrimAtPath("/MyRobot/Root_base_link").IsValid())
        for body_name in compiled_moz1_body_names():
            self.assertTrue(stage.GetPrimAtPath(f"/{body_map[body_name]}").IsValid(), body_name)

    def test_ensure_render_bundle_writes_body_map_for_moz1_usda(self):
        body_map = json.loads(moz1_bundle().body_map_json.read_text(encoding="utf-8"))
        for body_name in compiled_moz1_body_names():
            self.assertIn(body_name, body_map)


if __name__ == "__main__":
    unittest.main()
