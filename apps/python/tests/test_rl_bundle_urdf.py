from __future__ import annotations

import importlib
import importlib.util
import re
import sys
import tempfile
import unittest
from pathlib import Path

import json
import mujoco  # type: ignore
from pxr import Usd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.zapdos.bundle.bundle_builder import ensure_render_bundle
from utils.zapdos.bundle.camera_specs import build_render_cameras
from utils.zapdos.bundle.render_bundle import DEFAULT_SCENE_USD

R1PRO_USD = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"
MOZ1_URDF = REPO_ROOT / "deps" / "moz01" / "spirit01_model" / "urdf" / "moz1.urdf"
MOZ1_VISUAL_DIR = (
    REPO_ROOT
    / "deps"
    / "moz01"
    / "isaac_moz1"
    / "Issacsim_Assets"
    / "spirit01_model"
    / "spirit01_model"
    / "USD"
)
MOZ1_VISUAL_ROOT = "/World/MOZ1"


def compiled_moz1_body_names() -> list[str]:
    source = MOZ1_URDF.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmpdir:
        rewritten = Path(tmpdir) / "moz1.urdf"
        payload = re.sub(
            r'filename="package://spirit01_model/(.*?)"',
            lambda match: f'filename="{(MOZ1_URDF.parents[1] / match.group(1)).as_posix()}"',
            source,
        )
        rewritten.write_text(payload, encoding="utf-8")
        model = mujoco.MjModel.from_xml_path(str(rewritten))  # type: ignore
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

    def test_resolve_robot_assets_supports_r1pro_and_moz1_inputs(self):
        robot_assets = self.load_robot_assets()
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir)
            r1pro = robot_assets.resolve_robot_assets(R1PRO_USD, bundle_dir)
            moz1 = robot_assets.resolve_robot_assets(MOZ1_URDF, bundle_dir)

        self.assertEqual(r1pro.robot_input, R1PRO_USD.resolve())
        self.assertEqual(r1pro.physics_input, R1PRO_USD.resolve())
        self.assertEqual(r1pro.visual_usd, R1PRO_USD.resolve())
        self.assertEqual(r1pro.visual_root, "")
        self.assertEqual(r1pro.attachments_by_body, {})
        self.assertEqual(r1pro.static_visual_paths, [])
        self.assertEqual(r1pro.dependency_paths, [R1PRO_USD.resolve()])

        self.assertEqual(moz1.robot_input, MOZ1_URDF.resolve())
        self.assertEqual(moz1.visual_usd.parent.resolve(), MOZ1_VISUAL_DIR.resolve())
        self.assertEqual(moz1.visual_usd.name, "Moz1_omni_gripper_full.usd")
        self.assertEqual(moz1.visual_root, MOZ1_VISUAL_ROOT)
        self.assertEqual(sorted(moz1.attachments_by_body), sorted(compiled_moz1_body_names()))
        self.assertEqual(moz1.static_visual_paths, [f"{MOZ1_VISUAL_ROOT}/base_link"])
        self.assertIn(MOZ1_URDF.resolve(), moz1.dependency_paths)
        self.assertIn(moz1.visual_usd.resolve(), moz1.dependency_paths)
        self.assertIn(f"{MOZ1_VISUAL_ROOT}/waist03", moz1.attachments_by_body["waist03"])
        self.assertIn(f"{MOZ1_VISUAL_ROOT}/head21", moz1.attachments_by_body["waist03"])
        self.assertIn(f"{MOZ1_VISUAL_ROOT}/head22", moz1.attachments_by_body["waist03"])
        self.assertIn(f"{MOZ1_VISUAL_ROOT}/head23", moz1.attachments_by_body["waist03"])
        self.assertIn(f"{MOZ1_VISUAL_ROOT}/left_gripper_base_link", moz1.attachments_by_body["left07"])
        self.assertIn(f"{MOZ1_VISUAL_ROOT}/right_gripper_base_link", moz1.attachments_by_body["right07"])

    def test_rewrite_moz1_urdf_localizes_mesh_paths_without_mutating_source(self):
        robot_assets = self.load_robot_assets()
        original = MOZ1_URDF.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir)
            rewritten = robot_assets.rewrite_urdf_for_bundle(MOZ1_URDF, bundle_dir)

            self.assertNotEqual(rewritten.resolve(), MOZ1_URDF.resolve())
            self.assertTrue(rewritten.exists())
            payload = rewritten.read_text(encoding="utf-8")
            self.assertNotIn("package://spirit01_model/", payload)
            self.assertIn('filename="meshes/base_link.STL"', payload)
            self.assertIn('filename="meshes/head23.STL"', payload)
            self.assertTrue((bundle_dir / "meshes" / "base_link.STL").exists())
            self.assertTrue((bundle_dir / "meshes" / "head23.STL").exists())

        self.assertEqual(MOZ1_URDF.read_text(encoding="utf-8"), original)

    def test_ensure_render_bundle_creates_expected_outputs_for_moz1_urdf(self):
        bundle = ensure_render_bundle(MOZ1_URDF, DEFAULT_SCENE_USD)
        self.assertEqual(bundle.robot_usd, MOZ1_URDF.resolve())
        self.assertEqual(bundle.mjcf.name, "sim_scene.xml")
        self.assertEqual(bundle.robot_wrapper_usda.name, "robot_wrapper.usda")
        self.assertEqual(bundle.render_scene_usda.name, "render_scene.usda")
        self.assertTrue(bundle.mjcf.exists())
        self.assertTrue(bundle.robot_wrapper_usda.exists())
        self.assertTrue(bundle.render_scene_usda.exists())
        manifest = json.loads((bundle.bundle_dir / "manifest-v1.json").read_text(encoding="utf-8"))
        self.assertEqual(Path(manifest["robot_usd"]).resolve(), MOZ1_URDF.resolve())

    def test_build_render_cameras_synthesizes_main_for_camera_less_moz1(self):
        robot_assets = self.load_robot_assets()
        with tempfile.TemporaryDirectory() as tmpdir:
            rewritten = robot_assets.rewrite_urdf_for_bundle(MOZ1_URDF, Path(tmpdir))
            model = mujoco.MjModel.from_xml_path(str(rewritten))  # type: ignore

        body_paths = {body_name: f"MyRobot/{body_name}" for body_name in compiled_moz1_body_names()}
        cameras = build_render_cameras(model, body_paths)

        self.assertEqual([camera.name for camera in cameras], ["main"])
        self.assertEqual(cameras[0].prim, "/SceneRender/main")
        self.assertIsNone(cameras[0].body)

    def test_ensure_render_bundle_exposes_main_camera_for_moz1_urdf(self):
        bundle = ensure_render_bundle(MOZ1_URDF, DEFAULT_SCENE_USD)
        self.assertEqual(bundle.camera_names(), ["main"])

    def test_ensure_render_bundle_wraps_static_and_multi_visual_attachments_for_moz1(self):
        bundle = ensure_render_bundle(MOZ1_URDF, DEFAULT_SCENE_USD)
        stage = Usd.Stage.Open(str(bundle.robot_wrapper_usda))

        self.assertTrue(stage.GetPrimAtPath("/MyRobot/base_link/visuals").IsValid())
        for name in ("head21", "head22", "head23"):
            self.assertTrue(stage.GetPrimAtPath(f"/MyRobot/waist03/{name}/visuals").IsValid(), name)
        for body_name in compiled_moz1_body_names():
            self.assertTrue(stage.GetPrimAtPath(f"/MyRobot/{body_name}").IsValid(), body_name)


if __name__ == "__main__":
    unittest.main()
