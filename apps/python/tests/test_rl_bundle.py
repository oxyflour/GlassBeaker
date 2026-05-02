from __future__ import annotations

import json
import unittest
from pathlib import Path

import mujoco  # type: ignore
from pxr import Usd, UsdGeom

from utils.rl_bundle import DEFAULT_SCENE_USD, ensure_render_bundle

REPO_ROOT = Path(__file__).resolve().parents[3]
ROBOT_USD = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"


class RLBundleTest(unittest.TestCase):
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
        self.assertEqual(bundle.main_camera_prim, "/default_viz_camera")
        wrapper_stage = Usd.Stage.Open(str(bundle.robot_wrapper_usda))
        render_stage = Usd.Stage.Open(str(bundle.render_scene_usda))
        self.assertEqual(str(UsdGeom.GetStageUpAxis(wrapper_stage)), "Z")
        self.assertEqual(str(UsdGeom.GetStageUpAxis(render_stage)), "Z")


if __name__ == "__main__":
    unittest.main()
