from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from pxr import Usd, UsdGeom

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.zapdos.bundle.bundle_builder import ensure_render_bundle

ROBOT_USD = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"


class RLBundleGroundTest(unittest.TestCase):
    def test_ensure_render_bundle_injects_default_ground_when_scene_is_missing_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_path = Path(tmpdir) / "scene_without_ground.usda"
            stage = Usd.Stage.CreateNew(str(scene_path))
            stage.SetMetadata("metersPerUnit", 1.0)
            UsdGeom.SetStageUpAxis(stage, "Z")
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            crate = UsdGeom.Xform.Define(stage, "/World/Crate")
            UsdGeom.Cube.Define(stage, "/World/Crate/Visual").CreateSizeAttr(0.5)
            stage.GetRootLayer().Save()

            bundle = ensure_render_bundle(ROBOT_USD, scene_path)

        sim_stage = Usd.Stage.Open(str(bundle.sim_scene_usda))
        self.assertTrue(sim_stage.GetPrimAtPath("/Scene/Crate").IsValid())
        sim_ground = sim_stage.GetPrimAtPath("/Scene/Ground")
        self.assertTrue(sim_ground.IsValid())
        self.assertTrue(sim_ground.IsA(UsdGeom.Mesh))

        render_stage = Usd.Stage.Open(str(bundle.render_scene_usda))
        self.assertTrue(render_stage.GetPrimAtPath("/RenderScene/Crate").IsValid())
        render_ground = render_stage.GetPrimAtPath("/RenderScene/Ground")
        self.assertTrue(render_ground.IsValid())
        self.assertTrue(render_ground.IsA(UsdGeom.Mesh))


if __name__ == "__main__":
    unittest.main()
