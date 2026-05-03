from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pxr import Gf, Usd, UsdGeom

from utils.genie_sim_bundle import write_scene_usda


class GenieSimBundleTest(unittest.TestCase):
    def make_asset_with_root_xform_ops(self, tmp: str) -> tuple[Path, dict[str, dict[str, str]], dict]:
        assets_root = Path(tmp) / "GenieSimAssets"
        asset_dir = assets_root / "objects" / "bed_001"
        asset_dir.mkdir(parents=True)
        asset_path = asset_dir / "Aligned.usda"

        stage = Usd.Stage.CreateNew(asset_path.as_posix())
        asset = UsdGeom.Xform.Define(stage, "/Asset")
        ops = UsdGeom.Xformable(asset)
        ops.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
        ops.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        ops.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        stage.SetDefaultPrim(asset.GetPrim())
        stage.GetRootLayer().Save()

        assets_index = {"bed_001": {"url": "objects/bed_001/Aligned.usda"}}
        layout_info = {
            "layout": {
                "bed_001_instance": {
                    "usd": "bed_001",
                    "xyzw": [0.0, 0.0, 0.0, 1.0],
                    "xyz": [0.5, -0.25, 0.0],
                }
            }
        }
        return assets_root, assets_index, layout_info

    def test_write_scene_usda_wraps_payloads_that_define_root_xform_ops(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_root, assets_index, layout_info = self.make_asset_with_root_xform_ops(tmp)
            bundle_dir = Path(tmp) / "bundle"
            bundle_dir.mkdir()

            scene_path = write_scene_usda(bundle_dir, assets_root, assets_index, layout_info)

            stage = Usd.Stage.Open(scene_path.as_posix())
            parent = stage.GetPrimAtPath("/World/Objects/bed_001_instance")
            payload = stage.GetPrimAtPath("/World/Objects/bed_001_instance/Payload")

            self.assertTrue(parent.IsValid())
            self.assertTrue(payload.IsValid())
            self.assertEqual(
                [op.GetOpName() for op in UsdGeom.Xformable(parent).GetOrderedXformOps()],
                ["xformOp:translate", "xformOp:orient", "xformOp:scale"],
            )


if __name__ == "__main__":
    unittest.main()
