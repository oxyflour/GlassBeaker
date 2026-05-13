from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

from pxr import Gf, Usd, UsdGeom

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.zapdos.editor.placement import normalize_placement
from utils.zapdos.editor.scene_writer import resolve_instance_pose, write_overlay_scene
from utils.zapdos.editor.state import default_overlay_state
from utils.zapdos.zapdos_asset_library import asset_local_bounds, resolve_asset_record


class ZapdosOverlaySceneTest(unittest.TestCase):
    def make_assets_root(self, tmp: str) -> Path:
        assets_root = Path(tmp) / "GenieSimAssets"
        asset_dir = assets_root / "objects" / "table_000"
        asset_dir.mkdir(parents=True)
        asset_path = asset_dir / "Aligned.usda"
        stage = Usd.Stage.CreateNew(asset_path.as_posix())
        asset = UsdGeom.Xform.Define(stage, "/Asset")
        geom = UsdGeom.Cube.Define(stage, "/Asset/Cube")
        geom.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.375))
        geom.CreateSizeAttr(0.75)
        stage.SetDefaultPrim(asset.GetPrim())
        stage.GetRootLayer().Save()
        (assets_root / "__init__.py").write_text(
            "\n".join(
                [
                    "from pathlib import Path",
                    "ASSETS_PATH = Path(__file__).parent",
                    "ASSETS_INDEX = {'table_000': {'url': 'objects/table_000/Aligned.usda', 'description': {'semantic_name': ['table']}}}",
                    "ASSETS_INDEX_HASH = 'overlay-test-hash'",
                ]
            ),
            encoding="utf-8",
        )
        return assets_root

    def test_editor_package_owns_placement_and_scene_writer_without_overlay_imports(self):
        placement_source = (
            REPO_ROOT
            / "apps"
            / "python"
            / "utils"
            / "zapdos"
            / "editor"
            / "placement.py"
        ).read_text(encoding="utf-8")
        writer_source = (
            REPO_ROOT
            / "apps"
            / "python"
            / "utils"
            / "zapdos"
            / "editor"
            / "scene_writer.py"
        ).read_text(encoding="utf-8")

        self.assertIn("def normalize_placement", placement_source)
        self.assertIn("def write_overlay_scene", writer_source)
        self.assertNotIn("from utils.zapdos.overlay.overlay_placement import", placement_source)
        self.assertNotIn("from utils.zapdos.overlay.overlay_scene_writer import", writer_source)

    def test_resolve_asset_record_and_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_root = self.make_assets_root(tmp)
            record = resolve_asset_record("table_000", assets_root)
            bounds = asset_local_bounds(assets_root / record["url"])

            self.assertEqual(record["asset_id"], "table_000")
            self.assertEqual(bounds["min"][2], 0.0)
            self.assertGreater(bounds["max"][2], 0.7)

    def test_resolve_instance_pose_supports_floor_and_support_body(self):
        pose = resolve_instance_pose(
            {
                "id": "mug_001_01",
                "asset_id": "mug_001",
                "url": "objects/mug_001/Aligned.usda",
                "motion": "dynamic",
                "placement": {
                    "kind": "on_top_of_body",
                    "body": "Scene_table_000_01",
                    "xy": [0.1, 0.2],
                    "gap": 0.0,
                    "yaw": 0.0,
                },
            },
            asset_bounds={"min": [-0.05, -0.05, 0.0], "max": [0.05, 0.05, 0.1]},
            support_infos={"Scene_table_000_01": {"top_z": 0.75}},
            pose_overrides={},
        )

        self.assertEqual(pose["pos"], [0.1, 0.2, 0.75])

    def test_resolve_instance_pose_supports_floor_quaternion_alignment(self):
        pose = resolve_instance_pose(
            {
                "id": "benchmark_table_000_01",
                "asset_id": "benchmark_table_000",
                "url": "objects/benchmark/table/benchmark_table_000/Aligned.usda",
                "motion": "static",
                "placement": {
                    "kind": "floor_at_xy",
                    "xy": [0.0, 0.0],
                    "z_offset": 0.0,
                    "payload_quat": [math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0],
                },
            },
            asset_bounds={"min": [-0.3, -0.6, -0.37], "max": [0.3, 0.6, 0.37]},
            support_infos={},
            pose_overrides={},
        )

        self.assertEqual(pose["quat"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(pose["payload_quat"], [math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0])
        self.assertAlmostEqual(pose["pos"][2], 0.6, places=6)

    def test_resolve_instance_pose_keeps_payload_quat_when_pose_override_exists(self):
        pose = resolve_instance_pose(
            {
                "id": "benchmark_table_000_01",
                "asset_id": "benchmark_table_000",
                "url": "objects/benchmark/table/benchmark_table_000/Aligned.usda",
                "motion": "static",
                "placement": {
                    "kind": "floor_at_xy",
                    "xy": [0.0, 0.0],
                    "z_offset": 0.0,
                    "payload_quat": [math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0],
                },
            },
            asset_bounds={"min": [-0.3, -0.6, -0.37], "max": [0.3, 0.6, 0.37]},
            support_infos={},
            pose_overrides={
                "Scene_benchmark_table_000_01": {
                    "pos": [1.0, 2.0, 3.0],
                    "quat": [1.0, 0.0, 0.0, 0.0],
                },
            },
        )

        self.assertEqual(pose["pos"], [1.0, 2.0, 3.0])
        self.assertEqual(pose["quat"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(pose["payload_quat"], [math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0])

    def test_normalize_placement_infers_floor_kind_from_xy_payload(self):
        placement = normalize_placement({"xy": [0.0, 0.0], "z_offset": 0.0, "yaw": 0.0})

        self.assertEqual(placement["kind"], "floor_at_xy")

    def test_normalize_placement_accepts_position_alias_for_world_pose(self):
        placement = normalize_placement({"position": [0.0, 0.0, 0.0]})

        self.assertEqual(placement["kind"], "world_pose")
        self.assertEqual(placement["pos"], [0.0, 0.0, 0.0])
        self.assertEqual(placement["quat"], [1.0, 0.0, 0.0, 0.0])

    def test_normalize_placement_rejects_ambiguous_payload(self):
        with self.assertRaises(ValueError) as err:
            normalize_placement({})

        self.assertIn("placement.kind", str(err.exception))

    def test_write_overlay_scene_references_base_scene_and_marks_static_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_root = self.make_assets_root(tmp)
            base_scene = Path(tmp) / "scene.usda"
            stage = Usd.Stage.CreateNew(base_scene.as_posix())
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            stage.GetRootLayer().Save()

            overlay = default_overlay_state(str(assets_root))
            overlay["instances"].append({
                "id": "table_000_01",
                "asset_id": "table_000",
                "url": "objects/table_000/Aligned.usda",
                "motion": "static",
                "placement": {"kind": "floor_at_xy", "xy": [1.0, -0.5], "z_offset": 0.0, "yaw": 0.0},
            })

            scene_path = write_overlay_scene(
                Path(tmp) / "overlay_scene.usda",
                base_scene,
                assets_root,
                overlay,
                support_infos={},
                asset_bounds_by_instance={"table_000_01": {"min": [-0.375, -0.375, 0.0], "max": [0.375, 0.375, 0.75]}},
            )

            stage = Usd.Stage.Open(scene_path.as_posix())
            table = stage.GetPrimAtPath("/World/table_000_01")
            self.assertTrue(table.IsValid())
            self.assertEqual(UsdGeom.GetStageUpAxis(stage), UsdGeom.Tokens.z)
            self.assertEqual(UsdGeom.GetStageMetersPerUnit(stage), 1.0)
            kinematic = table.GetAttribute("physics:kinematicEnabled")
            self.assertTrue(kinematic.IsValid())
            self.assertTrue(kinematic.HasAuthoredValueOpinion())

    def test_write_overlay_scene_authors_payload_local_correction_without_tilting_root_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_root = self.make_assets_root(tmp)
            base_scene = Path(tmp) / "scene.usda"
            stage = Usd.Stage.CreateNew(base_scene.as_posix())
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            stage.GetRootLayer().Save()

            overlay = default_overlay_state(str(assets_root))
            overlay["instances"].append({
                "id": "table_000_01",
                "asset_id": "table_000",
                "url": "objects/table_000/Aligned.usda",
                "motion": "static",
                "placement": {
                    "kind": "floor_at_xy",
                    "xy": [0.0, 0.0],
                    "z_offset": 0.0,
                    "payload_quat": [math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0],
                },
            })

            scene_path = write_overlay_scene(
                Path(tmp) / "overlay_scene.usda",
                base_scene,
                assets_root,
                overlay,
                support_infos={},
                asset_bounds_by_instance={"table_000_01": {"min": [-0.375, -0.375, 0.0], "max": [0.375, 0.375, 0.75]}},
            )

            stage = Usd.Stage.Open(scene_path.as_posix())
            table = stage.GetPrimAtPath("/World/table_000_01")
            payload = stage.GetPrimAtPath("/World/table_000_01/Payload")
            self.assertEqual(table.GetAttribute("xformOp:orient").Get().GetReal(), 1.0)
            self.assertAlmostEqual(payload.GetAttribute("xformOp:orient").Get().GetReal(), math.sqrt(0.5), places=6)

    def test_write_overlay_scene_supports_on_top_of_body_for_earlier_overlay_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_root = self.make_assets_root(tmp)
            base_scene = Path(tmp) / "scene.usda"
            stage = Usd.Stage.CreateNew(base_scene.as_posix())
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            stage.GetRootLayer().Save()

            overlay = default_overlay_state(str(assets_root))
            overlay["instances"] = [
                {
                    "id": "table_000_01",
                    "asset_id": "table_000",
                    "url": "objects/table_000/Aligned.usda",
                    "motion": "static",
                    "placement": {"kind": "floor_at_xy", "xy": [0.0, 0.0], "z_offset": 0.0, "yaw": 0.0},
                },
                {
                    "id": "mug_000_01",
                    "asset_id": "mug_000",
                    "url": "objects/table_000/Aligned.usda",
                    "motion": "dynamic",
                    "placement": {
                        "kind": "on_top_of_body",
                        "body": "Scene_table_000_01",
                        "xy": [0.1, 0.2],
                        "gap": 0.0,
                        "yaw": 0.0,
                    },
                },
            ]

            scene_path = write_overlay_scene(
                Path(tmp) / "overlay_scene.usda",
                base_scene,
                assets_root,
                overlay,
                support_infos={},
                asset_bounds_by_instance={
                    "table_000_01": {"min": [-0.375, -0.375, 0.0], "max": [0.375, 0.375, 0.75]},
                    "mug_000_01": {"min": [-0.05, -0.05, 0.0], "max": [0.05, 0.05, 0.1]},
                },
            )

            stage = Usd.Stage.Open(scene_path.as_posix())
            mug = stage.GetPrimAtPath("/World/mug_000_01")
            translate = mug.GetAttribute("xformOp:translate").Get()
            self.assertEqual((translate[0], translate[1]), (0.1, 0.2))
            self.assertAlmostEqual(translate[2], 0.75, places=6)


if __name__ == "__main__":
    unittest.main()

