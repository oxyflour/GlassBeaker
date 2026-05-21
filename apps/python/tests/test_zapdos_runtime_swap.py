from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import mujoco  # type: ignore
from pxr import Usd, UsdGeom, UsdPhysics

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.zapdos.bundle import ensure_render_bundle  # noqa: E402
from utils.zapdos.bundle.usd_to_mjcf_adapter import sanitize_name  # noqa: E402
from utils.genie_sim import resolve_assets_root  # noqa: E402
from utils.zapdos.editor.scene_writer import write_overlay_scene  # noqa: E402
from utils.zapdos.editor.state import default_overlay_state  # noqa: E402
from utils.zapdos.editor.zapdos_editor import ZapdosEditor  # noqa: E402
from utils.zapdos.physics.mujoco_physics import MujocoPhysics  # noqa: E402
from utils.zapdos.zapdos_asset_library import asset_local_bounds, resolve_asset_record  # noqa: E402

ROBOT_USD = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"


class ZapdosRuntimeSwapTest(unittest.TestCase):
    def test_benchmark_cube_settles_on_visual_tabletop(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = ensure_render_bundle(ROBOT_USD, _write_benchmark_table_scene(Path(tmp)))
            physics = _physics("benchmark-cube-support", bundle)
            cube = "Scene_benchmark_building_blocks_006_01"
            table = "Scene_benchmark_table_000_01"
            try:
                for _ in range(500):
                    physics.step()

                cube_bottom = float(physics.body_world_aabb(cube)["min"][2])
                table_top = float(physics.body_world_aabb(table)["max"][2])
                self.assertGreaterEqual(cube_bottom, table_top - 0.005)
            finally:
                physics.close()

    def test_swap_runtime_bundle_copies_robot_joint_state_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_bundle = ensure_render_bundle(ROBOT_USD, _write_dynamic_cube(root / "old.usda", "OldBox", [0.0, 0.0, 0.1]))
            new_bundle = ensure_render_bundle(ROBOT_USD, _write_dynamic_cube(root / "new.usda", "NewBox", [0.34, 0.24, 0.1]))
            old_physics = _physics("swap-old-joint", old_bundle)
            new_physics = _physics("swap-new-joint", new_bundle)
            joint_name = old_physics.joint_state_msg()["name"][0]
            old_joint_id = mujoco.mj_name2id(old_physics.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)  # type: ignore
            old_qpos_adr = int(old_physics.model.jnt_qposadr[old_joint_id])
            expected_qpos = float(old_physics.data.qpos[old_qpos_adr]) + 0.01
            old_physics.data.qpos[old_qpos_adr] = expected_qpos
            actuator_name = next(name for name in old_physics.actuator_name_to_id if name in new_physics.actuator_name_to_id)
            expected_ctrl = 0.123
            old_physics.data.ctrl[old_physics.actuator_name_to_id[actuator_name]] = expected_ctrl
            session = _swap_session(old_bundle, old_physics, new_physics)
            editor = ZapdosEditor.__new__(ZapdosEditor)
            editor.session = session

            editor._swap_runtime_bundle(new_bundle, {"pose_overrides": {}})

            new_joint_id = mujoco.mj_name2id(session.physics.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)  # type: ignore
            new_qpos_adr = int(session.physics.model.jnt_qposadr[new_joint_id])
            self.assertEqual(float(session.physics.data.qpos[new_qpos_adr]), expected_qpos)
            self.assertEqual(float(session.physics.data.ctrl[session.physics.actuator_name_to_id[actuator_name]]), expected_ctrl)
            session.physics.close()

    def test_swap_runtime_bundle_does_not_copy_old_dynamic_body_qpos_into_new_scene_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_bundle = ensure_render_bundle(ROBOT_USD, _write_dynamic_cube(root / "old.usda", "OldBox", [0.0, 0.0, 0.1]))
            new_bundle = ensure_render_bundle(ROBOT_USD, _write_dynamic_cube(root / "new.usda", "NewBox", [0.34, 0.24, 0.1]))
            old_physics = _physics("swap-old", old_bundle)
            new_physics = _physics("swap-new", new_bundle)
            new_body = sanitize_name("/Scene/NewBox")
            expected_xyz = [float(new_physics.get_pose()[new_body][index]) for index in (12, 13, 14)]
            session = _swap_session(old_bundle, old_physics, new_physics)
            editor = ZapdosEditor.__new__(ZapdosEditor)
            editor.session = session

            old_physics.set_body_pose(sanitize_name("/Scene/OldBox"), [384.0, 0.0, 0.1], [1.0, 0.0, 0.0, 0.0])
            editor._swap_runtime_bundle(new_bundle, {"pose_overrides": {}})

            actual_xyz = [float(session.physics.get_pose()[new_body][index]) for index in (12, 13, 14)]
            self.assertEqual(actual_xyz, expected_xyz)
            session.physics.close()


def _write_dynamic_cube(path: Path, name: str, xyz: list[float]) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    stage.SetMetadata("metersPerUnit", 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    body = UsdGeom.Xform.Define(stage, f"/World/{name}")
    UsdGeom.Xformable(body.GetPrim()).AddTranslateOp().Set(tuple(xyz))
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(0.2)
    UsdGeom.Cube.Define(stage, f"/World/{name}/Visual").CreateSizeAttr(0.05)
    stage.GetRootLayer().Save()
    return path


def _write_benchmark_table_scene(root: Path) -> Path:
    assets_root = resolve_assets_root(None)
    overlay = default_overlay_state(str(assets_root))
    instances = []
    for asset_id, motion, placement in (
        ("benchmark_table_000", "static", {"kind": "floor_at_xy", "xy": [0.5, 0.0], "z_offset": 0.0, "yaw": 0.0}),
        ("benchmark_building_blocks_006", "dynamic", {
            "kind": "on_top_of_body",
            "body": "Scene_benchmark_table_000_01",
            "xy": [0.34, 0.24],
            "gap": 0.0,
            "yaw": 0.0,
        }),
    ):
        record = resolve_asset_record(asset_id, assets_root)
        instances.append({
            "id": f"{asset_id}_01",
            "asset_id": asset_id,
            "url": record["url"],
            "motion": motion,
            "placement": placement,
        })
    overlay["instances"] = instances
    return write_overlay_scene(
        root / "benchmark-table.usda",
        REPO_ROOT / "apps" / "python" / "assets" / "default_scene.usda",
        assets_root,
        overlay,
        support_infos={},
        asset_bounds_by_instance={item["id"]: asset_local_bounds(assets_root / item["url"]) for item in instances},
    )


def _physics(sess: str, bundle) -> MujocoPhysics:
    return MujocoPhysics(sess, bundle, json.loads(bundle.body_map_json.read_text(encoding="utf-8")))


def _swap_session(old_bundle, old_physics: MujocoPhysics, new_physics: MujocoPhysics) -> SimpleNamespace:
    return SimpleNamespace(
        sess="sess-swap",
        physics=old_physics,
        renderer=SimpleNamespace(reload_scene=mock.Mock(), set_bundle=mock.Mock(), close=mock.Mock()),
        bundle=old_bundle,
        _create_physics=mock.Mock(return_value=new_physics),
    )


if __name__ == "__main__":
    unittest.main()
