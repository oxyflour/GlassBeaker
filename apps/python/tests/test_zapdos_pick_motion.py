from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.teleop.arm_config import get_arm_config  # noqa: E402
from utils.teleop.ik_controller import IKController  # noqa: E402
from utils.zapdos.bundle.bundle_builder import ensure_render_bundle  # noqa: E402
from utils.zapdos.bundle.usd_to_mjcf_adapter import sanitize_name  # noqa: E402
from utils.zapdos.manipulation.executor import PickExecutor  # noqa: E402
from utils.zapdos.physics.mujoco_physics import MujocoPhysics  # noqa: E402

ROBOT_USD = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"


class _SurfacePhysics:
    def __init__(self) -> None:
        self.attached: list[tuple[str, str]] = []
        self.pose = {"Scene_apple_1": _matrix_at(0.5, 0.0, 0.83)}
        self.aabbs = {"Scene_apple_1": {"min": [0.454, -0.046, 0.784], "max": [0.546, 0.046, 0.876]}}

    def joint_state_msg(self) -> dict[str, list[object]]:
        return {"name": [], "position": []}

    def apply_joint_command(self, message: dict[str, object]) -> None:
        del message

    def step(self) -> None:
        pass

    def attach_body(self, parent_body: str, child_body: str) -> dict[str, object]:
        self.attached.append((parent_body, child_body))
        return {"parent_body": parent_body, "child_body": child_body}

    def detach_body(self, child_body: str) -> dict[str, object]:
        del child_body
        return {"ok": True}

    def get_attachment(self, child_body: str) -> dict[str, object] | None:
        if self.attached and self.attached[-1][1] == child_body:
            return {"parent_body": self.attached[-1][0], "child_body": child_body}
        return None

    def get_pose(self) -> dict[str, list[float]]:
        return self.pose

    def body_world_aabb(self, body_name: str) -> dict[str, list[float]] | None:
        return self.aabbs.get(body_name)


class _StaticIK:
    def __init__(self, pose: dict[str, tuple[float, ...]]) -> None:
        self.pose = pose

    def sync_joint_state(self, joint_state: dict[str, list[object]]) -> None:
        del joint_state

    def get_end_effector_pose(self, arm: str) -> dict[str, tuple[float, ...]]:
        del arm
        return self.pose

    def solve_step(
        self,
        arm: str,
        target_pose: dict[str, tuple[float, ...]],
        gripper_opening: float,
        **kwargs: object,
    ) -> dict[str, list[object]]:
        del arm, target_pose, gripper_opening, kwargs
        return {"name": [], "position": []}


class ZapdosPickMotionTest(unittest.TestCase):
    def test_execute_accepts_attach_when_gripper_reaches_target_aabb_surface(self):
        physics = _SurfacePhysics()
        ik = _StaticIK({
            "position": (0.405, 0.125, 0.911),
            "rotation": (1.0, 0.0, 0.0, 0.0),
        })
        executor = PickExecutor(physics, bundle=_bundle(), ik_controller=ik)

        result = executor.execute({
            "arm": "left",
            "target_body": "Scene_apple_1",
            "pick_tolerance": 0.16,
            "attach_tolerance": 0.11,
            "stages": [
                {
                    "name": "descend_to_pick",
                    "kind": "move_pose",
                    "pose": {"position": [0.49, 0.05, 0.83], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
                    "position_only": True,
                    "tolerance": 0.16,
                },
                {"name": "close_gripper", "kind": "gripper", "width": 0.0},
            ],
        })

        self.assertTrue(result["ok"])
        self.assertEqual(result["attachment"]["child_body"], "Scene_apple_1")

    def test_execute_arm_only_surface_plan_attaches_and_retreats_in_mujoco_loop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_path = Path(tmpdir) / "scene_pick.usda"
            stage = Usd.Stage.CreateNew(str(scene_path))
            stage.SetMetadata("metersPerUnit", 1.0)
            UsdGeom.SetStageUpAxis(stage, "Z")
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            crate = UsdGeom.Xform.Define(stage, "/World/Crate")
            UsdGeom.Xformable(crate.GetPrim()).AddTranslateOp().Set((0.5, 0.0, 0.83))
            UsdPhysics.MassAPI.Apply(crate.GetPrim()).CreateMassAttr(0.2)
            UsdGeom.Cube.Define(stage, "/World/Crate/Visual").CreateSizeAttr(0.092)
            stage.GetRootLayer().Save()

            bundle = ensure_render_bundle(ROBOT_USD, scene_path)
            body_map = json.loads(bundle.body_map_json.read_text(encoding="utf-8"))
            physics = MujocoPhysics("pick-apple-arm-only", bundle, body_map)
            physics.model.opt.gravity[:] = 0.0
            target_body = sanitize_name("/Scene/Crate")
            executor = PickExecutor(physics, bundle)
            try:
                ik = executor._ensure_ik()
                arm = "left"
                ik.sync_joint_state(physics.joint_state_msg())
                start = ik.get_end_effector_pose(arm)
                target_pose_before = physics.get_pose()[target_body]
                before_xyz = tuple(float(target_pose_before[index]) for index in (12, 13, 14))

                result = executor.execute({
                    "arm": arm,
                    "target_body": target_body,
                    "pick_tolerance": 0.16,
                    "attach_tolerance": 0.11,
                    "stages": [
                        {
                            "name": "descend_to_pick",
                            "kind": "move_pose",
                            "pose": {"position": [0.48, 0.06, 0.82], "quat_wxyz": list(start["rotation"])},
                            "position_only": True,
                            "tolerance": 0.16,
                        },
                        {"name": "close_gripper", "kind": "gripper", "width": 0.0},
                        {
                            "name": "retreat",
                            "kind": "move_pose",
                            "pose": {"position": [0.4, 0.18, 0.92], "quat_wxyz": list(start["rotation"])},
                            "position_only": True,
                            "tolerance": 0.08,
                        },
                    ],
                })

                ik.sync_joint_state(physics.joint_state_msg())
                reached = ik.get_end_effector_pose(arm)
                target_pose_after = physics.get_pose()[target_body]
                after_xyz = tuple(float(target_pose_after[index]) for index in (12, 13, 14))

                self.assertTrue(result["ok"])
                self.assertEqual(result["arm"], arm)
                self.assertEqual(result["attachment"]["child_body"], target_body)
                self.assertEqual(result["attachment"]["parent_body"], get_arm_config(arm).end_effector_body)
                self.assertGreater(reached["position"][0], start["position"][0] + 0.4)
                self.assertGreater(reached["position"][1], 0.0)
                self.assertLess(math.dist(reached["position"], (0.4, 0.18, 0.92)), 0.08)
                self.assertGreater(math.dist(after_xyz, before_xyz), 0.05)
            finally:
                physics.close()

    def test_attachment_uses_gripper_joint_as_parent_reference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_path = Path(tmpdir) / "scene_pick.usda"
            stage = Usd.Stage.CreateNew(str(scene_path))
            stage.SetMetadata("metersPerUnit", 1.0)
            UsdGeom.SetStageUpAxis(stage, "Z")
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            crate = UsdGeom.Xform.Define(stage, "/World/Crate")
            UsdGeom.Xformable(crate.GetPrim()).AddTranslateOp().Set((0.0, 0.0, 0.0))
            UsdPhysics.MassAPI.Apply(crate.GetPrim()).CreateMassAttr(0.2)
            UsdGeom.Cube.Define(stage, "/World/Crate/Visual").CreateSizeAttr(0.02)
            stage.GetRootLayer().Save()

            bundle = ensure_render_bundle(ROBOT_USD, scene_path)
            body_map = json.loads(bundle.body_map_json.read_text(encoding="utf-8"))
            physics = MujocoPhysics("pick-apple-gripper-joint", bundle, body_map)
            physics.model.opt.gravity[:] = 0.0
            target_body = sanitize_name("/Scene/Crate")
            try:
                arm = "left"
                gripper_pose = physics.get_pose()[get_arm_config(arm).end_effector_body]
                gripper_position = [float(gripper_pose[index]) for index in (12, 13, 14)]
                physics.set_body_pose(target_body, gripper_position, [1.0, 0.0, 0.0, 0.0])

                attachment = physics.attach_body(get_arm_config(arm).end_effector_body, target_body)

                self.assertLess(math.dist(attachment["relative_position"], (0.0, 0.0, 0.0)), 0.01)
            finally:
                physics.close()


def _matrix_at(x: float, y: float, z: float) -> list[float]:
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        x, y, z, 1.0,
    ]


def _bundle():
    return type("Bundle", (), {"robot_usd": REPO_ROOT / "robot.usda", "scene_usd": REPO_ROOT / "scene.usda"})()


if __name__ == "__main__":
    unittest.main()
