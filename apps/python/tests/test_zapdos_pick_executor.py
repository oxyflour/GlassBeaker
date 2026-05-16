from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException
from pxr import Usd, UsdGeom, UsdPhysics

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.zapdos.bundle.bundle_builder import ensure_render_bundle  # noqa: E402
from utils.zapdos.bundle.usd_to_mjcf_adapter import sanitize_name  # noqa: E402
from utils.zapdos.manipulation.executor import PickExecutor  # noqa: E402
from utils.zapdos.physics.mujoco_physics import MujocoPhysics  # noqa: E402


class _FakePhysics:
    def __init__(self) -> None:
        self.attached: list[tuple[str, str]] = []
        self.detached: list[str] = []
        self.step_count = 0
        self.pose = {
            "Scene_Crate": _matrix_at(0.0, 0.0, 0.0),
        }
        self.aabbs: dict[str, dict[str, list[float]] | None] = {}

    def joint_state_msg(self) -> dict[str, list[object]]:
        return {"name": [], "position": []}

    def apply_joint_command(self, message: dict[str, object]) -> None:
        pass

    def step(self) -> None:
        self.step_count += 1

    def attach_body(self, parent_body: str, child_body: str) -> dict[str, object]:
        self.attached.append((parent_body, child_body))
        return {"parent_body": parent_body, "child_body": child_body}

    def detach_body(self, child_body: str) -> dict[str, object]:
        self.detached.append(child_body)
        return {"ok": True}

    def get_attachment(self, child_body: str) -> dict[str, object] | None:
        if self.attached and self.attached[-1][1] == child_body and child_body not in self.detached:
            return {"child_body": child_body}
        return None

    def get_pose(self) -> dict[str, list[float]]:
        return self.pose

    def body_world_aabb(self, body_name: str) -> dict[str, list[float]] | None:
        return self.aabbs.get(body_name)


class _StaticIK:
    def __init__(self, pose: dict[str, tuple[float, ...]]) -> None:
        self.pose = pose

    def sync_joint_state(self, joint_state: dict[str, list[object]]) -> None:
        pass

    def get_end_effector_pose(self, arm: str) -> dict[str, tuple[float, ...]]:
        return self.pose

    def solve_step(
        self,
        arm: str,
        target_pose: dict[str, tuple[float, ...]],
        gripper_opening: float,
        **_kwargs: object,
    ) -> dict[str, list[object]]:
        return {"name": [], "position": []}


class _MutableIK:
    def __init__(self, pose: dict[str, tuple[float, ...]]) -> None:
        self.pose = pose

    def sync_joint_state(self, joint_state: dict[str, list[object]]) -> None:
        pass

    def get_end_effector_pose(self, arm: str) -> dict[str, tuple[float, ...]]:
        return self.pose

    def solve_step(
        self,
        arm: str,
        target_pose: dict[str, tuple[float, ...]],
        gripper_opening: float,
        **_kwargs: object,
    ) -> dict[str, list[object]]:
        self.pose = target_pose
        return {"name": [], "position": []}


class _RecordingIK(_MutableIK):
    def __init__(self, pose: dict[str, tuple[float, ...]]) -> None:
        super().__init__(pose)
        self.solve_kwargs: list[dict[str, object]] = []

    def solve_step(
        self,
        arm: str,
        target_pose: dict[str, tuple[float, ...]],
        gripper_opening: float,
        **kwargs: object,
    ) -> dict[str, list[object]]:
        self.solve_kwargs.append(dict(kwargs))
        return super().solve_step(arm, target_pose, gripper_opening, **kwargs)


class _FingerCenterIK(_MutableIK):
    def __init__(self, pose: dict[str, tuple[float, ...]]) -> None:
        super().__init__(pose)
        self.finger_pose = pose
        self.solve_targets: list[dict[str, object]] = []

    def get_gripper_finger_center_pose(self, arm: str) -> dict[str, tuple[float, ...]]:
        del arm
        return self.finger_pose

    def solve_step(
        self,
        arm: str,
        target_pose: dict[str, object],
        gripper_opening: float,
        **_kwargs: object,
    ) -> dict[str, list[object]]:
        del arm, gripper_opening
        self.solve_targets.append(target_pose)
        self.finger_pose = {
            "position": target_pose["position"],  # type: ignore
            "rotation": target_pose["rotation"],  # type: ignore
        }
        return {"name": [], "position": []}


class PickExecutorTest(unittest.TestCase):
    def test_execute_runs_stage_sequence_in_order(self):
        physics = _FakePhysics()
        ik = _MutableIK(_pose(0.25, 0.0, 0.2))
        executor = PickExecutor(physics, bundle=_bundle(), ik_controller=ik)
        visited: list[tuple[float, ...]] = []
        plan = {
            "arm": "left",
            "target_body": "Scene_Crate",
            "stages": [
                {"name": "escape_xy", "kind": "move_pose", "pose": {"position": [0.46, 0.0, 0.2], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
                {"name": "raise_to_transit", "kind": "move_pose", "pose": {"position": [0.46, 0.0, 0.92], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
                {"name": "close_gripper", "kind": "gripper", "width": 0.0},
            ],
        }

        def drive(_ik_controller, _arm, target, _gripper, steps=12, **_kwargs):
            del steps
            visited.append(target["position"])
            ik.pose = target

        with mock.patch.object(executor, "_drive_pose", side_effect=drive):
            with self.assertRaises(HTTPException):
                executor.execute(plan)

        self.assertEqual(visited[:2], [(0.46, 0.0, 0.2), (0.46, 0.0, 0.92)])

    def test_execute_drives_finger_center_move_target_through_ik(self):
        physics = _FakePhysics()
        ik = _FingerCenterIK(_pose(0.0, 0.0, 0.0))
        executor = PickExecutor(physics, bundle=_bundle(), ik_controller=ik)

        executor.execute({
            "arm": "left",
            "target_body": "Scene_Crate",
            "stages": [
                {
                    "name": "descend_to_grasp",
                    "kind": "move_pose",
                    "target_point": "finger_center",
                    "pose": {"position": [0.1, 0.2, 0.3], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
                },
            ],
        })

        self.assertTrue(ik.solve_targets)
        self.assertEqual(ik.solve_targets[-1]["target_point"], "finger_center")
        self.assertEqual(ik.finger_pose["position"], (0.1, 0.2, 0.3))

    def test_execute_reports_stage_name_when_motion_stage_fails(self):
        executor = PickExecutor(_FakePhysics(), bundle=_bundle(), ik_controller=_StaticIK(_pose(1.0, 0.0, 0.0)))

        with self.assertRaises(HTTPException) as err:
            executor.execute(_staged_plan())

        self.assertIn("descend_to_grasp", err.exception.detail)

    def test_execute_uses_arm_only_full_pose_ik_for_staged_moves(self):
        ik = _RecordingIK(_pose(0.0, 0.0, 0.08))
        executor = PickExecutor(_FakePhysics(), bundle=_bundle(), ik_controller=ik)

        executor.execute({
            "arm": "left",
            "target_body": "Scene_Crate",
            "stages": [
                {
                    "name": "raise_to_transit",
                    "kind": "move_pose",
                    "pose": {"position": [0.0, 0.0, 0.16], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
                },
            ],
        })

        self.assertTrue(ik.solve_kwargs)
        self.assertTrue(all(call.get("include_torso") in {None, False} for call in ik.solve_kwargs))
        self.assertTrue(all(call.get("position_only") in {None, False} for call in ik.solve_kwargs))

    def test_execute_opens_gripper_before_descend_and_uses_stage_steps(self):
        physics = _FakePhysics()
        physics.aabbs["Scene_Crate"] = {
            "min": [-0.01, -0.01, 0.01],
            "max": [0.01, 0.01, 0.03],
        }
        ik = _MutableIK(_pose(0.1, 0.2, 0.3))
        executor = PickExecutor(physics, bundle=_bundle(), ik_controller=ik)
        driven: list[tuple[tuple[float, ...], float, int]] = []

        def drive(_ik_controller, _arm, target, gripper, steps=12, **_kwargs):
            driven.append((target["position"], gripper, steps))
            ik.pose = target

        plan = {
            "arm": "left",
            "target_body": "Scene_Crate",
            "grasp_tolerance": 0.16,
            "attach_tolerance": 0.11,
            "stages": [
                {"name": "open_gripper", "kind": "gripper", "width": 0.05, "steps": 18},
                {
                    "name": "descend_to_grasp",
                    "kind": "move_pose",
                    "pose": {"position": [0.0, 0.0, 0.02], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
                    "position_only": True,
                    "steps": 24,
                    "tolerance": 0.16,
                },
                {"name": "close_gripper", "kind": "gripper", "width": 0.0},
            ],
        }

        with mock.patch.object(executor, "_drive_pose", side_effect=drive):
            result = executor.execute(plan)

        self.assertTrue(result["ok"])
        self.assertEqual(driven, [
            ((0.1, 0.2, 0.3), 0.05, 18),
            ((0.0, 0.0, 0.02), 0.05, 24),
            ((0.0, 0.0, 0.02), 0.0, 6),
        ])

    def test_execute_position_only_stage_uses_explicit_steps_to_slow_motion(self):
        target = {"position": [0.3, 0.0, 0.08], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}
        default_physics = _FakePhysics()
        slow_physics = _FakePhysics()
        default_executor = PickExecutor(default_physics, bundle=_bundle(), ik_controller=_MutableIK(_pose(0.0, 0.0, 0.08)))
        slow_executor = PickExecutor(slow_physics, bundle=_bundle(), ik_controller=_MutableIK(_pose(0.0, 0.0, 0.08)))

        default_executor.execute({
            "arm": "left",
            "target_body": "Scene_Crate",
            "stages": [
                {
                    "name": "descend_to_grasp",
                    "kind": "move_pose",
                    "pose": target,
                    "position_only": True,
                    "tolerance": 0.4,
                },
            ],
        })
        slow_executor.execute({
            "arm": "left",
            "target_body": "Scene_Crate",
            "stages": [
                {
                    "name": "descend_to_grasp",
                    "kind": "move_pose",
                    "pose": target,
                    "position_only": True,
                    "steps": 20,
                    "tolerance": 0.4,
                },
            ],
        })

        self.assertGreater(slow_physics.step_count, default_physics.step_count)

    def test_execute_rejects_gripper_stage_before_grasp_stage(self):
        physics = _FakePhysics()
        ik = _MutableIK(_pose(0.0, 0.0, 0.08))
        executor = PickExecutor(physics, bundle=_bundle(), ik_controller=ik)

        with self.assertRaises(HTTPException) as err:
            executor.execute(_out_of_order_gripper_plan())

        self.assertEqual(err.exception.status_code, 409)
        self.assertIn("missing descend_to_grasp", err.exception.detail)
        self.assertEqual(physics.attached, [])

    def test_execute_release_opens_gripper_and_detaches_attached_target(self):
        physics = _FakePhysics()
        physics.attached.append(("Root_r1_pro_with_gripper_left_gripper_link", "Scene_Crate"))
        ik = _MutableIK(_pose(0.0, 0.0, 0.08))
        executor = PickExecutor(physics, bundle=_bundle(), ik_controller=ik)
        driven: list[tuple[tuple[float, ...], float, int]] = []

        def drive(_ik_controller, _arm, target, gripper, steps=12, **_kwargs):
            driven.append((target["position"], gripper, steps))
            ik.pose = target

        with mock.patch.object(executor, "_drive_pose", side_effect=drive):
            result = executor.execute({
                "kind": "release",
                "arm": "left",
                "target_body": "Scene_Crate",
                "stages": [
                    {"name": "open_gripper", "kind": "gripper", "width": 0.05, "steps": 18},
                ],
            })

        self.assertTrue(result["ok"])
        self.assertEqual(driven, [((0.0, 0.0, 0.08), 0.05, 18)])
        self.assertEqual(physics.detached, ["Scene_Crate"])
        self.assertIsNone(result["attachment"])

    def test_execute_release_rejects_target_that_is_not_attached(self):
        executor = PickExecutor(_FakePhysics(), bundle=_bundle(), ik_controller=_MutableIK(_pose(0.0, 0.0, 0.08)))

        with self.assertRaises(HTTPException) as err:
            executor.execute({
                "kind": "release",
                "arm": "left",
                "target_body": "Scene_Crate",
                "stages": [
                    {"name": "open_gripper", "kind": "gripper", "width": 0.05, "steps": 18},
                ],
            })

        self.assertEqual(err.exception.status_code, 409)
        self.assertIn("not attached", err.exception.detail)

    def test_execute_rejects_unsupported_stage_kind(self):
        executor = PickExecutor(_FakePhysics(), bundle=_bundle(), ik_controller=_MutableIK(_pose(0.0, 0.0, 0.08)))

        with self.assertRaises(HTTPException) as err:
            executor.execute(_unsupported_stage_plan())

        self.assertEqual(err.exception.status_code, 409)
        self.assertIn("unsupported stage kind", err.exception.detail)

    def test_execute_rejects_attach_when_gripper_never_reaches_target(self):
        physics = _FakePhysics()
        ik = _StaticIK(_pose(1.0, 0.0, 0.0))
        executor = PickExecutor(physics, bundle=_bundle(), ik_controller=ik)

        with self.assertRaises(HTTPException) as err:
            executor.execute(_attach_reject_plan())

        self.assertEqual(err.exception.status_code, 409)
        self.assertEqual(physics.attached, [])

    def test_execute_uses_body_world_aabb_center_for_attach_proximity(self):
        physics = _FakePhysics()
        physics.pose["Scene_Crate"] = _matrix_at(1.0, 1.0, 1.0)
        physics.aabbs["Scene_Crate"] = {
            "min": [-0.01, -0.01, 0.01],
            "max": [0.01, 0.01, 0.03],
        }
        ik = _MutableIK(_pose(0.0, 0.0, 0.08))
        executor = PickExecutor(physics, bundle=_bundle(), ik_controller=ik)

        result = executor.execute(_aabb_attach_plan())

        self.assertTrue(result["ok"])
        self.assertEqual(physics.attached[-1], ("Root_r1_pro_with_gripper_right_gripper_link", "Scene_Crate"))
        self.assertEqual(physics.detached, [])

    def test_execute_detaches_when_lift_does_not_succeed(self):
        physics = _FakePhysics()
        ik = _MutableIK(_pose(0.0, 0.0, 0.08))
        executor = PickExecutor(physics, bundle=_bundle(), ik_controller=ik)

        def drive(_ik_controller, _arm, target, _gripper, steps=12, **_kwargs):
            del _ik_controller, _arm, _gripper, steps
            if target["position"] != (0.0, 0.0, 0.22):
                ik.pose = target

        with mock.patch.object(executor, "_drive_pose", side_effect=drive):
            with self.assertRaises(HTTPException) as err:
                executor.execute(_staged_plan())

        self.assertEqual(err.exception.status_code, 409)
        self.assertEqual(physics.attached[-1], ("Root_r1_pro_with_gripper_right_gripper_link", "Scene_Crate"))
        self.assertEqual(physics.detached, ["Scene_Crate"])

    def test_execute_detaches_when_staged_retreat_misses_tight_tolerance(self):
        physics = _FakePhysics()
        ik = _MutableIK(_pose(0.0, 0.0, 0.08))
        executor = PickExecutor(physics, bundle=_bundle(), ik_controller=ik)

        def drive(_ik_controller, _arm, target, _gripper, steps=12, **_kwargs):
            del _ik_controller, _arm, _gripper, steps
            if target["position"] != (0.0, 0.03, 0.02):
                ik.pose = target

        with mock.patch.object(executor, "_drive_pose", side_effect=drive):
            with self.assertRaises(HTTPException) as err:
                executor.execute(_tight_retreat_plan())

        self.assertEqual(err.exception.status_code, 409)
        self.assertIn("retreat", err.exception.detail)
        self.assertEqual(physics.attached[-1], ("Root_r1_pro_with_gripper_right_gripper_link", "Scene_Crate"))
        self.assertEqual(physics.detached, ["Scene_Crate"])

    def test_execute_detaches_when_any_stage_fails_after_attachment(self):
        physics = _FakePhysics()
        ik = _MutableIK(_pose(0.0, 0.0, 0.08))
        executor = PickExecutor(physics, bundle=_bundle(), ik_controller=ik)

        with self.assertRaises(HTTPException) as err:
            executor.execute(_post_attach_unsupported_stage_plan())

        self.assertEqual(err.exception.status_code, 409)
        self.assertIn("unsupported stage kind", err.exception.detail)
        self.assertEqual(physics.attached[-1], ("Root_r1_pro_with_gripper_right_gripper_link", "Scene_Crate"))
        self.assertEqual(physics.detached, ["Scene_Crate"])

    def test_execute_respects_explicit_empty_stages_without_using_legacy_fields(self):
        physics = _FakePhysics()
        executor = PickExecutor(physics, bundle=_bundle(), ik_controller=_MutableIK(_pose(0.0, 0.0, 0.08)))

        result = executor.execute({
            "arm": "right",
            "target_body": "Scene_Crate",
            "stages": [],
            "pre_grasp": {"position": [9.0, 9.0, 9.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
            "grasp": {"position": [9.0, 9.0, 9.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
            "lift": {"position": [9.0, 9.0, 9.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
        })

        self.assertTrue(result["ok"])
        self.assertEqual(physics.attached, [])

    def test_execute_attaches_and_lifts_collisionless_target_in_mujoco_physics_loop(self):
        robot_usd = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_path = Path(tmpdir) / "scene_pick.usda"
            stage = Usd.Stage.CreateNew(str(scene_path))
            stage.SetMetadata("metersPerUnit", 1.0)
            UsdGeom.SetStageUpAxis(stage, "Z")
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            crate = UsdGeom.Xform.Define(stage, "/World/Crate")
            UsdGeom.Xformable(crate.GetPrim()).AddTranslateOp().Set((0.0, 0.0, 0.2))
            UsdPhysics.MassAPI.Apply(crate.GetPrim()).CreateMassAttr(0.2)
            UsdGeom.Cube.Define(stage, "/World/Crate/Visual").CreateSizeAttr(0.02)
            stage.GetRootLayer().Save()

            bundle = ensure_render_bundle(robot_usd, scene_path)
            body_map = json.loads(bundle.body_map_json.read_text(encoding="utf-8"))
            physics = MujocoPhysics("pick-executor", bundle, body_map)
            physics.model.opt.gravity[:] = 0.0
            target_body = sanitize_name("/Scene/Crate")
            executor = PickExecutor(physics, bundle)
            try:
                ik = executor._ensure_ik()
                arm = "left"
                ik.sync_joint_state(physics.joint_state_msg())
                start = ik.get_end_effector_pose(arm)
                crate_pos = [start["position"][0], start["position"][1], start["position"][2] - 0.01]
                physics.set_body_pose(target_body, crate_pos, [1.0, 0.0, 0.0, 0.0])
                plan = {
                    "arm": arm,
                    "target_body": target_body,
                    "open_gripper": 0.02,
                    "stages": [
                        {
                            "name": "approach_xy",
                            "kind": "move_pose",
                            "pose": {"position": [crate_pos[0], crate_pos[1], start["position"][2]], "quat_wxyz": list(start["rotation"])},
                            "include_torso": True,
                            "position_only": True,
                            "tolerance": 0.03,
                        },
                        {
                            "name": "descend_to_pregrasp",
                            "kind": "move_pose",
                            "pose": {"position": [crate_pos[0], crate_pos[1], crate_pos[2] + 0.02], "quat_wxyz": list(start["rotation"])},
                            "include_torso": True,
                            "position_only": True,
                            "tolerance": 0.03,
                        },
                        {
                            "name": "descend_to_grasp",
                            "kind": "move_pose",
                            "pose": {"position": crate_pos, "quat_wxyz": list(start["rotation"])},
                            "include_torso": True,
                            "position_only": True,
                            "tolerance": 0.03,
                        },
                        {"name": "close_gripper", "kind": "gripper", "width": 0.02},
                        {
                            "name": "retreat",
                            "kind": "move_pose",
                            "pose": {"position": [crate_pos[0], crate_pos[1] + 0.015, start["position"][2]], "quat_wxyz": list(start["rotation"])},
                            "include_torso": True,
                            "position_only": True,
                            "tolerance": 0.03,
                        },
                    ],
                }

                result = executor.execute(plan)
                ik.sync_joint_state(physics.joint_state_msg())
                gripper = ik.get_end_effector_pose(arm)
                pose = physics.get_pose()[target_body]

                self.assertEqual(result["arm"], arm)
                self.assertEqual(result["target_body"], target_body)
                self.assertEqual(result["attachment"]["child_body"], target_body)
                self.assertEqual(result["attachment"]["parent_body"], "Root_r1_pro_with_gripper_left_gripper_link")
                self.assertGreater(gripper["position"][1], crate_pos[1] + 0.005)
                body_xyz = tuple(float(pose[index]) for index in (12, 13, 14))
                self.assertGreater(math.dist(body_xyz, crate_pos), 0.005)
            finally:
                physics.close()

    def test_execute_vertical_stage_reports_unreachable_motion(self):
        robot_usd = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"
        scene_usd = REPO_ROOT / "apps" / "python" / "assets" / "default_scene.usda"
        bundle = ensure_render_bundle(robot_usd, scene_usd)
        body_map = json.loads(bundle.body_map_json.read_text(encoding="utf-8"))
        physics = MujocoPhysics("pick-executor-raise", bundle, body_map)
        executor = PickExecutor(physics, bundle)
        try:
            ik = executor._ensure_ik()
            arm = "left"
            ik.sync_joint_state(physics.joint_state_msg())
            start = ik.get_end_effector_pose(arm)
            target = [start["position"][0], start["position"][1], start["position"][2] + 0.08]

            with self.assertRaises(HTTPException) as err:
                executor.execute({
                    "arm": arm,
                    "target_body": "unused",
                    "stages": [
                        {
                            "name": "raise_to_transit",
                            "kind": "move_pose",
                            "pose": {"position": target, "quat_wxyz": list(start["rotation"])},
                        },
                    ],
                })

            self.assertEqual(err.exception.status_code, 409)
            self.assertIn("raise_to_transit", err.exception.detail)
        finally:
            physics.close()

    def test_execute_right_vertical_stage_stops_before_large_backward_drift(self):
        robot_usd = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"
        scene_usd = REPO_ROOT / "apps" / "python" / "assets" / "default_scene.usda"
        bundle = ensure_render_bundle(robot_usd, scene_usd)
        body_map = json.loads(bundle.body_map_json.read_text(encoding="utf-8"))
        physics = MujocoPhysics("pick-executor-right-raise", bundle, body_map)
        executor = PickExecutor(physics, bundle)
        try:
            ik = executor._ensure_ik()
            arm = "right"
            ik.sync_joint_state(physics.joint_state_msg())
            start = ik.get_end_effector_pose(arm)
            target = [start["position"][0], start["position"][1], start["position"][2] + 0.08]

            with self.assertRaises(HTTPException) as err:
                executor.execute({
                    "arm": arm,
                    "target_body": "unused",
                    "stages": [
                        {
                            "name": "raise_to_transit",
                            "kind": "move_pose",
                            "pose": {"position": target, "quat_wxyz": list(start["rotation"])},
                        },
                    ],
                })

            ik.sync_joint_state(physics.joint_state_msg())
            reached = ik.get_end_effector_pose(arm)
            self.assertEqual(err.exception.status_code, 409)
            self.assertIn("raise_to_transit", err.exception.detail)
            self.assertLess(abs(reached["position"][0] - start["position"][0]), 0.06)
        finally:
            physics.close()

    def test_execute_escapes_support_footprint_before_pregrasp_when_start_under_support(self):
        physics = _FakePhysics()
        physics.pose["Scene_Crate"] = _matrix_at(0.1, 0.0, 0.2)
        ik = _MutableIK(_pose(0.25, 0.0, 0.2))
        executor = PickExecutor(physics, bundle=_bundle(), ik_controller=ik)
        driven: list[tuple[tuple[float, ...], float]] = []

        def drive(ik_controller, arm: str, target: dict[str, tuple[float, ...]], gripper: float, steps: int = 12, **_kwargs: object) -> None:
            del ik_controller, arm, steps
            driven.append((target["position"], gripper))
            ik.pose = target

        plan = {
            "arm": "right",
            "target_body": "Scene_Crate",
            "support_surface": {
                "top_z": 0.8,
                "xy_min": [-0.4, -0.3],
                "xy_max": [0.4, 0.3],
            },
            "pre_grasp": {"position": [0.1, 0.0, 0.92], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
            "grasp": {"position": [0.1, 0.0, 0.2], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
            "close": {"command": "close", "width": 0.0},
            "lift": {"position": [0.1, 0.03, 0.2], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
        }

        with mock.patch.object(executor, "_drive_pose", side_effect=drive):
            executor.execute(plan)

        self.assertEqual(driven[0][0], (0.46, 0.0, 0.2))
        self.assertEqual(driven[1][0], (0.46, 0.0, 0.92))
        self.assertEqual(driven[2][0], (0.1, 0.0, 0.92))
def _staged_plan() -> dict[str, object]:
    return {
        "arm": "right",
        "target_body": "Scene_Crate",
        "stages": [
            {"name": "approach_xy", "kind": "move_pose", "pose": {"position": [1.0, 0.0, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
            {"name": "descend_to_pregrasp", "kind": "move_pose", "pose": {"position": [1.0, 0.0, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
            {"name": "descend_to_grasp", "kind": "move_pose", "pose": {"position": [0.0, 0.0, 0.02], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
            {"name": "close_gripper", "kind": "gripper", "width": 0.0},
            {"name": "retreat", "kind": "move_pose", "pose": {"position": [0.0, 0.0, 0.22], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
        ],
    }


def _attach_reject_plan() -> dict[str, object]:
    return {
        "arm": "right",
        "target_body": "Scene_Crate",
        "stages": [
            {"name": "approach_xy", "kind": "move_pose", "pose": {"position": [1.0, 0.0, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
            {"name": "descend_to_pregrasp", "kind": "move_pose", "pose": {"position": [1.0, 0.0, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
            {"name": "descend_to_grasp", "kind": "move_pose", "pose": {"position": [1.0, 0.0, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
            {"name": "close_gripper", "kind": "gripper", "width": 0.0},
        ],
    }


def _aabb_attach_plan() -> dict[str, object]:
    return {
        "arm": "right",
        "target_body": "Scene_Crate",
        "stages": [
            {"name": "descend_to_grasp", "kind": "move_pose", "pose": {"position": [0.0, 0.0, 0.02], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
            {"name": "close_gripper", "kind": "gripper", "width": 0.0},
        ],
    }


def _out_of_order_gripper_plan() -> dict[str, object]:
    return {
        "arm": "right",
        "target_body": "Scene_Crate",
        "stages": [
            {"name": "approach_xy", "kind": "move_pose", "pose": {"position": [0.0, 0.0, 0.08], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
            {"name": "close_gripper", "kind": "gripper", "width": 0.0},
            {"name": "descend_to_grasp", "kind": "move_pose", "pose": {"position": [0.0, 0.0, 0.02], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
        ],
    }


def _unsupported_stage_plan() -> dict[str, object]:
    return {
        "arm": "right",
        "target_body": "Scene_Crate",
        "stages": [
            {"name": "approach_xy", "kind": "move_pose", "pose": {"position": [0.0, 0.0, 0.08], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
            {"name": "spin_wrist", "kind": "joint_delta"},
        ],
    }


def _post_attach_unsupported_stage_plan() -> dict[str, object]:
    return {
        "arm": "right",
        "target_body": "Scene_Crate",
        "stages": [
            {"name": "descend_to_grasp", "kind": "move_pose", "pose": {"position": [0.0, 0.0, 0.02], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
            {"name": "close_gripper", "kind": "gripper", "width": 0.0},
            {"name": "spin_wrist", "kind": "joint_delta"},
        ],
    }


def _tight_retreat_plan() -> dict[str, object]:
    return {
        "arm": "right",
        "target_body": "Scene_Crate",
        "stages": [
            {"name": "approach_xy", "kind": "move_pose", "pose": {"position": [0.0, 0.0, 0.08], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
            {"name": "descend_to_grasp", "kind": "move_pose", "pose": {"position": [0.0, 0.0, 0.02], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
            {"name": "close_gripper", "kind": "gripper", "width": 0.0},
            {"name": "retreat", "kind": "move_pose", "pose": {"position": [0.0, 0.03, 0.02], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
        ],
    }


def _pose(x: float, y: float, z: float) -> dict[str, tuple[float, ...]]:
    return {"position": (x, y, z), "rotation": (1.0, 0.0, 0.0, 0.0)}


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
