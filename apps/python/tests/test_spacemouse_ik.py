from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.teleop.arm_config import get_arm_config  # noqa: E402
from utils.teleop.ik_controller import IKController  # noqa: E402
from utils.zapdos.physics.mujoco_physics import MujocoPhysics  # noqa: E402

ROBOT_USD = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"
SCENE_USD = REPO_ROOT / "apps" / "python" / "assets" / "default_scene.usda"


def pose_error(pose: dict[str, tuple[float, ...]], target: dict[str, tuple[float, ...]]) -> float:
    return math.dist(pose["position"], target["position"])


def rotation_error_radians(current: tuple[float, ...], target: tuple[float, ...]) -> float:
    dot = sum(a * b for a, b in zip(current, target))
    dot = max(-1.0, min(1.0, abs(dot)))
    return 2.0 * math.acos(dot)


class SpaceMouseIKControllerTest(unittest.TestCase):
    def setUp(self):
        self.controller = IKController(ROBOT_USD, SCENE_USD)

    def test_right_arm_step_reduces_small_position_error(self):
        pose = self.controller.get_end_effector_pose("right")
        target = {
            "position": (pose["position"][0] + 0.01, pose["position"][1], pose["position"][2]),
            "rotation": pose["rotation"],
        }

        before = pose_error(pose, target)
        command = self.controller.solve_step("right", target, 0.01)
        self.controller.sync_joint_state(command)
        after = pose_error(self.controller.get_end_effector_pose("right"), target)

        self.assertLess(after, before)

    def test_left_arm_step_respects_joint_limits(self):
        pose = self.controller.get_end_effector_pose("left")
        target = {
            "position": (pose["position"][0], pose["position"][1] + 0.03, pose["position"][2]),
            "rotation": pose["rotation"],
        }

        command = self.controller.solve_step("left", target, 0.02)
        self.assertEqual(command["name"][:7], list(self.controller.arm_joint_names("left")))
        for joint_name, position in zip(command["name"][:7], command["position"][:7]):
            lower, upper = self.controller.joint_limits(joint_name)
            self.assertGreaterEqual(position, lower)
            self.assertLessEqual(position, upper)

    def test_left_arm_reaches_nearby_target_using_gripper_joint_endpoint(self):
        body_map = json.loads(self.controller.bundle.body_map_json.read_text(encoding="utf-8"))
        physics = MujocoPhysics("ik-test", self.controller.bundle, body_map)
        arm = "left"
        config = get_arm_config(arm)
        try:
            self.controller.sync_joint_state(physics.joint_state_msg())
            start = self.controller.get_end_effector_pose(arm)
            target = {
                "position": (start["position"][0], start["position"][1] + 0.03, start["position"][2]),
                "rotation": start["rotation"],
            }

            before = pose_error(start, target)
            for _ in range(80):
                self.controller.sync_joint_state(physics.joint_state_msg())
                physics.apply_joint_command(self.controller.solve_step(arm, target, 0.02))
                physics.step()

            pose = physics.get_pose()[config.end_effector_body]
            self.controller.sync_joint_state(physics.joint_state_msg())
            reached = self.controller.get_end_effector_pose(arm)
            after = pose_error(reached, target)
            rotation_error = rotation_error_radians(reached["rotation"], target["rotation"])

            pose_matrix = np.array(pose, dtype=float).reshape(4, 4).T
            expected = pose_matrix[:3, 3]
            self.assertLess(after, before)
            self.assertLess(after, 0.01)
            self.assertLess(math.dist(tuple(expected.tolist()), reached["position"]), 0.002)
            self.assertLess(rotation_error, 0.05)
        finally:
            physics.close()

    def test_finger_center_target_converts_to_gripper_joint_target_with_rotation(self):
        arm = "left"
        end_effector = self.controller.get_end_effector_pose(arm)
        finger_center = self.controller.get_gripper_finger_center_pose(arm)
        current_rotation = _quat_matrix(end_effector["rotation"])
        local_offset = current_rotation.T @ (
            np.asarray(finger_center["position"], dtype=float) - np.asarray(end_effector["position"], dtype=float)
        )
        target_rotation = (
            0.7071067811865476,
            0.0,
            0.7071067811865475,
            0.0,
        )
        desired_finger_center = (0.5, -0.1, 0.83)

        gripper_joint_target = self.controller.finger_center_pose_to_end_effector_pose(arm, {
            "position": desired_finger_center,
            "rotation": target_rotation,
        })

        expected = np.asarray(desired_finger_center, dtype=float) - _quat_matrix(target_rotation) @ local_offset
        self.assertLess(math.dist(gripper_joint_target["position"], tuple(expected.tolist())), 1e-9)
        self.assertEqual(gripper_joint_target["rotation"], target_rotation)

    def test_left_arm_position_only_with_torso_reaches_vertical_target(self):
        body_map = json.loads(self.controller.bundle.body_map_json.read_text(encoding="utf-8"))
        physics = MujocoPhysics("ik-test", self.controller.bundle, body_map)
        arm = "left"
        try:
            self.controller.sync_joint_state(physics.joint_state_msg())
            start = self.controller.get_end_effector_pose(arm)
            target = {
                "position": (start["position"][0], start["position"][1], start["position"][2] + 0.08),
                "rotation": start["rotation"],
            }

            for _ in range(300):
                self.controller.sync_joint_state(physics.joint_state_msg())
                physics.apply_joint_command(self.controller.solve_step(
                    arm,
                    target,
                    0.02,
                    include_torso=True,
                    position_only=True,
                ))
                physics.step()

            self.controller.sync_joint_state(physics.joint_state_msg())
            reached = self.controller.get_end_effector_pose(arm)

            self.assertLess(pose_error(reached, target), 0.015)
            self.assertGreater(reached["position"][2] - start["position"][2], 0.06)
            self.assertLess(abs(reached["position"][0] - start["position"][0]), 0.02)
            self.assertLess(abs(reached["position"][1] - start["position"][1]), 0.02)
        finally:
            physics.close()


def _quat_matrix(quat_wxyz: tuple[float, ...]) -> np.ndarray:
    w, x, y, z = quat_wxyz
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


if __name__ == "__main__":
    unittest.main()
