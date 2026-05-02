from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from teleop.ik_controller import IKController  # noqa: E402

ROBOT_USD = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"
SCENE_USD = REPO_ROOT / "apps" / "python" / "assets" / "default_scene.usda"


def pose_error(pose: dict[str, tuple[float, ...]], target: dict[str, tuple[float, ...]]) -> float:
    return math.dist(pose["position"], target["position"])


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


if __name__ == "__main__":
    unittest.main()
