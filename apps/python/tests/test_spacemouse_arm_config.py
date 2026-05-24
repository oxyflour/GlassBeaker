from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.teleop.arm_config import ARM_CONFIGS, get_arm_config, load_arm_configs  # noqa: E402


class SpaceMouseArmConfigTest(unittest.TestCase):
    def test_left_arm_config_matches_r1pro_names(self):
        config = get_arm_config("left")

        self.assertEqual(config.arm, "left")
        self.assertEqual(
            config.joint_names,
            (
                "left_arm_joint1",
                "left_arm_joint2",
                "left_arm_joint3",
                "left_arm_joint4",
                "left_arm_joint5",
                "left_arm_joint6",
                "left_arm_joint7",
            ),
        )
        self.assertEqual(
            config.gripper_joint_names,
            ("left_gripper_finger_joint1", "left_gripper_finger_joint2"),
        )
        self.assertEqual(config.end_effector_body, "Root_r1_pro_with_gripper_left_gripper_link")
        self.assertEqual(
            config.gripper_finger_body_names,
            (
                "Root_r1_pro_with_gripper_left_gripper_finger_link1",
                "Root_r1_pro_with_gripper_left_gripper_finger_link2",
            ),
        )
        self.assertEqual(config.target_offset, (0.0, 0.0, 0.0))

    def test_right_arm_config_matches_r1pro_names(self):
        config = get_arm_config("right")

        self.assertEqual(config.arm, "right")
        self.assertEqual(
            config.joint_names,
            (
                "right_arm_joint1",
                "right_arm_joint2",
                "right_arm_joint3",
                "right_arm_joint4",
                "right_arm_joint5",
                "right_arm_joint6",
                "right_arm_joint7",
            ),
        )
        self.assertEqual(
            config.gripper_joint_names,
            ("right_gripper_finger_joint1", "right_gripper_finger_joint2"),
        )
        self.assertEqual(config.end_effector_body, "Root_r1_pro_with_gripper_right_gripper_link")
        self.assertEqual(
            config.gripper_finger_body_names,
            (
                "Root_r1_pro_with_gripper_right_gripper_finger_link1",
                "Root_r1_pro_with_gripper_right_gripper_finger_link2",
            ),
        )
        self.assertEqual(config.target_offset, (0.0, 0.0, 0.0))

    def test_configs_only_expose_left_and_right(self):
        self.assertEqual(sorted(ARM_CONFIGS), ["left", "right"])

    def test_load_arm_configs_applies_target_offset_from_config(self):
        configs = load_arm_configs("r1pro", config={
            "override": {
                "target_offset": {
                    "r1pro": {
                        "left": [0.01, -0.02, 0.03],
                    },
                },
            },
        })

        self.assertEqual(configs["left"].target_offset, (0.01, -0.02, 0.03))
        self.assertEqual(configs["right"].target_offset, (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
