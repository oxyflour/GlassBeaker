from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArmConfig:
    arm: str
    joint_names: tuple[str, ...]
    gripper_joint_names: tuple[str, str]
    end_effector_body: str


ARM_CONFIGS = {
    "left": ArmConfig(
        arm="left",
        joint_names=(
            "left_arm_joint1",
            "left_arm_joint2",
            "left_arm_joint3",
            "left_arm_joint4",
            "left_arm_joint5",
            "left_arm_joint6",
            "left_arm_joint7",
        ),
        gripper_joint_names=("left_gripper_finger_joint1", "left_gripper_finger_joint2"),
        end_effector_body="Root_r1_pro_with_gripper_left_gripper_link",
    ),
    "right": ArmConfig(
        arm="right",
        joint_names=(
            "right_arm_joint1",
            "right_arm_joint2",
            "right_arm_joint3",
            "right_arm_joint4",
            "right_arm_joint5",
            "right_arm_joint6",
            "right_arm_joint7",
        ),
        gripper_joint_names=("right_gripper_finger_joint1", "right_gripper_finger_joint2"),
        end_effector_body="Root_r1_pro_with_gripper_right_gripper_link",
    ),
}


def get_arm_config(arm: str) -> ArmConfig:
    try:
        return ARM_CONFIGS[arm]
    except KeyError as exc:
        raise ValueError(f"Unsupported arm: {arm}") from exc
