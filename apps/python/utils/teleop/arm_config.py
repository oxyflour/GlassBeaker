from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from utils.user_config import read_user_config


@dataclass(frozen=True)
class ArmConfig:
    arm: str
    joint_names: tuple[str, ...]
    gripper_joint_names: tuple[str, str]
    end_effector_body: str
    gripper_finger_body_names: tuple[str, str]
    target_offset: tuple[float, float, float]


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
        gripper_finger_body_names=(
            "Root_r1_pro_with_gripper_left_gripper_finger_link1",
            "Root_r1_pro_with_gripper_left_gripper_finger_link2",
        ),
        target_offset=(0.0, 0.0, 0.0),
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
        gripper_finger_body_names=(
            "Root_r1_pro_with_gripper_right_gripper_finger_link1",
            "Root_r1_pro_with_gripper_right_gripper_finger_link2",
        ),
        target_offset=(0.0, 0.0, 0.0),
    ),
}


def load_arm_configs(robot_key: str | None, *, config: Mapping[str, object] | None = None) -> dict[str, ArmConfig]:
    if robot_key is None:
        return dict(ARM_CONFIGS)
    payload = config if config is not None else read_user_config()
    offsets = _parse_target_offsets(payload.get("override"), robot_key)
    return {
        arm: replace(base, target_offset=offsets.get(arm, base.target_offset))
        for arm, base in ARM_CONFIGS.items()
    }


def get_arm_config(arm: str) -> ArmConfig:
    try:
        return ARM_CONFIGS[arm]
    except KeyError as exc:
        raise ValueError(f"Unsupported arm: {arm}") from exc


def _parse_target_offsets(override: object, robot_key: str) -> dict[str, tuple[float, float, float]]:
    if override is None:
        return {}
    if not isinstance(override, dict):
        raise RuntimeError("override must be a JSON object.")
    target_offset = override.get("target_offset")
    if target_offset is None:
        return {}
    if not isinstance(target_offset, dict):
        raise RuntimeError("override.target_offset must be a JSON object.")
    robot_offsets = target_offset.get(robot_key)
    if robot_offsets is None:
        return {}
    if not isinstance(robot_offsets, dict):
        raise RuntimeError(f"override.target_offset.{robot_key} must be a JSON object.")
    parsed: dict[str, tuple[float, float, float]] = {}
    for arm, value in robot_offsets.items():
        path = f"override.target_offset.{robot_key}.{arm}"
        if arm not in ARM_CONFIGS:
            raise RuntimeError(f"{path} must target a supported arm.")
        if (
            not isinstance(value, list)
            or len(value) != 3
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
        ):
            raise RuntimeError(f"{path} must be a [x, y, z] numeric list.")
        parsed[str(arm)] = (float(value[0]), float(value[1]), float(value[2]))
    return parsed
