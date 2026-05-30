from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from utils.user_config import read_user_config
from .robot_model import get_robot_model_key_from_usd


class JointDriveOverride(TypedDict, total=False):
    damping: float
    stiffness: float
    kp: float
    forcerange: tuple[float, float]


def load_joint_drive_overrides(robot_usd: str | Path | None) -> dict[str, JointDriveOverride]:
    return parse_joint_drive_overrides(read_user_config().get("override"), get_robot_model_key_from_usd(robot_usd))


def parse_joint_drive_overrides(override: object, robot_key: str | None) -> dict[str, JointDriveOverride]:
    if robot_key is None or override is None:
        return {}
    if not isinstance(override, dict):
        raise RuntimeError("override must be a JSON object.")
    joint_drive = override.get("joint_drive")
    if joint_drive is None:
        return {}
    if not isinstance(joint_drive, dict):
        raise RuntimeError("override.joint_drive must be a JSON object.")
    robot_drive = joint_drive.get(robot_key)
    if robot_drive is None:
        return {}
    if not isinstance(robot_drive, dict):
        raise RuntimeError(f"override.joint_drive.{robot_key} must be a JSON object.")
    parsed: dict[str, JointDriveOverride] = {}
    for joint_name, payload in robot_drive.items():
        path = f"override.joint_drive.{robot_key}.{joint_name}"
        if not isinstance(payload, dict):
            raise RuntimeError(f"{path} must be a JSON object.")
        spec: JointDriveOverride = {}
        for key, value in payload.items():
            if key in {"damping", "stiffness", "kp"}:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise RuntimeError(f"{path}.{key} must be numeric.")
                spec[key] = float(value)
                continue
            if key == "forcerange":
                if (
                    not isinstance(value, list)
                    or len(value) != 2
                    or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
                ):
                    raise RuntimeError(f"{path}.forcerange must be a [min, max] numeric pair.")
                lower = float(value[0])
                upper = float(value[1])
                if lower > upper:
                    raise RuntimeError(f"{path}.forcerange must be ordered as [min, max].")
                spec["forcerange"] = (lower, upper)
                continue
            raise RuntimeError(f"{path}.{key} is not supported.")
        parsed[joint_name] = spec
    return parsed


__all__ = ["JointDriveOverride", "load_joint_drive_overrides", "parse_joint_drive_overrides"]
