from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from utils.user_config import read_user_config
from utils.zapdos.robot_model import get_robot_model_key_from_usd


@dataclass(frozen=True)
class GripperCollisionOverride:
    geom_name: str
    kind: str
    pos: tuple[float, float, float]
    size: tuple[float, float, float]


_R1PRO_FINGER_GEOMS = {
    "left": {
        "finger1": "Root_r1_pro_with_gripper_left_gripper_finger_link1_collisions_left_gripper_finger_link1_Mesh_geom",
        "finger2": "Root_r1_pro_with_gripper_left_gripper_finger_link2_collisions_left_gripper_finger_link2_Mesh_geom",
    },
    "right": {
        "finger1": "Root_r1_pro_with_gripper_right_gripper_finger_link1_collisions_right_gripper_finger_link1_Mesh_geom",
        "finger2": "Root_r1_pro_with_gripper_right_gripper_finger_link2_collisions_right_gripper_finger_link2_Mesh_geom",
    },
}

_GRIPPER_GEOMS_BY_ROBOT = {
    "r1pro": _R1PRO_FINGER_GEOMS,
}


def load_gripper_collision_overrides(robot_usd: str | Path | None) -> list[GripperCollisionOverride]:
    return parse_gripper_collision_overrides(
        read_user_config().get("override"),
        get_robot_model_key_from_usd(robot_usd),
    )


def parse_gripper_collision_overrides(
    override: object,
    robot_key: str | None,
) -> list[GripperCollisionOverride]:
    if robot_key is None or override is None:
        return []
    if not isinstance(override, dict):
        raise RuntimeError("override must be a JSON object.")
    gripper_collision = override.get("gripper_collision")
    if gripper_collision is None:
        return []
    if not isinstance(gripper_collision, dict):
        raise RuntimeError("override.gripper_collision must be a JSON object.")
    robot_payload = gripper_collision.get(robot_key)
    if robot_payload is None:
        return []
    if not isinstance(robot_payload, dict):
        raise RuntimeError(f"override.gripper_collision.{robot_key} must be a JSON object.")
    robot_targets = _GRIPPER_GEOMS_BY_ROBOT.get(robot_key)
    if robot_targets is None:
        raise RuntimeError(f"override.gripper_collision.{robot_key} is not supported.")

    parsed: list[GripperCollisionOverride] = []
    for arm, arm_payload in robot_payload.items():
        arm_path = f"override.gripper_collision.{robot_key}.{arm}"
        if arm not in robot_targets:
            raise RuntimeError(f"{arm_path} must target a supported arm.")
        if not isinstance(arm_payload, dict):
            raise RuntimeError(f"{arm_path} must be a JSON object.")
        for finger, spec in arm_payload.items():
            path = f"{arm_path}.{finger}"
            geom_name = robot_targets[arm].get(str(finger))
            if geom_name is None:
                raise RuntimeError(f"{path} must target finger1 or finger2.")
            if not isinstance(spec, dict):
                raise RuntimeError(f"{path} must be a JSON object.")
            kind = spec.get("type", "box")
            if kind != "box":
                raise RuntimeError(f"{path}.type must be 'box'.")
            parsed.append(
                GripperCollisionOverride(
                    geom_name=geom_name,
                    kind="box",
                    pos=_parse_vec3(spec.get("pos", [0.0, 0.0, 0.0]), f"{path}.pos", positive=False),
                    size=_parse_vec3(spec.get("size"), f"{path}.size", positive=True),
                )
            )
    return parsed


def apply_gripper_collision_overrides(mjcf: Path, overrides: list[GripperCollisionOverride]) -> None:
    if not overrides:
        return
    tree = ET.parse(mjcf)
    root = tree.getroot()
    for spec in overrides:
        geom = root.find(f".//geom[@name='{spec.geom_name}']")
        if geom is None:
            raise RuntimeError(f"Missing gripper collision geom: {spec.geom_name}")
        geom.set("type", spec.kind)
        geom.set("pos", _fmt_vec(spec.pos))
        geom.set("size", _fmt_vec(spec.size))
        for attr in ("mesh", "material"):
            geom.attrib.pop(attr, None)
    tree.write(mjcf, encoding="utf-8", xml_declaration=True)


def _parse_vec3(value: object, path: str, *, positive: bool) -> tuple[float, float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise RuntimeError(f"{path} must be a [x, y, z] numeric list.")
    parsed = (float(value[0]), float(value[1]), float(value[2]))
    if positive and any(item <= 0.0 for item in parsed):
        raise RuntimeError(f"{path} values must be positive.")
    return parsed


def _fmt_vec(value: tuple[float, float, float]) -> str:
    return " ".join(f"{item:g}" for item in value)


__all__ = [
    "GripperCollisionOverride",
    "apply_gripper_collision_overrides",
    "load_gripper_collision_overrides",
    "parse_gripper_collision_overrides",
]
