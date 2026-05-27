from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from utils.user_config import read_user_config
from utils.zapdos.robot_model import get_robot_model_key_from_usd


@dataclass(frozen=True)
class GripperCollisionOverride:
    geom_name: str
    base_geom_name: str
    kind: str
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
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
            for index, geom_spec in enumerate(_iter_geom_specs(spec, path)):
                parsed.append(
                    GripperCollisionOverride(
                        geom_name=_override_geom_name(geom_name, geom_spec, index, path),
                        base_geom_name=geom_name,
                        kind="box",
                        pos=_parse_vec3(geom_spec.get("pos", [0.0, 0.0, 0.0]), f"{path}.pos", positive=False),
                        quat=_parse_vec4(geom_spec.get("quat", [1.0, 0.0, 0.0, 0.0]), f"{path}.quat"),
                        size=_parse_vec3(geom_spec.get("size"), f"{path}.size", positive=True),
                    )
                )
    return parsed


def apply_gripper_collision_overrides(mjcf: Path, overrides: list[GripperCollisionOverride]) -> None:
    if not overrides:
        return
    tree = ET.parse(mjcf)
    root = tree.getroot()
    specs_by_base: dict[str, list[GripperCollisionOverride]] = {}
    for spec in overrides:
        specs_by_base.setdefault(spec.base_geom_name, []).append(spec)
    for base_geom_name, specs in specs_by_base.items():
        parent, base_geom = _find_geom_parent(root, base_geom_name)
        if base_geom is None or parent is None:
            raise RuntimeError(f"Missing gripper collision geom: {base_geom_name}")
        insert_at = list(parent).index(base_geom)
        for index, spec in enumerate(specs):
            geom = base_geom if index == 0 else ET.Element("geom", dict(base_geom.attrib))
            _apply_geom_override(geom, spec)
            if index > 0:
                parent.insert(insert_at + index, geom)
    tree.write(mjcf, encoding="utf-8", xml_declaration=True)


def _iter_geom_specs(spec: dict[str, object], path: str) -> list[dict[str, object]]:
    geoms = spec.get("geoms")
    if geoms is None:
        return [spec]
    if not isinstance(geoms, list) or not geoms:
        raise RuntimeError(f"{path}.geoms must be a non-empty list.")
    parsed = []
    for index, item in enumerate(geoms):
        if not isinstance(item, dict):
            raise RuntimeError(f"{path}.geoms[{index}] must be a JSON object.")
        parsed.append(item)
    return parsed


def _override_geom_name(base_name: str, spec: dict[str, object], index: int, path: str) -> str:
    if index == 0:
        return base_name
    suffix = spec.get("name", f"box_{index + 1}")
    if not isinstance(suffix, str) or not suffix or any(not (char.isalnum() or char == "_") for char in suffix):
        raise RuntimeError(f"{path}.geoms[{index}].name must contain only letters, digits, and underscores.")
    return f"{base_name}_{suffix}"


def _find_geom_parent(root: ET.Element, name: str) -> tuple[ET.Element | None, ET.Element | None]:
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom" and child.get("name") == name:
                return parent, child
    return None, None


def _apply_geom_override(geom: ET.Element, spec: GripperCollisionOverride) -> None:
    geom.set("name", spec.geom_name)
    geom.set("type", spec.kind)
    geom.set("pos", _fmt_vec(spec.pos))
    geom.set("quat", _fmt_vec(spec.quat))
    geom.set("size", _fmt_vec(spec.size))
    for attr in ("mesh", "material"):
        geom.attrib.pop(attr, None)


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


def _parse_vec4(value: object, path: str) -> tuple[float, float, float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise RuntimeError(f"{path} must be a [w, x, y, z] numeric list.")
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _fmt_vec(value: tuple[float, ...]) -> str:
    return " ".join(f"{item:g}" for item in value)


__all__ = [
    "GripperCollisionOverride",
    "apply_gripper_collision_overrides",
    "load_gripper_collision_overrides",
    "parse_gripper_collision_overrides",
]
