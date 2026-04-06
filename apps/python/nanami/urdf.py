from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from .models import ControlGroup, JointControl, JointDef, LinkDef, RobotManifest, SourceConfig, control_group_for_joint


def float_list(raw: str | None, fallback: list[float]) -> list[float]:
    if not raw:
        return fallback[:]
    return [float(item) for item in raw.split()]


def element_attr(node: ET.Element | None, name: str, fallback: str | None = None) -> str | None:
    if node is None:
        return fallback
    return node.attrib.get(name, fallback)


def build_controls(joints: list[JointDef]) -> list[ControlGroup]:
    groups: dict[str, ControlGroup] = {}
    for joint in joints:
        if joint.type == "fixed":
            continue
        group_name = control_group_for_joint(joint.name)
        if group_name is None:
            continue
        lower = joint.lower if joint.lower is not None else -3.141592653589793
        upper = joint.upper if joint.upper is not None else 3.141592653589793
        if joint.type == "prismatic":
            lower = joint.lower if joint.lower is not None else 0.0
            upper = joint.upper if joint.upper is not None else 0.05
        groups.setdefault(group_name, ControlGroup(name=group_name)).joints.append(
            JointControl(name=joint.name, kind=joint.type, lower=lower, upper=upper)
        )
    ordered = ["base", "torso", "left_arm", "left_gripper", "right_arm", "right_gripper"]
    return [groups[name] for name in ordered if name in groups]


def parse_robot_manifest(urdf_path: Path, source: SourceConfig, resolved_commit: str) -> RobotManifest:
    root = ET.fromstring(urdf_path.read_text(encoding="utf-8"))
    package_name = root.attrib["name"]
    links: list[LinkDef] = []
    joints: list[JointDef] = []
    parent_joint_for_link: dict[str, str] = {}

    for joint_node in root.findall("joint"):
        limit = joint_node.find("limit")
        origin = joint_node.find("origin")
        axis = joint_node.find("axis")
        joint = JointDef(
            name=joint_node.attrib["name"],
            type=joint_node.attrib["type"],
            parent_link=joint_node.find("parent").attrib["link"],  # type: ignore[union-attr]
            child_link=joint_node.find("child").attrib["link"],  # type: ignore[union-attr]
            origin_xyz=float_list(element_attr(origin, "xyz"), [0.0, 0.0, 0.0]),
            origin_rpy=float_list(element_attr(origin, "rpy"), [0.0, 0.0, 0.0]),
            axis_xyz=float_list(element_attr(axis, "xyz"), [0.0, 0.0, 0.0]),
            lower=float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None,
            upper=float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None,
        )
        joints.append(joint)
        parent_joint_for_link[joint.child_link] = joint.name

    for link_node in root.findall("link"):
        visual = link_node.find("visual")
        color = visual.find("material/color").attrib.get("rgba") if visual is not None and visual.find("material/color") is not None else None  # type: ignore[union-attr]
        mesh = visual.find("geometry/mesh").attrib.get("filename") if visual is not None and visual.find("geometry/mesh") is not None else None  # type: ignore[union-attr]
        links.append(
            LinkDef(
                name=link_node.attrib["name"],
                color_rgba=float_list(color, [0.8, 0.8, 0.8, 1.0]),
                mesh_source=mesh,
                parent_joint=parent_joint_for_link.get(link_node.attrib["name"]),
            )
        )

    movable_joint_count = sum(joint.type != "fixed" for joint in joints)
    return RobotManifest(
        robot_id=package_name,
        package_name=package_name,
        resolved_commit=resolved_commit,
        source=source,
        links=links,
        joints=joints,
        movable_joint_count=movable_joint_count,
        link_count=len(links),
        controls=build_controls(joints),
    )
