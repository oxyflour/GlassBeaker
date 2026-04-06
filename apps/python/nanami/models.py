from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class SourceConfig:
    repo: str = "local"
    ref: str = "workspace"
    robot_path: str = "R1"
    proxy: str | None = None


@dataclass(slots=True)
class JointDef:
    name: str
    type: str
    parent_link: str
    child_link: str
    origin_xyz: list[float]
    origin_rpy: list[float]
    axis_xyz: list[float]
    lower: float | None = None
    upper: float | None = None


@dataclass(slots=True)
class LinkDef:
    name: str
    color_rgba: list[float]
    mesh_source: str | None = None
    mesh_obj_path: str | None = None
    parent_joint: str | None = None


@dataclass(slots=True)
class JointControl:
    name: str
    kind: str
    lower: float
    upper: float


@dataclass(slots=True)
class ControlGroup:
    name: str
    joints: list[JointControl] = field(default_factory=list)


@dataclass(slots=True)
class RobotManifest:
    robot_id: str
    package_name: str
    resolved_commit: str
    source: SourceConfig
    links: list[LinkDef]
    joints: list[JointDef]
    movable_joint_count: int
    link_count: int
    controls: list[ControlGroup]

    def to_dict(self) -> dict:
        return asdict(self)


def manifest_from_dict(data: dict) -> RobotManifest:
    return RobotManifest(
        robot_id=data["robot_id"],
        package_name=data["package_name"],
        resolved_commit=data["resolved_commit"],
        source=SourceConfig(**data["source"]),
        links=[LinkDef(**item) for item in data["links"]],
        joints=[JointDef(**item) for item in data["joints"]],
        movable_joint_count=data["movable_joint_count"],
        link_count=data["link_count"],
        controls=[
            ControlGroup(name=item["name"], joints=[JointControl(**joint) for joint in item["joints"]])
            for item in data["controls"]
        ],
    )


def control_group_for_joint(name: str) -> str | None:
    if name.startswith(("steer_motor_", "wheel_motor_")):
        return "base"
    if name.startswith("torso_"):
        return "torso"
    if name.startswith("left_arm_"):
        return "left_arm"
    if name.startswith("left_gripper_"):
        return "left_gripper"
    if name.startswith("right_arm_"):
        return "right_arm"
    if name.startswith("right_gripper_"):
        return "right_gripper"
    return None
