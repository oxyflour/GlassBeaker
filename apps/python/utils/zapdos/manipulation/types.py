from typing import NotRequired, TypedDict


class WorldAabb(TypedDict):
    min: list[float]
    max: list[float]


class SceneObject(TypedDict):
    body: str
    label: str
    asset_id: str | None
    motion: str | None
    tags: list[str]
    support_body: str | None
    position: list[float] | None
    matrix: list[float] | None
    top_z: float | None
    bounds_min: list[float] | None
    bounds_max: list[float] | None
    world_aabb: WorldAabb | None


class GroundedPick(TypedDict):
    target: SceneObject
    support: SceneObject | None


class PickPose(TypedDict):
    frame: str
    position: list[float]
    quat_wxyz: list[float]


class PickOrientation(TypedDict):
    mode: str
    quat_wxyz: list[float]


class PlanningPose(TypedDict):
    position: list[float]
    quat_wxyz: list[float]


class PickCommand(TypedDict):
    command: str
    width: float


class SupportSurface(TypedDict):
    top_z: float
    xy_min: list[float]
    xy_max: list[float]


class PickStage(TypedDict):
    name: str
    kind: str
    pose: NotRequired[PickPose]
    width: NotRequired[float]


class PickPlan(TypedDict):
    kind: str
    target_body: str
    orientation: PickOrientation
    stages: NotRequired[list[PickStage]]
    pre_grasp: NotRequired[PickPose]
    grasp: NotRequired[PickPose]
    close: NotRequired[PickCommand]
    lift: NotRequired[PickPose]
    support_surface: NotRequired[SupportSurface]


__all__ = [
    "GroundedPick",
    "PickCommand",
    "PickOrientation",
    "PickPlan",
    "PickPose",
    "PickStage",
    "PlanningPose",
    "SceneObject",
    "SupportSurface",
    "WorldAabb",
]
