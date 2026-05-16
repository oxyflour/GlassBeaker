from __future__ import annotations

from utils.zapdos.manipulation.types import PickPlan, PickPose, PickStage, PlanningPose, SceneObject, SupportSurface

TOP_DOWN_QUAT_WXYZ = [1.0, 0.0, 0.0, 0.0]
GRIPPER_LINK_TO_FINGER_MID_Z = 0.03689


def plan_pick(
    target: SceneObject,
    *,
    support: SceneObject | None,
    scene_objects: list[SceneObject],
    arm: str,
    start_pose: PlanningPose,
    xy_margin: float = 0.06,
    z_margin: float = 0.05,
) -> PickPlan:
    support_surface = _support_surface(support)
    if support_surface is None:
        raise ValueError("planner_insufficient_geometry: support surface bounds are required")
    grasp_center = _grasp_center(target)
    if grasp_center is None:
        raise ValueError(f"Target has no world position: {target['body']}")
    grasp_z = _grasp_z(target, grasp_center[2])
    pre_grasp = _pose(grasp_center[0], grasp_center[1], grasp_z + 0.12)
    grasp = _pose(grasp_center[0], grasp_center[1], grasp_z)
    stages: list[PickStage] = []
    if _needs_escape(start_pose["position"], support_surface):
        escape_xy = _escape_xy(start_pose["position"], support_surface, xy_margin)
        transit_z = _transit_z(
            _pose(escape_xy[0], escape_xy[1], start_pose["position"][2]),
            pre_grasp,
            scene_objects,
            ignore={target["body"]},
            z_margin=z_margin,
        )
        stages.append(_move_stage("escape_xy", escape_xy[0], escape_xy[1], start_pose["position"][2], start_pose["quat_wxyz"]))
        stages.append(_move_stage("raise_to_transit", escape_xy[0], escape_xy[1], transit_z, pre_grasp["quat_wxyz"]))
    else:
        transit_z = _transit_z(start_pose, pre_grasp, scene_objects, ignore={target["body"]}, z_margin=z_margin)
    if not stages and start_pose["position"][2] < transit_z:
        stages.append(_move_stage("raise_to_transit", start_pose["position"][0], start_pose["position"][1], transit_z, pre_grasp["quat_wxyz"]))
    stages.extend([
        _move_stage("approach_xy", pre_grasp["position"][0], pre_grasp["position"][1], transit_z, pre_grasp["quat_wxyz"]),
        _move_stage("descend_to_pregrasp", *pre_grasp["position"], pre_grasp["quat_wxyz"]),
        _move_stage("descend_to_grasp", *grasp["position"], grasp["quat_wxyz"]),
        {"name": "close_gripper", "kind": "gripper", "width": 0.0},
        _move_stage("retreat", pre_grasp["position"][0], pre_grasp["position"][1], transit_z, pre_grasp["quat_wxyz"]),
    ])
    return {
        "kind": "pick",
        "target_body": target["body"],
        "orientation": {"mode": "top_down", "quat_wxyz": list(TOP_DOWN_QUAT_WXYZ)},
        "stages": stages,
        "pre_grasp": pre_grasp,
        "grasp": grasp,
        "close": {"command": "close", "width": 0.0},
        "lift": _pose(pre_grasp["position"][0], pre_grasp["position"][1], transit_z),
        "support_surface": support_surface,
    }


def _grasp_center(target: SceneObject) -> tuple[float, float, float] | None:
    aabb = target.get("world_aabb") or _world_aabb_from_bounds(target)
    if aabb is not None:
        return (
            0.5 * (float(aabb["min"][0]) + float(aabb["max"][0])),
            0.5 * (float(aabb["min"][1]) + float(aabb["max"][1])),
            0.5 * (float(aabb["min"][2]) + float(aabb["max"][2])),
        )
    if target["position"] is None:
        return None
    return float(target["position"][0]), float(target["position"][1]), float(target["position"][2])


def _grasp_z(target: SceneObject, center_z: float) -> float:
    if target.get("world_aabb") is not None or _world_aabb_from_bounds(target) is not None:
        return center_z + GRIPPER_LINK_TO_FINGER_MID_Z
    if target["bounds_min"] is not None and target["bounds_max"] is not None:
        center_z += 0.5 * (float(target["bounds_min"][2]) + float(target["bounds_max"][2]))
    return center_z + GRIPPER_LINK_TO_FINGER_MID_Z


def _pose(x: float, y: float, z: float) -> PickPose:
    return {
        "frame": "world",
        "position": [round(float(x), 6), round(float(y), 6), round(float(z), 6)],
        "quat_wxyz": list(TOP_DOWN_QUAT_WXYZ),
    }


def _move_stage(
    name: str,
    x: float,
    y: float,
    z: float,
    quat_wxyz: list[float],
) -> PickStage:
    return {
        "name": name,
        "kind": "move_pose",
        "pose": {
            "frame": "world",
            "position": [round(float(x), 6), round(float(y), 6), round(float(z), 6)],
            "quat_wxyz": [round(float(value), 6) for value in quat_wxyz],
        },
    }


def _support_surface(support: SceneObject | None) -> SupportSurface | None:
    if support is None or support["top_z"] is None:
        return None
    aabb = support.get("world_aabb")
    if aabb is not None:
        return {
            "top_z": float(support["top_z"]),
            "xy_min": [round(float(aabb["min"][0]), 6), round(float(aabb["min"][1]), 6)],
            "xy_max": [round(float(aabb["max"][0]), 6), round(float(aabb["max"][1]), 6)],
        }
    if support["bounds_min"] is None or support["bounds_max"] is None:
        return None
    if support["position"] is None:
        return None
    center_x, center_y = (float(support["position"][0]), float(support["position"][1]))
    return {
        "top_z": float(support["top_z"]),
        "xy_min": [
            round(center_x + float(support["bounds_min"][0]), 6),
            round(center_y + float(support["bounds_min"][1]), 6),
        ],
        "xy_max": [
            round(center_x + float(support["bounds_max"][0]), 6),
            round(center_y + float(support["bounds_max"][1]), 6),
        ],
    }


def _needs_escape(position: list[float], support_surface: SupportSurface) -> bool:
    return (
        float(position[2]) < float(support_surface["top_z"])
        and float(support_surface["xy_min"][0]) <= float(position[0]) <= float(support_surface["xy_max"][0])
        and float(support_surface["xy_min"][1]) <= float(position[1]) <= float(support_surface["xy_max"][1])
    )


def _escape_xy(position: list[float], support_surface: SupportSurface, xy_margin: float) -> tuple[float, float]:
    x, y = float(position[0]), float(position[1])
    x_min, y_min = (float(support_surface["xy_min"][0]), float(support_surface["xy_min"][1]))
    x_max, y_max = (float(support_surface["xy_max"][0]), float(support_surface["xy_max"][1]))
    candidates = (
        (x_min - xy_margin, y),
        (x_max + xy_margin, y),
        (x, y_min - xy_margin),
        (x, y_max + xy_margin),
    )
    return min(candidates, key=lambda candidate: abs(candidate[0] - x) + abs(candidate[1] - y))


def _transit_z(
    start_pose: PlanningPose,
    goal_pose: PickPose,
    scene_objects: list[SceneObject],
    *,
    ignore: set[str],
    z_margin: float,
) -> float:
    x_min = min(float(start_pose["position"][0]), float(goal_pose["position"][0])) - z_margin
    x_max = max(float(start_pose["position"][0]), float(goal_pose["position"][0])) + z_margin
    y_min = min(float(start_pose["position"][1]), float(goal_pose["position"][1])) - z_margin
    y_max = max(float(start_pose["position"][1]), float(goal_pose["position"][1])) + z_margin
    transit_z = max(float(start_pose["position"][2]), float(goal_pose["position"][2]))
    for obj in scene_objects:
        if obj["body"] in ignore:
            continue
        aabb = obj.get("world_aabb")
        if aabb is None:
            aabb = _world_aabb_from_bounds(obj)
        if aabb is None:
            continue
        if float(aabb["max"][0]) < x_min or float(aabb["min"][0]) > x_max:
            continue
        if float(aabb["max"][1]) < y_min or float(aabb["min"][1]) > y_max:
            continue
        obstacle_top = float(aabb["max"][2])
        if obj.get("top_z") is not None:
            obstacle_top = max(obstacle_top, float(obj["top_z"]))
        transit_z = max(transit_z, obstacle_top + z_margin)
    return round(transit_z, 6)


def _world_aabb_from_bounds(obj: SceneObject) -> dict[str, list[float]] | None:
    if obj["position"] is None or obj["bounds_min"] is None or obj["bounds_max"] is None:
        return None
    center_x, center_y, center_z = (float(obj["position"][0]), float(obj["position"][1]), float(obj["position"][2]))
    return {
        "min": [
            center_x + float(obj["bounds_min"][0]),
            center_y + float(obj["bounds_min"][1]),
            center_z + float(obj["bounds_min"][2]),
        ],
        "max": [
            center_x + float(obj["bounds_max"][0]),
            center_y + float(obj["bounds_max"][1]),
            center_z + float(obj["bounds_max"][2]),
        ],
    }


__all__ = ["TOP_DOWN_QUAT_WXYZ", "plan_pick"]
