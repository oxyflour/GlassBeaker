from __future__ import annotations

from fastapi import HTTPException

from utils.zapdos.manipulation.types import SceneObject


def build_pick_apple_plan(
    target: SceneObject,
    current_pose: dict[str, list[float]],
    *,
    arm: str,
    open_width: float,
) -> dict[str, object]:
    center = _target_center(target)
    side = 1.0 if arm == "left" else -1.0
    quat_wxyz = list(current_pose["quat_wxyz"])
    pick_position = [round(center[0], 6), round(center[1], 6), round(center[2], 6)]
    above_position = [pick_position[0], pick_position[1], round(center[2] + 0.12, 6)]
    retreat_position = [
        round(center[0] - 0.10, 6),
        round(center[1] + side * 0.18, 6),
        round(center[2] + 0.09, 6),
    ]
    return {
        "kind": "pick",
        "arm": arm,
        "target_body": target["body"],
        "pick_tolerance": 0.025,
        "attach_tolerance": 0.015,
        "stages": [
            {"name": "open_gripper", "kind": "gripper", "width": open_width, "steps": 18},
            {"name": "approach_above", "kind": "move_pose", "pose": {"position": above_position, "quat_wxyz": quat_wxyz}, "target_point": "finger_center", "position_only": True, "steps": 20, "tolerance": 0.05},
            {"name": "descend_to_pick", "kind": "move_pose", "pose": {"position": pick_position, "quat_wxyz": quat_wxyz}, "target_point": "finger_center", "position_only": True, "steps": 24, "tolerance": 0.05},
            {"name": "close_gripper", "kind": "gripper", "width": 0.0},
            {"name": "retreat", "kind": "move_pose", "pose": {"position": retreat_position, "quat_wxyz": quat_wxyz}, "target_point": "finger_center", "position_only": True, "steps": 20, "tolerance": 0.08},
        ],
    }


def _target_center(target: SceneObject) -> tuple[float, float, float]:
    aabb = target.get("world_aabb")
    if aabb is not None:
        return (
            0.5 * (float(aabb["min"][0]) + float(aabb["max"][0])),
            0.5 * (float(aabb["min"][1]) + float(aabb["max"][1])),
            0.5 * (float(aabb["min"][2]) + float(aabb["max"][2])),
        )
    position = target.get("position")
    if position is not None:
        return float(position[0]), float(position[1]), float(position[2])
    raise HTTPException(status_code=400, detail=f"Pick apple requires world position for {target['body']}")
