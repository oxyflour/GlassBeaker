from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco  # type: ignore
import numpy as np

from utils.user_config import read_user_config
from utils.zapdos.robot_model import get_robot_model_key_from_usd

ARMS = ("left", "right")


@dataclass(frozen=True)
class GraspFrame:
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


def load_grasp_frames(robot_usd: Path | None) -> dict[str, GraspFrame]:
    frames = {arm: GraspFrame() for arm in ARMS}
    robot_key = get_robot_model_key_from_usd(robot_usd)
    if robot_key is None:
        return frames
    override = read_user_config().get("override")
    if override is None:
        return frames
    if not isinstance(override, dict):
        raise RuntimeError("override must be a JSON object.")
    ik = override.get("ik")
    if ik is None:
        return frames
    if not isinstance(ik, dict):
        raise RuntimeError("override.ik must be a JSON object.")
    robot_ik = ik.get(robot_key)
    if robot_ik is None:
        return frames
    if not isinstance(robot_ik, dict):
        raise RuntimeError(f"override.ik.{robot_key} must be a JSON object.")
    for arm in ARMS:
        frames[arm] = _read_grasp_frame(robot_ik, arm, robot_key)
    return frames


def pose_matrix(pos: tuple[float, ...], quat_wxyz: tuple[float, ...]) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=float)
    matrix = np.eye(4)
    rotation = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(rotation, quat)  # type: ignore
    matrix[:3, :3] = rotation.reshape(3, 3)
    matrix[:3, 3] = np.asarray(pos, dtype=float)
    return matrix


def matrix_to_pose(matrix: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    quat = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix[:3, :3], dtype=float).reshape(-1))  # type: ignore
    if quat[0] < 0:
        quat = -quat
    pos = tuple(float(value) for value in matrix[:3, 3])
    return pos, tuple(float(value) for value in quat)


def apply_grasp_frame(body_matrix: np.ndarray, frame: GraspFrame) -> np.ndarray:
    return body_matrix @ pose_matrix(frame.pos, frame.quat_wxyz)


def _read_grasp_frame(robot_ik: dict[str, object], arm: str, robot_key: str) -> GraspFrame:
    gripper = robot_ik.get(f"{arm}_gripper")
    path = f"override.ik.{robot_key}.{arm}_gripper"
    if gripper is None:
        return GraspFrame()
    if not isinstance(gripper, dict):
        raise RuntimeError(f"{path} must be a JSON object.")
    grasp_frame = gripper.get("grasp_frame")
    if grasp_frame is None:
        return GraspFrame()
    if not isinstance(grasp_frame, dict):
        raise RuntimeError(f"{path}.grasp_frame must be a JSON object.")
    pos = _vector(grasp_frame.get("pos", [0.0, 0.0, 0.0]), 3, f"{path}.grasp_frame.pos")
    quat = _vector(grasp_frame.get("quat", [1.0, 0.0, 0.0, 0.0]), 4, f"{path}.grasp_frame.quat")
    norm = float(np.linalg.norm(np.asarray(quat, dtype=float)))
    if norm <= 1e-12:
        raise RuntimeError(f"{path}.grasp_frame.quat must be non-zero.")
    return GraspFrame(
        pos=pos,
        quat_wxyz=tuple(float(value) / norm for value in quat),
    )


def _vector(value: object, size: int, path: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise RuntimeError(f"{path} must be a JSON array of length {size}.")
    items = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (float, int)):
            raise RuntimeError(f"{path} must contain only numeric values.")
        items.append(float(item))
    return tuple(items)


__all__ = ["ARMS", "GraspFrame", "apply_grasp_frame", "load_grasp_frames", "matrix_to_pose", "pose_matrix"]
