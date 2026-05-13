from __future__ import annotations

from dataclasses import dataclass

import mujoco  # type: ignore
import numpy as np


@dataclass(frozen=True)
class BodyAttachment:
    parent_body: str
    child_body: str
    relative_pose: np.ndarray

    def to_payload(self) -> dict[str, object]:
        pos, quat = matrix_to_pose(self.relative_pose)
        return {
            "parent_body": self.parent_body,
            "child_body": self.child_body,
            "relative_position": pos.tolist(),
            "relative_quat": quat.tolist(),
        }


def create_attachment(
    parent_body: str,
    child_body: str,
    parent_pose: np.ndarray,
    child_pose: np.ndarray,
) -> BodyAttachment:
    return BodyAttachment(parent_body, child_body, np.linalg.inv(parent_pose) @ child_pose)


def attachment_world_pose(attachment: BodyAttachment, parent_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return matrix_to_pose(parent_pose @ attachment.relative_pose)


def matrix_to_pose(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quat = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix[:3, :3], dtype=float).reshape(-1))  # type: ignore
    if quat[0] < 0:
        quat = -quat
    return np.asarray(matrix[:3, 3], dtype=float), quat
