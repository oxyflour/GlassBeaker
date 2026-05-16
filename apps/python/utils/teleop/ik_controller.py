from __future__ import annotations

import math
from pathlib import Path

import mujoco  # type: ignore
import numpy as np

from .arm_config import get_arm_config
from .grasp_frame import apply_grasp_frame, load_grasp_frames, matrix_to_pose
from utils.zapdos.bundle import ensure_render_bundle
from utils.zapdos.physics.mujoco_tools import body_world_pose

TORSO_JOINT_NAMES = ("torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4")


class IKController:
    def __init__(self, robot_usd: Path, scene_usd: Path) -> None:
        self.bundle = ensure_render_bundle(robot_usd.resolve(), scene_usd.resolve())
        self.model = mujoco.MjModel.from_xml_path(str(self.bundle.mjcf))  # type: ignore
        self.data = mujoco.MjData(self.model)  # type: ignore
        self._joint_ids = {}
        self._body_ids = {}
        self._joint_limits = {}
        self._torso_joint_names = []
        self._grasp_frames = load_grasp_frames(robot_usd.resolve())
        for name in TORSO_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)  # type: ignore
            if joint_id >= 0:
                lower, upper = self.model.jnt_range[joint_id]
                self._joint_limits[name] = (float(lower), float(upper))
                self._torso_joint_names.append(name)
        for arm in ("left", "right"):
            config = get_arm_config(arm)
            self._joint_ids[arm] = tuple(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)  # type: ignore
                for name in config.joint_names
            )
            self._body_ids[arm] = mujoco.mj_name2id(  # type: ignore
                self.model,
                mujoco.mjtObj.mjOBJ_BODY, # type: ignore
                config.end_effector_body,
            )
            for name in (*config.joint_names, *config.gripper_joint_names):
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)  # type: ignore
                lower, upper = self.model.jnt_range[joint_id]
                self._joint_limits[name] = (float(lower), float(upper))
        mujoco.mj_forward(self.model, self.data)  # type: ignore

    def arm_joint_names(self, arm: str) -> tuple[str, ...]:
        return get_arm_config(arm).joint_names

    def joint_limits(self, joint_name: str) -> tuple[float, float]:
        return self._joint_limits[joint_name]

    def sync_joint_state(self, joint_state: dict) -> None:
        for name, position in zip(joint_state.get("name") or [], joint_state.get("position") or []):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)  # type: ignore
            if joint_id < 0:
                continue
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            self.data.qpos[qpos_adr] = float(position)
        mujoco.mj_forward(self.model, self.data)  # type: ignore

    def get_end_effector_pose(self, arm: str) -> dict[str, tuple[float, ...]]:
        position, rotation = matrix_to_pose(self._grasp_world_pose(arm))
        return {
            "position": position,
            "rotation": rotation,
        }

    def solve_step(
        self,
        arm: str,
        target_pose: dict,
        gripper_opening: float,
        *,
        include_torso: bool = False,
        position_only: bool = False,
    ) -> dict[str, list[float]]:
        config = get_arm_config(arm)
        joint_names = config.joint_names
        if include_torso:
            joint_names = (*self._torso_joint_names, *joint_names)
        joint_ids = tuple(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)  # type: ignore
            for name in joint_names
        )
        current = self.get_end_effector_pose(arm)
        pos_err = np.asarray(target_pose["position"], dtype=float) - np.asarray(current["position"], dtype=float)
        rot_err = self._rotation_error(current["rotation"], target_pose["rotation"])
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jac(self.model, self.data, jacp, jacr, np.asarray(current["position"], dtype=float), self._body_ids[arm])  # type: ignore
        dof_ids = [int(self.model.jnt_dofadr[joint_id]) for joint_id in joint_ids]
        J = jacp[:, dof_ids] if position_only else np.vstack([jacp[:, dof_ids], jacr[:, dof_ids]])
        err = pos_err if position_only else np.concatenate([pos_err, rot_err])
        damping = 1e-3
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(len(err)), err)
        dq = np.clip(dq, -0.05, 0.05)
        positions = []
        for joint_name, joint_id, delta in zip(joint_names, joint_ids, dq):
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            lower, upper = self._joint_limits[joint_name]
            next_pos = float(np.clip(self.data.qpos[qpos_adr] + delta, lower, upper))
            positions.append(next_pos)
        grip = float(np.clip(gripper_opening, 0.0, self._joint_limits[config.gripper_joint_names[0]][1]))
        return {
            "name": [*joint_names, *config.gripper_joint_names], # type: ignore
            "position": [*positions, grip, -grip],
        }

    def _grasp_world_pose(self, arm: str) -> np.ndarray:
        return apply_grasp_frame(body_world_pose(self.data, self._body_ids[arm]), self._grasp_frames[arm])

    def _rotation_error(self, current: tuple[float, ...], target: tuple[float, ...]) -> np.ndarray:
        q_current = np.asarray(current, dtype=float)
        q_target = np.asarray(target, dtype=float)
        q_err = self._quat_mul(q_target, self._quat_conj(q_current))
        if q_err[0] < 0:
            q_err = -q_err
        norm = np.linalg.norm(q_err[1:])
        if norm < 1e-9:
            return np.zeros(3, dtype=float)
        angle = 2.0 * math.atan2(norm, max(q_err[0], 1e-9))
        return q_err[1:] / norm * angle

    def _quat_conj(self, quat: np.ndarray) -> np.ndarray:
        return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=float)

    def _quat_mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dtype=float,
        )

