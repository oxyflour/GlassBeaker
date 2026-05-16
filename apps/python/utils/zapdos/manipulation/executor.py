from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException

from utils.teleop.arm_config import get_arm_config
from utils.teleop.ik_controller import IKController

GRASP_POSE_TOLERANCE = 0.08
TARGET_ATTACH_TOLERANCE = 0.10
LIFT_POSE_TOLERANCE = 0.08
DRIVE_POSE_TOLERANCE = 0.01
DRIVE_ROTATION_TOLERANCE = 0.05
DRIVE_SETTLE_STEPS = 72
DRIVE_STAGNATION_STEPS = 24
DRIVE_MIN_PROGRESS = 0.01
DRIVE_DIVERGENCE_MARGIN = 0.001
SUPPORT_ESCAPE_MARGIN = 0.06
POSITION_STAGE_SEGMENT_LENGTH = 0.05
POSITION_STAGE_SEGMENT_STEPS = 32
POSITION_STAGE_SUBGOAL_TOLERANCE = 0.02
POSITION_STAGE_STEP_SCALE = 0.0005


class PickExecutor:
    def __init__(self, physics, bundle, ik_controller: IKController | None = None) -> None:
        self.physics = physics
        self.bundle = bundle
        self.ik_controller = ik_controller

    def current_pose(self, arm: str) -> dict[str, list[float]]:
        ik = self._ensure_ik()
        ik.sync_joint_state(self.physics.joint_state_msg())
        pose = ik.get_end_effector_pose(arm)
        return {
            "position": [float(value) for value in pose["position"]],
            "quat_wxyz": [float(value) for value in pose["rotation"]],
        }

    def execute(self, plan: dict[str, object]) -> dict[str, object]:
        if str(plan.get("kind") or "pick") == "release":
            return self._execute_release(plan)

        arm = str(plan.get("arm") or "left")
        ik = self._ensure_ik()
        ik.sync_joint_state(self.physics.joint_state_msg())
        target_body = str(plan["target_body"])
        open_width = float(plan.get("open_gripper", 0.04))
        closed_width = 0.0
        grasp_tolerance = float(plan.get("grasp_tolerance", GRASP_POSE_TOLERANCE))
        attach_tolerance = float(plan.get("attach_tolerance", TARGET_ATTACH_TOLERANCE))
        attached = False
        stages = plan["stages"] if "stages" in plan else self._legacy_stages(plan)
        grasp_pose: dict[str, tuple[float, ...]] | None = None
        try:
            for raw_stage in stages:
                stage = raw_stage if isinstance(raw_stage, dict) else {}
                stage_name = str(stage["name"])
                stage_kind = str(stage["kind"])
                if stage_kind == "move_pose":
                    target = self._pose(stage["pose"])
                    grasp_pose = target if stage_name == "descend_to_grasp" else grasp_pose
                    include_torso = bool(stage.get("include_torso", False))
                    position_only = bool(stage.get("position_only", False))
                    self._drive_pose(
                        ik,
                        arm,
                        target,
                        closed_width if attached else open_width,
                        steps=max(1, int(stage.get("steps", 12))),
                        include_torso=include_torso,
                        position_only=position_only,
                    )
                    self._require_pose_reached(
                        ik,
                        arm,
                        target,
                        float(stage.get("tolerance", DRIVE_POSE_TOLERANCE)),
                        stage_name,
                    )
                    continue
                if stage_kind != "gripper":
                    raise HTTPException(status_code=409, detail=f"Pick failed: unsupported stage kind {stage_kind} at {stage_name}")
                stage_width = float(stage.get("width", 0.0))
                stage_steps = max(1, int(stage.get("steps", 6)))
                if stage_name != "close_gripper":
                    hold_pose = ik.get_end_effector_pose(arm)
                    self._drive_pose(ik, arm, hold_pose, stage_width, steps=stage_steps)
                    if attached:
                        closed_width = stage_width
                    else:
                        open_width = stage_width
                    continue
                closed_width = stage_width
                if grasp_pose is None:
                    raise HTTPException(status_code=409, detail=f"Pick failed: missing descend_to_grasp stage before {stage_name}")
                self._drive_pose(ik, arm, grasp_pose, closed_width, steps=stage_steps)
                self._require_pose_reached(ik, arm, grasp_pose, grasp_tolerance, stage_name)
                self._require_target_near_gripper(ik, arm, target_body, attach_tolerance)
                self.physics.attach_body(str(plan.get("gripper_body") or get_arm_config(arm).end_effector_body), target_body)
                attached = True
        except Exception:
            if attached:
                self.physics.detach_body(target_body)
            raise
        return {"ok": True, "arm": arm, "target_body": target_body, "attachment": self.physics.get_attachment(target_body)}

    def _execute_release(self, plan: dict[str, object]) -> dict[str, object]:
        arm = str(plan.get("arm") or "left")
        target_body = str(plan["target_body"])
        attachment = self.physics.get_attachment(target_body)
        if attachment is None:
            raise HTTPException(status_code=409, detail=f"Release failed: {target_body} is not attached")

        ik = self._ensure_ik()
        ik.sync_joint_state(self.physics.joint_state_msg())
        hold_pose = ik.get_end_effector_pose(arm)
        for raw_stage in plan.get("stages", []):
            stage = raw_stage if isinstance(raw_stage, dict) else {}
            if str(stage.get("kind")) != "gripper":
                raise HTTPException(status_code=409, detail=f"Release failed: unsupported stage kind {stage.get('kind')}")
            self._drive_pose(
                ik,
                arm,
                hold_pose,
                float(stage.get("width", 0.0)),
                steps=max(1, int(stage.get("steps", 6))),
            )
        self.physics.detach_body(target_body)
        return {"ok": True, "arm": arm, "target_body": target_body, "attachment": self.physics.get_attachment(target_body)}

    def _ensure_ik(self) -> IKController:
        if self.ik_controller is None:
            self.ik_controller = IKController(Path(self.bundle.robot_usd), Path(self.bundle.scene_usd))
        return self.ik_controller

    def _drive_pose(
        self,
        ik: IKController,
        arm: str,
        target: dict[str, tuple[float, ...]],
        gripper: float,
        steps: int = 12,
        *,
        include_torso: bool = False,
        position_only: bool = False,
    ) -> None:
        start = ik.get_end_effector_pose(arm)
        requested_steps = max(steps, 1)
        if position_only:
            dx = abs(float(target["position"][0]) - float(start["position"][0]))
            dy = abs(float(target["position"][1]) - float(start["position"][1]))
            position_error = self._distance(start["position"], target["position"])
            if dx <= POSITION_STAGE_SUBGOAL_TOLERANCE and dy <= POSITION_STAGE_SUBGOAL_TOLERANCE:
                max_steps = max(DRIVE_SETTLE_STEPS, requested_steps, int(math.ceil(position_error / POSITION_STAGE_STEP_SCALE)))
                for _ in range(max_steps):
                    current = ik.get_end_effector_pose(arm)
                    if self._distance(current["position"], target["position"]) <= DRIVE_POSE_TOLERANCE:
                        return
                    self.physics.apply_joint_command(ik.solve_step(
                        arm,
                        target,
                        gripper,
                        include_torso=include_torso,
                        position_only=True,
                    ))
                    self.physics.step()
                    ik.sync_joint_state(self.physics.joint_state_msg())
                return
            segments = max(requested_steps, int(math.ceil(position_error / POSITION_STAGE_SEGMENT_LENGTH)))
            subgoal_tolerance = min(
                POSITION_STAGE_SUBGOAL_TOLERANCE,
                max(position_error / float(segments), 1e-6) * 0.5,
            )
            for index in range(segments):
                alpha = float(index + 1) / float(segments)
                subgoal = {
                    "position": tuple((1.0 - alpha) * a + alpha * b for a, b in zip(start["position"], target["position"])),
                    "rotation": target["rotation"],
                }
                for _ in range(POSITION_STAGE_SEGMENT_STEPS):
                    current = ik.get_end_effector_pose(arm)
                    if self._distance(current["position"], subgoal["position"]) <= subgoal_tolerance:
                        break
                    self.physics.apply_joint_command(ik.solve_step(
                        arm,
                        subgoal,
                        gripper,
                        include_torso=include_torso,
                        position_only=True,
                    ))
                    self.physics.step()
                    ik.sync_joint_state(self.physics.joint_state_msg())
            return
        initial_error = self._distance(start["position"], target["position"])
        required_progress = min(DRIVE_MIN_PROGRESS, initial_error * 0.25)
        best_error = initial_error
        total_steps = 0
        for index in range(requested_steps):
            alpha = float(index + 1) / float(requested_steps)
            pose = {
                "position": tuple((1.0 - alpha) * a + alpha * b for a, b in zip(start["position"], target["position"])),
                "rotation": tuple(self._normalize((1.0 - alpha) * np.asarray(start["rotation"]) + alpha * np.asarray(target["rotation"]))),
            }
            self.physics.apply_joint_command(ik.solve_step(
                arm,
                pose,
                gripper,
                include_torso=include_torso,
                position_only=position_only,
            ))
            self.physics.step()
            ik.sync_joint_state(self.physics.joint_state_msg())
            current = ik.get_end_effector_pose(arm)
            current_error = self._distance(current["position"], target["position"])
            best_error = min(best_error, current_error)
            total_steps += 1
            if (
                total_steps >= DRIVE_STAGNATION_STEPS
                and best_error >= initial_error - required_progress
                and current_error >= initial_error + DRIVE_DIVERGENCE_MARGIN
            ):
                return
        for _ in range(DRIVE_SETTLE_STEPS):
            current = ik.get_end_effector_pose(arm)
            if self._distance(current["position"], target["position"]) <= DRIVE_POSE_TOLERANCE:
                if position_only or self._rotation_error(current["rotation"], target["rotation"]) <= DRIVE_ROTATION_TOLERANCE:
                    return
            self.physics.apply_joint_command(ik.solve_step(
                arm,
                target,
                gripper,
                include_torso=include_torso,
                position_only=position_only,
            ))
            self.physics.step()
            ik.sync_joint_state(self.physics.joint_state_msg())
            current = ik.get_end_effector_pose(arm)
            current_error = self._distance(current["position"], target["position"])
            best_error = min(best_error, current_error)
            total_steps += 1
            if (
                total_steps >= DRIVE_STAGNATION_STEPS
                and best_error >= initial_error - required_progress
                and current_error >= initial_error + DRIVE_DIVERGENCE_MARGIN
            ):
                return

    def _pose(self, raw: object) -> dict[str, tuple[float, ...]]:
        pose = raw if isinstance(raw, dict) else {}
        position = tuple(float(v) for v in pose.get("position", (0.0, 0.0, 0.0)))
        rotation = tuple(float(v) for v in pose.get("quat_wxyz") or pose.get("rotation", (1.0, 0.0, 0.0, 0.0)))
        return {"position": position, "rotation": rotation}

    def _normalize(self, quat: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(quat)
        return quat if norm <= 1e-9 else quat / norm

    def _close_width(self, plan: dict[str, object]) -> float:
        command = plan.get("close")
        if isinstance(command, dict) and "width" in command:
            return float(command["width"])
        return float(plan.get("closed_gripper", 0.0))

    def _legacy_stages(self, plan: dict[str, object]) -> list[dict[str, object]]:
        stages: list[dict[str, object]] = []
        support = plan.get("support_surface")
        if isinstance(support, dict):
            top_z = support.get("top_z")
            xy_min = self._xy_pair(support.get("xy_min"))
            xy_max = self._xy_pair(support.get("xy_max"))
            current = self.current_pose(str(plan.get("arm") or "left"))["position"]
            if xy_min is not None and xy_max is not None and isinstance(top_z, (float, int)):
                if current[2] < float(top_z) and xy_min[0] <= current[0] <= xy_max[0] and xy_min[1] <= current[1] <= xy_max[1]:
                    escape_x, escape_y = self._escape_xy((current[0], current[1]), xy_min, xy_max)
                    pre_grasp = self._pose(plan["pre_grasp"])
                    stages.extend([
                        {"name": "escape_xy", "kind": "move_pose", "pose": {"position": [escape_x, escape_y, current[2]], "quat_wxyz": self.current_pose(str(plan.get("arm") or "left"))["quat_wxyz"]}},
                        {"name": "raise_to_transit", "kind": "move_pose", "pose": {"position": [escape_x, escape_y, pre_grasp["position"][2]], "quat_wxyz": list(pre_grasp["rotation"])}},
                    ])
        stages.extend([
            {"name": "pre_grasp", "kind": "move_pose", "pose": plan["pre_grasp"]},
            {"name": "descend_to_grasp", "kind": "move_pose", "pose": plan["grasp"]},
            {"name": "close_gripper", "kind": "gripper", "width": self._close_width(plan)},
            {"name": "lift", "kind": "move_pose", "pose": plan["lift"], "tolerance": LIFT_POSE_TOLERANCE},
        ])
        return stages

    def _require_pose_reached(
        self,
        ik: IKController,
        arm: str,
        target: dict[str, tuple[float, ...]],
        tolerance: float,
        stage: str,
    ) -> None:
        current = ik.get_end_effector_pose(arm)
        error = self._distance(current["position"], target["position"])
        if error > tolerance:
            raise HTTPException(status_code=409, detail=f"Pick failed: {stage} pose error {error:.3f} exceeds tolerance")

    def _require_target_near_gripper(
        self,
        ik: IKController,
        arm: str,
        target_body: str,
        tolerance: float,
    ) -> None:
        gripper = ik.get_end_effector_pose(arm)
        target_position: tuple[float, ...] | None = None
        body_world_aabb = getattr(self.physics, "body_world_aabb", None)
        if callable(body_world_aabb):
            aabb = body_world_aabb(target_body)
            if aabb is not None:
                target_position = (
                    min(max(float(gripper["position"][0]), float(aabb["min"][0])), float(aabb["max"][0])),
                    min(max(float(gripper["position"][1]), float(aabb["min"][1])), float(aabb["max"][1])),
                    min(max(float(gripper["position"][2]), float(aabb["min"][2])), float(aabb["max"][2])),
                )
        if target_position is None:
            target_pose = self.physics.get_pose().get(target_body)
            if target_pose is None:
                raise HTTPException(status_code=409, detail=f"Pick failed: target pose unavailable for {target_body}")
            target_position = tuple(float(target_pose[index]) for index in (12, 13, 14))
        error = self._distance(gripper["position"], target_position)
        if error > tolerance:
            raise HTTPException(status_code=409, detail=f"Pick failed: gripper is too far from {target_body} to attach")

    def _distance(self, left: tuple[float, ...], right: tuple[float, ...]) -> float:
        return float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))

    def _rotation_error(self, left: tuple[float, ...], right: tuple[float, ...]) -> float:
        dot = float(abs(np.dot(np.asarray(left, dtype=float), np.asarray(right, dtype=float))))
        dot = float(np.clip(dot, -1.0, 1.0))
        return float(2.0 * np.arccos(dot))

    def _xy_pair(self, value: object) -> tuple[float, float] | None:
        if not isinstance(value, list) or len(value) != 2:
            return None
        return float(value[0]), float(value[1])

    def _escape_xy(
        self,
        current_xy: tuple[float, float],
        xy_min: tuple[float, float],
        xy_max: tuple[float, float],
    ) -> tuple[float, float]:
        x, y = current_xy
        candidates = (
            (xy_min[0] - SUPPORT_ESCAPE_MARGIN, y),
            (xy_max[0] + SUPPORT_ESCAPE_MARGIN, y),
            (x, xy_min[1] - SUPPORT_ESCAPE_MARGIN),
            (x, xy_max[1] + SUPPORT_ESCAPE_MARGIN),
        )
        return min(candidates, key=lambda candidate: self._distance(candidate, current_xy))
