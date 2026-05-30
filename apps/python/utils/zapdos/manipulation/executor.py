from __future__ import annotations

from collections.abc import Generator, Iterator
import math
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException

from utils.teleop.arm_config import get_arm_config
from utils.teleop.ik_controller import IKController

PICK_POSE_TOLERANCE = 0.08
TARGET_ATTACH_TOLERANCE = 0.10
LIFT_POSE_TOLERANCE = 0.08
DRIVE_POSE_TOLERANCE = 0.01
DRIVE_ROTATION_TOLERANCE = 0.05
DRIVE_SETTLE_STEPS = 72
DRIVE_STAGNATION_STEPS = 24
DRIVE_MIN_PROGRESS = 0.01
DRIVE_DIVERGENCE_MARGIN = 0.001
SUPPORT_ESCAPE_MARGIN = 0.06
GRIPPER_STAGE_STEPS = 100
GRIPPER_SETTLE_STEPS = 300
GRIPPER_WIDTH_TOLERANCE = 0.002
POSITION_STAGE_SEGMENT_LENGTH = 0.05
POSITION_STAGE_SEGMENT_STEPS = 32
POSITION_STAGE_SUBGOAL_TOLERANCE = 0.02


class PickExecutor:
    def __init__(self, physics, bundle, ik_controller: IKController | None = None) -> None:
        self.physics = physics
        self.bundle = bundle
        self.ik_controller = ik_controller

    def current_pose(self, arm: str, *, target_point: str = "end_effector") -> dict[str, list[float]]:
        ik = self._ensure_ik()
        ik.sync_joint_state(self.physics.joint_state_msg())
        if target_point == "finger_center":
            pose = ik.get_gripper_finger_center_pose(arm)
        elif target_point == "end_effector":
            pose = ik.get_end_effector_pose(arm)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported target point: {target_point}")
        return {
            "position": [float(value) for value in pose["position"]],
            "quat_wxyz": [float(value) for value in pose["rotation"]],
        }

    def execute(self, plan: dict[str, object]) -> dict[str, object]:
        iterator = self.iter_execute(plan)
        while True:
            try:
                next(iterator)
            except StopIteration as stop:
                return stop.value

    def iter_execute(self, plan: dict[str, object]) -> Generator[None, None, dict[str, object]]:
        if str(plan.get("kind") or "pick") == "release":
            return (yield from self._iter_execute_release(plan))

        arm = str(plan.get("arm") or "left")
        ik = self._ensure_ik()
        ik.sync_joint_state(self.physics.joint_state_msg())
        target_body = str(plan["target_body"])
        open_width = float(plan.get("open_gripper", 0.04))
        current_gripper_width = open_width
        pick_tolerance = float(plan.get("pick_tolerance", PICK_POSE_TOLERANCE))
        attach_tolerance = float(plan.get("attach_tolerance", TARGET_ATTACH_TOLERANCE))
        stages = plan["stages"] if "stages" in plan else self._legacy_stages(plan)
        pick_pose: dict[str, tuple[float, ...]] | None = None
        grasp_contact_required = False
        for raw_stage in stages:
            stage = raw_stage if isinstance(raw_stage, dict) else {}
            stage_name = str(stage["name"])
            stage_kind = str(stage["kind"])
            if stage_kind == "move_pose":
                target = self._pose(stage["pose"])
                target_point = str(stage.get("target_point") or "end_effector")
                if target_point == "finger_center":
                    target["target_point"] = target_point
                elif target_point != "end_effector":
                    raise HTTPException(status_code=409, detail=f"Pick failed: unsupported target point {target_point} at {stage_name}")
                pick_pose = target if stage_name == "descend_to_pick" else pick_pose
                include_torso = bool(stage.get("include_torso", False))
                position_only = bool(stage.get("position_only", False))
                yield from self._yield_drive_pose(
                    ik,
                    arm,
                    target,
                    current_gripper_width,
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
                if grasp_contact_required:
                    self._require_bilateral_gripper_contact(arm, target_body, stage_name)
                continue
            if stage_kind != "gripper":
                raise HTTPException(status_code=409, detail=f"Pick failed: unsupported stage kind {stage_kind} at {stage_name}")
            stage_width = float(stage.get("width", 0.0))
            stage_steps = max(1, int(stage.get("steps", GRIPPER_STAGE_STEPS)))
            if stage_name != "close_gripper":
                hold_pose = ik.get_end_effector_pose(arm)
                yield from self._yield_drive_pose(
                    ik,
                    arm,
                    hold_pose,
                    stage_width,
                    steps=stage_steps,
                    start_gripper=current_gripper_width,
                )
                current_gripper_width = stage_width
                continue
            if pick_pose is None:
                raise HTTPException(status_code=409, detail=f"Pick failed: missing descend_to_pick stage before {stage_name}")
            hold_pose = self._current_pose_for_target(ik, arm, pick_pose)
            if pick_pose.get("target_point") == "finger_center":
                hold_pose["target_point"] = "finger_center"  # type: ignore
            gripper_body = str(plan.get("gripper_body") or get_arm_config(arm).end_effector_body)
            current_gripper_width = yield from self._yield_close_gripper(
                ik,
                arm,
                hold_pose,
                current_gripper_width,
                stage_width,
                stage_steps,
                gripper_body,
                target_body,
            )
            self._require_pose_reached(ik, arm, pick_pose, pick_tolerance, stage_name)
            self._require_target_near_gripper(ik, arm, target_body, attach_tolerance)
            self._require_bilateral_gripper_contact(arm, target_body, stage_name)
            grasp_contact_required = True
        return {"ok": True, "arm": arm, "target_body": target_body, "attachment": self.physics.get_attachment(target_body)}

    def _iter_execute_release(self, plan: dict[str, object]) -> Generator[None, None, dict[str, object]]:
        arm = str(plan.get("arm") or "left")
        target_body = str(plan["target_body"])
        ik = self._ensure_ik()
        ik.sync_joint_state(self.physics.joint_state_msg())
        hold_pose = ik.get_end_effector_pose(arm)
        current_width = float(plan.get("closed_gripper", 0.0))
        for raw_stage in plan.get("stages", []):
            stage = raw_stage if isinstance(raw_stage, dict) else {}
            if str(stage.get("kind")) != "gripper":
                raise HTTPException(status_code=409, detail=f"Release failed: unsupported stage kind {stage.get('kind')}")
            stage_width = float(stage.get("width", 0.0))
            yield from self._yield_drive_pose(
                ik,
                arm,
                hold_pose,
                stage_width,
                steps=max(1, int(stage.get("steps", GRIPPER_STAGE_STEPS))),
                start_gripper=current_width,
            )
            current_width = stage_width
        if self.physics.get_attachment(target_body) is not None:
            self.physics.detach_body(target_body)
        return {"ok": True, "arm": arm, "target_body": target_body, "attachment": self.physics.get_attachment(target_body)}

    def _ensure_ik(self) -> IKController:
        if self.ik_controller is None:
            self.ik_controller = IKController(Path(self.bundle.robot_usd), Path(self.bundle.scene_usd))
        return self.ik_controller

    def _yield_close_gripper(
        self,
        ik: IKController,
        arm: str,
        hold_pose: dict[str, tuple[float, ...]],
        start_width: float,
        target_width: float,
        steps: int,
        gripper_body: str,
        target_body: str,
    ) -> Generator[None, None, float]:
        for index in range(max(steps, 1)):
            alpha = float(index + 1) / float(max(steps, 1))
            command_width = self._interpolate_gripper(start_width, target_width, alpha)
            self.physics.apply_joint_command(ik.solve_step(arm, hold_pose, command_width))
            self.physics.step()
            ik.sync_joint_state(self.physics.joint_state_msg())
            yield
        if self._gripper_width_error(arm, target_width) is None:
            return target_width
        if self._bodies_in_contact(gripper_body, target_body):
            return target_width
        for _ in range(GRIPPER_SETTLE_STEPS):
            if self._bodies_in_contact(gripper_body, target_body):
                return target_width
            error = self._gripper_width_error(arm, target_width)
            if error is None or error <= GRIPPER_WIDTH_TOLERANCE:
                return target_width
            self.physics.apply_joint_command(ik.solve_step(arm, hold_pose, target_width))
            self.physics.step()
            ik.sync_joint_state(self.physics.joint_state_msg())
            yield
        error = self._gripper_width_error(arm, target_width)
        if error is not None and error > GRIPPER_WIDTH_TOLERANCE:
            raise HTTPException(status_code=409, detail=f"Pick failed: close_gripper width error {error:.3f} exceeds tolerance")
        return target_width

    def _yield_drive_pose(
        self,
        ik: IKController,
        arm: str,
        target: dict[str, tuple[float, ...]],
        gripper: float,
        steps: int = 12,
        *,
        include_torso: bool = False,
        position_only: bool = False,
        start_gripper: float | None = None,
    ) -> Generator[None, None, None]:
        result = self._drive_pose(
            ik,
            arm,
            target,
            gripper,
            steps=steps,
            include_torso=include_torso,
            position_only=position_only,
            start_gripper=start_gripper,
        )
        if isinstance(result, Iterator):
            yield from result

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
        start_gripper: float | None = None,
    ) -> Generator[None, None, None]:
        return self._drive_pose_iter(
            ik,
            arm,
            target,
            gripper,
            steps=steps,
            include_torso=include_torso,
            position_only=position_only,
            start_gripper=start_gripper,
        )

    def _drive_pose_iter(
        self,
        ik: IKController,
        arm: str,
        target: dict[str, tuple[float, ...]],
        gripper: float,
        steps: int = 12,
        *,
        include_torso: bool = False,
        position_only: bool = False,
        start_gripper: float | None = None,
    ) -> Generator[None, None, None]:
        start = self._current_pose_for_target(ik, arm, target)
        requested_steps = max(steps, 1)
        if position_only:
            dx = abs(float(target["position"][0]) - float(start["position"][0]))
            dy = abs(float(target["position"][1]) - float(start["position"][1]))
            position_error = self._distance(start["position"], target["position"])
            if dx <= POSITION_STAGE_SUBGOAL_TOLERANCE and dy <= POSITION_STAGE_SUBGOAL_TOLERANCE:
                segments = requested_steps
            else:
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
                if target.get("target_point") == "finger_center":
                    subgoal["target_point"] = "finger_center"  # type: ignore
                segment_start = self._current_pose_for_target(ik, arm, subgoal)
                segment_initial_error = self._distance(segment_start["position"], subgoal["position"])
                segment_required_progress = min(DRIVE_MIN_PROGRESS, segment_initial_error * 0.25)
                segment_best_error = segment_initial_error
                segment_steps = 0
                for _ in range(POSITION_STAGE_SEGMENT_STEPS):
                    current = self._current_pose_for_target(ik, arm, subgoal)
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
                    yield
                    current = self._current_pose_for_target(ik, arm, subgoal)
                    current_error = self._distance(current["position"], subgoal["position"])
                    segment_best_error = min(segment_best_error, current_error)
                    segment_steps += 1
                    if (
                        segment_steps >= DRIVE_STAGNATION_STEPS
                        and segment_best_error >= segment_initial_error - segment_required_progress
                    ):
                        break
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
            if target.get("target_point") == "finger_center":
                pose["target_point"] = "finger_center"  # type: ignore
            self.physics.apply_joint_command(ik.solve_step(
                arm,
                pose,
                self._interpolate_gripper(start_gripper, gripper, alpha),
                include_torso=include_torso,
                position_only=position_only,
            ))
            self.physics.step()
            ik.sync_joint_state(self.physics.joint_state_msg())
            yield
            current = self._current_pose_for_target(ik, arm, target)
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
            current = self._current_pose_for_target(ik, arm, target)
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
            yield
            current = self._current_pose_for_target(ik, arm, target)
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

    def _current_pose_for_target(self, ik: IKController, arm: str, target: dict[str, object]) -> dict[str, tuple[float, ...]]:
        if target.get("target_point") == "finger_center":
            finger_center_pose = getattr(ik, "get_gripper_finger_center_pose", None)
            if callable(finger_center_pose):
                return finger_center_pose(arm)
        return ik.get_end_effector_pose(arm)

    def _normalize(self, quat: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(quat)
        return quat if norm <= 1e-9 else quat / norm

    def _interpolate_gripper(self, start: float | None, target: float, alpha: float) -> float:
        if start is None:
            return target
        return float((1.0 - alpha) * start + alpha * target)

    def _gripper_width_error(self, arm: str, target: float) -> float | None:
        current = self._current_gripper_width(arm)
        return None if current is None else abs(current - target)

    def _current_gripper_width(self, arm: str) -> float | None:
        joint_state = self.physics.joint_state_msg()
        values = dict(zip(joint_state.get("name") or [], joint_state.get("position") or []))
        joint1, joint2 = get_arm_config(arm).gripper_joint_names
        openings: list[float] = []
        if joint1 in values:
            openings.append(float(values[joint1]))
        if joint2 in values:
            openings.append(-float(values[joint2]))
        return max(openings, default=0.0) if openings else None

    def _bodies_in_contact(self, body_a: str, body_b: str) -> bool:
        bodies_in_contact = getattr(self.physics, "bodies_in_contact", None)
        return bool(callable(bodies_in_contact) and bodies_in_contact(body_a, body_b))

    def _require_bilateral_gripper_contact(self, arm: str, target_body: str, stage: str) -> None:
        bodies_in_contact = getattr(self.physics, "bodies_in_contact", None)
        if not callable(bodies_in_contact):
            return
        missing = [
            finger_body
            for finger_body in get_arm_config(arm).gripper_finger_body_names
            if not bool(bodies_in_contact(finger_body, target_body))
        ]
        if missing:
            raise HTTPException(status_code=409, detail=f"Pick failed: {stage} target is not held by both fingers")

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
                    pre_pick = self._pose(plan["pre_pick"])
                    stages.extend([
                        {"name": "escape_xy", "kind": "move_pose", "pose": {"position": [escape_x, escape_y, current[2]], "quat_wxyz": self.current_pose(str(plan.get("arm") or "left"))["quat_wxyz"]}},
                        {"name": "raise_to_transit", "kind": "move_pose", "pose": {"position": [escape_x, escape_y, pre_pick["position"][2]], "quat_wxyz": list(pre_pick["rotation"])}},
                    ])
        stages.extend([
            {"name": "pre_pick", "kind": "move_pose", "pose": plan["pre_pick"]},
            {"name": "descend_to_pick", "kind": "move_pose", "pose": plan["pick"]},
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
        current = self._current_pose_for_target(ik, arm, target)
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
        finger_center_pose = getattr(ik, "get_gripper_finger_center_pose", None)
        gripper = finger_center_pose(arm) if callable(finger_center_pose) else ik.get_end_effector_pose(arm)
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
