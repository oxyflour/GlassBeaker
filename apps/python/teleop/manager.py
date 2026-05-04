from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from teleop.arm_config import get_arm_config
from teleop.device import SpaceMouseDevice, SpaceMouseSample
from teleop.ik_controller import IKController
from teleop.ros_client import RosBridgeClient
from utils.rl_bundle import DEFAULT_SCENE_USD

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROBOT_USD = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"


class SpaceMouseManager:
    def __init__(self, device=None, ros_client=None, ik_controller=None) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.device = device
        self.ros_client = ros_client
        self._injected_ik_controller = ik_controller
        self.ik_controller = ik_controller
        self._running = False
        self._active_arm = "right"
        self._mode = "off"
        self._config = self._default_config()
        self._target_poses: dict[str, dict[str, tuple[float, ...]]] = {}
        self._gripper_openings = {"left": 0.0, "right": 0.0}
        self._pending_reset = {"right"}
        self._device_connected = False
        self._ros_connected = False
        self._last_joint_state_at: float | None = None

    def start(self, mode_override: str | None = None, **config: Any) -> dict[str, Any]:
        with self._lock:
            if self._running:
                self._stop_locked()
            self._config = {**self._default_config(), **config}
            self.device = self.device or SpaceMouseDevice()
            self.ros_client = self.ros_client or RosBridgeClient()
            self.ik_controller = self._injected_ik_controller
            self._target_poses.clear()
            self._gripper_openings = {"left": 0.0, "right": 0.0}
            self._pending_reset = {"left", "right", self._active_arm}
            self._mode = mode_override or self._active_arm
            self._stop.clear()
            self._running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            return self._status_unlocked()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._stop_locked()
            return self._status_unlocked()

    def shutdown(self) -> None:
        self.stop()

    def set_active_arm(self, arm: str) -> dict[str, Any]:
        if arm not in {"left", "right"}:
            raise ValueError(f"Unsupported arm: {arm}")
        with self._lock:
            self._active_arm = arm
            self._mode = arm
            self._pending_reset.add(arm)
            return self._status_unlocked()

    def set_mode(self, mode: str) -> dict[str, Any]:
        if mode not in {"off", "left", "right"}:
            raise ValueError(f"Unsupported mode: {mode}")
        with self._lock:
            self._mode = mode
            if mode in {"left", "right"}:
                self._active_arm = mode
                self._pending_reset.add(mode)
            return self._status_unlocked()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_unlocked()

    def step_once(self) -> None:
        if self.device is None or self.ros_client is None:
            return
        self.ros_client.poll_messages()
        self._ros_connected = bool(self.ros_client.status().get("connected"))
        sample = self.device.poll()
        self._device_connected = bool(self.device.status().get("connected"))
        joint_state = self.ros_client.latest_joint_state()
        if joint_state is None:
            return
        self._last_joint_state_at = time.time()
        self._sync_gripper_openings(joint_state)
        if self._mode == "off":
            return
        ik_controller = self._ensure_ik_controller()
        ik_controller.sync_joint_state(joint_state)
        active_arm = self._active_arm
        target = self._ensure_target_pose(active_arm)
        if sample is not None:
            if sample.buttons == (True, True):
                self._pending_reset.add(active_arm)
                target = self._ensure_target_pose(active_arm, force=True)
            else:
                self._apply_gripper_buttons(active_arm, sample)
                target = self._apply_sample_to_target(target, sample)
                self._target_poses[active_arm] = target
        if not self._device_connected or not self._ros_connected:
            return
        command = ik_controller.solve_step(active_arm, target, self._gripper_openings[active_arm])
        self.ros_client.publish_joint_command(command)

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            self.step_once()
            time.sleep(max(1.0 / float(self._config["rate_hz"]), 0.001))

    def _default_config(self) -> dict[str, Any]:
        return {
            "robot_usd": str(DEFAULT_ROBOT_USD),
            "scene_usd": str(DEFAULT_SCENE_USD),
            "rate_hz": 60.0,
            "linear_scale": 0.5,
            "angular_scale": 0.8,
            "gripper_step": 0.005,
        }

    def _status_unlocked(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "mode": self._mode,
            "active_arm": self._active_arm,
            "device_connected": self._device_connected,
            "ros_connected": self._ros_connected,
            "last_joint_state_at": self._last_joint_state_at,
            "config": dict(self._config),
        }

    def _stop_locked(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self._running = False
        for dependency in (self.device, self.ros_client):
            if dependency is not None and hasattr(dependency, "close"):
                dependency.close()

    def _ensure_ik_controller(self) -> IKController:
        if self.ik_controller is None:
            self.ik_controller = IKController(
                Path(self._config["robot_usd"]),
                Path(self._config["scene_usd"]),
            )
        return self.ik_controller

    def _ensure_target_pose(self, arm: str, force: bool = False) -> dict[str, tuple[float, ...]]:
        ik_controller = self._ensure_ik_controller()
        if force or arm in self._pending_reset or arm not in self._target_poses:
            self._target_poses[arm] = ik_controller.get_end_effector_pose(arm)
            self._pending_reset.discard(arm)
        return self._target_poses[arm]

    def _sync_gripper_openings(self, joint_state: dict) -> None:
        values = dict(zip(joint_state.get("name") or [], joint_state.get("position") or []))
        for arm in ("left", "right"):
            joint1, joint2 = get_arm_config(arm).gripper_joint_names
            if joint1 in values:
                self._gripper_openings[arm] = float(max(values[joint1], -values.get(joint2, 0.0), 0.0))

    def _apply_gripper_buttons(self, arm: str, sample: SpaceMouseSample) -> None:
        left, right = sample.buttons
        if left:
            self._gripper_openings[arm] += float(self._config["gripper_step"])
        if right:
            self._gripper_openings[arm] -= float(self._config["gripper_step"])
        self._gripper_openings[arm] = float(np.clip(self._gripper_openings[arm], 0.0, 0.05))

    def _apply_sample_to_target(self, target: dict, sample: SpaceMouseSample) -> dict[str, tuple[float, ...]]:
        dt = 1.0 / float(self._config["rate_hz"])
        pos = np.asarray(target["position"], dtype=float) + np.asarray(sample.translation, dtype=float) * float(self._config["linear_scale"]) * dt
        rot = np.asarray(target["rotation"], dtype=float)
        for axis, value in zip(np.eye(3), sample.rotation):
            if value != 0.0:
                rot = self._quat_mul(rot, self._axis_angle_quat(axis, value * float(self._config["angular_scale"]) * dt))
        return {"position": tuple(float(v) for v in pos), "rotation": tuple(float(v) for v in self._quat_normalize(rot))}

    def _axis_angle_quat(self, axis: np.ndarray, angle: float) -> np.ndarray:
        half = angle * 0.5
        return np.array([np.cos(half), *(axis * np.sin(half))], dtype=float)

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

    def _quat_normalize(self, quat: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(quat)
        return quat if norm == 0.0 else quat / norm
