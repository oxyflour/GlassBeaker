from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from teleop.device import SpaceMouseSample  # noqa: E402
from teleop.manager import SpaceMouseManager  # noqa: E402


class _FakeDevice:
    def __init__(self, samples: list[SpaceMouseSample], connected: bool = True) -> None:
        self.samples = list(samples)
        self.connected = connected
        self.poll_count = 0

    def poll(self) -> SpaceMouseSample | None:
        self.poll_count += 1
        if self.samples:
            return self.samples.pop(0)
        return None

    def status(self) -> dict[str, bool]:
        return {"connected": self.connected}

    def close(self) -> None:
        return None


class _FakeRosClient:
    def __init__(self, joint_state: dict | None, connected: bool = True) -> None:
        self.joint_state = joint_state
        self.connected = connected
        self.published: list[dict] = []

    def poll_messages(self) -> None:
        return None

    def latest_joint_state(self) -> dict | None:
        return self.joint_state

    def publish_joint_command(self, command: dict) -> None:
        self.published.append(command)

    def status(self) -> dict[str, bool]:
        return {"connected": self.connected}

    def close(self) -> None:
        return None


class _FakeIKController:
    def __init__(self) -> None:
        self.synced: list[dict] = []
        self.solve_calls: list[tuple[str, dict, float]] = []
        self.poses = {
            "left": {"position": (1.0, 0.0, 0.0), "rotation": (1.0, 0.0, 0.0, 0.0)},
            "right": {"position": (2.0, 0.0, 0.0), "rotation": (1.0, 0.0, 0.0, 0.0)},
        }

    def sync_joint_state(self, joint_state: dict) -> None:
        self.synced.append(joint_state)

    def get_end_effector_pose(self, arm: str) -> dict:
        return self.poses[arm]

    def solve_step(self, arm: str, target_pose: dict, gripper_opening: float) -> dict:
        self.solve_calls.append((arm, target_pose, gripper_opening))
        return {"name": [f"{arm}_joint"], "position": [gripper_opening]}


class SpaceMouseManagerTest(unittest.TestCase):
    def call_with_timeout(self, fn, *args, timeout: float = 0.5, **kwargs):
        result: dict[str, object] = {}
        error: dict[str, BaseException] = {}

        def target() -> None:
            try:
                result["value"] = fn(*args, **kwargs)
            except BaseException as exc:  # pragma: no cover - re-raised below
                error["value"] = exc

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            self.fail(f"{fn.__name__} did not return within {timeout:.1f}s")
        if "value" in error:
            raise error["value"]
        return result.get("value")

    def sample(self, *, left: bool = False, right: bool = False) -> SpaceMouseSample:
        return SpaceMouseSample(
            translation=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0),
            buttons=(left, right),
        )

    def test_start_and_stop_return_status_snapshots(self):
        manager = SpaceMouseManager(
            device=_FakeDevice([]),
            ros_client=_FakeRosClient(None),
            ik_controller=_FakeIKController(),
        )

        started = self.call_with_timeout(manager.start)
        stopped = self.call_with_timeout(manager.stop)

        self.assertEqual(started["running"], True)
        self.assertEqual(stopped["running"], False)

    def test_start_does_not_eagerly_build_ik_controller(self):
        manager = SpaceMouseManager(
            device=_FakeDevice([]),
            ros_client=_FakeRosClient(None),
        )

        class _SlowIKController:
            def __init__(self, *_args, **_kwargs) -> None:
                raise AssertionError("start() should not construct IKController")

        with mock.patch("teleop.manager.IKController", _SlowIKController):
            started = self.call_with_timeout(manager.start)

        self.assertEqual(started["running"], True)
        self.assertIsNone(manager.ik_controller)

    def test_step_once_builds_missing_ik_controller_on_demand(self):
        manager = SpaceMouseManager(
            device=_FakeDevice([self.sample()]),
            ros_client=_FakeRosClient({"name": [], "position": []}),
        )
        manager.set_mode("right")
        built: list[tuple[Path, Path]] = []

        class _LazyIKController(_FakeIKController):
            def __init__(self, robot_usd: Path, scene_usd: Path) -> None:
                super().__init__()
                built.append((robot_usd, scene_usd))

        with mock.patch("teleop.manager.IKController", _LazyIKController):
            manager.step_once()

        self.assertEqual(len(built), 1)
        self.assertEqual(len(manager.ik_controller.solve_calls), 1)

    def test_off_mode_keeps_thread_alive_but_suppresses_publish(self):
        manager = SpaceMouseManager(
            device=_FakeDevice([self.sample()]),
            ros_client=_FakeRosClient({"name": [], "position": []}),
            ik_controller=_FakeIKController(),
        )

        status = self.call_with_timeout(manager.set_mode, "off")
        manager.step_once()

        self.assertEqual(status["mode"], "off")
        self.assertEqual(manager.ros_client.published, [])

    def test_switching_arm_snaps_target_to_new_pose(self):
        manager = SpaceMouseManager(
            device=_FakeDevice([self.sample()]),
            ros_client=_FakeRosClient({"name": [], "position": []}),
            ik_controller=_FakeIKController(),
        )

        status = self.call_with_timeout(manager.set_active_arm, "left")
        manager.step_once()

        arm, target, _ = manager.ik_controller.solve_calls[-1]
        self.assertEqual(status["active_arm"], "left")
        self.assertEqual(arm, "left")
        self.assertEqual(target, manager.ik_controller.poses["left"])

    def test_double_button_resets_target_to_current_pose(self):
        manager = SpaceMouseManager(
            device=_FakeDevice([self.sample(left=True, right=True)]),
            ros_client=_FakeRosClient({"name": [], "position": []}),
            ik_controller=_FakeIKController(),
        )

        manager.set_mode("right")
        manager.step_once()

        _, target, _ = manager.ik_controller.solve_calls[-1]
        self.assertEqual(target, manager.ik_controller.poses["right"])

    def test_safe_idle_when_ros_or_device_is_unavailable(self):
        manager = SpaceMouseManager(
            device=_FakeDevice([], connected=False),
            ros_client=_FakeRosClient(None, connected=False),
            ik_controller=_FakeIKController(),
        )

        manager.step_once()

        self.assertEqual(manager.status()["ros_connected"], False)
        self.assertEqual(manager.status()["device_connected"], False)
        self.assertEqual(manager.ros_client.published, [])

    def test_device_is_polled_before_first_joint_state_arrives(self):
        device = _FakeDevice([])
        manager = SpaceMouseManager(
            device=device,
            ros_client=_FakeRosClient(None),
            ik_controller=_FakeIKController(),
        )

        manager.step_once()

        self.assertEqual(device.poll_count, 1)


if __name__ == "__main__":
    unittest.main()
