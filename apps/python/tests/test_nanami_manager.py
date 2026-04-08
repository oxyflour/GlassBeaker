from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nanami.github_cache import AssetBundle
from nanami.manager import NanamiManager
from nanami.models import ControlGroup, JointControl, RobotManifest, SourceConfig
from nanami.runtime import RuntimeBusyError, RuntimeHost


def fake_manifest() -> RobotManifest:
    return RobotManifest(
        robot_id="r1",
        package_name="r1",
        resolved_commit="deadbeef",
        source=SourceConfig(robot_path="R1"),
        links=[],
        joints=[],
        movable_joint_count=1,
        link_count=2,
        controls=[ControlGroup(name="left_arm", joints=[JointControl(name="j1", kind="revolute", lower=-1.0, upper=1.0)])],
    )


def fake_bundle(tmp: str) -> AssetBundle:
    cache_root = Path(tmp) / "asset-cache" / "R1_local" / "abc123"
    return AssetBundle(
        manifest=fake_manifest(),
        cache_hit=True,
        cache_root=cache_root,
        runtime_key=str(cache_root),
        asset_key="r1_local_abc123",
    )


def fake_start(self: RuntimeHost) -> None:
    self.worker = SimpleNamespace(send=mock.Mock(), stop=mock.Mock())


class NanamiManagerTest(unittest.TestCase):
    def make_manager(self) -> NanamiManager:
        return NanamiManager(runtime_ttl_seconds=0.05, reaper_poll_seconds=0.01)

    def test_create_session_reuses_warm_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = fake_bundle(tmp)
            manager = self.make_manager()
            try:
                with mock.patch("nanami.manager.ensure_r1_asset_cache", return_value=bundle):
                    with mock.patch.object(RuntimeHost, "start", fake_start):
                        first = manager.create_session(None)
                        runtime_id = first["runtime_id"]
                        manager.on_worker_event(bundle.runtime_key, {"type": "hello"})
                        manager.on_worker_event(bundle.runtime_key, {"type": "robot_loaded"})
                        manager.destroy_session(first["session_id"])

                        second = manager.create_session(None)

                self.assertFalse(first["runtime_reused"])
                self.assertTrue(second["runtime_reused"])
                self.assertEqual(second["runtime_id"], runtime_id)
            finally:
                manager.shutdown()

    def test_create_session_rejects_busy_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = fake_bundle(tmp)
            manager = self.make_manager()
            try:
                with mock.patch("nanami.manager.ensure_r1_asset_cache", return_value=bundle):
                    with mock.patch.object(RuntimeHost, "start", fake_start):
                        manager.create_session(None)
                        with self.assertRaises(RuntimeBusyError):
                            manager.create_session(None)
            finally:
                manager.shutdown()

    def test_destroy_session_keeps_runtime_until_ttl_then_reaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = fake_bundle(tmp)
            manager = self.make_manager()
            try:
                with mock.patch("nanami.manager.ensure_r1_asset_cache", return_value=bundle):
                    with mock.patch.object(RuntimeHost, "start", fake_start):
                        data = manager.create_session(None)
                        runtime = manager.runtimes[bundle.runtime_key]
                        manager.destroy_session(data["session_id"])
                        self.assertIn(bundle.runtime_key, manager.runtimes)
                        time.sleep(0.08)
                        self.assertNotIn(bundle.runtime_key, manager.runtimes)
                        runtime.worker.stop.assert_called_once_with()
            finally:
                manager.shutdown()

    def test_failed_runtime_is_replaced_after_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = fake_bundle(tmp)
            manager = self.make_manager()
            try:
                with mock.patch("nanami.manager.ensure_r1_asset_cache", return_value=bundle):
                    with mock.patch.object(RuntimeHost, "start", fake_start):
                        first = manager.create_session(None)
                        first_runtime_id = first["runtime_id"]
                        manager.on_worker_event(bundle.runtime_key, {"type": "worker_exit"})
                        manager.destroy_session(first["session_id"])
                        second = manager.create_session(None)

                self.assertFalse(second["runtime_reused"])
                self.assertNotEqual(second["runtime_id"], first_runtime_id)
            finally:
                manager.shutdown()

    def test_robot_load_endpoint_reports_runtime_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = fake_bundle(tmp)
            manager = self.make_manager()
            try:
                with mock.patch("nanami.manager.ensure_r1_asset_cache", return_value=bundle):
                    with mock.patch.object(RuntimeHost, "start", fake_start):
                        data = manager.create_session(None)
                        starting = manager.load_robot(data["session_id"])
                        manager.on_worker_event(bundle.runtime_key, {"type": "hello"})
                        loading = manager.load_robot(data["session_id"])
                        manager.on_worker_event(bundle.runtime_key, {"type": "robot_loaded"})
                        loaded = manager.load_robot(data["session_id"])

                self.assertEqual(starting["status"], "starting")
                self.assertEqual(loading["status"], "loading")
                self.assertEqual(loaded["status"], "loaded")
                self.assertEqual(loaded["robot_id"], "r1")
            finally:
                manager.shutdown()


if __name__ == "__main__":
    unittest.main()
