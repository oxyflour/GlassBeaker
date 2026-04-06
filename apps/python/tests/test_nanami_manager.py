from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from nanami.manager import NanamiManager


class FakeWorker:
    def __init__(self, *_args, **_kwargs):
        self.stopped = False

    def write_session_config(self, _payload: dict) -> None:
        pass

    def start(self) -> None:
        raise RuntimeError("boom")

    def stop(self, timeout: float = 5.0) -> None:
        self.stopped = True


class NanamiManagerTest(unittest.TestCase):
    def seed_session(self, manager: NanamiManager, session_id: str, *, ready: bool, loaded: bool):
        session = mock.Mock()
        session.ready = ready
        session.robot_loaded = loaded
        manager.sessions[session_id] = session
        manager.manifests[session_id] = SimpleNamespace(
            robot_id="r1",
            link_count=8,
            movable_joint_count=6,
            to_dict=mock.Mock(return_value={"robot_id": "r1"}),
        )
        manager.cache_hit[session_id] = True
        manager.workers[session_id] = mock.Mock()
        return session, manager.workers[session_id]

    def test_destroy_session_clears_state_and_stops_worker(self):
        manager = NanamiManager()
        session = mock.Mock()
        worker = mock.Mock()
        manager.sessions["s1"] = session
        manager.manifests["s1"] = SimpleNamespace()
        manager.cache_hit["s1"] = True
        manager.workers["s1"] = worker

        manager.destroy_session("s1")

        session.close.assert_called_once_with()
        worker.stop.assert_called_once_with()
        self.assertNotIn("s1", manager.sessions)
        self.assertNotIn("s1", manager.manifests)
        self.assertNotIn("s1", manager.cache_hit)
        self.assertNotIn("s1", manager.workers)

    def test_create_session_rolls_back_when_worker_start_fails(self):
        manager = NanamiManager()
        manifest = SimpleNamespace(resolved_commit="deadbeef")
        with mock.patch("nanami.manager.ensure_r1_asset_cache", return_value=(manifest, False)):
            with mock.patch("nanami.manager.NanamiWorker", FakeWorker):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    manager.create_session(None)

        self.assertEqual(manager.sessions, {})
        self.assertEqual(manager.manifests, {})
        self.assertEqual(manager.cache_hit, {})
        self.assertEqual(manager.workers, {})

    def test_load_robot_queues_until_worker_is_ready(self):
        manager = NanamiManager()
        session, worker = self.seed_session(manager, "s1", ready=False, loaded=False)

        data = manager.load_robot("s1")

        self.assertEqual(data["status"], "queued")
        worker.send.assert_not_called()

        manager.on_worker_event("s1", {"type": "hello"})

        worker.send.assert_called_once_with({"type": "load_robot", "manifest": {"robot_id": "r1"}})
        session.publish.assert_any_call({"type": "session_ready", "session_id": "s1"})
        session.publish.assert_any_call({"type": "robot_load_started", "session_id": "s1"})

    def test_load_robot_is_idempotent_and_retriable_after_error(self):
        manager = NanamiManager()
        _session, worker = self.seed_session(manager, "s1", ready=True, loaded=False)

        first = manager.load_robot("s1")
        second = manager.load_robot("s1")

        self.assertEqual(first["status"], "loading")
        self.assertEqual(second["status"], "loading")
        worker.send.assert_called_once_with({"type": "load_robot", "manifest": {"robot_id": "r1"}})

        manager.on_worker_event("s1", {"type": "worker_error", "message": "boom"})

        third = manager.load_robot("s1")

        self.assertEqual(third["status"], "loading")
        self.assertEqual(worker.send.call_count, 2)


if __name__ == "__main__":
    unittest.main()
