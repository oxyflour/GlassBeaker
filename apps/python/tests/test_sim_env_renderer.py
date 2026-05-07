from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from utils.zapdos import sim_env as sim_env_module
from utils.zapdos.sim_env import IsaacRenderer


class IsaacRendererReadTest(unittest.TestCase):
    @staticmethod
    def _bundle(scene_usd: str, *camera_names: str):
        cameras = [
            SimpleNamespace(name=name, prim=f"/MyRobot/{name}")
            for name in camera_names
        ]
        return SimpleNamespace(render_scene_usda=Path(scene_usd), cameras=cameras)

    def test_isaac_request_bypasses_proxy_for_loopback_url(self):
        class DummyResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"ok": true}'

        opener = mock.Mock()
        opener.open.return_value = DummyResponse()

        with mock.patch.object(sim_env_module, "ISAAC_API_URL", "http://127.0.0.1:13000/api/isaac"):
            with mock.patch.object(sim_env_module, "build_opener", create=True, return_value=opener) as build_opener:
                with mock.patch.object(sim_env_module, "urlopen", side_effect=AssertionError("should bypass proxy")):
                    payload = sim_env_module._isaac_request("GET")

        self.assertEqual(payload, {"ok": True})
        build_opener.assert_called_once()
        opener.open.assert_called_once()

    def test_read_returns_frame_for_requested_camera(self):
        renderer = IsaacRenderer.__new__(IsaacRenderer)
        renderer._running = True
        renderer.proc_id = "renderer"
        renderer.proc = None
        renderer.shm = object()
        renderer.frame_counter = np.array([9], dtype=np.uint32)
        renderer.frames = np.zeros((1, 2, 2, 2, 3), dtype=np.uint8)
        renderer.frames[0, 0, :, :, :] = 17
        renderer.frames[0, 1, :, :, :] = 33
        renderer.camera_index = {"head_camera": 0, "left_wrist_camera": 1}
        renderer._bind_shm = lambda: None

        head_index, head_frame = renderer.read("head_camera")
        wrist_index, wrist_frame = renderer.read("left_wrist_camera")

        self.assertEqual(head_index, 9)
        self.assertEqual(wrist_index, 9)
        self.assertTrue(np.all(head_frame == 17))
        self.assertTrue(np.all(wrist_frame == 33))

    def test_snapshot_cameras_reads_renderer_response_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            renderer = IsaacRenderer.__new__(IsaacRenderer)
            renderer._running = True
            renderer.proc_id = "renderer"
            renderer.proc = None
            renderer.control_dir = Path(tmp)
            renderer.log_path = Path(tmp) / "renderer.log"
            renderer._refresh_process_state = lambda: True

            def respond() -> None:
                request_path = renderer.control_dir / "request.json"
                response_path = renderer.control_dir / "response.json"
                deadline = time.time() + 2.0
                while time.time() < deadline and not request_path.exists():
                    time.sleep(0.01)
                payload = json.loads(request_path.read_text(encoding="utf-8"))
                response_path.write_text(json.dumps({
                    "id": payload["id"],
                    "ok": True,
                    "cameras": [{"name": "head_camera", "parent_prim": "/MyRobot/zed_link"}],
                }), encoding="utf-8")

            thread = threading.Thread(target=respond, daemon=True)
            thread.start()

            cameras = renderer.snapshot_cameras(timeout=2.0)

        self.assertEqual(cameras[0]["name"], "head_camera")
        self.assertEqual(cameras[0]["parent_prim"], "/MyRobot/zed_link")

    def test_snapshot_cameras_waits_for_control_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            renderer = IsaacRenderer.__new__(IsaacRenderer)
            renderer._running = True
            renderer.proc_id = "renderer"
            renderer.proc = None
            renderer.control_dir = Path(tmp)
            renderer.log_path = Path(tmp) / "renderer.log"
            renderer._refresh_process_state = lambda: True
            renderer._control_lock = threading.Lock()

            def respond() -> None:
                request_path = renderer.control_dir / "request.json"
                response_path = renderer.control_dir / "response.json"
                deadline = time.time() + 2.0
                while time.time() < deadline and not request_path.exists():
                    time.sleep(0.01)
                payload = json.loads(request_path.read_text(encoding="utf-8"))
                response_path.write_text(json.dumps({
                    "id": payload["id"],
                    "ok": True,
                    "cameras": [{"name": "head_camera", "parent_prim": "/MyRobot/zed_link"}],
                }), encoding="utf-8")

            result: list[object] = []
            renderer._control_lock.acquire()
            thread = threading.Thread(
                target=lambda: result.append(renderer.snapshot_cameras(timeout=2.0)),
                daemon=True,
            )
            responder = threading.Thread(target=respond, daemon=True)
            thread.start()
            responder.start()
            time.sleep(0.1)
            self.assertFalse((renderer.control_dir / "request.json").exists())

            renderer._control_lock.release()
            thread.join(timeout=2.0)

        self.assertEqual(result[0][0]["name"], "head_camera")

    def test_snapshot_cameras_raises_renderer_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            renderer = IsaacRenderer.__new__(IsaacRenderer)
            renderer._running = True
            renderer.proc_id = "renderer"
            renderer.proc = None
            renderer.control_dir = Path(tmp)
            renderer.log_path = Path(tmp) / "renderer.log"
            renderer._refresh_process_state = lambda: True

            def respond() -> None:
                request_path = renderer.control_dir / "request.json"
                response_path = renderer.control_dir / "response.json"
                deadline = time.time() + 2.0
                while time.time() < deadline and not request_path.exists():
                    time.sleep(0.01)
                payload = json.loads(request_path.read_text(encoding="utf-8"))
                response_path.write_text(json.dumps({
                    "id": payload["id"],
                    "ok": False,
                    "error": "snapshot failed",
                }), encoding="utf-8")

            thread = threading.Thread(target=respond, daemon=True)
            thread.start()

            with self.assertRaises(RuntimeError) as err:
                renderer.snapshot_cameras(timeout=2.0)

        self.assertIn("snapshot failed", str(err.exception))

    def test_reload_scene_updates_bundle_and_camera_index_after_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            renderer = IsaacRenderer.__new__(IsaacRenderer)
            renderer._running = True
            renderer.proc_id = "renderer"
            renderer.proc = None
            renderer.control_dir = Path(tmp)
            renderer.log_path = Path(tmp) / "renderer.log"
            renderer._refresh_process_state = lambda: True
            renderer._control_lock = threading.Lock()
            renderer.bundle = self._bundle("old_scene.usda", "old_camera")
            renderer.camera_index = {"old_camera": 0}
            next_bundle = self._bundle("new_scene.usda", "head_camera", "wrist_camera")
            requests: list[dict[str, object]] = []

            def respond() -> None:
                request_path = renderer.control_dir / "request.json"
                response_path = renderer.control_dir / "response.json"
                deadline = time.time() + 2.0
                while time.time() < deadline and not request_path.exists():
                    time.sleep(0.01)
                payload = json.loads(request_path.read_text(encoding="utf-8"))
                requests.append(payload)
                response_path.write_text(json.dumps({
                    "id": payload["id"],
                    "ok": True,
                }), encoding="utf-8")

            responder = threading.Thread(target=respond, daemon=True)
            responder.start()

            renderer.reload_scene(next_bundle, timeout=2.0)

        self.assertIs(renderer.bundle, next_bundle)
        self.assertEqual(renderer.camera_index, {"head_camera": 0, "wrist_camera": 1})
        self.assertEqual(requests[0]["op"], "reload_scene")
        self.assertEqual(requests[0]["scene_usd"], "new_scene.usda")
        self.assertEqual(requests[0]["cameras"], [
            {"name": "head_camera", "prim": "/MyRobot/head_camera"},
            {"name": "wrist_camera", "prim": "/MyRobot/wrist_camera"},
        ])

    def test_reload_scene_keeps_existing_state_when_renderer_reports_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            renderer = IsaacRenderer.__new__(IsaacRenderer)
            renderer._running = True
            renderer.proc_id = "renderer"
            renderer.proc = None
            renderer.control_dir = Path(tmp)
            renderer.log_path = Path(tmp) / "renderer.log"
            renderer._refresh_process_state = lambda: True
            renderer._control_lock = threading.Lock()
            old_bundle = self._bundle("old_scene.usda", "old_camera")
            renderer.bundle = old_bundle
            renderer.camera_index = {"old_camera": 0}
            next_bundle = self._bundle("new_scene.usda", "head_camera")

            def respond() -> None:
                request_path = renderer.control_dir / "request.json"
                response_path = renderer.control_dir / "response.json"
                deadline = time.time() + 2.0
                while time.time() < deadline and not request_path.exists():
                    time.sleep(0.01)
                payload = json.loads(request_path.read_text(encoding="utf-8"))
                response_path.write_text(json.dumps({
                    "id": payload["id"],
                    "ok": False,
                    "error": "reload failed",
                }), encoding="utf-8")

            responder = threading.Thread(target=respond, daemon=True)
            responder.start()

            with self.assertRaises(RuntimeError) as err:
                renderer.reload_scene(next_bundle, timeout=2.0)

        self.assertIn("reload failed", str(err.exception))
        self.assertIs(renderer.bundle, old_bundle)
        self.assertEqual(renderer.camera_index, {"old_camera": 0})

    def test_close_can_skip_remote_delete(self):
        renderer = IsaacRenderer.__new__(IsaacRenderer)
        renderer.proc = None
        renderer._proc_log = None
        renderer.proc_id = "renderer"
        renderer.proc_pid = 123
        renderer._running = True
        renderer.shm = None
        renderer.frame_counter = None
        renderer.frames = None

        with mock.patch.object(sim_env_module, "_isaac_request") as isaac_request:
            renderer.close(stop_remote=False)

        isaac_request.assert_not_called()
        self.assertIsNone(renderer.proc_id)
        self.assertIsNone(renderer.proc_pid)
        self.assertFalse(renderer._running)


if __name__ == "__main__":
    unittest.main()

