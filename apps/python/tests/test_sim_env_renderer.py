from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.zapdos.renderer import isaac_renderer as renderer_module
from utils.zapdos.renderer.isaac_renderer import IsaacRenderer


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

        with mock.patch.object(renderer_module, "ISAAC_API_URL", "http://127.0.0.1:13000/api/isaac"):
            with mock.patch.object(renderer_module, "build_opener", create=True, return_value=opener) as build_opener:
                with mock.patch.object(renderer_module, "urlopen", side_effect=AssertionError("should bypass proxy")):
                    payload = renderer_module._isaac_request("GET")

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
        renderer = IsaacRenderer.__new__(IsaacRenderer)
        renderer._shm_prefix = "glassbeaker_sess_frames"
        renderer._spawn_generation = 0
        renderer.shm_name = "glassbeaker_sess_frames"
        renderer.width = 4
        renderer.height = 3
        renderer.bundle = self._bundle("old_scene.usda", "old_camera")
        renderer.camera_index = {"old_camera": 0}
        renderer.close = mock.Mock()
        renderer._spawn = mock.Mock()
        next_bundle = self._bundle("new_scene.usda", "head_camera", "wrist_camera")

        renderer.reload_scene(next_bundle, timeout=2.0)

        self.assertIs(renderer.bundle, next_bundle)
        self.assertEqual(renderer.camera_index, {"head_camera": 0, "wrist_camera": 1})
        self.assertEqual(renderer.frame_buffer.num_cameras, 2)
        renderer.close.assert_called_once_with()
        renderer._spawn.assert_called_once_with()

    def test_reload_scene_keeps_existing_state_when_restart_fails(self):
        renderer = IsaacRenderer.__new__(IsaacRenderer)
        renderer._shm_prefix = "glassbeaker_sess_frames"
        renderer._spawn_generation = 0
        renderer.shm_name = "glassbeaker_sess_frames"
        renderer.width = 4
        renderer.height = 3
        old_bundle = self._bundle("old_scene.usda", "old_camera")
        renderer.bundle = old_bundle
        renderer.camera_index = {"old_camera": 0}
        renderer.close = mock.Mock()
        renderer._spawn = mock.Mock(side_effect=RuntimeError("restart failed"))
        next_bundle = self._bundle("new_scene.usda", "head_camera")

        with self.assertRaises(RuntimeError) as err:
            renderer.reload_scene(next_bundle, timeout=2.0)

        self.assertIn("restart failed", str(err.exception))
        self.assertIs(renderer.bundle, old_bundle)
        self.assertEqual(renderer.camera_index, {"old_camera": 0})
        self.assertEqual(renderer.shm_name, "glassbeaker_sess_frames")
        self.assertEqual(renderer._spawn_generation, 0)

    def test_reload_scene_restarts_renderer_without_control_ipc(self):
        renderer = IsaacRenderer.__new__(IsaacRenderer)
        renderer._shm_prefix = "glassbeaker_sess_frames"
        renderer._spawn_generation = 0
        renderer.shm_name = "glassbeaker_sess_frames"
        renderer.width = 4
        renderer.height = 3
        renderer.bundle = self._bundle("old_scene.usda", "old_camera")
        renderer.camera_index = {"old_camera": 0}
        renderer.close = mock.Mock()
        renderer._spawn = mock.Mock()
        renderer._control_request = mock.Mock(side_effect=AssertionError("reload should restart instead of hot reloading"))
        next_bundle = self._bundle("new_scene.usda", "head_camera", "wrist_camera")

        renderer.reload_scene(next_bundle, timeout=2.0)

        renderer.close.assert_called_once_with()
        renderer._spawn.assert_called_once_with()
        self.assertIs(renderer.bundle, next_bundle)
        self.assertEqual(renderer.camera_index, {"head_camera": 0, "wrist_camera": 1})
        self.assertEqual(renderer.shm_name, "glassbeaker_sess_frames_r1")
        self.assertEqual(renderer.frame_buffer.shm_name, "glassbeaker_sess_frames_r1")
        self.assertEqual(renderer.frame_buffer.num_cameras, 2)

    def test_wait_ready_reports_log_path_when_renderer_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            renderer = IsaacRenderer.__new__(IsaacRenderer)
            renderer.log_path = Path(tmp) / "renderer.log"
            renderer.log_path.write_text("renderer traceback", encoding="utf-8")
            renderer.shm_name = "glassbeaker_renderer_frames"
            renderer._refresh_process_state = lambda: False
            renderer.close = mock.Mock()

            with self.assertRaises(RuntimeError) as err:
                asyncio.run(renderer.wait_ready(timeout=0.1))

        self.assertIn("IsaacSim failed to start, check", str(err.exception))
        self.assertIn(str(renderer.log_path), str(err.exception))
        self.assertIn("renderer traceback", str(err.exception))
        renderer.close.assert_called_once()

    def test_snapshot_cameras_reports_log_path_when_renderer_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            renderer = IsaacRenderer.__new__(IsaacRenderer)
            renderer._running = True
            renderer.proc_id = "renderer"
            renderer.proc = None
            renderer.control_dir = Path(tmp)
            renderer.log_path = Path(tmp) / "renderer.log"
            renderer.log_path.write_text("renderer exited during snapshot", encoding="utf-8")
            renderer._refresh_process_state = lambda: False

            with self.assertRaises(RuntimeError) as err:
                renderer.snapshot_cameras(timeout=0.1)

        self.assertIn("IsaacSim quit unexpectedly, check", str(err.exception))
        self.assertIn(str(renderer.log_path), str(err.exception))
        self.assertIn("renderer exited during snapshot", str(err.exception))

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

        with mock.patch.object(renderer_module, "_isaac_request") as isaac_request:
            renderer.close(stop_remote=False)

        isaac_request.assert_not_called()
        self.assertIsNone(renderer.proc_id)
        self.assertIsNone(renderer.proc_pid)
        self.assertFalse(renderer._running)


if __name__ == "__main__":
    unittest.main()
