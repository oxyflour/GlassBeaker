from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from utils.sim_env import IsaacRenderer


class IsaacRendererReadTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
