from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
