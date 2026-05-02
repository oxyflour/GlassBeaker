from __future__ import annotations

import unittest

from utils.rl_cameras import RenderCamera, camera_name_to_index


class RLCamerasTest(unittest.TestCase):
    def test_camera_name_to_index_requires_exact_match(self):
        cameras = [
            RenderCamera(
                name="head_camera",
                prim="/MyRobot/zed_link/head_camera",
                topic="/env_0/head_camera/image_raw",
                frame_id="head_camera",
                body="Root_r1_pro_with_gripper_zed_link",
                pos=[0.0, 0.0, 0.0],
                quat=[1.0, 0.0, 0.0, 0.0],
                fovy=45.0,
            ),
            RenderCamera(
                name="left_wrist_camera",
                prim="/MyRobot/left_link/left_wrist_camera",
                topic="/env_0/left_wrist_camera/image_raw",
                frame_id="left_wrist_camera",
                body="Root_r1_pro_with_gripper_left_realsense_link",
                pos=[0.0, 0.0, 0.0],
                quat=[1.0, 0.0, 0.0, 0.0],
                fovy=45.0,
            ),
        ]

        mapping = camera_name_to_index(cameras)

        self.assertEqual(mapping["head_camera"], 0)
        self.assertEqual(mapping["left_wrist_camera"], 1)
        self.assertNotIn("HEAD_CAMERA", mapping)
        self.assertNotIn("main", mapping)


if __name__ == "__main__":
    unittest.main()
