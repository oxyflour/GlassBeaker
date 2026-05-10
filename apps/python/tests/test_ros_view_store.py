from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.ros_view.ros_view_store import RosViewStore  # type: ignore


class _BridgeStub:
    def __init__(self) -> None:
        self.conns = {object()}
        self.calls: list[tuple[str, list[object]]] = []
        self.subscriptions: list[tuple[str, str]] = []

    async def call(self, method: str, args: list[object]):
        self.calls.append((method, args))
        if method == "list_topics":
            return [
                ("/env_0/head_camera/image_raw", ["sensor_msgs/msg/Image"]),
                ("/env_0/joint_states", ["sensor_msgs/msg/JointState"]),
                ("/tf", ["tf2_msgs/msg/TFMessage"]),
            ]
        raise AssertionError(f"unexpected method: {method}")

    async def subscribe(self, topic: str, type_name: str, callback) -> None:
        self.subscriptions.append((topic, type_name))


class RosViewStoreTest(unittest.IsolatedAsyncioTestCase):
    async def test_state_discovers_topics_from_ros(self):
        store = RosViewStore(_BridgeStub())

        state = await store.state()

        self.assertEqual(
            [topic["id"] for topic in state["topics"]],
            ["/env_0/head_camera/image_raw", "/env_0/joint_states", "/tf"],
        )
        self.assertEqual(state["topics"][0]["src"], "/python/ros_view/render/%2Fenv_0%2Fhead_camera%2Fimage_raw")
        self.assertEqual(state["topics"][1]["kind"], "plot")
        self.assertEqual(state["topics"][2]["kind"], "state")
        self.assertEqual(state["topics"][2]["fields"][1]["value"], "Unsupported")
        self.assertEqual(
            store.bridge.subscriptions,
            [
                ("/env_0/head_camera/image_raw", "sensor_msgs/msg/Image"),
                ("/env_0/joint_states", "sensor_msgs/msg/JointState"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
