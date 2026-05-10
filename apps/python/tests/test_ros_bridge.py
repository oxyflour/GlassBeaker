from __future__ import annotations

import sys
import unittest
from pathlib import Path

from fastapi import WebSocketDisconnect

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

import utils.ros_bridge as ros_bridge_module

Bridge = ros_bridge_module.Bridge
BridgeUnavailable = getattr(ros_bridge_module, "BridgeUnavailable", RuntimeError)


class _FailingWebSocket:
    async def send_bytes(self, data: bytes) -> None:
        raise WebSocketDisconnect(code=1006)


class RosBridgeTest(unittest.IsolatedAsyncioTestCase):
    async def test_call_discards_stale_connection_and_raises_bridge_unavailable(self):
        bridge = Bridge()
        bridge.conns = {_FailingWebSocket()}
        bridge.calls = {}
        bridge.subs = {}

        with self.assertRaises(BridgeUnavailable):
            await bridge.call("subscribe", ["/env_0/joint_command", "sensor_msgs/msg/JointState"])

        self.assertEqual(bridge.conns, set())
        self.assertEqual(bridge.calls, {})

    async def test_subscribe_rolls_back_callback_when_transport_setup_fails(self):
        bridge = Bridge()
        bridge.conns = {_FailingWebSocket()}
        bridge.calls = {}
        bridge.subs = {}

        def callback(topic: str, msg) -> None:
            return None

        with self.assertRaises(BridgeUnavailable):
            await bridge.subscribe("/env_0/joint_command", "sensor_msgs/msg/JointState", callback)

        self.assertNotIn("/env_0/joint_command", bridge.subs)


if __name__ == "__main__":
    unittest.main()
