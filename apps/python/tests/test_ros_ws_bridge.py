from __future__ import annotations

import pickle
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "ros"))

import ws_bridge


class RosWsBridgeTest(unittest.TestCase):
    def test_bridge_server_url_defaults_to_ipv4_loopback(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                ws_bridge.bridge_server_url(),
                "ws://127.0.0.1:13001/api/ros/ws",
            )

    def test_bridge_server_url_uses_explicit_env_override(self):
        with mock.patch.dict(os.environ, {"WS_ADDR": "ws://example.com/api/ros/ws"}, clear=True):
            self.assertEqual(
                ws_bridge.bridge_server_url(),
                "ws://example.com/api/ros/ws",
            )

    def test_loopback_bridge_disables_proxy(self):
        self.assertEqual(
            ws_bridge.bridge_connect_kwargs("ws://127.0.0.1:13001/api/ros/ws"),
            {"proxy": None},
        )
        self.assertEqual(
            ws_bridge.bridge_connect_kwargs("ws://localhost:13001/api/ros/ws"),
            {"proxy": None},
        )

    def test_remote_bridge_keeps_default_proxy_behavior(self):
        self.assertEqual(
            ws_bridge.bridge_connect_kwargs("ws://example.com/api/ros/ws"),
            {},
        )


class RosWsBridgeDispatchTest(unittest.IsolatedAsyncioTestCase):
    async def test_dispatch_bridge_request_spreads_pickled_args(self):
        session = mock.Mock()
        session.call = mock.AsyncMock(return_value="ok")
        payload = pickle.dumps(
            [
                "subscribe",
                ["/env_0/joint_command", "sensor_msgs/msg/JointState"],
                "call-1",
            ]
        )

        reply = await ws_bridge.dispatch_bridge_request(session, payload)

        self.assertEqual(reply, ("call-1", None, "ok"))
        session.call.assert_awaited_once_with(
            "subscribe",
            "/env_0/joint_command",
            "sensor_msgs/msg/JointState",
        )


if __name__ == "__main__":
    unittest.main()
