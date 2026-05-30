from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.teleop.ros_client import RosBridgeClient  # noqa: E402
from utils.zapdos.ros.topics import JOINT_COMMAND_TOPIC, JOINT_STATE_TYPE, JOINT_STATES_TOPIC  # noqa: E402


class _FakeBridge:
    def __init__(self) -> None:
        self.conns = {object()}
        self.calls: list[tuple[str, list[object]]] = []
        self.subscriptions: list[tuple[str, str, object]] = []

    async def call(self, method: str, args: list[object]) -> dict[str, bool]:
        self.calls.append((method, args))
        return {"ok": True}

    async def subscribe(self, topic: str, type_name: str, callback) -> None:
        self.subscriptions.append((topic, type_name, callback))

    def unsubscribe(self, topic: str, callback) -> None:
        self.subscriptions = [
            item for item in self.subscriptions if item[:2] != (topic, JOINT_STATE_TYPE) or item[2] is not callback
        ]


class RosBridgeClientTest(unittest.TestCase):
    def run_coro(self, coro):
        return asyncio.run(coro)

    def test_poll_messages_subscribes_and_receives_joint_states(self):
        bridge = _FakeBridge()
        client = RosBridgeClient(bridge=bridge, runner=self.run_coro)

        client.poll_messages()
        _, _, callback = bridge.subscriptions[0]
        callback(JOINT_STATES_TOPIC, {"name": ["joint1"], "position": [0.5]})

        self.assertEqual(client.status()["connected"], True)
        self.assertEqual(bridge.subscriptions[0][:2], (JOINT_STATES_TOPIC, JOINT_STATE_TYPE))
        self.assertEqual(client.latest_joint_state()["position"], [0.5])

    def test_publish_joint_command_forwards_to_bridge_call(self):
        bridge = _FakeBridge()
        client = RosBridgeClient(bridge=bridge, runner=self.run_coro)

        command = {"name": ["joint1"], "position": [0.1]}
        client.publish_joint_command(command)

        self.assertEqual(
            bridge.calls[0],
            ("publish", [JOINT_COMMAND_TOPIC, JOINT_STATE_TYPE, command]),
        )


if __name__ == "__main__":
    unittest.main()

