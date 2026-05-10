from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

import utils.zapdos.session.zapdos_session as session_module
import utils.ros_bridge as ros_bridge_module
from utils.zapdos.session.streaming_mixin import SessionStreamingMixin

BridgeUnavailable = getattr(ros_bridge_module, "BridgeUnavailable", RuntimeError)


class _BridgeStub:
    def __init__(self) -> None:
        self.conns = {object()}
        self.subs: dict[str, set[object]] = {}
        self.subscribe_calls: list[tuple[str, str]] = []
        self.publish_calls: list[tuple[str, str, object]] = []

    async def subscribe(self, topic: str, type_name: str, callback) -> None:
        self.subscribe_calls.append((topic, type_name))
        self.subs.setdefault(topic, set()).add(callback)

    async def call(self, method: str, args: list[object]):
        if method != "publish":
            raise AssertionError(f"unexpected method: {method}")
        topic, type_name, payload = args
        self.publish_calls.append((str(topic), str(type_name), payload))
        return {"ok": True}


class _FakeSession(SessionStreamingMixin):
    pass


class ZapdosSendRosTest(unittest.IsolatedAsyncioTestCase):
    def make_session(self) -> _FakeSession:
        session = _FakeSession()
        active = {"value": True}

        async def fake_wait_ready():
            return None

        async def fake_sleep(_: float):
            active["value"] = False

        session.renderer = SimpleNamespace(wait_ready=fake_wait_ready)
        session.physics = SimpleNamespace(
            joint_state_msg=lambda: {"name": ["joint"], "position": [0.0]},
            model=object(),
            data=object(),
        )
        session.bundle = SimpleNamespace(cameras=[SimpleNamespace(name="head_camera")])
        session.command_subscribed = False
        session.on_message = mock.Mock()
        session.is_active = lambda: active["value"]
        session._image_messages = mock.Mock(return_value=[
            (session_module.image_topic("head_camera"), {"data": b"rgb"}),
        ])
        self.sleep_patch = mock.patch.object(session_module.asyncio, "sleep", side_effect=fake_sleep)
        return session

    async def test_send_ros_skips_image_publish_without_local_image_subscriber(self):
        session = self.make_session()
        bridge = _BridgeStub()

        with self.sleep_patch:
            with mock.patch.object(session_module, "bridge", bridge):
                with mock.patch.object(session_module, "tf_message", return_value={"transforms": []}):
                    await session.send_ros()

        self.assertEqual(
            [topic for topic, _, _ in bridge.publish_calls],
            [
                session_module.JOINT_STATES_TOPIC,
                session_module.TF_RENDER_TOPIC,
            ],
        )
        session._image_messages.assert_not_called()

    async def test_send_ros_publishes_images_when_local_subscriber_exists(self):
        session = self.make_session()
        bridge = _BridgeStub()
        bridge.subs[session_module.image_topic("head_camera")] = {object()}

        with self.sleep_patch:
            with mock.patch.object(session_module, "bridge", bridge):
                with mock.patch.object(session_module, "tf_message", return_value={"transforms": []}):
                    await session.send_ros()

        self.assertEqual(
            [topic for topic, _, _ in bridge.publish_calls],
            [
                session_module.JOINT_STATES_TOPIC,
                session_module.TF_RENDER_TOPIC,
                session_module.image_topic("head_camera"),
            ],
        )
        session._image_messages.assert_called_once_with()

    async def test_send_ros_suppresses_traceback_for_bridge_disconnect(self):
        session = self.make_session()

        class _DisconnectingBridge:
            def __init__(self) -> None:
                self.conns = {object()}
                self.subs: dict[str, set[object]] = {}

            async def subscribe(self, topic: str, type_name: str, callback) -> None:
                self.conns.clear()
                raise BridgeUnavailable("no connections now")

            async def call(self, method: str, args: list[object]):
                raise AssertionError(f"unexpected method: {method}")

        with self.sleep_patch:
            with mock.patch.object(session_module, "bridge", _DisconnectingBridge()):
                with mock.patch.object(session_module.traceback, "print_exc") as print_exc:
                    await session.send_ros()

        print_exc.assert_not_called()
        self.assertEqual(session.command_subscribed, False)


if __name__ == "__main__":
    unittest.main()
