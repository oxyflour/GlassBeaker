from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from utils.ros_bridge import bridge as default_bridge
from utils.zapdos.ros.topics import JOINT_COMMAND_TOPIC, JOINT_STATE_TYPE, JOINT_STATES_TOPIC

CoroutineRunner = Callable[[Awaitable[Any]], Any]


class RosBridgeClient:
    def __init__(
        self,
        bridge=default_bridge,
        runner: CoroutineRunner | None = None,
        timeout: float = 1.0,
    ) -> None:
        self._bridge = bridge
        self.timeout = timeout
        self._runner = runner or self._default_runner(asyncio.get_running_loop())
        self._subscribed = False
        self._latest_joint_state: dict[str, Any] | None = None
        self.connected = False
        self.last_error: str | None = None

    def poll_messages(self) -> None:
        self.ensure_connected()

    def latest_joint_state(self) -> dict[str, Any] | None:
        return self._latest_joint_state

    def publish_joint_command(self, command: dict[str, Any]) -> None:
        if not self.ensure_connected():
            return
        try:
            self._runner(self._bridge.call("publish", [JOINT_COMMAND_TOPIC, JOINT_STATE_TYPE, command]))
            self.last_error = None
        except Exception as exc:
            self.last_error = str(exc)
            self.connected = False

    def status(self) -> dict[str, object]:
        self.connected = bool(self._bridge.conns)
        return {"connected": self.connected, "last_error": self.last_error}

    def close(self) -> None:
        if self._subscribed:
            self._bridge.unsubscribe(JOINT_STATES_TOPIC, self._on_joint_state)
        self._subscribed = False
        self.connected = False

    def ensure_connected(self) -> bool:
        self.connected = bool(self._bridge.conns)
        if not self.connected:
            return False
        if self._subscribed:
            return True
        try:
            self._runner(self._bridge.subscribe(JOINT_STATES_TOPIC, JOINT_STATE_TYPE, self._on_joint_state))
            self._subscribed = True
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.connected = False
            return False

    def _on_joint_state(self, topic: str, msg: Any) -> None:
        if topic == JOINT_STATES_TOPIC and isinstance(msg, dict):
            self._latest_joint_state = msg

    def _default_runner(self, loop: asyncio.AbstractEventLoop) -> CoroutineRunner:
        def run(coro: Awaitable[Any]) -> Any:
            return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=self.timeout)

        return run

