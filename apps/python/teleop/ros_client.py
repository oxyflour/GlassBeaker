from __future__ import annotations

import os
import pickle
import time
import uuid
from typing import Any

from websockets.sync.client import connect

from utils.sim_env import JOINT_COMMAND_TOPIC, JOINT_STATE_TYPE, JOINT_STATES_TOPIC


class RosBridgeClient:
    def __init__(self, url: str | None = None, timeout: float = 1.0) -> None:
        port = os.getenv("LISTEN_PORT", "13001")
        self.url = url or f"ws://127.0.0.1:{port}/api/ros/ws"
        self.timeout = timeout
        self._conn = None
        self._subscribed = False
        self._latest_joint_state: dict[str, Any] | None = None
        self.connected = False
        self.last_error: str | None = None

    def poll_messages(self) -> None:
        if not self.ensure_connected():
            return
        while True:
            try:
                call, _, ret = self._recv(timeout=0.0)
            except TimeoutError:
                return
            except Exception as exc:
                self.last_error = str(exc)
                self.close()
                return
            self._handle_message(call, ret)

    def latest_joint_state(self) -> dict[str, Any] | None:
        return self._latest_joint_state

    def publish_joint_command(self, command: dict[str, Any]) -> None:
        if self.ensure_connected():
            self._call("publish", [JOINT_COMMAND_TOPIC, JOINT_STATE_TYPE, command])

    def status(self) -> dict[str, object]:
        return {"connected": self.connected, "last_error": self.last_error}

    def close(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        self._conn = None
        self.connected = False
        self._subscribed = False

    def ensure_connected(self) -> bool:
        if self._conn is not None:
            return True
        try:
            self._conn = connect(self.url, max_size=None)
            self.connected = True
            self.last_error = None
            self._call("subscribe", [JOINT_STATES_TOPIC, JOINT_STATE_TYPE])
            self._subscribed = True
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.close()
            return False

    def _call(self, method: str, args: list[Any]) -> Any:
        if self._conn is None:
            raise RuntimeError("ROS bridge is not connected")
        call_id = str(uuid.uuid4())
        self._conn.send(pickle.dumps([method, args, call_id]))
        deadline = time.time() + self.timeout
        while True:
            remaining = max(deadline - time.time(), 0.0)
            call, err, ret = self._recv(timeout=remaining)
            if call == call_id:
                if err:
                    raise RuntimeError(str(err))
                return ret
            self._handle_message(call, ret)

    def _recv(self, timeout: float):
        if self._conn is None:
            raise RuntimeError("ROS bridge is not connected")
        raw = self._conn.recv(timeout=timeout, decode=False)
        return pickle.loads(raw)

    def _handle_message(self, call: str, ret: Any) -> None:
        if call or not isinstance(ret, dict):
            return
        if ret.get("topic") == JOINT_STATES_TOPIC:
            self._latest_joint_state = ret.get("msg")
