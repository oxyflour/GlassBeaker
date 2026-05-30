from __future__ import annotations

import asyncio
import copy
import io
import json
import queue
import threading
from collections import deque
from typing import Any

from PIL import Image

from utils.ros_bridge import bridge as default_bridge
from utils.ros_view.ros_view_topics import PLOT_COLORS, build_topic, normalize_topics, subscription_type
from utils.zapdos.ros.topics import IMAGE_TYPE, JOINT_STATE_TYPE


def _mjpeg_chunk(payload: bytes) -> bytes:
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"


def _sse_event(name: str, payload: dict[str, object]) -> str:
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def _placeholder_jpeg() -> bytes:
    image = Image.new("RGB", (640, 360), (10, 18, 28))
    data = io.BytesIO()
    image.save(data, format="JPEG", quality=70)
    return data.getvalue()


class RosViewStore:
    def __init__(self, bridge=default_bridge) -> None:
        self.bridge = bridge
        self.last_error: str | None = None
        self._lock = threading.Lock()
        self._topic_order: list[str] = []
        self._topics: dict[str, dict[str, object]] = {}
        self._subscription_types: dict[str, str] = {}
        self._history: dict[str, deque[dict[str, float]]] = {}
        self._images: dict[str, tuple[int, bytes]] = {}
        self._subscribed: set[str] = set()
        self._listeners: set[queue.Queue[dict[str, object]]] = set()
        self._placeholder = _placeholder_jpeg()

    async def state(self) -> dict[str, object]:
        await self.ensure_started()
        return self.snapshot()

    async def stream(self):
        await self.ensure_started()
        listener: queue.Queue[dict[str, object]] = queue.Queue(maxsize=1)
        self._listeners.add(listener)
        self._push(listener, self.snapshot())
        last_payload: dict[str, object] | None = None
        try:
            while True:
                try:
                    payload = await asyncio.to_thread(listener.get, True, 3.0)
                    last_payload = payload
                    yield _sse_event("state", payload)
                except queue.Empty:
                    payload = self.snapshot()
                    if payload != last_payload:
                        last_payload = payload
                        yield _sse_event("state", payload)
                    else:
                        yield ": keepalive\n\n"
        finally:
            self._listeners.discard(listener)

    async def render(self, topic_id: str):
        if not self.has_image_topic(topic_id):
            raise KeyError(topic_id)
        last_version = -1
        while True:
            version, payload = self._images.get(topic_id, (0, self._placeholder))
            if version != last_version:
                last_version = version
                yield _mjpeg_chunk(payload)
            await asyncio.sleep(0.1 if version else 1.0)

    def has_image_topic(self, topic_id: str) -> bool:
        return self._subscription_types.get(topic_id) == IMAGE_TYPE

    async def ensure_started(self) -> None:
        if not self.bridge.conns:
            return
        await self._discover_topics()
        for topic, type_name in self._subscription_types.items():
            if topic in self._subscribed:
                continue
            try:
                await self.bridge.subscribe(topic, type_name, self.on_message)
                self._subscribed.add(topic)
                self.last_error = None
            except Exception as exc:
                self.last_error = str(exc)

    async def _discover_topics(self) -> None:
        if self._topic_order:
            return
        try:
            items = normalize_topics(await self.bridge.call("list_topics", []))
        except Exception as exc:
            self.last_error = str(exc)
            return
        topics = [build_topic(topic, type_names) for topic, type_names in items]
        with self._lock:
            self._topic_order = [str(topic["id"]) for topic in topics]
            self._topics = {str(topic["id"]): topic for topic in topics}
            self._subscription_types = {
                topic: type_name
                for topic, type_names in items
                if (type_name := subscription_type(type_names))
            }
            self._history = {
                topic: deque(maxlen=8)
                for topic, type_name in self._subscription_types.items()
                if type_name == JOINT_STATE_TYPE
            }

    def on_message(self, topic: str, msg: Any) -> None:
        type_name = self._subscription_types.get(topic)
        if type_name == IMAGE_TYPE:
            self._update_image(topic, msg)
            return
        if type_name == JOINT_STATE_TYPE:
            self._update_plot(topic, msg)
            self._broadcast()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            topics = [copy.deepcopy(self._topics[topic_id]) for topic_id in self._topic_order]
        return {
            "connected": bool(self.bridge.conns),
            "last_error": self.last_error,
            "topics": topics,
        }

    def _update_image(self, topic_id: str, msg: Any) -> None:
        payload = self._encode_jpeg(msg)
        if payload is None:
            return
        version = self._images.get(topic_id, (0, b""))[0] + 1
        self._images[topic_id] = (version, payload)

    def _update_plot(self, topic_id: str, msg: Any) -> None:
        names = [str(name) for name in (msg.get("name") or [])]
        values = list(msg.get("position") or [])
        count = min(3, len(names), len(values))
        if count == 0:
            return
        sample = {names[index]: float(values[index]) for index in range(count)}
        history = self._history[topic_id]
        history.append(sample)
        stamps = [f"-{len(history) - index - 1}" if index < len(history) - 1 else "now" for index in range(len(history))]
        series = [{
            "label": name,
            "color": PLOT_COLORS[index],
            "values": [entry.get(name, 0.0) for entry in history],
        } for index, name in enumerate(sample)]
        with self._lock:
            self._topics[topic_id]["unit"] = "rad"
            self._topics[topic_id]["timestamps"] = stamps
            self._topics[topic_id]["series"] = series

    def _push(self, listener: queue.Queue[dict[str, object]], payload: dict[str, object]) -> None:
        try:
            listener.put_nowait(payload)
        except queue.Full:
            try:
                listener.get_nowait()
            except queue.Empty:
                pass
            listener.put_nowait(payload)

    def _broadcast(self) -> None:
        snapshot = self.snapshot()
        for listener in list(self._listeners):
            self._push(listener, snapshot)

    def _encode_jpeg(self, msg: Any) -> bytes | None:
        if not isinstance(msg, dict) or msg.get("encoding") != "rgb8":
            return None
        width = int(msg.get("width") or 0)
        height = int(msg.get("height") or 0)
        if width <= 0 or height <= 0:
            return None
        data = msg.get("data")
        if isinstance(data, list):
            payload = bytes(int(item) & 0xFF for item in data)
        elif isinstance(data, (bytes, bytearray)):
            payload = bytes(data)
        else:
            return None
        if len(payload) != width * height * 3:
            return None
        image = Image.frombytes("RGB", (width, height), payload)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80)
        return buffer.getvalue()


store = RosViewStore()

