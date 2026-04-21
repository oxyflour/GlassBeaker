import asyncio
import threading
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional
from uuid import uuid4

import websockets

import rclpy                                        # type: ignore
from rclpy.executors import MultiThreadedExecutor   # type: ignore
from rclpy.node import Node                         # type: ignore
from rclpy.qos import QoSProfile                    # type: ignore
from rclpy.serialization import deserialize_message, serialize_message    # type: ignore
from rosidl_runtime_py import message_to_ordereddict, set_message_fields  # type: ignore
from rosidl_runtime_py.utilities import get_message # type: ignore

from protocol import (
    decode_binary_envelope,
    dumps_json,
    encode_binary_envelope,
    loads_json,
    make_msg,
)


@dataclass
class RosSubscriptionHandle:
    topic: str
    ros_type: str
    subscription: Any
    mode: str


@dataclass
class RosPublisherHandle:
    topic: str
    ros_type: str
    publisher: Any
    mode: str


class RosWsBridge(Node):
    def __init__(self, server_url: str, peer_id: Optional[str] = None) -> None:
        super().__init__("ros_ws_bridge")
        self.server_url = server_url
        self.peer_id = peer_id or f"bridge-{uuid4().hex[:8]}"

        self._ros_subs: Dict[str, RosSubscriptionHandle] = {}
        self._ros_pubs: Dict[str, RosPublisherHandle] = {}

        self._loop = asyncio.new_event_loop()
        self._ws = None
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()
        self._send_lock = asyncio.Lock()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())

    async def _main(self) -> None:
        while rclpy.ok():
            try:
                async with websockets.connect(
                    self.server_url,
                    max_size=None,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    self._ws = ws
                    await ws.send(
                        dumps_json(
                            {
                                "op": "hello",
                                "role": "bridge",
                                "peer_id": self.peer_id,
                            }
                        )
                    )
                    ack = loads_json(str(await ws.recv()))
                    self.get_logger().info(f"connected to server: {ack}")
                    await self._recv_loop(ws)
            except Exception as exc:
                self.get_logger().error(f"websocket disconnected: {exc}")
                await asyncio.sleep(1.0)
            finally:
                self._ws = None

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            if isinstance(raw, str):
                msg = loads_json(raw)
                await self._handle_text(msg)
            else:
                await self._handle_binary(raw)

    async def _handle_text(self, msg: Dict[str, Any]) -> None:
        op = msg.get("op")
        request_id = msg.get("request_id")
        try:
            if op == "list_topics":
                topics = self.get_topic_names_and_types()
                await self._send_json(
                    make_msg(
                        "list_topics_result",
                        request_id=request_id,
                        ok=True,
                        data={
                            "topics": [
                                {"name": name, "types": types}
                                for name, types in topics
                            ]
                        },
                    )
                )
                return

            if op == "create_subscriber":
                topic = msg["topic"]
                ros_type = msg["ros_type"]
                mode = msg.get("mode", "json")
                qos_depth = int(msg.get("qos_depth", 10))
                self._create_subscriber(topic, ros_type, mode, qos_depth)
                await self._send_json(
                    make_msg(
                        "create_subscriber_result",
                        request_id=request_id,
                        ok=True,
                        data={"topic": topic, "ros_type": ros_type, "mode": mode},
                    )
                )
                return

            if op == "remove_subscriber":
                topic = msg["topic"]
                self._remove_subscriber(topic)
                await self._send_json(
                    make_msg(
                        "remove_subscriber_result",
                        request_id=request_id,
                        ok=True,
                        data={"topic": topic},
                    )
                )
                return

            if op == "create_publisher":
                topic = msg["topic"]
                ros_type = msg["ros_type"]
                mode = msg.get("mode", "json")
                qos_depth = int(msg.get("qos_depth", 10))
                self._create_publisher(topic, ros_type, mode, qos_depth)
                await self._send_json(
                    make_msg(
                        "create_publisher_result",
                        request_id=request_id,
                        ok=True,
                        data={"topic": topic, "ros_type": ros_type, "mode": mode},
                    )
                )
                return

            if op == "remove_publisher":
                topic = msg["topic"]
                self._remove_publisher(topic)
                await self._send_json(
                    make_msg(
                        "remove_publisher_result",
                        request_id=request_id,
                        ok=True,
                        data={"topic": topic},
                    )
                )
                return

            if op == "publish":
                topic = msg["topic"]
                data = msg.get("data", { })
                encoding = msg.get("encoding", "json")
                self._publish_json(topic, data, encoding)
                await self._send_json(
                    make_msg(
                        "publish_result",
                        request_id=request_id,
                        ok=True,
                        data={"topic": topic},
                    )
                )
                return

            await self._send_json(
                make_msg(
                    f"{op}_result",
                    request_id=request_id,
                    ok=False,
                    error=f"unsupported op: {op}",
                )
            )
        except Exception as exc:
            self.get_logger().exception("request failed")
            await self._send_json(
                make_msg(
                    f"{op}_result",
                    request_id=request_id,
                    ok=False,
                    error=str(exc),
                )
            )

    async def _handle_binary(self, raw: bytes) -> None:
        env = decode_binary_envelope(raw)
        header = env.header
        op = header.get("op")
        request_id = header.get("request_id")
        try:
            if op != "publish":
                return
            topic = header["topic"]
            encoding = header.get("encoding", "cdr")
            self._publish_binary(topic, env.payload, encoding)
            await self._send_json(
                make_msg(
                    "publish_result",
                    request_id=request_id,
                    ok=True,
                    data={"topic": topic},
                )
            )
        except Exception as exc:
            await self._send_json(
                make_msg(
                    "publish_result",
                    request_id=request_id,
                    ok=False,
                    error=str(exc),
                )
            )

    def _create_subscriber(self, topic: str, ros_type: str, mode: str, qos_depth: int) -> None:
        self._remove_subscriber(topic)
        msg_cls = get_message(ros_type)
        qos = QoSProfile(depth=qos_depth)
        callback = partial(self._on_ros_message, topic, ros_type, mode)
        sub = self.create_subscription(msg_cls, topic, callback, qos)
        self._ros_subs[topic] = RosSubscriptionHandle(topic, ros_type, sub, mode)
        self.get_logger().info(f"subscriber created: {topic} [{ros_type}] mode={mode}")

    def _remove_subscriber(self, topic: str) -> None:
        handle = self._ros_subs.pop(topic, None)
        if handle is not None:
            self.destroy_subscription(handle.subscription)
            self.get_logger().info(f"subscriber removed: {topic}")

    def _create_publisher(self, topic: str, ros_type: str, mode: str, qos_depth: int) -> None:
        self._remove_publisher(topic)
        msg_cls = get_message(ros_type)
        qos = QoSProfile(depth=qos_depth)
        pub = self.create_publisher(msg_cls, topic, qos)
        self._ros_pubs[topic] = RosPublisherHandle(topic, ros_type, pub, mode)
        self.get_logger().info(f"publisher created: {topic} [{ros_type}] mode={mode}")

    def _remove_publisher(self, topic: str) -> None:
        handle = self._ros_pubs.pop(topic, None)
        if handle is not None:
            self.destroy_publisher(handle.publisher)
            self.get_logger().info(f"publisher removed: {topic}")

    def _on_ros_message(self, topic: str, ros_type: str, mode: str, msg: Any) -> None:
        if mode == "json":
            payload = {
                "op": "topic_data",
                "topic": topic,
                "ros_type": ros_type,
                "encoding": "json",
                "data": message_to_ordereddict(msg),
            }
            asyncio.run_coroutine_threadsafe(self._send_json(payload), self._loop)
            return

        if mode == "cdr":
            data = serialize_message(msg)
            header = {
                "op": "topic_data",
                "topic": topic,
                "ros_type": ros_type,
                "encoding": "cdr",
                "content_type": "application/x-ros2-cdr",
            }
            asyncio.run_coroutine_threadsafe(self._send_binary(header, data), self._loop)
            return

        raise ValueError(f"unsupported subscriber mode: {mode}")

    def _publish_json(self, topic: str, data: Dict[str, Any], encoding: str) -> None:
        if topic not in self._ros_pubs:
            raise KeyError(f"publisher not found for topic: {topic}")
        handle = self._ros_pubs[topic]
        if encoding != "json":
            raise ValueError(f"json publish only supports encoding=json, got {encoding}")
        msg_cls = get_message(handle.ros_type)
        msg = msg_cls()
        set_message_fields(msg, data)
        handle.publisher.publish(msg)

    def _publish_binary(self, topic: str, payload: bytes, encoding: str) -> None:
        if topic not in self._ros_pubs:
            raise KeyError(f"publisher not found for topic: {topic}")
        handle = self._ros_pubs[topic]
        if encoding != "cdr":
            raise ValueError(f"binary publish only supports encoding=cdr, got {encoding}")
        msg_cls = get_message(handle.ros_type)
        msg = deserialize_message(payload, msg_cls)
        handle.publisher.publish(msg)

    async def _send_json(self, msg: Dict[str, Any]) -> None:
        ws = self._ws
        if ws is None:
            return
        async with self._send_lock:
            await ws.send(dumps_json(msg))

    async def _send_binary(self, header: Dict[str, Any], payload: bytes) -> None:
        ws = self._ws
        if ws is None:
            return
        async with self._send_lock:
            await ws.send(encode_binary_envelope(header, payload))


def main() -> None:
    rclpy.init()
    server_url = os.environ.get('WS_ADDR', "ws://127.0.0.1:13001/api/ros/ws")
    bridge = RosWsBridge(server_url=server_url)
    executor = MultiThreadedExecutor()
    executor.add_node(bridge)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
