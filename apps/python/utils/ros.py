import asyncio
import json
import sys
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set
from uuid import uuid4

from fastapi import WebSocket, HTTPException

sys.path.append(os.path.normpath(f"{__file__}/../../../"))
from ros.protocol import decode_binary_envelope, dumps_json, make_msg, encode_binary_envelope


@dataclass
class Peer:
    peer_id: str
    role: str
    websocket: WebSocket
    subscriptions: Set[str] = field(default_factory=set)


class Hub:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.peers: Dict[str, Peer] = {}
        self.bridge_id: Optional[str] = None
        self.pending_rpcs: Dict[str, asyncio.Future] = {}
        self.topic_subscribers: Dict[str, Set[str]] = defaultdict(set)

    async def register(self, websocket: WebSocket) -> Peer:
        await websocket.accept()
        hello = await websocket.receive_text()
        hello_msg = json.loads(hello)
        if hello_msg.get("op") != "hello":
            await websocket.close(code=4400, reason="first frame must be hello")
            raise HTTPException(400, "first websocket frame must be hello")

        role = hello_msg.get("role")
        if role not in {"bridge", "client"}:
            await websocket.close(code=4400, reason="invalid role")
            raise HTTPException(400, "invalid websocket role")

        peer_id = hello_msg.get("peer_id") or f"{role}-{uuid4().hex[:8]}"
        peer = Peer(peer_id=peer_id, role=role, websocket=websocket)

        async with self._lock:
            self.peers[peer_id] = peer
            if role == "bridge":
                self.bridge_id = peer_id

        await websocket.send_text(
            dumps_json(
                make_msg(
                    "hello_ack",
                    ok=True,
                    data={"peer_id": peer_id, "role": role},
                )
            )
        )
        return peer

    async def unregister(self, peer: Peer) -> None:
        async with self._lock:
            self.peers.pop(peer.peer_id, None)
            for topic in list(peer.subscriptions):
                self.topic_subscribers[topic].discard(peer.peer_id)
                if not self.topic_subscribers[topic]:
                    self.topic_subscribers.pop(topic, None)

            if self.bridge_id == peer.peer_id:
                self.bridge_id = None
                for request_id, fut in list(self.pending_rpcs.items()):
                    if not fut.done():
                        fut.set_exception(RuntimeError("bridge disconnected"))
                    self.pending_rpcs.pop(request_id, None)

    async def get_bridge(self) -> Peer:
        async with self._lock:
            if self.bridge_id is None:
                raise RuntimeError("no bridge connected")
            bridge = self.peers.get(self.bridge_id)
            if bridge is None:
                raise RuntimeError("bridge not available")
            return bridge

    async def send_to_bridge(self, msg: Dict[str, Any]) -> None:
        bridge = await self.get_bridge()
        await bridge.websocket.send_text(dumps_json(msg))

    async def rpc_bridge(self, msg: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any]:
        request_id = msg.get("request_id") or uuid4().hex
        msg["request_id"] = request_id

        fut = asyncio.get_running_loop().create_future()
        self.pending_rpcs[request_id] = fut
        try:
            await self.send_to_bridge(msg)
            result = await asyncio.wait_for(fut, timeout=timeout)
            return result
        finally:
            self.pending_rpcs.pop(request_id, None)

    async def handle_bridge_text(self, peer: Peer, msg: Dict[str, Any]) -> None:
        op = msg.get("op", '')

        if op.endswith("_result"):
            request_id = msg.get("request_id", '')
            fut = self.pending_rpcs.get(request_id)
            if fut is not None and not fut.done():
                fut.set_result(msg)
            return

        if op == "topic_data":
            topic = msg.get("topic")
            data = msg.get("data")
            if topic:
                await self.broadcast_json(
                    topic,
                    {
                        "op": "topic_data",
                        "topic": topic,
                        "encoding": "json",
                        "data": data,
                    },
                )
            return

    async def handle_bridge_binary(self, peer: Peer, payload: bytes) -> None:
        env = decode_binary_envelope(payload)
        header = env.header
        if header.get("op") != "topic_data":
            return
        topic = header.get("topic")
        if not topic:
            return
        await self.broadcast_binary(topic, header, env.payload)

    async def handle_client_text(self, peer: Peer, msg: Dict[str, Any]) -> None:
        op = msg.get("op")
        request_id = msg.get("request_id")

        if op == "subscribe_topic":
            topic = msg["topic"]
            peer.subscriptions.add(topic)
            self.topic_subscribers[topic].add(peer.peer_id)
            await peer.websocket.send_text(
                dumps_json(
                    make_msg(
                        "subscribe_topic_result",
                        request_id=request_id,
                        ok=True,
                        data={"topic": topic},
                    )
                )
            )
            return

        if op == "unsubscribe_topic":
            topic = msg["topic"]
            peer.subscriptions.discard(topic)
            self.topic_subscribers[topic].discard(peer.peer_id)
            if not self.topic_subscribers[topic]:
                self.topic_subscribers.pop(topic, None)
            await peer.websocket.send_text(
                dumps_json(
                    make_msg(
                        "unsubscribe_topic_result",
                        request_id=request_id,
                        ok=True,
                        data={"topic": topic},
                    )
                )
            )
            return

        if op in {
            "list_topics",
            "create_subscriber",
            "remove_subscriber",
            "create_publisher",
            "remove_publisher",
            "publish",
        }:
            result = await self.rpc_bridge(msg)
            await peer.websocket.send_text(dumps_json(result))
            return

        await peer.websocket.send_text(
            dumps_json(
                make_msg(
                    f"{op}_result",
                    request_id=request_id,
                    ok=False,
                    error=f"unsupported op: {op}",
                )
            )
        )

    async def broadcast_json(self, topic: str, msg: Dict[str, Any]) -> None:
        peer_ids = list(self.topic_subscribers.get(topic, set()))
        for peer_id in peer_ids:
            peer = self.peers.get(peer_id)
            if peer is None:
                continue
            await peer.websocket.send_text(dumps_json(msg))

    async def broadcast_binary(self, topic: str, header: Dict[str, Any], payload: bytes) -> None:
        peer_ids = list(self.topic_subscribers.get(topic, set()))

        binary = encode_binary_envelope(header, payload)
        for peer_id in peer_ids:
            peer = self.peers.get(peer_id)
            if peer is None:
                continue
            await peer.websocket.send_bytes(binary)


hub = Hub()
