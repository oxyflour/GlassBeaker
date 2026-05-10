from __future__ import annotations

import ipaddress
import os
import pickle
from urllib.parse import urlparse
from typing import Any

DEFAULT_WS_ADDR = "ws://127.0.0.1:13001/api/ros/ws"


def bridge_server_url() -> str:
    return os.environ.get("WS_ADDR", DEFAULT_WS_ADDR)


def bridge_connect_kwargs(server_url: str) -> dict[str, object]:
    hostname = (urlparse(server_url).hostname or "").strip().lower()
    if _is_loopback_host(hostname):
        return {"proxy": None}
    return {}


def _is_loopback_host(hostname: str) -> bool:
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


async def dispatch_bridge_request(session: Any, payload: bytes) -> tuple[str, Exception | None, Any]:
    method, args, call = pickle.loads(payload)
    err = None
    ret = None
    try:
        ret = await session.call(method, *args)
    except Exception as exception:
        err = exception
    return call, err, ret


__all__ = [
    "DEFAULT_WS_ADDR",
    "bridge_connect_kwargs",
    "bridge_server_url",
    "dispatch_bridge_request",
]
