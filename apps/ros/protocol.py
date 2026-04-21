import json
import struct
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class BinaryEnvelope:
    header: Dict[str, Any]
    payload: bytes


def encode_binary_envelope(header: Dict[str, Any], payload: bytes) -> bytes:
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(header_bytes)) + header_bytes + payload


def decode_binary_envelope(data: bytes) -> BinaryEnvelope:
    if len(data) < 4:
        raise ValueError("binary frame too short")
    header_len = struct.unpack(">I", data[:4])[0]
    if len(data) < 4 + header_len:
        raise ValueError("incomplete binary frame")
    header = json.loads(data[4 : 4 + header_len].decode("utf-8"))
    payload = data[4 + header_len :]
    return BinaryEnvelope(header=header, payload=payload)


def make_msg(
    op: str,
    *,
    request_id: str | None = None,
    ok: bool | None = None,
    data: Dict[str, Any] | None = None,
    error: str | None = None,
) -> Dict[str, Any]:
    msg: Dict[str, Any] = {"op": op}
    if request_id is not None:
        msg["request_id"] = request_id
    if ok is not None:
        msg["ok"] = ok
    if data is not None:
        msg["data"] = data
    if error is not None:
        msg["error"] = error
    return msg


def dumps_json(msg: Dict[str, Any]) -> str:
    return json.dumps(msg, separators=(",", ":"), ensure_ascii=False)


def loads_json(text: str) -> Dict[str, Any]:
    return json.loads(text)