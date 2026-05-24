from __future__ import annotations

import io
import struct
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def write_mp4(path: Path, frames: list[np.ndarray], *, fps: int) -> None:
    if not frames:
        raise ValueError("frames must not be empty")
    if fps <= 0:
        raise ValueError("fps must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_frames = [_rgb8_frame(frame) for frame in frames]
    imageio.mimsave(path, rgb_frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)


def write_mjpeg_avi(path: Path, frames: list[np.ndarray], *, fps: int) -> None:
    if not frames:
        raise ValueError("frames must not be empty")
    if fps <= 0:
        raise ValueError("fps must be positive")
    encoded = [_jpeg_bytes(frame) for frame in frames]
    height, width = _frame_size(frames[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        _write_avi(handle, encoded, width=width, height=height, fps=fps)


def _jpeg_bytes(frame: np.ndarray) -> bytes:
    image = Image.fromarray(_rgb8_frame(frame), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _rgb8_frame(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("frame must be an RGB-like array")
    array = array[:, :, :3]
    if np.issubdtype(array.dtype, np.floating):
        array = np.maximum(array, 0.0)
        array = array / (1.0 + array)
        array = np.power(array, 1.0 / 2.2) * 255.0
    return np.asarray(np.clip(array, 0, 255), dtype=np.uint8)


def _frame_size(frame: np.ndarray) -> tuple[int, int]:
    array = np.asarray(frame)
    if array.ndim != 3:
        raise ValueError("frame must have height, width, channels")
    return int(array.shape[0]), int(array.shape[1])


def _write_avi(handle, frames: list[bytes], *, width: int, height: int, fps: int) -> None:
    max_frame = max(len(frame) for frame in frames)
    movi_payload = bytearray()
    offsets: list[tuple[int, int]] = []
    for frame in frames:
        offsets.append((4 + len(movi_payload), len(frame)))
        movi_payload.extend(_chunk(b"00dc", frame))
    movi = _list_chunk(b"movi", bytes(movi_payload))
    idx = _idx1(offsets)
    hdrl = _list_chunk(b"hdrl", _avih(len(frames), width, height, fps, max_frame) + _strl(len(frames), width, height, fps, max_frame))
    body = hdrl + movi + idx
    handle.write(b"RIFF")
    handle.write(struct.pack("<I", len(body) + 4))
    handle.write(b"AVI ")
    handle.write(body)


def _avih(frame_count: int, width: int, height: int, fps: int, max_frame: int) -> bytes:
    payload = struct.pack(
        "<IIIIIIIIIIIIII",
        int(1_000_000 / fps), max_frame * fps, 0, 0x10, frame_count, 0, 1,
        max_frame, width, height, 0, 0, 0, 0,
    )
    return _chunk(b"avih", payload)


def _strl(frame_count: int, width: int, height: int, fps: int, max_frame: int) -> bytes:
    stream = struct.pack(
        "<4s4sIHHIIIIIIIIiiii",
        b"vids", b"MJPG", 0, 0, 0, 0, 1, fps, 0, frame_count,
        max_frame, 0xFFFFFFFF, 0, 0, 0, width, height,
    )
    bitmap = struct.pack("<IiiHH4sIiiII", 40, width, height, 1, 24, b"MJPG", width * height * 3, 0, 0, 0, 0)
    return _list_chunk(b"strl", _chunk(b"strh", stream) + _chunk(b"strf", bitmap))


def _idx1(offsets: list[tuple[int, int]]) -> bytes:
    payload = bytearray()
    for offset, size in offsets:
        payload.extend(struct.pack("<4sIII", b"00dc", 0x10, offset, size))
    return _chunk(b"idx1", bytes(payload))


def _list_chunk(kind: bytes, payload: bytes) -> bytes:
    return b"LIST" + struct.pack("<I", len(payload) + 4) + kind + payload + (b"\0" if len(payload) % 2 else b"")


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return kind + struct.pack("<I", len(payload)) + payload + (b"\0" if len(payload) % 2 else b"")
