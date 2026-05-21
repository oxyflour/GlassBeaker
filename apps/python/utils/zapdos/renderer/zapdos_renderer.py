from __future__ import annotations

import asyncio
import io
import os
import traceback
from collections.abc import AsyncIterator, Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from utils.zapdos.bundle.camera_specs import camera_name_to_index
from utils.camera_override import save_camera_overrides
from .base import RendererBackend
from .isaac_renderer import mjpeg_chunk, placeholder_jpeg

if TYPE_CHECKING:
    from utils.zapdos.bundle import RenderBundle

DEFAULT_FRAME_DELAY = 0.03333


class ZapdosRenderer:
    def __init__(
        self,
        *,
        backend: RendererBackend,
        bundle: "RenderBundle",
        render_size: tuple[int, int],
        is_active: Callable[[], bool],
        image_topic: Callable[[str], str],
        image_subscriptions: Mapping[str, object] | None = None,
        frame_delay: float = DEFAULT_FRAME_DELAY,
    ) -> None:
        self.backend = backend
        self.render_size = render_size
        self.is_active = is_active
        self.image_topic = image_topic
        self.image_subscriptions = image_subscriptions if image_subscriptions is not None else {}
        self.frame_delay = frame_delay
        self.set_bundle(bundle)

    def set_bundle(self, bundle: "RenderBundle") -> None:
        self.bundle = bundle
        self.camera_index = camera_name_to_index(bundle.cameras)
        self.last_frame_index = {camera.name: -1 for camera in bundle.cameras}

    async def wait_ready(self, timeout: float = 300.0) -> dict[str, Any]:
        if timeout == 300.0:
            return await self.backend.wait_ready()
        return await self.backend.wait_ready(timeout=timeout)

    def reload_scene(self, bundle: "RenderBundle", timeout: float = 30.0) -> None:
        if timeout == 30.0:
            self.backend.reload_scene(bundle)
        else:
            self.backend.reload_scene(bundle, timeout=timeout)
        self.set_bundle(bundle)

    def close(self, stop_remote: bool = True) -> None:
        if stop_remote:
            self.backend.close()
            return
        self.backend.close(stop_remote=False)

    def _encode_jpeg(self, frame: np.ndarray) -> bytes:
        data = io.BytesIO()
        Image.fromarray(frame).save(data, format="JPEG", quality=80)
        return data.getvalue()

    def _placeholder_frame(self, text: str, camera_count: int = 1) -> bytes:
        return placeholder_jpeg(
            self.render_size[0] * camera_count,
            self.render_size[1],
            text,
        )

    def _start_backend_if_supported(self) -> None:
        start = getattr(self.backend, "start", None)
        if callable(start):
            start()

    def should_publish_camera_images(self) -> bool:
        mode = os.getenv("ZAPDOS_PUBLISH_CAMERA_IMAGES", "").strip().lower()
        if mode in {"1", "true", "yes", "always"}:
            return True
        if mode in {"0", "false", "no", "never"}:
            return False
        return any(
            self.image_subscriptions.get(self.image_topic(camera.name))
            for camera in self.bundle.cameras
        )

    def image_messages(self) -> list[tuple[str, dict[str, Any]]]:
        messages: list[tuple[str, dict[str, Any]]] = []
        for camera in self.bundle.cameras:
            frame_state = self.backend.read(camera.name)
            if frame_state is None:
                continue
            index, frame = frame_state
            if index == self.last_frame_index.get(camera.name, -1):
                continue
            self.last_frame_index[camera.name] = index
            messages.append(
                (
                    self.image_topic(camera.name),
                    {
                        "header": {"frame_id": camera.frame_id},
                        "height": int(frame.shape[0]),
                        "width": int(frame.shape[1]),
                        "encoding": "rgb8",
                        "is_bigendian": 0,
                        "step": int(frame.shape[1] * 3),
                        "data": frame.tobytes(),
                    },
                )
            )
        return messages

    def snapshot(self, camera_name: str) -> bytes:
        if not self.backend.ready:
            self._start_backend_if_supported()
            return self._placeholder_frame("Waiting")
        frame_state = self.backend.read(camera_name)
        if frame_state is None:
            return self._placeholder_frame("Waiting" if self.is_active() else "Closed")
        _, frame = frame_state
        return self._encode_jpeg(frame)

    def save_camera_override(self) -> dict[str, Any]:
        path, saved = save_camera_overrides(self.backend.snapshot_cameras())
        return {"ok": True, "saved": saved, "path": str(path)}

    def _composite_camera_names(self) -> list[str]:
        return [camera.name for camera in self.bundle.cameras[:3]]

    def _read_composite_frame(self, camera_names: list[str]) -> tuple[int, np.ndarray] | None:
        frames: list[np.ndarray] = []
        frame_index = -1
        for camera_name in camera_names:
            frame_state = self.backend.read(camera_name)
            if frame_state is None:
                return None
            index, frame = frame_state
            if frame_index < 0:
                frame_index = index
            frames.append(frame)
        if frame_index < 0:
            return None
        return frame_index, np.concatenate(frames, axis=1)

    async def render(self, camera_name: str) -> AsyncIterator[bytes]:
        last_frame_index = -1
        while self.is_active():
            while not self.backend.ready:
                self._start_backend_if_supported()
                yield mjpeg_chunk(self._placeholder_frame("Waiting"))
                await asyncio.sleep(1)
            try:
                frame_state = self.backend.read(camera_name)
                if frame_state is None:
                    await asyncio.sleep(self.frame_delay)
                    continue
                index, frame = frame_state
                if index == last_frame_index:
                    await asyncio.sleep(self.frame_delay)
                    continue
                last_frame_index = index
                yield mjpeg_chunk(self._encode_jpeg(frame))
                await asyncio.sleep(self.frame_delay)
            except Exception:
                traceback.print_exc()
                await asyncio.sleep(1)
        yield mjpeg_chunk(self._placeholder_frame("Closed"))

    async def render_multi_camera(self) -> AsyncIterator[bytes]:
        last_frame_index = -1
        camera_names = self._composite_camera_names()
        if not camera_names:
            yield mjpeg_chunk(self._placeholder_frame("No Cameras"))
            return
        while self.is_active():
            while not self.backend.ready:
                self._start_backend_if_supported()
                yield mjpeg_chunk(self._placeholder_frame("Waiting", len(camera_names)))
                await asyncio.sleep(1)
            try:
                frame_state = self._read_composite_frame(camera_names)
                if frame_state is None:
                    await asyncio.sleep(self.frame_delay)
                    continue
                index, frame = frame_state
                if index == last_frame_index:
                    await asyncio.sleep(self.frame_delay)
                    continue
                last_frame_index = index
                yield mjpeg_chunk(self._encode_jpeg(frame))
                await asyncio.sleep(self.frame_delay)
            except Exception:
                traceback.print_exc()
                await asyncio.sleep(1)
        yield mjpeg_chunk(self._placeholder_frame("Closed", len(camera_names)))
