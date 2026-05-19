from __future__ import annotations

import asyncio
import importlib
import shutil
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from utils.zapdos.bundle.camera_specs import camera_name_to_index
from utils.zapdos.renderer.mitsuba_scene import apply_mitsuba_transforms, build_mitsuba_scene_dict

if TYPE_CHECKING:
    from utils.zapdos.bundle import RenderBundle

REPO_ROOT = Path(__file__).resolve().parents[5]
MITSUBA_VARIANT = "cuda_ad_rgb"
MITSUBA_HINT = "Mitsuba CUDA rendering failed. Ensure the cuda_ad_rgb backend and CUDA runtime are available."


def load_mitsuba():
    mi = importlib.import_module("mitsuba")
    mi.set_variant(MITSUBA_VARIANT)
    return mi


class MitsubaRenderer:
    def __init__(
        self,
        sess: str,
        bundle: "RenderBundle",
        width: int,
        height: int,
        render_hz: float,
        headless: bool,
        ros_domain_id: int,
    ) -> None:
        tag = "".join(ch if ch.isalnum() else "_" for ch in sess)[:20] or "default"
        self.bundle = bundle
        self.width = width
        self.height = height
        self.render_hz = render_hz
        self.headless = headless
        self.ros_domain_id = ros_domain_id
        self.work_dir = REPO_ROOT / "apps" / "python" / "tmp" / f"mitsuba_{tag}"
        self.camera_index = camera_name_to_index(bundle.cameras)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frames: dict[str, tuple[int, np.ndarray]] = {}
        self._frame_index = 0
        self._error: BaseException | None = None
        self._ready = False
        self._scene = None
        self._snapshots: list[dict[str, Any]] = []
        self._mi = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def ready(self) -> bool:
        return self._ready and self._error is None

    async def wait_ready(self, timeout: float = 300.0) -> dict[str, Any]:
        if self._thread is None:
            try:
                self._start()
            except BaseException as exc:
                self._error = exc
                self._raise_error()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._error is not None:
                self._raise_error()
            if self.ready:
                return self.status()
            await asyncio.sleep(0.01)
        raise TimeoutError(f"MitsubaRenderer did not produce a frame in {timeout:.0f}s")

    def read(self, camera_name: str) -> tuple[int, np.ndarray] | None:
        if camera_name not in self.camera_index or not self.running:
            return None
        with self._lock:
            frame = self._frames.get(camera_name)
            if frame is None:
                return None
            index, image = frame
            return index, image.copy()

    def reload_scene(self, bundle: "RenderBundle", timeout: float = 30.0) -> None:
        del timeout
        self.bundle = bundle
        self.camera_index = camera_name_to_index(bundle.cameras)
        with self._lock:
            self._frames.clear()
            self._ready = False
            self._frame_index = 0
        self._load_scene()

    def snapshot_cameras(self, timeout: float = 5.0) -> list[dict[str, Any]]:
        del timeout
        return list(self._snapshots)

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "ready": self.ready,
            "backend": "mitsuba",
            "ros_domain_id": self.ros_domain_id,
            "width": self.width,
            "height": self.height,
            "work_dir": str(self.work_dir),
            "error": None if self._error is None else str(self._error),
        }

    def close(self, stop_remote: bool = True) -> None:
        del stop_remote
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._lock:
            self._frames.clear()
            self._ready = False

    def _start(self) -> None:
        self._load_scene()
        self._stop.clear()
        self._thread = threading.Thread(target=self._render_loop, name="mitsuba-renderer", daemon=True)
        self._thread.start()

    def _load_scene(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        mesh_dir = self.work_dir / "meshes"
        if mesh_dir.exists():
            shutil.rmtree(mesh_dir)
        scene_dict, self._snapshots = build_mitsuba_scene_dict(
            self.bundle,
            mesh_dir,
            self.width,
            self.height,
            spp=max(1, int(self.render_hz // 10) or 1),
        )
        self._mi = self._mi or load_mitsuba()
        self._scene = self._mi.load_dict(apply_mitsuba_transforms(scene_dict, self._mi))

    def _render_loop(self) -> None:
        delay = 1.0 / max(float(self.render_hz), 1.0)
        while not self._stop.is_set():
            try:
                for sensor, camera in enumerate(self.bundle.cameras):
                    image = self._render_camera(sensor)
                    with self._lock:
                        self._frame_index += 1
                        self._frames[camera.name] = (self._frame_index, image)
                        self._ready = True
            except BaseException as exc:
                self._error = exc
                return
            self._stop.wait(delay)

    def _render_camera(self, sensor: int) -> np.ndarray:
        rendered = self._mi.render(self._scene, sensor=sensor, spp=max(1, int(self.render_hz // 10) or 1))
        frame = np.asarray(rendered)
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        frame = frame[:, :, :3]
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.maximum(frame, 0.0)
            frame = frame / (1.0 + frame)
            frame = np.power(frame, 1.0 / 2.2) * 255.0
        return np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)

    def _raise_error(self) -> None:
        error = self._error
        if error is None:
            return
        raise RuntimeError(f"{MITSUBA_HINT} Detail: {error}") from error
