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
        self._scene_params = None
        self._scene_dict: dict[str, Any] | None = None
        self._mesh_vertices: dict[str, np.ndarray] = {}
        self._snapshots: list[dict[str, Any]] = []
        self._mi = None
        self._pose: dict[str, list[float]] = {}
        self._pose_version = 0
        self._loaded_pose_version = -1

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def ready(self) -> bool:
        return self._ready and self._error is None

    async def wait_ready(self, timeout: float = 300.0) -> dict[str, Any]:
        self.start()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._error is not None:
                self._raise_error()
            if self.ready:
                return self.status()
            await asyncio.sleep(0.01)
        raise TimeoutError(f"MitsubaRenderer did not produce a frame in {timeout:.0f}s")

    def start(self) -> None:
        if self._error is not None:
            self._raise_error()
        if self._thread is not None:
            return
        try:
            self._start()
        except BaseException as exc:
            self._error = exc
            self._raise_error()

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
            self._scene = None
            self._scene_params = None
            self._scene_dict = None
            self._mesh_vertices = {}
            self._loaded_pose_version = -1
        self._load_scene()

    def update_pose(self, pose: dict[str, list[float]]) -> None:
        with self._lock:
            if pose == self._pose:
                return
            self._pose = {name: list(matrix) for name, matrix in pose.items()}
            self._pose_version += 1

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
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mitsuba-renderer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._load_scene()
        except BaseException as exc:
            self._error = exc
            return
        self._render_loop()

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
        with self._lock:
            self._scene_dict = scene_dict
        self._load_pose_scene()

    def _render_loop(self) -> None:
        delay = 1.0 / max(float(self.render_hz), 1.0)
        while not self._stop.is_set():
            try:
                self._load_pose_scene()
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

    def _load_pose_scene(self) -> None:
        with self._lock:
            scene_dict = self._scene_dict
            pose = dict(self._pose)
            pose_version = self._pose_version
            if scene_dict is None or pose_version == self._loaded_pose_version:
                return
        posed_scene = self._scene_for_pose(scene_dict, pose)
        if self._scene is None or self._scene_params is None:
            self._load_full_pose_scene(posed_scene)
        else:
            self._update_scene_params(posed_scene)
        with self._lock:
            self._loaded_pose_version = pose_version

    def _load_full_pose_scene(self, posed_scene: dict[str, Any]) -> None:
        self._scene = self._mi.load_dict(apply_mitsuba_transforms(posed_scene, self._mi))
        traverse = getattr(self._mi, "traverse", None)
        self._scene_params = traverse(self._scene) if callable(traverse) else None
        self._mesh_vertices = {}
        if self._scene_params is None:
            return
        for key, value in posed_scene.items():
            if not isinstance(value, dict) or value.get("type") != "ply":
                continue
            if not isinstance(value.get("_zapdos_body"), str):
                continue
            vertex_key = f"{key}.vertex_positions"
            filename = value.get("filename")
            if vertex_key in self._scene_params and isinstance(filename, str):
                self._mesh_vertices[key] = _read_ply_vertices(Path(filename))

    def _update_scene_params(self, posed_scene: dict[str, Any]) -> None:
        params = self._scene_params
        if params is None:
            self._load_full_pose_scene(posed_scene)
            return
        changed = False
        for key, value in posed_scene.items():
            if not isinstance(value, dict):
                continue
            dynamic = isinstance(value.get("_zapdos_body"), str)
            look_at = value.get("to_world_look_at")
            transform_key = f"{key}.to_world"
            if dynamic and look_at is not None and transform_key in params:
                params[transform_key] = self._mi.ScalarTransform4f.look_at(
                    origin=look_at["origin"],
                    target=look_at["target"],
                    up=look_at["up"],
                )
                changed = True
            matrix = value.get("to_world_matrix")
            vertices = self._mesh_vertices.get(key)
            vertex_key = f"{key}.vertex_positions"
            if dynamic and matrix is not None and vertices is not None and vertex_key in params:
                transform = np.asarray(matrix, dtype=float).reshape(4, 4)
                homogeneous = np.concatenate(
                    [vertices, np.ones((vertices.shape[0], 1), dtype=float)],
                    axis=1,
                )
                params[vertex_key] = (homogeneous @ transform.T)[:, :3].reshape(-1)
                changed = True
        if changed:
            params.update()

    def _scene_for_pose(self, scene_dict: dict[str, Any], pose: dict[str, list[float]]) -> dict[str, Any]:
        scene = dict(scene_dict)
        for key, value in scene_dict.items():
            if not isinstance(value, dict):
                continue
            body = value.get("_zapdos_body")
            local_matrix = value.get("_zapdos_body_local_matrix")
            body_matrix = pose.get(body) if isinstance(body, str) else None
            if body_matrix is None:
                continue
            entry = dict(value)
            body_transform = np.asarray(body_matrix, dtype=float).reshape(4, 4).T
            if local_matrix is not None:
                entry["to_world_matrix"] = (
                    body_transform
                    @ np.asarray(local_matrix, dtype=float).reshape(4, 4)
                ).tolist()
            self._update_camera_pose(entry, body_transform)
            scene[key] = entry
        return scene

    def _update_camera_pose(self, entry: dict[str, Any], body_transform: np.ndarray) -> None:
        origin = entry.get("_zapdos_camera_local_origin")
        target = entry.get("_zapdos_camera_local_target")
        up = entry.get("_zapdos_camera_local_up")
        if origin is None or target is None or up is None:
            return
        world_origin = body_transform @ np.asarray(origin, dtype=float).reshape(4)
        world_target = body_transform @ np.asarray(target, dtype=float).reshape(4)
        world_up = body_transform[:3, :3] @ np.asarray(up, dtype=float).reshape(3)
        entry["to_world_look_at"] = {
            "origin": world_origin[:3].tolist(),
            "target": world_target[:3].tolist(),
            "up": world_up.tolist(),
        }

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


def _read_ply_vertices(path: Path) -> np.ndarray:
    vertex_count: int | None = None
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
            if parts == ["end_header"]:
                break
        if vertex_count is None:
            raise RuntimeError(f"PLY missing vertex count: {path}")
        vertices = [
            [float(value) for value in file.readline().split()[:3]]
            for _ in range(vertex_count)
        ]
    return np.asarray(vertices, dtype=float)
