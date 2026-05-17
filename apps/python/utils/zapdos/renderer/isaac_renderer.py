from __future__ import annotations

import asyncio
import io
import json
import os
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import ProxyHandler, Request, build_opener, urlopen

import mujoco  # type: ignore
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.zapdos.bundle.camera_specs import camera_name_to_index, cameras_json
from utils.zapdos.renderer.control_channel import send_control_request
from utils.zapdos.renderer.frame_buffer import SharedFrameBuffer
from utils.zapdos.renderer.isaac_process import (
    ISAAC_PYTHON,
    RENDERER_ENTRY,
    REPO_ROOT,
    close_local_renderer,
    format_isaacsim_failure,
    setup_renderer_env,
    spawn_local_renderer,
)

if TYPE_CHECKING:
    from utils.zapdos.bundle import RenderBundle

ISAAC_API_URL = os.getenv("ISAAC_API_URL", "http://127.0.0.1:13000/api/isaac")


def mjpeg_chunk(payload: bytes) -> bytes:
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"


def _isaac_request(
    method: str,
    payload: dict[str, Any] | None = None,
    query: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    url = ISAAC_API_URL
    if query:
        url = f"{url}?{urlencode(query)}"
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(url, data=body, headers=headers, method=method)
    try:
        if (urlparse(url).hostname or "").lower() in {"127.0.0.1", "localhost", "::1"}:
            opener = build_opener(ProxyHandler({}))
            response_ctx = opener.open(request, timeout=timeout)
        else:
            response_ctx = urlopen(request, timeout=timeout)
        with response_ctx as response:
            data = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Isaac API {method} failed: {exc.code} {detail or exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError(f"Isaac API unavailable at {ISAAC_API_URL}: {exc.reason}") from exc
    return json.loads(data.decode("utf-8")) if data else {}


@lru_cache(maxsize=8)
def placeholder_jpeg(width: int, height: int, text: str) -> bytes:
    image = Image.new("RGB", (width, height), (20, 24, 32))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", size=int(height * 0.2))
    draw.text((16, int(height * 0.4)), text, font=font, fill=(236, 239, 244))
    data = io.BytesIO()
    image.save(data, format="JPEG", quality=80)
    return data.getvalue()


def tf_message(model, data, frame_names: dict[str, str] | None = None) -> dict[str, Any]:
    transforms: list[dict[str, Any]] = []
    quat = np.empty(4)
    for body_id in range(1, model.nbody):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)  # type: ignore
        if not body:
            continue
        child_frame = frame_names.get(body, body) if frame_names else body
        mujoco.mju_mat2Quat(quat, np.array(data.xmat[body_id], dtype=float).reshape(-1))  # type: ignore
        pos = np.array(data.xpos[body_id], dtype=float)
        transforms.append({
            "header": {"frame_id": "world"},
            "child_frame_id": child_frame,
            "transform": {
                "translation": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
                "rotation": {"w": float(quat[0]), "x": float(quat[1]), "y": float(quat[2]), "z": float(quat[3])},
            },
        })
    return {"transforms": transforms}


class IsaacRenderer:
    def __init__(self, sess: str, bundle: "RenderBundle", width: int, height: int, render_hz: float, headless: bool, ros_domain_id: int) -> None:
        tag = "".join(ch if ch.isalnum() else "_" for ch in sess)[:20] or "default"
        self.bundle = bundle
        self.width = width
        self.height = height
        self.render_hz = render_hz
        self.headless = headless
        self.ros_domain_id = ros_domain_id
        self.proc_id: str | None = None
        self.proc_pid: int | None = None
        self.proc = None
        self._running = False
        self.shm_name = f"glassbeaker_{tag}_frames"
        self.log_path = REPO_ROOT / "apps" / "python" / "tmp" / f"renderer_{tag}.log"
        self.control_dir = REPO_ROOT / "apps" / "python" / "tmp" / f"renderer_{tag}_ipc"
        self.frame_buffer = SharedFrameBuffer(self.shm_name, len(bundle.cameras), self.width, self.height)
        self.camera_index = camera_name_to_index(bundle.cameras)
        self._control_lock = threading.Lock()
        self._spawn()

    @property
    def shm(self):
        frame_buffer = self.__dict__.get("frame_buffer")
        if frame_buffer is None:
            return self.__dict__.get("_shm")
        return frame_buffer.shm

    @shm.setter
    def shm(self, value):
        frame_buffer = self.__dict__.get("frame_buffer")
        if frame_buffer is None:
            self.__dict__["_shm"] = value
            return
        frame_buffer.shm = value

    @property
    def frame_counter(self):
        frame_buffer = self.__dict__.get("frame_buffer")
        if frame_buffer is None:
            return self.__dict__.get("_frame_counter")
        return frame_buffer.frame_counter

    @frame_counter.setter
    def frame_counter(self, value):
        frame_buffer = self.__dict__.get("frame_buffer")
        if frame_buffer is None:
            self.__dict__["_frame_counter"] = value
            return
        frame_buffer.frame_counter = value

    @property
    def frames(self):
        frame_buffer = self.__dict__.get("frame_buffer")
        if frame_buffer is None:
            return self.__dict__.get("_frames")
        return frame_buffer.frames

    @frames.setter
    def frames(self, value):
        frame_buffer = self.__dict__.get("frame_buffer")
        if frame_buffer is None:
            self.__dict__["_frames"] = value
            return
        frame_buffer.frames = value

    @property
    def running(self) -> bool:
        return self._running and (self.proc_id is not None or self.proc is not None)

    @property
    def ready(self) -> bool:
        return self.shm is not None and self.frame_counter is not None and self.frames is not None

    def _bind_shm(self) -> None:
        self.frame_buffer.bind()

    def _refresh_process_state(self) -> bool:
        if self.proc is not None:
            self._running = self.proc.poll() is None
            self.proc_pid = self.proc.pid if self._running else self.proc_pid
            return self._running
        if self.proc_id is None:
            self._running = False
            return False
        state = _isaac_request("GET", query={"id": self.proc_id}, timeout=5.0)
        self._running = bool(state.get("running"))
        pid = state.get("pid")
        self.proc_pid = int(pid) if pid is not None else self.proc_pid
        return self._running

    def _spawn(self) -> None:
        if not ISAAC_PYTHON.exists():
            raise FileNotFoundError(f"Isaac Python not found: {ISAAC_PYTHON}")
        if not RENDERER_ENTRY.exists():
            raise FileNotFoundError(f"Renderer entry not found: {RENDERER_ENTRY}")
        self.close()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.control_dir.mkdir(parents=True, exist_ok=True)
        env = setup_renderer_env(os.environ.copy(), self.ros_domain_id)
        env["GB_RENDERER_CONTROL_DIR"] = str(self.control_dir)
        cmd = [
            str(ISAAC_PYTHON),
            "-u",
            str(RENDERER_ENTRY),
            "--scene-usd", str(self.bundle.render_scene_usda),
            "--num-envs", "1",
            "--render-hz", str(self.render_hz),
            "--cam-width", str(self.width),
            "--cam-height", str(self.height),
            "--main-cam-prim", self.bundle.cameras[0].prim,
            "--cameras-json", cameras_json(self.bundle.cameras),
            "--shm-name", self.shm_name,
            "--ros-domain-id", str(self.ros_domain_id),
            *([] if os.environ.get("DEBUG_ISAAC_SHOW") else ["--headless"]),
        ]
        if not self.headless:
            cmd.pop()
        try:
            state = _isaac_request("POST", {
                "id": self.shm_name,
                "cmd": cmd,
                "cwd": str(REPO_ROOT / "apps" / "isaac"),
                "env": env,
                "logPath": str(self.log_path),
            })
            self.proc_id = str(state.get("id") or self.shm_name)
            self.proc_pid = int(state["pid"]) if state.get("pid") is not None else None
            self._running = bool(state.get("running", True))
        except RuntimeError:
            self.proc = spawn_local_renderer(
                cmd,
                cwd=str(REPO_ROOT / "apps" / "isaac"),
                env=env,
                log_path=self.log_path,
            )
            self.proc_id = None
            self.proc_pid = self.proc.pid
            self._running = True

    async def wait_ready(self, timeout: float = 300.0) -> dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self._refresh_process_state():
                await asyncio.sleep(1)
                try:
                    raise RuntimeError(format_isaacsim_failure(
                        "IsaacSim failed to start",
                        self.log_path,
                        f"renderer exited before creating shared memory '{self.shm_name}'",
                    ))
                finally:
                    self.close()
            try:
                self._bind_shm()
                return self.status()
            except FileNotFoundError:
                await asyncio.sleep(5)
        try:
            raise TimeoutError(format_isaacsim_failure(
                "IsaacSim did not become ready",
                self.log_path,
                f"renderer did not create shared memory '{self.shm_name}' in {timeout:.0f}s",
            ))
        finally:
            self.close()

    def read(self, camera_name: str) -> tuple[int, np.ndarray] | None:
        if not self.running:
            return None
        try:
            self._bind_shm()
        except FileNotFoundError:
            return None
        frame_buffer = self.__dict__.get("frame_buffer")
        if frame_buffer is None:
            if self.frame_counter is None or self.frames is None:
                return None
            return int(self.frame_counter[0]), self.frames[0, self.camera_index[camera_name]].copy()
        return frame_buffer.read(self.camera_index[camera_name])

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "ready": self.ready,
            "proc_id": self.proc_id,
            "proc_pid": self.proc_pid,
            "ros_domain_id": self.ros_domain_id,
            "shm_name": self.shm_name,
            "width": self.width,
            "height": self.height,
            "log_path": str(self.log_path),
        }

    def _control_request(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        if not hasattr(self, "_control_lock"):
            self._control_lock = threading.Lock()
        return send_control_request(
            control_dir=self.control_dir,
            payload=payload,
            timeout=timeout,
            refresh_process_state=self._refresh_process_state,
            quit_message=lambda detail: format_isaacsim_failure(
                "IsaacSim quit unexpectedly",
                self.log_path,
                detail,
            ),
            control_lock=self._control_lock,
        )

    def snapshot_cameras(self, timeout: float = 5.0) -> list[dict[str, Any]]:
        response = self._control_request({"op": "snapshot_cameras"}, timeout)
        cameras = response.get("cameras")
        if not isinstance(cameras, list):
            raise RuntimeError("renderer snapshot returned invalid cameras")
        return cameras

    def reload_scene(self, bundle: "RenderBundle", timeout: float = 30.0) -> None:
        self._control_request({
            "op": "reload_scene",
            "scene_usd": str(bundle.render_scene_usda),
            "cameras": [
                {"name": camera.name, "prim": camera.prim}
                for camera in bundle.cameras
            ],
        }, timeout)
        self.bundle = bundle
        self.camera_index = camera_name_to_index(bundle.cameras)

    def close(self, stop_remote: bool = True) -> None:
        if self.proc is not None:
            close_local_renderer(self.proc)
            self.proc = None
            self._running = False
            self.proc_pid = None
        if self.proc_id is not None and stop_remote:
            try:
                _isaac_request("DELETE", {"id": self.proc_id}, timeout=10.0)
            except Exception:
                pass
        if self.proc_id is not None:
            self.proc_id = None
            self.proc_pid = None
            self._running = False
        frame_buffer = self.__dict__.get("frame_buffer")
        if frame_buffer is not None:
            frame_buffer.close()
        else:
            self.shm = None
            self.frame_counter = None
            self.frames = None
