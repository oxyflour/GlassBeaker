from __future__ import annotations

import asyncio
import io
import json
import os
import queue
import subprocess
import time
from functools import lru_cache
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import mujoco  # type: ignore
import mujoco.viewer  # type: ignore
import numpy as np
from PIL import Image, ImageDraw

from utils.mujoco_tools import create_xml, flatten_matrix
from utils.ros_bridge import bridge
from utils.session import Session

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_USD = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"
ISAAC_PYTHON = REPO_ROOT / "apps" / "isaac" / ".venv" / "Scripts" / "python.exe"
ISAAC_SITE = REPO_ROOT / "apps" / "isaac" / ".venv" / "Lib" / "site-packages" / "isaacsim" / "exts" / "isaacsim.ros2.bridge"
RENDERER_SCRIPT = REPO_ROOT / "deps" / "genie_sim" / "source" / "geniesim" / "rl" / "renderer" / "rl_renderer.py"
MAIN_CAM_PRIM = "/default_viz_camera"
TF_RENDER_TOPIC = "/env_0/tf_render"
TF_RENDER_TYPE = "tf2_msgs/msg/TFMessage"
SHM_HEADER_BYTES = 4


def _tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="ignore").splitlines()[-lines:])


def _consume_future(task) -> None:
    try:
        task.exception()
    except Exception:
        pass


def _mjpeg_chunk(payload: bytes) -> bytes:
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"


@lru_cache(maxsize=8)
def _placeholder_jpeg(width: int, height: int, text: str) -> bytes:
    image = Image.new("RGB", (width, height), (20, 24, 32))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, height - 56, width, height), fill=(10, 12, 18))
    draw.text((16, height - 38), text, fill=(236, 239, 244))
    data = io.BytesIO()
    image.save(data, format="JPEG", quality=80)
    return data.getvalue()


def _tf_message(model, data) -> dict[str, Any]:
    transforms: list[dict[str, Any]] = []
    quat = np.empty(4)
    for body_id in range(1, model.nbody):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)  # type: ignore
        if not body:
            continue
        mujoco.mju_mat2Quat(quat, np.array(data.xmat[body_id], dtype=float).reshape(-1))  # type: ignore
        pos = np.array(data.xpos[body_id], dtype=float)
        transforms.append({
            "header": {"frame_id": "world"},
            "child_frame_id": body,
            "transform": {
                "translation": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
                "rotation": {"w": float(quat[0]), "x": float(quat[1]), "y": float(quat[2]), "z": float(quat[3])},
            },
        })
    return {"transforms": transforms}


def _isaac_ros_root() -> Path | None:
    for name in ("jazzy", "humble"):
        root = ISAAC_SITE / name
        if (root / "rclpy" / "rclpy").exists() and (root / "lib").exists():
            return root
    return None


class IsaacRenderer:
    def __init__(self, sess: str, scene_usd: Path, width: int, height: int, render_hz: float, headless: bool, ros_domain_id: int) -> None:
        tag = "".join(ch if ch.isalnum() else "_" for ch in sess)[:20] or "default"
        self.scene_usd = scene_usd
        self.width = width
        self.height = height
        self.render_hz = render_hz
        self.headless = headless
        self.ros_domain_id = ros_domain_id
        self.shm_name = f"glassbeaker_{tag}_frames"
        self.log_path = REPO_ROOT / "apps" / "python" / "tmp" / f"renderer_{tag}.log"
        self.proc: subprocess.Popen[str] | None = None
        self.log_file = None
        self.shm: shared_memory.SharedMemory | None = None
        self.frame_counter: np.ndarray | None = None
        self.frames: np.ndarray | None = None

        self._spawn()

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def ready(self) -> bool:
        return self.shm is not None and self.frame_counter is not None and self.frames is not None

    def _bind_shm(self) -> None:
        if self.shm is not None:
            return
        self.shm = shared_memory.SharedMemory(name=self.shm_name)
        self.frame_counter = np.ndarray((1,), dtype=np.uint32, buffer=self.shm.buf, offset=0)
        self.frames = np.ndarray((1, 1, self.height, self.width, 3), dtype=np.uint8, buffer=self.shm.buf, offset=SHM_HEADER_BYTES)

    def _spawn(self) -> None:
        if not ISAAC_PYTHON.exists():
            raise FileNotFoundError(f"Isaac Python not found: {ISAAC_PYTHON}")
        if not RENDERER_SCRIPT.exists():
            raise FileNotFoundError(f"Renderer script not found: {RENDERER_SCRIPT}")
        self.close()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, "w", encoding="utf-8")
        env = os.environ.copy()
        env.pop("ELECTRON_RUN_AS_NODE", None)
        env["SIM_REPO_ROOT"] = str(REPO_ROOT / "deps" / "genie_sim")
        env["ROS_DOMAIN_ID"] = str(self.ros_domain_id)
        env["PYTHONUNBUFFERED"] = "1"
        ros_root = _isaac_ros_root()
        py_paths = [str(REPO_ROOT / "deps" / "genie_sim" / "source")]
        if ros_root is not None:
            py_paths.append(str(ros_root / "rclpy"))
            env["PATH"] = os.pathsep.join([str(ros_root / "lib"), env.get("PATH", "")])
            print(f'set PATH=%PATH%;' + str(ros_root / "lib"))
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [*py_paths, env.get("PYTHONPATH", "")]))
        for key in ['SIM_REPO_ROOT', 'ROS_DOMAIN_ID', 'PYTHONUNBUFFERED', 'PYTHONPATH']:
            print(f'set {key}=' + env[key])
        cmd = [
            str(ISAAC_PYTHON),
            "-u",
            str(RENDERER_SCRIPT),
            "--scene-usd", str(self.scene_usd),
            "--num-envs", "1",
            "--render-hz", str(self.render_hz),
            "--cam-width", str(self.width),
            "--cam-height", str(self.height),
            "--main-cam-prim", MAIN_CAM_PRIM,
            "--shm-name", self.shm_name,
            "--ros-domain-id", str(self.ros_domain_id),
            "--headless",
        ]
        if not self.headless:
            cmd.pop()
        print('CMD: ' + ' '.join(cmd))
        print('INFO: check log ' + str(self.log_path))
        self.proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            text=True)

    async def wait_ready(self, timeout: float = 300.0) -> dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.running:
                try:
                    raise RuntimeError(_tail(self.log_path) or f"renderer exited with code {self.proc.returncode if self.proc else '?'}")
                finally:
                    self.close()
            try:
                self._bind_shm()
                return self.status()
            except FileNotFoundError:
                print('shared memory not ready, waiting...')
                await asyncio.sleep(5)
        try:
            raise TimeoutError(f"renderer did not create shared memory '{self.shm_name}' in {timeout:.0f}s")
        finally:
            self.close()

    def read(self) -> tuple[int, np.ndarray] | None:
        if not self.running:
            return None
        try:
            self._bind_shm()
        except FileNotFoundError:
            return None
        if self.frame_counter is None or self.frames is None:
            return None
        return int(self.frame_counter[0]), self.frames[0, 0].copy()

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "ready": self.ready,
            "ros_domain_id": self.ros_domain_id,
            "shm_name": self.shm_name,
            "width": self.width,
            "height": self.height,
            "log_path": str(self.log_path),
        }

    def close(self) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self.proc = None
        if self.shm is not None:
            self.shm.close()
            self.shm = None
            self.frame_counter = None
            self.frames = None
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

