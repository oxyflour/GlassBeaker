from __future__ import annotations

import asyncio
import os
import subprocess
import time
from pathlib import Path

from utils.ros_bridge import bridge

REPO_ROOT = Path(__file__).resolve().parents[3]
ROS_APP_DIR = REPO_ROOT / "apps" / "ros"
ROS_LOG_PATH = REPO_ROOT / "apps" / "python" / "tmp" / "ros_worker.log"

_proc: subprocess.Popen[bytes] | None = None
_log = None
_refs = 0


def acquire_ros_worker() -> None:
    global _refs
    _refs += 1
    _ensure_started()


def release_ros_worker() -> None:
    global _refs
    _refs = max(0, _refs - 1)
    if _refs == 0:
        _stop()


async def wait_for_ros_bridge(timeout: float = 20.0) -> bool:
    _ensure_started()
    deadline = time.time() + timeout
    while time.time() < deadline:
        if bridge.conns:
            return True
        if _proc is not None and _proc.poll() is not None:
            _ensure_started(force=True)
        await asyncio.sleep(0.2)
    return bool(bridge.conns)


def _ensure_started(force: bool = False) -> None:
    global _proc, _log
    if not force and _proc is not None and _proc.poll() is None:
        return
    _stop()
    ROS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _log = ROS_LOG_PATH.open("ab")
    env = os.environ.copy()
    env.pop("ELECTRON_RUN_AS_NODE", None)
    env["PYTHONUNBUFFERED"] = "1"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    _proc = subprocess.Popen(
        ["pixi", "run", "-e", "default", "python", "app.py"],
        cwd=str(ROS_APP_DIR),
        env=env,
        stdout=_log,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )


def _stop() -> None:
    global _proc, _log
    if _proc is not None:
        if _proc.poll() is None:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(_proc.pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
            except Exception:
                try:
                    _proc.terminate()
                    _proc.wait(timeout=5)
                except Exception:
                    pass
        _proc = None
    if _log is not None:
        _log.close()
        _log = None
