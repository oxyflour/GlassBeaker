from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
ISAAC_PYTHON = REPO_ROOT / "apps" / "isaac" / ".venv" / "Scripts" / "python.exe"
ISAAC_SITE = REPO_ROOT / "apps" / "isaac" / ".venv" / "Lib" / "site-packages" / "isaacsim" / "exts" / "isaacsim.ros2.bridge"
RENDERER_ENTRY = REPO_ROOT / "apps" / "isaac" / "rl_renderer_entry.py"


def tail_log(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="ignore").splitlines()[-lines:])


def format_isaacsim_failure(
    summary: str,
    log_path: Path,
    detail: str = "",
    *,
    lines: int = 40,
) -> str:
    parts = [f"{summary}, check {log_path}"]
    detail = detail.strip()
    if detail:
        parts.append(detail)
    log_tail = tail_log(log_path, lines)
    if log_tail:
        parts.append(f"Last log lines:\n{log_tail}")
    return "\n".join(parts)


def _isaac_ros_root() -> Path | None:
    for name in ("jazzy", "humble"):
        root = ISAAC_SITE / name
        if (root / "rclpy" / "rclpy").exists() and (root / "lib").exists():
            return root
    return None


def setup_renderer_env(env: dict[str, str], ros_domain_id: int) -> dict[str, str]:
    env = dict(env)
    env.pop("ELECTRON_RUN_AS_NODE", None)
    env["ROS_DOMAIN_ID"] = str(ros_domain_id)
    env["SIM_REPO_ROOT"] = str(REPO_ROOT / "deps" / "genie_sim")
    env["PYTHONUNBUFFERED"] = "1"
    ros_root = _isaac_ros_root()
    py_paths = [str(REPO_ROOT / "deps" / "genie_sim" / "source")]
    if ros_root is not None:
        py_paths.append(str(ros_root / "rclpy"))
        env["PATH"] = os.pathsep.join([str(ros_root / "lib"), env.get("PATH", "")])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [*py_paths, env.get("PYTHONPATH", "")]))
    return env


def spawn_local_renderer(
    cmd: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    log_path: Path,
) -> subprocess.Popen[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("wb")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
    except Exception:
        log_file.close()
        raise
    process._glassbeaker_log_file = log_file  # type: ignore[attr-defined]
    return process


def close_local_renderer(process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except Exception:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                pass
    log_file = getattr(process, "_glassbeaker_log_file", None)
    if log_file is not None:
        log_file.close()
