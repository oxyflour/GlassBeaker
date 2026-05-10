from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable


def request_path(control_dir: Path) -> Path:
    return control_dir / "request.json"


def response_path(control_dir: Path) -> Path:
    return control_dir / "response.json"


def send_control_request(
    *,
    control_dir: Path,
    payload: dict[str, Any],
    timeout: float,
    refresh_process_state: Callable[[], bool],
    tail_log: Callable[[], str],
    control_lock: threading.Lock,
) -> dict[str, Any]:
    req_path = request_path(control_dir)
    res_path = response_path(control_dir)
    req_id = str(time.time_ns())
    operation = str(payload.get("op") or "control request")
    with control_lock:
        req_path.parent.mkdir(parents=True, exist_ok=True)
        res_path.unlink(missing_ok=True)
        req_path.unlink(missing_ok=True)
        req_path.write_text(json.dumps({"id": req_id, **payload}), encoding="utf-8")
        deadline = time.time() + timeout
        try:
            while time.time() < deadline:
                if res_path.exists():
                    response = json.loads(res_path.read_text(encoding="utf-8"))
                    if response.get("id") != req_id:
                        time.sleep(0.02)
                        continue
                    if not response.get("ok"):
                        raise RuntimeError(str(response.get("error") or f"renderer {operation} failed"))
                    return response
                if not refresh_process_state():
                    raise RuntimeError(tail_log() or f"renderer exited while waiting for {operation}")
                time.sleep(0.02)
        finally:
            res_path.unlink(missing_ok=True)
            req_path.unlink(missing_ok=True)
    raise TimeoutError(f"renderer {operation} did not complete in {timeout:.1f}s")
