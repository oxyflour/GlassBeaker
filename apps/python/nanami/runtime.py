from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import HDR_PATH
from .models import RobotManifest
from .worker import NanamiWorker


class RuntimeBusyError(RuntimeError):
    pass


@dataclass(slots=True)
class RuntimeHost:
    runtime_key: str
    asset_key: str
    manifest: RobotManifest
    config_path: Path
    on_event: object
    runtime_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    worker: NanamiWorker | None = None
    active_session_id: str | None = None
    ready: bool = False
    loading: bool = False
    robot_loaded: bool = False
    failed: bool = False
    last_error: str | None = None
    idle_since: float | None = None

    def start(self) -> None:
        self.worker = NanamiWorker(self.runtime_id, self.config_path, self.on_event)
        self.worker.write_runtime_config(
            {
                "runtime_id": self.runtime_id,
                "asset_key": self.asset_key,
                "hdr_path": str(HDR_PATH),
            }
        )
        self.worker.start()

    def stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()

    def attach(self, session_id: str) -> None:
        if self.active_session_id and self.active_session_id != session_id:
            raise RuntimeBusyError(f"Runtime busy for session {self.active_session_id}.")
        self.active_session_id = session_id
        self.idle_since = None

    def release(self, session_id: str) -> bool:
        if self.active_session_id != session_id:
            return False
        self.active_session_id = None
        self.idle_since = time.monotonic()
        return True

    def send(self, payload: dict) -> None:
        if self.worker is None:
            raise RuntimeError("Runtime worker is not available.")
        self.worker.send(payload)

    def warm(self) -> bool:
        return self.ready and self.robot_loaded and not self.failed and self.last_error is None

    def runtime_state(self, session_id: str | None = None) -> str:
        if self.failed or self.last_error:
            return "error"
        if session_id and self.active_session_id and self.active_session_id != session_id:
            return "busy"
        if self.robot_loaded:
            return "loaded"
        if not self.ready:
            return "starting"
        if self.loading:
            return "loading"
        return "starting"

    def ready_event(self, session_id: str) -> dict:
        return {"type": "session_ready", "session_id": session_id, "runtime_id": self.runtime_id}

    def loaded_event(self) -> dict:
        return {
            "type": "robot_loaded",
            "link_count": self.manifest.link_count,
            "movable_joint_count": self.manifest.movable_joint_count,
            "controls": [asdict(group) for group in self.manifest.controls],
        }

    def expired(self, ttl_seconds: float, now: float) -> bool:
        return self.active_session_id is None and self.idle_since is not None and now - self.idle_since >= ttl_seconds
