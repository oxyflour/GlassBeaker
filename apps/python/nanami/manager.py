from __future__ import annotations

import threading
import uuid
from pathlib import Path

from .config import SESSION_DEFAULT_SOURCE, SESSION_ROOT, SHOTS_ROOT, ensure_roots
from .github_cache import ensure_r1_asset_cache
from .models import RobotManifest, SourceConfig
from .session import NanamiSession
from .worker import NanamiWorker


class NanamiManager:
    def __init__(self):
        ensure_roots()
        self.sessions: dict[str, NanamiSession] = {}
        self.manifests: dict[str, RobotManifest] = {}
        self.workers: dict[str, NanamiWorker] = {}
        self.cache_hit: dict[str, bool] = {}
        self.pending_loads: set[str] = set()
        self.active_loads: set[str] = set()
        self.load_lock = threading.Lock()

    def create_session(self, source_data: dict | None) -> dict:
        source = SourceConfig(**(source_data or SESSION_DEFAULT_SOURCE))
        session_id = uuid.uuid4().hex[:12]
        preview_dir = SESSION_ROOT / session_id / "preview"
        final_dir = SHOTS_ROOT / session_id
        session = NanamiSession(session_id, preview_dir, final_dir)
        manifest, cache_hit = ensure_r1_asset_cache(source)
        self.sessions[session_id] = session
        self.manifests[session_id] = manifest
        self.cache_hit[session_id] = cache_hit
        session.publish({"type": "assets_cached", "cache_hit": cache_hit, "resolved_commit": manifest.resolved_commit})

        worker = NanamiWorker(session, SESSION_ROOT / session_id / "worker_config.json", lambda event: self.on_worker_event(session_id, event))
        try:
            worker.write_session_config(
                {
                    "session_id": session_id,
                    "preview_dir": str(preview_dir),
                    "final_dir": str(final_dir),
                }
            )
            worker.start()
        except Exception:
            session.close()
            worker.stop()
            self.sessions.pop(session_id, None)
            self.manifests.pop(session_id, None)
            self.cache_hit.pop(session_id, None)
            raise
        self.workers[session_id] = worker
        return {"session_id": session_id, "resolved_commit": manifest.resolved_commit}

    def get_session(self, session_id: str) -> NanamiSession:
        if session_id not in self.sessions:
            raise KeyError(session_id)
        return self.sessions[session_id]

    def load_robot(self, session_id: str) -> dict:
        session = self.get_session(session_id)
        print(f"[nanami:{session_id}] load_robot called, ready={session.ready}, loaded={session.robot_loaded}")
        with self.load_lock:
            if session.robot_loaded:
                return self._load_response(session_id, "loaded")
            self.pending_loads.add(session_id)
        if self._dispatch_robot_load(session_id):
            return self._load_response(session_id, "loading")
        return self._load_response(session_id, self._load_status(session_id))

    def queue_state(self, session_id: str, joints: dict[str, float], camera: dict | None) -> None:
        session = self.get_session(session_id)
        if not session.robot_loaded:
            raise RuntimeError("Robot must be loaded before state updates.")
        session.queue_state(joints, camera)

    def start_export(self, session_id: str, profile: str) -> str:
        job_id = uuid.uuid4().hex[:10]
        session = self.get_session(session_id)
        if not session.robot_loaded:
            raise RuntimeError("Robot must be loaded before export.")
        payload = {"type": "export_final", "job_id": job_id, "profile": profile, "output_dir": str(session.final_dir / job_id)}
        self.workers[session_id].send(payload)
        session.publish({"type": "export_started", "job_id": job_id})
        return job_id

    def destroy_session(self, session_id: str) -> None:
        session = self.get_session(session_id)
        worker = self.workers.pop(session_id, None)
        self.manifests.pop(session_id, None)
        self.cache_hit.pop(session_id, None)
        self.sessions.pop(session_id, None)
        self._clear_load_state(session_id)
        session.close()
        if worker is not None:
            worker.stop()

    def on_worker_event(self, session_id: str, event: dict) -> None:
        session = self.sessions.get(session_id)
        if session is None:
            return
        print(f"[nanami:{session_id}] Worker event: {event.get('type')}")
        if event.get("type") == "hello":
            session.ready = True
            session.publish({"type": "session_ready", "session_id": session_id})
            self._dispatch_robot_load(session_id)
        elif event.get("type") == "frame_ready":
            session.set_frame_from_path(Path(event["path"]))
            session.publish(event)
        elif event.get("type") == "robot_loaded":
            self._clear_load_state(session_id)
            session.robot_loaded = True
            session.publish(event)
        elif event.get("type") == "worker_error":
            if not event.get("job_id"):
                self._clear_load_state(session_id)
            session.publish(event)
        elif event.get("type") == "worker_exit":
            self._clear_load_state(session_id)
            session.publish(event)
        elif event.get("type") == "export_done":
            session.publish(event)

    def _dispatch_robot_load(self, session_id: str) -> bool:
        session = self.sessions.get(session_id)
        manifest = self.manifests.get(session_id)
        worker = self.workers.get(session_id)
        if session is None or manifest is None or worker is None or not session.ready or session.robot_loaded:
            return False
        with self.load_lock:
            if session_id not in self.pending_loads or session_id in self.active_loads:
                return False
            self.pending_loads.discard(session_id)
            self.active_loads.add(session_id)
        try:
            worker.send({"type": "load_robot", "manifest": manifest.to_dict()})
        except Exception:
            self._clear_load_state(session_id)
            raise
        session.publish({"type": "robot_load_started", "session_id": session_id})
        return True

    def _clear_load_state(self, session_id: str) -> None:
        with self.load_lock:
            self.pending_loads.discard(session_id)
            self.active_loads.discard(session_id)

    def _load_status(self, session_id: str) -> str:
        session = self.sessions[session_id]
        if session.robot_loaded:
            return "loaded"
        with self.load_lock:
            if session_id in self.active_loads:
                return "loading"
            if session_id in self.pending_loads:
                return "queued"
        return "idle"

    def _load_response(self, session_id: str, status: str) -> dict:
        manifest = self.manifests[session_id]
        return {
            "status": status,
            "robot_id": manifest.robot_id,
            "cache_hit": self.cache_hit[session_id],
            "link_count": manifest.link_count,
            "movable_joint_count": manifest.movable_joint_count,
        }


MANAGER = NanamiManager()
