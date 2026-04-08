from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

from .config import RUNTIME_ROOT, SESSION_DEFAULT_SOURCE, SESSION_ROOT, SHOTS_ROOT, ensure_roots
from .github_cache import AssetBundle, ensure_r1_asset_cache
from .models import SourceConfig
from .runtime import RuntimeBusyError, RuntimeHost
from .session import NanamiSession


class NanamiManager:
    def __init__(self, runtime_ttl_seconds: float = 900.0, reaper_poll_seconds: float = 5.0):
        ensure_roots()
        self.runtime_ttl_seconds = runtime_ttl_seconds
        self.reaper_poll_seconds = reaper_poll_seconds
        self.sessions: dict[str, NanamiSession] = {}
        self.session_runtime: dict[str, str] = {}
        self.runtimes: dict[str, RuntimeHost] = {}
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.reaper = threading.Thread(target=self._reaper_loop, daemon=True)
        self.reaper.start()

    def create_session(self, source_data: dict | None) -> dict:
        bundle = ensure_r1_asset_cache(SourceConfig(**(source_data or SESSION_DEFAULT_SOURCE)))
        session_id = uuid.uuid4().hex[:12]
        preview_dir = SESSION_ROOT / session_id / "preview"
        final_dir = SHOTS_ROOT / session_id
        session = NanamiSession(session_id, preview_dir, final_dir)
        host: RuntimeHost | None = None
        reused = False
        try:
            with self.lock:
                host, reused = self._claim_runtime(bundle)
                host.attach(session_id)
                self.sessions[session_id] = session
                self.session_runtime[session_id] = host.runtime_key
                session.attach_command_sink(lambda payload: self._send_session_command(session_id, payload))
                session.publish(self._assets_event(bundle))
                host.send({"type": "attach_session", "session_id": session_id, "preview_dir": str(preview_dir)})
                self._publish_attach_events(session_id, host)
        except Exception:
            session.close()
            if host is not None:
                with self.lock:
                    host.release(session_id)
                    self.sessions.pop(session_id, None)
                    self.session_runtime.pop(session_id, None)
            raise
        return {
            "session_id": session_id,
            "resolved_commit": bundle.manifest.resolved_commit,
            "runtime_id": host.runtime_id,
            "runtime_state": host.runtime_state(session_id),
            "runtime_reused": reused,
        }

    def get_session(self, session_id: str) -> NanamiSession:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def load_robot(self, session_id: str) -> dict:
        session = self.get_session(session_id)
        host = self._runtime_for(session_id)
        return self._status_response(session_id, host, session)

    def queue_state(self, session_id: str, joints: dict[str, float], camera: dict | None) -> None:
        session = self.get_session(session_id)
        host = self._runtime_for(session_id)
        if host.active_session_id != session_id:
            raise RuntimeError("Session is no longer attached to a runtime.")
        if host.runtime_state(session_id) != "loaded":
            raise RuntimeError("Robot must be loaded before state updates.")
        session.queue_state(joints, camera)

    def start_export(self, session_id: str, profile: str) -> str:
        host = self._runtime_for(session_id)
        session = self.get_session(session_id)
        if host.runtime_state(session_id) != "loaded":
            raise RuntimeError("Robot must be loaded before export.")
        job_id = uuid.uuid4().hex[:10]
        host.send(
            {
                "type": "export_final",
                "session_id": session_id,
                "job_id": job_id,
                "profile": profile,
                "output_dir": str(session.final_dir / job_id),
            }
        )
        session.publish({"type": "export_started", "job_id": job_id})
        return job_id

    def destroy_session(self, session_id: str) -> None:
        with self.lock:
            session = self.get_session(session_id)
            host = self._runtime_for(session_id)
            self.sessions.pop(session_id, None)
            self.session_runtime.pop(session_id, None)
            if host.release(session_id):
                try:
                    host.send({"type": "detach_session", "session_id": session_id})
                except OSError:
                    pass
        session.close()

    def on_worker_event(self, runtime_key: str, event: dict) -> None:
        with self.lock:
            host = self.runtimes.get(runtime_key)
            if host is None:
                return
            event_type = event.get("type")
            if event_type == "hello":
                host.ready = True
                self._publish_to_active(host, host.ready_event(host.active_session_id or ""))
                return
            if event_type == "robot_loaded":
                host.loading = False
                host.robot_loaded = True
                host.last_error = None
                self._publish_to_active(host, host.loaded_event())
                return
            if event_type == "frame_ready":
                self._handle_frame(event)
                return
            if event_type == "worker_error":
                if not event.get("job_id"):
                    host.loading = False
                    host.last_error = event.get("message") or "Worker error"
                if event.get("session_id"):
                    self._publish_event(event)
                else:
                    self._publish_to_active(host, event)
                return
            if event_type == "worker_exit":
                host.ready = False
                host.loading = False
                host.failed = True
                self._publish_to_active(host, event)
                return
            self._publish_event(event)

    def shutdown(self) -> None:
        self.stop_event.set()
        self.reaper.join(timeout=1.0)
        for host in list(self.runtimes.values()):
            host.stop()

    def _claim_runtime(self, bundle: AssetBundle) -> tuple[RuntimeHost, bool]:
        host = self.runtimes.get(bundle.runtime_key)
        if host and (host.failed or (host.last_error and not host.robot_loaded)):
            host.stop()
            self.runtimes.pop(bundle.runtime_key, None)
            host = None
        if host is None:
            host = RuntimeHost(
                runtime_key=bundle.runtime_key,
                asset_key=bundle.asset_key,
                manifest=bundle.manifest,
                config_path=RUNTIME_ROOT / bundle.asset_key / "runtime_config.json",
                on_event=lambda event: self.on_worker_event(bundle.runtime_key, event),
            )
            try:
                host.start()
            except Exception:
                host.stop()
                raise
            self.runtimes[bundle.runtime_key] = host
            return host, False
        if host.active_session_id:
            raise RuntimeBusyError(f"Runtime busy for session {host.active_session_id}.")
        return host, True

    def _publish_attach_events(self, session_id: str, host: RuntimeHost) -> None:
        session = self.sessions[session_id]
        if host.ready:
            session.publish(host.ready_event(session_id))
        if host.robot_loaded:
            session.publish(host.loaded_event())
            return
        if not host.loading:
            host.loading = True
            host.send({"type": "load_robot", "manifest": host.manifest.to_dict()})
            session.publish({"type": "robot_load_started", "session_id": session_id})

    def _send_session_command(self, session_id: str, payload: dict) -> None:
        host = self._runtime_for(session_id)
        try:
            host.send({**payload, "session_id": session_id})
        except OSError as exc:
            host.failed = True
            host.last_error = str(exc)
            session = self.sessions.get(session_id)
            if session is not None:
                session.publish({"type": "worker_error", "session_id": session_id, "message": str(exc)})

    def _runtime_for(self, session_id: str) -> RuntimeHost:
        runtime_key = self.session_runtime.get(session_id)
        if runtime_key is None:
            raise KeyError(session_id)
        host = self.runtimes.get(runtime_key)
        if host is None:
            raise RuntimeError("Runtime is not available.")
        return host

    def _publish_to_active(self, host: RuntimeHost, event: dict) -> None:
        session_id = host.active_session_id
        if not session_id:
            return
        session = self.sessions.get(session_id)
        if session is not None:
            session.publish(event)

    def _publish_event(self, event: dict) -> None:
        session = self.sessions.get(event.get("session_id", ""))
        if session is not None:
            session.publish(event)

    def _handle_frame(self, event: dict) -> None:
        session = self.sessions.get(event.get("session_id", ""))
        if session is None:
            return
        session.set_frame_from_path(Path(event["path"]))
        session.publish(event)

    def _assets_event(self, bundle: AssetBundle) -> dict:
        return {
            "type": "assets_cached",
            "cache_hit": bundle.cache_hit,
            "resolved_commit": bundle.manifest.resolved_commit,
            "runtime_key": bundle.runtime_key,
        }

    def _status_response(self, session_id: str, host: RuntimeHost, session: NanamiSession) -> dict:
        return {
            "status": host.runtime_state(session_id),
            "runtime_id": host.runtime_id,
            "robot_id": host.manifest.robot_id,
            "cache_hit": self._assets_event_from_session(session).get("cache_hit", False),
            "link_count": host.manifest.link_count,
            "movable_joint_count": host.manifest.movable_joint_count,
        }

    def _assets_event_from_session(self, session: NanamiSession) -> dict:
        for event in reversed(session.history):
            if event.get("type") == "assets_cached":
                return event
        return {}

    def _reaper_loop(self) -> None:
        while not self.stop_event.wait(self.reaper_poll_seconds):
            now = time.monotonic()
            stale: list[str] = []
            with self.lock:
                for runtime_key, host in self.runtimes.items():
                    if host.expired(self.runtime_ttl_seconds, now):
                        stale.append(runtime_key)
                for runtime_key in stale:
                    self.runtimes.pop(runtime_key).stop()


MANAGER = NanamiManager()
