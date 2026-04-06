from __future__ import annotations

import asyncio
import base64
import queue
import threading
import time
from pathlib import Path

PLACEHOLDER_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxAQDxAQEA8QDw8PDw8PEA8QDw8QDw8QFREWFhURFRUYHSggGBolGxUVITEhJSkrLi4uFx8zODMsNygtLisBCgoKDg0OGxAQGy0lICUtLS0tLS8tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAAEAAQMBIgACEQEDEQH/xAAcAAABBQEBAQAAAAAAAAAAAAAAAQIDBAUGBwj/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAHh0W0xZf/EABoQAQEBAQEBAQAAAAAAAAAAAAERAhIhMUH/2gAIAQEAAT8A0M7LQmY0Yw9f/8QAFBEBAAAAAAAAAAAAAAAAAAAAEP/aAAgBAgEBPwA//8QAFBEBAAAAAAAAAAAAAAAAAAAAEP/aAAgBAwEBPwA//9k="
)


class NanamiSession:
    def __init__(self, session_id: str, preview_dir: Path, final_dir: Path):
        self.session_id = session_id
        self.preview_dir = preview_dir
        self.final_dir = final_dir
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = []
        self.subscribers: list[queue.Queue[dict]] = []
        self.event_cv = threading.Condition()
        self.frame_cv = threading.Condition()
        self.latest_frame = PLACEHOLDER_JPEG
        self.frame_version = 0
        self.ready = False
        self.robot_loaded = False
        self.closed = False
        self._command_sink = None
        self._state_lock = threading.Lock()
        self._pending_state: dict | None = None
        self._state_event = threading.Event()
        self._state_thread = threading.Thread(target=self._state_loop, daemon=True)
        self._state_thread.start()

    def attach_command_sink(self, sink) -> None:
        self._command_sink = sink

    def publish(self, event: dict) -> None:
        with self.event_cv:
            self.history.append(event)
            self.history = self.history[-32:]
            for subscriber in list(self.subscribers):
                subscriber.put(event)
            self.event_cv.notify_all()

    def subscribe(self) -> queue.Queue[dict]:
        feed: queue.Queue[dict] = queue.Queue()
        with self.event_cv:
            for event in self.history:
                feed.put(event)
            self.subscribers.append(feed)
        return feed

    def unsubscribe(self, feed: queue.Queue[dict]) -> None:
        with self.event_cv:
            if feed in self.subscribers:
                self.subscribers.remove(feed)

    def wait_for_event(self, event_type: str, timeout: float) -> dict | None:
        deadline = time.time() + timeout
        with self.event_cv:
            while True:
                for event in reversed(self.history):
                    if event.get("type") == event_type:
                        return event
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self.event_cv.wait(remaining)

    def set_frame_from_path(self, frame_path: Path) -> None:
        data = None
        for _ in range(10):
            try:
                data = frame_path.read_bytes()
                break
            except OSError:
                time.sleep(0.02)
        if data is None:
            raise FileNotFoundError(frame_path)
        with self.frame_cv:
            self.latest_frame = data
            self.frame_version += 1
            self.frame_cv.notify_all()

    def wait_for_frame(self, version: int, timeout: float) -> tuple[int, bytes]:
        with self.frame_cv:
            if self.frame_version == version and not self.closed:
                self.frame_cv.wait(timeout)
            return self.frame_version, self.latest_frame

    def queue_state(self, joints: dict[str, float], camera: dict | None) -> None:
        with self._state_lock:
            self._pending_state = {"type": "set_state", "joints": joints, "camera": camera}
        self._state_event.set()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._state_event.set()
        with self.event_cv:
            for subscriber in list(self.subscribers):
                subscriber.put({"type": "session_closed", "session_id": self.session_id})
            self.event_cv.notify_all()
        with self.frame_cv:
            self.frame_cv.notify_all()

    def _state_loop(self) -> None:
        while not self.closed:
            self._state_event.wait()
            if self.closed:
                return
            time.sleep(0.05)
            self._state_event.clear()
            with self._state_lock:
                payload = self._pending_state
                self._pending_state = None
            if payload and self._command_sink:
                self._command_sink(payload)

    async def stream_events(self):
        feed = self.subscribe()
        try:
            while True:
                if self.closed:
                    return
                try:
                    event = await asyncio.to_thread(feed.get, True, 15.0)
                except queue.Empty:
                    yield {"type": "keepalive"}
                    continue
                if event.get("type") == "session_closed":
                    return
                yield event
        finally:
            self.unsubscribe(feed)
