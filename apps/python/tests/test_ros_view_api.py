from __future__ import annotations

import sys
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

try:
    from api import ros_view  # type: ignore
except ImportError:
    ros_view = None


class _StubStore:
    def __init__(self) -> None:
        self.calls = 0
        self.stream_calls = 0
        self.render_calls: list[str] = []

    async def state(self):
        self.calls += 1
        return {
            "connected": True,
            "last_error": None,
            "topics": [{
                "id": "/env_0/head_camera/image_raw",
                "kind": "image",
                "label": "/env_0/head_camera/image_raw",
                "description": "sensor_msgs/msg/Image",
                "src": "/python/ros_view/render/%2Fenv_0%2Fhead_camera%2Fimage_raw",
            }],
        }

    async def stream(self):
        self.stream_calls += 1
        yield 'event: state\ndata: {"connected": true}\n\n'

    async def render(self, topic_id: str):
        self.render_calls.append(topic_id)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nfake\r\n"

    def has_image_topic(self, topic_id: str) -> bool:
        return topic_id == "/env_0/head_camera/image_raw"


class RosViewApiTest(unittest.TestCase):
    def test_state_endpoint_returns_ros_view_topics(self):
        self.assertIsNotNone(ros_view, "api.ros_view module missing")

        app = FastAPI()
        stub = _StubStore()
        ros_view.store = stub
        app.include_router(ros_view.router, prefix="/api/ros_view")
        client = TestClient(app)

        response = client.get("/api/ros_view/state")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["topics"][0]["id"], "/env_0/head_camera/image_raw")
        self.assertEqual(response.json()["topics"][0]["src"], "/python/ros_view/render/%2Fenv_0%2Fhead_camera%2Fimage_raw")
        self.assertEqual(stub.calls, 1)

    def test_stream_endpoint_returns_event_stream(self):
        self.assertIsNotNone(ros_view, "api.ros_view module missing")

        app = FastAPI()
        stub = _StubStore()
        ros_view.store = stub
        app.include_router(ros_view.router, prefix="/api/ros_view")
        client = TestClient(app)

        response = client.get("/api/ros_view/stream")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")
        self.assertIn('event: state\ndata: {"connected": true}', response.text)
        self.assertEqual(stub.stream_calls, 1)

    def test_render_endpoint_returns_mjpeg_stream(self):
        self.assertIsNotNone(ros_view, "api.ros_view module missing")

        app = FastAPI()
        stub = _StubStore()
        ros_view.store = stub
        app.include_router(ros_view.router, prefix="/api/ros_view")
        client = TestClient(app)

        response = client.get("/api/ros_view/render/%2Fenv_0%2Fhead_camera%2Fimage_raw")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "multipart/x-mixed-replace; boundary=frame")
        self.assertIn(b"Content-Type: image/jpeg", response.content)
        self.assertEqual(stub.render_calls, ["/env_0/head_camera/image_raw"])


if __name__ == "__main__":
    unittest.main()
