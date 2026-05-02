from __future__ import annotations

import sys
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from api.teleop import spacemouse  # noqa: E402


class _StubManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def start(self, **config):
        self.calls.append(("start", config))
        return {"running": True, "active_arm": "right", "config": config}

    def stop(self):
        self.calls.append(("stop", None))
        return {"running": False}

    def status(self):
        self.calls.append(("status", None))
        return {"running": True, "active_arm": "right"}

    def set_active_arm(self, arm: str):
        self.calls.append(("set_active_arm", arm))
        return {"running": True, "active_arm": arm}

    def shutdown(self) -> None:
        self.calls.append(("shutdown", None))


class SpaceMouseApiTest(unittest.TestCase):
    def make_client(self) -> tuple[TestClient, _StubManager]:
        app = FastAPI()
        stub = _StubManager()
        spacemouse.manager = stub
        app.include_router(spacemouse.router, prefix="/api/teleop/spacemouse")
        return TestClient(app), stub

    def test_start_endpoint_passes_config_to_manager(self):
        client, stub = self.make_client()

        response = client.post(
            "/api/teleop/spacemouse/start",
            json={
                "robot_usd": "deps/galaxea/object/r1pro/r1pro.usda",
                "scene_usd": "apps/python/assets/default_scene.usda",
                "rate_hz": 60.0,
                "linear_scale": 0.2,
                "angular_scale": 0.7,
                "gripper_step": 0.01,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(stub.calls[0][0], "start")
        self.assertEqual(stub.calls[0][1]["rate_hz"], 60.0)
        self.assertEqual(response.json()["active_arm"], "right")

    def test_set_active_arm_only_accepts_left_or_right(self):
        client, stub = self.make_client()

        ok = client.post("/api/teleop/spacemouse/set_active_arm", json={"arm": "left"})
        bad = client.post("/api/teleop/spacemouse/set_active_arm", json={"arm": "both"})

        self.assertEqual(ok.status_code, 200)
        self.assertEqual(stub.calls[0], ("set_active_arm", "left"))
        self.assertEqual(bad.status_code, 422)

    def test_shutdown_event_stops_manager(self):
        client, stub = self.make_client()

        with client:
            response = client.get("/api/teleop/spacemouse/status")
            self.assertEqual(response.status_code, 200)

        self.assertIn(("shutdown", None), stub.calls)


if __name__ == "__main__":
    unittest.main()
