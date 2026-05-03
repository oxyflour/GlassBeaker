from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from api import genie_sim  # noqa: E402


class GenieSimApiTest(unittest.TestCase):
    def make_client(self) -> TestClient:
        app = FastAPI()
        app.include_router(genie_sim.router, prefix="/api/genie_sim")
        return TestClient(app)

    def test_execute_returns_scene_usda_path(self):
        client = self.make_client()
        payload = {
            "assetsRoot": "C:/assets",
            "code": "def root_scene(): return []",
            "description": "table_scene",
            "objects": [],
            "relations": {"nodes": [], "links": []},
            "sceneUsdaPath": "C:/tmp/scene.usda",
            "seed": 7,
        }
        with mock.patch.object(genie_sim, "execute_scene_code", return_value=payload):
            response = client.post("/api/genie_sim/execute", json={"code": payload["code"]})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sceneUsdaPath"], "C:/tmp/scene.usda")
        self.assertNotIn("bundleId", response.json())

    def test_execute_returns_traceback_in_500_detail(self):
        client = self.make_client()

        def explode(*_args, **_kwargs):
            raise AttributeError("'list' object has no attribute 'add'")

        with mock.patch.object(genie_sim, "execute_scene_code", side_effect=explode):
            response = client.post("/api/genie_sim/execute", json={"code": "def root_scene(): return []"})

        detail = response.json()["detail"]
        self.assertEqual(response.status_code, 500)
        self.assertIn("Traceback", detail)
        self.assertIn("AttributeError", detail)
        self.assertIn("'list' object has no attribute 'add'", detail)

    def test_render_route_is_removed(self):
        client = self.make_client()
        response = client.post("/api/genie_sim/render", json={"bundle_id": "bundle-1"})
        self.assertEqual(response.status_code, 404)

    def test_artifact_route_is_removed(self):
        client = self.make_client()
        response = client.get("/api/genie_sim/artifacts/bundle-1/rendering_traj_999.png")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
