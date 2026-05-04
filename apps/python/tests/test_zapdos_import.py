from __future__ import annotations

import asyncio
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.parse import urlencode

import mujoco  # type: ignore
from fastapi import Request
from utils.rl_bundle import ensure_render_bundle

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "apps" / "python" / "api" / "zapdos" / "{session}" / "{action}.py"
SPEC = importlib.util.spec_from_file_location("zapdos_session_action_import", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class ZapdosImportTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        MODULE.sessions.clear()

    def make_request(self, query: str = ""):
        return Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/python/zapdos/sess-1/init/start",
                "path_params": {"session": "sess-1", "action": "init", "name": "start"},
                "query_string": query.encode("utf-8"),
                "headers": [],
            }
        )

    def test_input_path_accepts_absolute_scene_usd(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene.usda"
            scene.write_text("#usda 1.0\n", encoding="utf-8")
            req = self.make_request(urlencode({"scene_usd": str(scene)}))

            resolved = MODULE._input_path(req, "scene_usd", Path("unused.usda"))

            self.assertEqual(resolved, scene.resolve())

    def test_input_path_accepts_repo_relative_scene_usd(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            scene = repo_root / "tmp" / "scene.usda"
            scene.parent.mkdir(parents=True)
            scene.write_text("#usda 1.0\n", encoding="utf-8")
            req = self.make_request(urlencode({"scene_usd": "tmp/scene.usda"}))

            with mock.patch.object(MODULE, "REPO_ROOT", repo_root):
                resolved = MODULE._input_path(req, "scene_usd", Path("unused.usda"))

            self.assertEqual(resolved, scene.resolve())

    async def test_failed_bootstrap_is_evicted_and_can_retry(self):
        req = self.make_request()
        create = mock.AsyncMock(side_effect=[RuntimeError("boom"), SimpleNamespace(camera_index={})])
        with mock.patch.object(MODULE, "_input_path", side_effect=[Path("robot.usda"), Path("scene.usda"), Path("robot.usda"), Path("scene.usda")]):
            with mock.patch.object(MODULE.ZapdosSession, "create", new=create):
                future = MODULE._get_or_create_session_future(req, "sess-1")
                with self.assertRaises(RuntimeError):
                    await MODULE._await_session_future("sess-1", future)
                self.assertNotIn("sess-1", MODULE.sessions)

                retry = MODULE._get_or_create_session_future(req, "sess-1")
                self.assertIs(MODULE.sessions["sess-1"], retry)
                await MODULE._await_session_future("sess-1", retry)

    async def test_init_recreates_inactive_cached_session(self):
        req = self.make_request()
        stale: asyncio.Future[object] = asyncio.Future()
        stale.set_result(SimpleNamespace(is_active=lambda: False))
        MODULE.sessions["sess-1"] = stale
        create = mock.AsyncMock(return_value=SimpleNamespace(camera_index={}))

        with mock.patch.object(MODULE, "_input_path", side_effect=[Path("robot.usda"), Path("scene.usda")]):
            with mock.patch.object(MODULE.ZapdosSession, "create", new=create):
                fresh = MODULE._get_or_create_session_future(req, "sess-1")

        self.assertIs(MODULE.sessions["sess-1"], fresh)
        self.assertIsNot(fresh, stale)
        await MODULE._await_session_future("sess-1", fresh)

    async def test_init_stream_emits_readable_error(self):
        future: asyncio.Future[object] = asyncio.Future()
        future.set_exception(RuntimeError("scene_usd not found"))
        MODULE.sessions["sess-1"] = future

        events = [chunk async for chunk in MODULE._init_stream("sess-1", future)]

        self.assertEqual(events, ["data: loading\n\n", "data: error: scene_usd not found\n\n"])

    async def test_init_stream_emits_heartbeat_while_bootstrap_is_pending(self):
        future = asyncio.create_task(asyncio.sleep(0.03, result=SimpleNamespace(camera_index={})))
        MODULE.sessions["sess-1"] = future

        with mock.patch.object(MODULE, "INIT_STREAM_HEARTBEAT_SEC", 0.01):
            events = [chunk async for chunk in MODULE._init_stream("sess-1", future)]

        self.assertEqual(events[0], "data: loading\n\n")
        self.assertIn("data: loading: preparing render bundle\n\n", events[:-1])
        self.assertEqual(events[-1], "data: started\n\n")

    def test_build_geometry_only_assigns_texture_when_material_has_rgb_texture(self):
        bundle = ensure_render_bundle(MODULE.DEFAULT_ROBOT_USD, MODULE.DEFAULT_SCENE_USD)
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.model = mujoco.MjModel.from_xml_path(str(bundle.mjcf))  # type: ignore
        session.assets = {}

        geoms = MODULE.ZapdosSession._build_geometry(session, bundle.mjcf.parent)

        textured = [geom for geom in geoms.values() if geom.kind == "mesh" and geom.texture]
        untextured = [geom for geom in geoms.values() if geom.kind == "mesh" and not geom.texture]

        self.assertTrue(textured)
        self.assertTrue(untextured)
        for geom in untextured:
            mat_id = int(session.model.geom_matid[geom.geom_id])
            if mat_id < 0:
                continue
            tex_id = int(session.model.mat_texid[mat_id, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
            self.assertLess(tex_id, 0)

    def test_require_session_future_rejects_missing_runtime_session(self):
        with self.assertRaises(MODULE.HTTPException) as err:
            MODULE._require_session_future("sess-1")

        self.assertEqual(err.exception.status_code, 409)
        self.assertEqual(err.exception.detail, "Session not initialized")

    async def test_require_session_future_evicts_inactive_session(self):
        future: asyncio.Future[object] = asyncio.Future()
        future.set_result(SimpleNamespace(is_active=lambda: False))
        MODULE.sessions["sess-1"] = future

        with self.assertRaises(MODULE.HTTPException) as err:
            MODULE._require_session_future("sess-1")

        self.assertEqual(err.exception.status_code, 409)
        self.assertEqual(err.exception.detail, "Session expired")
        self.assertNotIn("sess-1", MODULE.sessions)


if __name__ == "__main__":
    unittest.main()
