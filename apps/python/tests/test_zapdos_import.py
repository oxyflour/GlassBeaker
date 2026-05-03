from __future__ import annotations

import asyncio
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.parse import urlencode

from fastapi import Request

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

    async def test_init_stream_emits_readable_error(self):
        future: asyncio.Future[object] = asyncio.Future()
        future.set_exception(RuntimeError("scene_usd not found"))
        MODULE.sessions["sess-1"] = future

        events = [chunk async for chunk in MODULE._init_stream("sess-1", future)]

        self.assertEqual(events, ["data: loading\n\n", "data: error: scene_usd not found\n\n"])


if __name__ == "__main__":
    unittest.main()
