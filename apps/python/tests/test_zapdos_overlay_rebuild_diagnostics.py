from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))
SCENE_OPS_PATH = REPO_ROOT / "apps" / "python" / "utils" / "zapdos" / "editor" / "rebuild_manager.py"
SCRIPT_PATH = REPO_ROOT / "apps" / "python" / "scripts" / "prepare_zapdos_overlay_rebuild.py"

from utils.zapdos.editor import rebuild_events
from utils.zapdos.editor.state import default_overlay_state


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _stub_modules():
    bundle = types.ModuleType("utils.zapdos.bundle")
    rebuild_runner = types.ModuleType("utils.zapdos.editor.rebuild_runner")

    class RenderBundle:
        @classmethod
        def from_json(cls, data):
            return {"bundle": data}

    bundle.RenderBundle = RenderBundle
    bundle.ensure_render_bundle = lambda robot_usd, scene_usd: SimpleNamespace(
        to_json=lambda: {"robot_usd": str(robot_usd), "scene_usd": str(scene_usd), "cameras": []}
    )

    rebuild_runner.prepare_overlay_rebuild_request = lambda request, stage_logger=None: (
        stage_logger("write_overlay_scene") if stage_logger else None,
        stage_logger("ensure_render_bundle") if stage_logger else None,
        {"bundle": {"bundle_dir": "bundle", "cameras": []}, "next_revision": "rev-2"},
    )[-1]

    return {
        "fastapi": types.SimpleNamespace(HTTPException=type("HTTPException", (Exception,), {})),
        "utils.genie_sim": types.SimpleNamespace(resolve_assets_root=lambda value=None: Path(str(value or "C:/assets"))),
        "utils.zapdos.bundle": bundle,
        "utils.zapdos.editor.rebuild_runner": rebuild_runner,
        "utils.zapdos.zapdos_asset_library": types.SimpleNamespace(
            resolve_asset_record=lambda asset_id, assets_root: {"asset_id": asset_id, "url": "objects/item.usda"},
            asset_local_bounds=lambda path: {"min": [0, 0, 0], "max": [1, 1, 1]},
        ),
    }


class ZapdosOverlayRebuildDiagnosticsTest(unittest.TestCase):
    @staticmethod
    def build_rebuild_state():
        return rebuild_events.SceneRebuildState()

    def load_scene_ops(self):
        with mock.patch.dict(sys.modules, _stub_modules(), clear=False):
            return _load_module(SCENE_OPS_PATH, "zapdos_scene_ops_diagnostics_test")

    def load_script(self):
        with mock.patch.dict(sys.modules, _stub_modules(), clear=False):
            return _load_module(SCRIPT_PATH, "prepare_zapdos_overlay_rebuild_diagnostics_test")

    def test_prepare_overlay_rebuild_runs_inline_without_spawning_subprocess(self):
        module = self.load_scene_ops()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = SimpleNamespace(
                robot_usd=root / "robot.usda",
                base_scene_usd=root / "base_scene.usda",
                composed_scene_usd=root / "scene-overlay.usda",
                session_dir=root,
            )
            next_overlay = default_overlay_state("C:/assets")
            previous_overlay = default_overlay_state("C:/assets")
            fake_bundle = SimpleNamespace(bundle_dir=Path("bundle"))

            payload = {
                "bundle": {"bundle_dir": "bundle", "cameras": []},
                "next_revision": "rev-2",
            }
            with mock.patch("subprocess.Popen") as popen:
                with mock.patch.object(module, "_run_overlay_rebuild_inline", return_value=payload) as run_inline:
                    with mock.patch.object(module.RenderBundle, "from_json", return_value=fake_bundle):
                        prepared = module.prepare_overlay_rebuild(
                            session,
                            next_overlay,
                            {},
                            previous_overlay,
                            "rev-1",
                        )

        self.assertIs(prepared.bundle, fake_bundle)
        self.assertEqual(prepared.next_revision, "rev-2")
        popen.assert_not_called()
        run_inline.assert_called_once()

    def test_prepare_overlay_rebuild_emits_inline_progress_stages(self):
        module = self.load_scene_ops()
        session = SimpleNamespace(
            scene_rebuild_state=self.build_rebuild_state(),
        )
        session.scene_rebuild_future = lambda op_id: rebuild_events.scene_rebuild_future(session, op_id)
        session.discard_scene_rebuild_job = lambda op_id: rebuild_events.discard_scene_rebuild_job(session, op_id)
        rebuild_events.create_scene_rebuild_job(session, "op-1", {"ok": True, "items": []})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session.robot_usd = root / "robot.usda"
            session.base_scene_usd = root / "base_scene.usda"
            session.composed_scene_usd = root / "scene-overlay.usda"
            session.session_dir = root
            def fake_run_inline(request_payload, stage_logger):
                stage_logger("write_overlay_scene")
                stage_logger("ensure_render_bundle")
                return {
                    "bundle": {"bundle_dir": "bundle", "cameras": []},
                    "next_revision": "rev-2",
                }

            with mock.patch.object(module, "_run_overlay_rebuild_inline", side_effect=fake_run_inline):
                prepared = module.prepare_overlay_rebuild(
                    session,
                    default_overlay_state("C:/assets"),
                    {},
                    default_overlay_state("C:/assets"),
                    "rev-1",
                    op_id="op-1",
                )

        events = rebuild_events.drain_scene_rebuild_events(session, "op-1")
        stages = [payload["stage"] for name, payload in events if name == "progress"]
        self.assertIsNotNone(prepared)
        self.assertIn("prepare_overlay_rebuild.inline.started", stages)
        self.assertIn("prepare_overlay_rebuild.write_overlay_scene", stages)
        self.assertIn("prepare_overlay_rebuild.ensure_render_bundle", stages)
        self.assertIn("prepare_overlay_rebuild.inline.done", stages)

    def test_prepare_overlay_rebuild_uses_per_operation_scene_path(self):
        module = self.load_scene_ops()
        session = SimpleNamespace(
            scene_rebuild_state=self.build_rebuild_state(),
        )
        session.scene_rebuild_future = lambda op_id: rebuild_events.scene_rebuild_future(session, op_id)
        session.discard_scene_rebuild_job = lambda op_id: rebuild_events.discard_scene_rebuild_job(session, op_id)
        rebuild_events.create_scene_rebuild_job(session, "op-1", {"ok": True, "items": []})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session.robot_usd = root / "robot.usd"
            session.base_scene_usd = root / "base_scene.usda"
            session.composed_scene_usd = root / "scene-overlay.usda"
            session.session_dir = root
            with mock.patch.object(
                module,
                "_run_overlay_rebuild_inline",
                return_value={"bundle": {"bundle_dir": "bundle", "cameras": []}, "next_revision": "rev-2"},
            ) as run_inline:
                module.prepare_overlay_rebuild(
                    session,
                    default_overlay_state("C:/assets"),
                    {},
                    default_overlay_state("C:/assets"),
                    "rev-1",
                    op_id="op-1",
                )

        request_payload = run_inline.call_args.args[0]
        self.assertEqual(Path(request_payload["composed_scene_usd"]).name, "scene-overlay-op-1.usda")

    def test_prepare_overlay_rebuild_script_logs_progress_stages(self):
        script = self.load_script()
        request = {
            "robot_usd": "robot.usda",
            "base_scene_usd": "base_scene.usda",
            "composed_scene_usd": "scene-overlay.usda",
            "next_overlay": {"version": 1, "assets_root": "C:/assets", "instances": [], "pose_overrides": {}},
            "support_infos": {},
        }
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            script.prepare_overlay_rebuild(request)

        output = stderr.getvalue()
        self.assertIn("write_overlay_scene", output)
        self.assertIn("ensure_render_bundle", output)

    def test_background_prepare_error_completes_scene_rebuild_job_without_drain(self):
        module = self.load_scene_ops()
        session = SimpleNamespace(
            rebuilding_scene=True,
            scene_rebuild_state=self.build_rebuild_state(),
        )
        session.scene_rebuild_future = lambda op_id: rebuild_events.scene_rebuild_future(session, op_id)
        session.discard_scene_rebuild_job = lambda op_id: rebuild_events.discard_scene_rebuild_job(session, op_id)
        session._prepare_overlay_rebuild = mock.Mock(side_effect=RuntimeError("subprocess crashed"))
        rebuild_events.create_scene_rebuild_job(session, "op-1", {"ok": True, "items": []})

        async def run_sync(fn, world_token=None):
            return fn(session)

        @contextlib.asynccontextmanager
        async def reserve_world():
            token = object()
            yield token

        session.run_sync = run_sync
        session.reserve_world = reserve_world

        async def consume():
            stream = rebuild_events.stream_scene_rebuild_job(session, "op-1")
            started = await asyncio.wait_for(anext(stream), timeout=0.1)
            with mock.patch.object(session, "_capture_support_info_inputs", return_value=object(), create=True):
                with mock.patch.object(module, "resolve_support_infos", return_value={}):
                    await module.run_overlay_rebuild(
                        session,
                        "op-1",
                        default_overlay_state("C:/assets"),
                        default_overlay_state("C:/assets"),
                        "rev-1",
                    )
            progress = await asyncio.wait_for(anext(stream), timeout=0.2)
            failed = await asyncio.wait_for(anext(stream), timeout=0.2)
            self.assertEqual(started, 'event: started\ndata: {"op_id": "op-1"}\n\n')
            self.assertEqual(progress, 'event: progress\ndata: {"stage": "prepare_overlay_rebuild.started"}\n\n')
            self.assertEqual(failed, 'event: failed\ndata: {"detail": "subprocess crashed"}\n\n')
            self.assertFalse(session.rebuilding_scene)
            with self.assertRaises(StopAsyncIteration):
                await anext(stream)

        asyncio.run(consume())

    def test_stream_scene_rebuild_job_emits_progress_event_before_completion(self):
        module = self.load_scene_ops()
        session = SimpleNamespace(
            scene_rebuild_state=self.build_rebuild_state(),
        )
        session.scene_rebuild_future = lambda op_id: rebuild_events.scene_rebuild_future(session, op_id)
        session.discard_scene_rebuild_job = lambda op_id: rebuild_events.discard_scene_rebuild_job(session, op_id)
        rebuild_events.create_scene_rebuild_job(session, "op-1", {"ok": True, "items": []})

        async def consume():
            stream = rebuild_events.stream_scene_rebuild_job(session, "op-1")
            started = await asyncio.wait_for(anext(stream), timeout=0.1)
            rebuild_events.emit_scene_rebuild_progress(session, "op-1", "prepare_overlay_rebuild.done")
            progress = await asyncio.wait_for(anext(stream), timeout=0.2)
            session.scene_rebuild_state.jobs["op-1"].future.set_result({"ok": True, "items": [], "scene_revision": "rev-2"})
            done = await asyncio.wait_for(anext(stream), timeout=0.2)
            self.assertEqual(started, 'event: started\ndata: {"op_id": "op-1"}\n\n')
            self.assertEqual(progress, 'event: progress\ndata: {"stage": "prepare_overlay_rebuild.done"}\n\n')
            self.assertEqual(done, 'event: done\ndata: {"ok": true, "items": [], "scene_revision": "rev-2"}\n\n')

        asyncio.run(consume())


if __name__ == "__main__":
    unittest.main()
