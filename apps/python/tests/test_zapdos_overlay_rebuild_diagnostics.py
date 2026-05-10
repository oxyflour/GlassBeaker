from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import queue
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))
SCENE_OPS_PATH = REPO_ROOT / "apps" / "python" / "utils" / "zapdos" / "rebuild" / "scene_rebuild_manager.py"
SCRIPT_PATH = REPO_ROOT / "apps" / "python" / "scripts" / "prepare_zapdos_overlay_rebuild.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _stub_modules():
    bundle = types.ModuleType("utils.zapdos.bundle")
    rebuild_runner = types.ModuleType("utils.zapdos.rebuild.overlay_rebuild_runner")

    class RenderBundle:
        @classmethod
        def from_json(cls, data):
            return {"bundle": data}

    bundle.RenderBundle = RenderBundle
    bundle.ensure_render_bundle = lambda robot_usd, scene_usd: SimpleNamespace(
        to_json=lambda: {"robot_usd": str(robot_usd), "scene_usd": str(scene_usd), "cameras": []}
    )

    overlay_state = types.ModuleType("utils.zapdos.overlay.overlay_state")
    overlay_repository = types.ModuleType("utils.zapdos.overlay.overlay_repository")
    overlay_scene_writer = types.ModuleType("utils.zapdos.overlay.overlay_scene_writer")
    overlay_state.default_overlay_state = lambda assets_root=None: {
        "version": 1,
        "assets_root": assets_root,
        "instances": [],
        "pose_overrides": {},
    }
    overlay_state.scene_revision = lambda base_scene_usd, next_overlay: "rev-2"
    overlay_repository.save_overlay_state = lambda path, state: None
    overlay_scene_writer.write_overlay_scene = lambda *args, **kwargs: None
    rebuild_runner.prepare_overlay_rebuild_request = lambda request, stage_logger=None: (
        stage_logger("write_overlay_scene") if stage_logger else None,
        stage_logger("ensure_render_bundle") if stage_logger else None,
        {"bundle": {"bundle_dir": "bundle", "cameras": []}, "next_revision": "rev-2"},
    )[-1]

    return {
        "fastapi": types.SimpleNamespace(HTTPException=type("HTTPException", (Exception,), {})),
        "utils.genie_sim": types.SimpleNamespace(resolve_assets_root=lambda value=None: Path(str(value or "C:/assets"))),
        "utils.zapdos.bundle": bundle,
        "utils.zapdos.rebuild.overlay_rebuild_runner": rebuild_runner,
        "utils.zapdos.zapdos_asset_library": types.SimpleNamespace(
            resolve_asset_record=lambda asset_id, assets_root: {"asset_id": asset_id, "url": "objects/item.usda"},
            asset_local_bounds=lambda path: {"min": [0, 0, 0], "max": [1, 1, 1]},
        ),
        "utils.zapdos.overlay.overlay_state": overlay_state,
        "utils.zapdos.overlay.overlay_repository": overlay_repository,
        "utils.zapdos.overlay.overlay_scene_writer": overlay_scene_writer,
    }


class ZapdosOverlayRebuildDiagnosticsTest(unittest.TestCase):
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
            next_overlay = module.default_overlay_state("C:/assets")
            previous_overlay = module.default_overlay_state("C:/assets")
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
            overlay_completions=queue.Queue(),
            scene_rebuild_jobs={
                "op-1": module.SceneRebuildJob(
                    future=module.ConcurrentFuture(),
                    success_payload={"ok": True, "items": []},
                    events=queue.Queue(),
                ),
            },
            scene_rebuild_jobs_lock=threading.Lock(),
        )
        session.scene_rebuild_future = lambda op_id: module.scene_rebuild_future(session, op_id)
        session.discard_scene_rebuild_job = lambda op_id: module.discard_scene_rebuild_job(session, op_id)

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
                    module.default_overlay_state("C:/assets"),
                    {},
                    module.default_overlay_state("C:/assets"),
                    "rev-1",
                    op_id="op-1",
                )

        events = module._drain_scene_rebuild_events(session, "op-1")
        stages = [payload["stage"] for name, payload in events if name == "progress"]
        self.assertIsNotNone(prepared)
        self.assertIn("prepare_overlay_rebuild.inline.started", stages)
        self.assertIn("prepare_overlay_rebuild.write_overlay_scene", stages)
        self.assertIn("prepare_overlay_rebuild.ensure_render_bundle", stages)
        self.assertIn("prepare_overlay_rebuild.inline.done", stages)

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
            overlay_completions=queue.Queue(),
            rebuilding_scene=True,
            scene_rebuild_jobs={
                "op-1": module.SceneRebuildJob(
                    future=module.ConcurrentFuture(),
                    success_payload={"ok": True, "items": []},
                    events=queue.Queue(),
                ),
            },
            scene_rebuild_jobs_lock=threading.Lock(),
        )
        session.scene_rebuild_future = lambda op_id: module.scene_rebuild_future(session, op_id)
        session.discard_scene_rebuild_job = lambda op_id: module.discard_scene_rebuild_job(session, op_id)
        session._prepare_overlay_rebuild = mock.Mock(side_effect=RuntimeError("subprocess crashed"))

        async def consume():
            stream = module.stream_scene_rebuild_job(session, "op-1")
            started = await asyncio.wait_for(anext(stream), timeout=0.1)
            module.run_overlay_rebuild_background(
                session,
                "op-1",
                module.default_overlay_state("C:/assets"),
                {},
                module.default_overlay_state("C:/assets"),
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
            scene_rebuild_jobs={
                "op-1": module.SceneRebuildJob(
                    future=module.ConcurrentFuture(),
                    success_payload={"ok": True, "items": []},
                    events=queue.Queue(),
                ),
            },
            scene_rebuild_jobs_lock=threading.Lock(),
        )
        session.scene_rebuild_future = lambda op_id: module.scene_rebuild_future(session, op_id)
        session.discard_scene_rebuild_job = lambda op_id: module.discard_scene_rebuild_job(session, op_id)

        async def consume():
            stream = module.stream_scene_rebuild_job(session, "op-1")
            started = await asyncio.wait_for(anext(stream), timeout=0.1)
            module.emit_scene_rebuild_progress(session, "op-1", "prepare_overlay_rebuild.done")
            progress = await asyncio.wait_for(anext(stream), timeout=0.2)
            session.scene_rebuild_jobs["op-1"].future.set_result({"ok": True, "items": [], "scene_revision": "rev-2"})
            done = await asyncio.wait_for(anext(stream), timeout=0.2)
            self.assertEqual(started, 'event: started\ndata: {"op_id": "op-1"}\n\n')
            self.assertEqual(progress, 'event: progress\ndata: {"stage": "prepare_overlay_rebuild.done"}\n\n')
            self.assertEqual(done, 'event: done\ndata: {"ok": true, "items": [], "scene_revision": "rev-2"}\n\n')

        asyncio.run(consume())


if __name__ == "__main__":
    unittest.main()
