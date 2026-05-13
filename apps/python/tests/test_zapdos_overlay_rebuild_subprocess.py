from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

import utils.zapdos.editor.rebuild_manager as EDITOR_REBUILD_MANAGER
from utils.zapdos.editor.state import default_overlay_state

SCRIPT_PATH = REPO_ROOT / "apps" / "python" / "scripts" / "prepare_zapdos_overlay_rebuild.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module

class ZapdosOverlayRebuildSubprocessTest(unittest.TestCase):
    def _build_session(self, root: Path):
        session = SimpleNamespace(
            session_dir=root,
            base_scene_usd=root / "base_scene.usda",
            robot_usd=root / "robot.usda",
            composed_scene_usd=root / "scene-overlay.usda",
        )
        session.base_scene_usd.write_text("#usda 1.0\n", encoding="utf-8")
        session.robot_usd.write_text("#usda 1.0\n", encoding="utf-8")
        return session

    def test_prepare_overlay_rebuild_runs_inline_without_spawning_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._build_session(Path(tmp))
            next_overlay = default_overlay_state("C:/assets")
            next_overlay["instances"] = [{
                "id": "table_000_01",
                "asset_id": "table_000",
                "url": "objects/table_000/Aligned.usda",
                "motion": "static",
                "placement": {"kind": "floor_at_xy", "xy": [0.0, 0.0], "z_offset": 0.0, "yaw": 0.0},
            }]
            previous_overlay = default_overlay_state("C:/assets")
            fake_bundle = SimpleNamespace(bundle_dir=Path("bundle"))
            payload = {
                "bundle": {"bundle_dir": "bundle", "cameras": []},
                "next_revision": "rev-2",
            }

            with mock.patch("subprocess.Popen") as popen:
                with mock.patch.object(
                    EDITOR_REBUILD_MANAGER,
                    "_run_overlay_rebuild_inline",
                    return_value=payload,
                ) as run_inline:
                    with mock.patch.object(EDITOR_REBUILD_MANAGER.RenderBundle, "from_json", return_value=fake_bundle) as from_json:
                        prepared = EDITOR_REBUILD_MANAGER.prepare_overlay_rebuild(
                            session,
                            next_overlay,
                            {"Scene_table_000_01": {"top_z": 0.75}},
                            previous_overlay,
                            "rev-1",
                        )

        popen.assert_not_called()
        run_inline.assert_called_once_with({
            "robot_usd": str(session.robot_usd),
            "base_scene_usd": str(session.base_scene_usd),
            "composed_scene_usd": str(session.composed_scene_usd),
            "next_overlay": next_overlay,
            "support_infos": {"Scene_table_000_01": {"top_z": 0.75}},
        }, mock.ANY)
        from_json.assert_called_once_with({"bundle_dir": "bundle", "cameras": []})
        self.assertIs(prepared.bundle, fake_bundle)
        self.assertEqual(prepared.next_overlay, next_overlay)
        self.assertEqual(prepared.previous_overlay, previous_overlay)
        self.assertEqual(prepared.previous_revision, "rev-1")
        self.assertEqual(prepared.next_revision, "rev-2")

    def test_prepare_overlay_rebuild_script_returns_bundle_payload(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"missing script: {SCRIPT_PATH}")
        script = _load_module(SCRIPT_PATH, "prepare_zapdos_overlay_rebuild_test")
        request = {
            "robot_usd": "robot.usda",
            "base_scene_usd": "base_scene.usda",
            "composed_scene_usd": "scene-overlay.usda",
            "next_overlay": {
                "version": 1,
                "assets_root": "C:/assets",
                "instances": [{
                    "id": "table_000_01",
                    "asset_id": "table_000",
                    "url": "objects/table_000/Aligned.usda",
                    "motion": "static",
                    "placement": {"kind": "floor_at_xy", "xy": [0.0, 0.0], "z_offset": 0.0, "yaw": 0.0},
                }],
                "pose_overrides": {},
            },
            "support_infos": {"Scene_table_000_01": {"top_z": 0.75}},
        }
        expected = {
            "bundle": {"bundle_dir": "bundle", "cameras": []},
            "next_revision": "rev-2",
        }

        with mock.patch.object(script, "prepare_overlay_rebuild_request", return_value=expected) as runner:
            result = script.prepare_overlay_rebuild(request)

        runner.assert_called_once_with(request, script._log_stage)
        self.assertEqual(result, expected)

    def test_prepare_overlay_rebuild_script_imports_editor_runner(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("from utils.zapdos.editor.rebuild_runner import prepare_overlay_rebuild_request", source)


if __name__ == "__main__":
    unittest.main()
