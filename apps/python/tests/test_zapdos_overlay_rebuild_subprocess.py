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

import utils.zapdos.session.zapdos_session as SESSION_MODULE

ACTION_PATH = REPO_ROOT / "apps" / "python" / "api" / "zapdos" / "{session}" / "{action}.py"
SCRIPT_PATH = REPO_ROOT / "apps" / "python" / "scripts" / "prepare_zapdos_overlay_rebuild.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


ACTION = _load_module(ACTION_PATH, "zapdos_overlay_rebuild_action_test")


class ZapdosOverlayRebuildSubprocessTest(unittest.TestCase):
    def _build_session(self, root: Path):
        session = ACTION.ZapdosSession.__new__(ACTION.ZapdosSession)
        session.session_dir = root
        session.base_scene_usd = root / "base_scene.usda"
        session.robot_usd = root / "robot.usda"
        session.composed_scene_usd = root / "scene-overlay.usda"
        session.base_scene_usd.write_text("#usda 1.0\n", encoding="utf-8")
        session.robot_usd.write_text("#usda 1.0\n", encoding="utf-8")
        return session

    def test_prepare_overlay_rebuild_runs_inline_without_spawning_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._build_session(Path(tmp))
            next_overlay = ACTION.default_overlay_state("C:/assets")
            next_overlay["instances"] = [{
                "id": "table_000_01",
                "asset_id": "table_000",
                "url": "objects/table_000/Aligned.usda",
                "motion": "static",
                "placement": {"kind": "floor_at_xy", "xy": [0.0, 0.0], "z_offset": 0.0, "yaw": 0.0},
            }]
            previous_overlay = ACTION.default_overlay_state("C:/assets")
            fake_bundle = SimpleNamespace(bundle_dir=Path("bundle"))
            payload = {
                "bundle": {"bundle_dir": "bundle", "cameras": []},
                "next_revision": "rev-2",
            }

            with mock.patch("subprocess.Popen") as popen:
                with mock.patch.object(
                    SESSION_MODULE.rebuild_manager,
                    "_run_overlay_rebuild_inline",
                    return_value=payload,
                ) as run_inline:
                    with mock.patch.object(
                        SESSION_MODULE.rebuild_manager.RenderBundle,
                        "from_json",
                        return_value=fake_bundle,
                    ) as from_json:
                        prepared = ACTION.ZapdosSession._prepare_overlay_rebuild(
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


if __name__ == "__main__":
    unittest.main()
