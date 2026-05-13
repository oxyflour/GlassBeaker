from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))


class RendererImportBoundariesTest(unittest.TestCase):
    def _fresh_import(self, module_name: str, prefixes: tuple[str, ...]):
        saved = {
            name: module
            for name, module in sys.modules.items()
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
        }
        original_import = __import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "mujoco" or name.startswith("mujoco."):
                raise AssertionError("mujoco import should not occur during this import path")
            return original_import(name, globals, locals, fromlist, level)

        for name in saved:
            del sys.modules[name]
        try:
            with mock.patch("builtins.__import__", side_effect=guarded_import):
                return importlib.import_module(module_name)
        finally:
            for name in list(sys.modules):
                if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
                    del sys.modules[name]
            sys.modules.update(saved)

    def test_control_channel_import_does_not_require_mujoco(self):
        module = self._fresh_import(
            "utils.zapdos.renderer.control_channel",
            ("utils.zapdos.renderer", "utils.zapdos.bundle.camera_specs", "mujoco"),
        )

        self.assertEqual(module.request_path(Path("C:/tmp")), Path("C:/tmp/request.json"))
        self.assertEqual(module.response_path(Path("C:/tmp")), Path("C:/tmp/response.json"))

    def test_camera_specs_pure_helpers_import_without_mujoco(self):
        module = self._fresh_import(
            "utils.zapdos.bundle.camera_specs",
            ("utils.zapdos.bundle.camera_specs", "mujoco"),
        )

        camera = module.RenderCamera(
            name="head_camera",
            prim="/MyRobot/head_camera",
            topic="/env_0/head_camera/image_raw",
            frame_id="head_camera",
            body="Root_head_link",
            pos=[0.0, 0.0, 0.0],
            quat=[1.0, 0.0, 0.0, 0.0],
            fovy=45.0,
        )

        self.assertEqual(module.camera_name_to_index([camera]), {"head_camera": 0})


if __name__ == "__main__":
    unittest.main()
