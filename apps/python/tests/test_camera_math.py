from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from utils.camera_math import focal_length_from_fovy, fovy_from_focal_length

REPO_ROOT = Path(__file__).resolve().parents[3]
RENDERER_ENTRY = REPO_ROOT / "apps" / "isaac" / "rl_renderer_entry.py"


class CameraMathTest(unittest.TestCase):
    def test_fovy_conversion_round_trip_is_available_without_rl_camera_module(self):
        focal_length = focal_length_from_fovy(60.0, 20.0)

        self.assertAlmostEqual(focal_length, 17.320508075688775)
        self.assertAlmostEqual(fovy_from_focal_length(focal_length, 20.0), 60.0)

    def test_renderer_entry_imports_dependency_light_camera_math(self):
        source = RENDERER_ENTRY.read_text(encoding="utf-8")

        self.assertIn("from utils.camera_math import fovy_from_focal_length", source)
        self.assertNotIn("from utils.zapdos.rl_cameras import fovy_from_focal_length", source)

    def test_renderer_entry_uses_renderer_package_helpers(self):
        source = RENDERER_ENTRY.read_text(encoding="utf-8")

        self.assertIn(
            "from utils.zapdos.renderer.isaac_renderer_reload import (",
            source,
        )
        self.assertIn(
            "from utils.zapdos.renderer.control_channel import request_path, response_path",
            source,
        )
        self.assertNotIn("from utils.zapdos.isaac_renderer_reload import (", source)
        self.assertNotIn(
            "from utils.zapdos.renderer_ipc import request_path, response_path",
            source,
        )

    def test_renderer_control_channel_import_does_not_require_mujoco(self):
        script = """
import builtins

original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "mujoco":
        raise ModuleNotFoundError("No module named 'mujoco'")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
import utils.zapdos.renderer.control_channel
print("ok")
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT / "apps" / "python",
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\\n{result.stdout}\\nstderr:\\n{result.stderr}",
        )
        self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
