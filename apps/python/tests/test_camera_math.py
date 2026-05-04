from __future__ import annotations

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
        self.assertNotIn("from utils.rl_cameras import fovy_from_focal_length", source)


if __name__ == "__main__":
    unittest.main()
