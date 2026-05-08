from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))
MODULE_PATH = REPO_ROOT / "apps" / "python" / "api" / "zapdos" / "{session}" / "{action}.py"
SPEC = importlib.util.spec_from_file_location("zapdos_session_action", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class ZapdosRenderCameraTest(unittest.TestCase):
    def test_require_camera_name_returns_exact_match_only(self):
        session = SimpleNamespace(camera_index={"head_camera": 0})

        self.assertEqual(MODULE._require_camera_name(session, "head_camera"), "head_camera")
        with self.assertRaises(HTTPException) as ctx:
            MODULE._require_camera_name(session, "HEAD_CAMERA")

        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
