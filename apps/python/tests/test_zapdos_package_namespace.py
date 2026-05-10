from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))


class ZapdosPackageNamespaceTest(unittest.TestCase):
    def test_moved_zapdos_modules_are_importable_from_package_namespace(self):
        module_names = [
            "utils.zapdos.rebuild.overlay_rebuild_runner",
            "utils.zapdos.session.request_router",
            "utils.zapdos.session.session_state",
            "utils.zapdos.physics.base",
            "utils.zapdos.renderer.base",
            "utils.zapdos.rebuild.scene_rebuild_job",
        ]

        for module_name in module_names:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
