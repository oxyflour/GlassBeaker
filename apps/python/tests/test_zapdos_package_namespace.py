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
            "utils.zapdos.zapdos_session",
            "utils.zapdos.request_router",
            "utils.zapdos.editor.state",
            "utils.zapdos.editor.repository",
            "utils.zapdos.editor.placement",
            "utils.zapdos.editor.commands",
            "utils.zapdos.editor.scene_writer",
            "utils.zapdos.editor.rebuild_runner",
            "utils.zapdos.editor.rebuild_manager",
            "utils.zapdos.editor.rebuild_job",
            "utils.zapdos.editor.rebuild_types",
            "utils.zapdos.editor.rebuild_events",
            "utils.zapdos.editor.zapdos_editor",
            "utils.zapdos.physics.base",
            "utils.zapdos.renderer.base",
            "utils.zapdos.renderer.zapdos_renderer",
        ]

        for module_name in module_names:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
