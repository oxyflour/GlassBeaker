from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.zapdos.task_streams import FINITE_TASK_NAMES


class ZapdosTaskStreamsTest(unittest.TestCase):
    def test_place_object_is_a_finite_task(self):
        self.assertIn("place_object", FINITE_TASK_NAMES)

    def test_scene_asset_mutations_are_finite_tasks(self):
        self.assertIn("add_assets_to_scene", FINITE_TASK_NAMES)
        self.assertIn("remove_assets_from_scene", FINITE_TASK_NAMES)
        self.assertIn("remove_asset_from_scene", FINITE_TASK_NAMES)


if __name__ == "__main__":
    unittest.main()
