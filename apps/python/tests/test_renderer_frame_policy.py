from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.zapdos.renderer.frame_policy import should_render_frame


class RendererFramePolicyTest(unittest.TestCase):
    def test_first_frame_renders_even_without_dirty_subscribers(self):
        subscribers = [SimpleNamespace(_dirty=False)]

        self.assertTrue(should_render_frame(subscribers, frame_counter=0))

    def test_startup_warmup_renders_first_few_frames_without_dirty_subscribers(self):
        subscribers = [SimpleNamespace(_dirty=False)]

        self.assertTrue(should_render_frame(subscribers, frame_counter=1))
        self.assertTrue(should_render_frame(subscribers, frame_counter=2))

    def test_later_frame_skips_when_no_subscriber_is_dirty(self):
        subscribers = [SimpleNamespace(_dirty=False), SimpleNamespace(_dirty=False)]

        self.assertFalse(should_render_frame(subscribers, frame_counter=3))

    def test_later_frame_renders_when_any_subscriber_is_dirty(self):
        subscribers = [SimpleNamespace(_dirty=False), SimpleNamespace(_dirty=True)]

        self.assertTrue(should_render_frame(subscribers, frame_counter=3))


if __name__ == "__main__":
    unittest.main()
