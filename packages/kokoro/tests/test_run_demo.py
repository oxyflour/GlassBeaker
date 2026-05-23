from __future__ import annotations

import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from run_demo import (
    DEFAULT_FEATURE_PERIOD_M,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_FRAMES,
    render_output_path,
    video_output_path,
)


class RunDemoTest(unittest.TestCase):
    def test_default_output_dir_stays_inside_kokoro_package(self) -> None:
        self.assertEqual(DEFAULT_OUTPUT_DIR, Path("packages/kokoro/tmp"))

    def test_default_feature_period_matches_default_height_field(self) -> None:
        self.assertEqual(DEFAULT_FEATURE_PERIOD_M, 500e-6)

    def test_default_video_duration_is_five_seconds_per_orbit(self) -> None:
        self.assertEqual(DEFAULT_VIDEO_FRAMES / DEFAULT_VIDEO_FPS, 5.0)

    def test_render_output_path_is_png(self) -> None:
        self.assertEqual(render_output_path(Path("tmp/kokoro")), Path("tmp/kokoro/kokoro_render.png"))

    def test_video_output_path_is_avi(self) -> None:
        self.assertEqual(video_output_path(Path("tmp/kokoro")), Path("tmp/kokoro/kokoro_orbit.avi"))


if __name__ == "__main__":
    unittest.main()
