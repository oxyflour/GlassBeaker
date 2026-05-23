from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from kokoro.video import write_mjpeg_avi


class VideoTest(unittest.TestCase):
    def test_write_mjpeg_avi_creates_indexed_avi_file(self) -> None:
        frames = [
            np.full((8, 10, 3), [220, 40, 20], dtype=np.uint8),
            np.full((8, 10, 3), [20, 80, 220], dtype=np.uint8),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "orbit.avi"

            write_mjpeg_avi(path, frames, fps=12)

            data = path.read_bytes()
            self.assertEqual(data[:4], b"RIFF")
            self.assertEqual(data[8:12], b"AVI ")
            self.assertIn(b"movi", data)
            self.assertIn(b"idx1", data)
            self.assertIn(b"MJPG", data)


if __name__ == "__main__":
    unittest.main()
