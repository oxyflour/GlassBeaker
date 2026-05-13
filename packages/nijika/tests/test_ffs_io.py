from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "nijika"))
sys.path.insert(0, str(REPO_ROOT / "chinatsu"))

from baseline.ffs_io import load_ffs_sample, write_ffs_sample
from farfield import load_ffs


class FfsIoTest(unittest.TestCase):
    def test_round_trip_parses_with_chinatsu(self):
        sample_path = Path(
            "C:/Projects/GlassBeaker/tmp/dataset-v3-ffs/antenna_000/1-[f=1400000000].ffs"
        )

        metadata, field = load_ffs_sample(sample_path)

        self.assertEqual(field.shape[0], 1)
        self.assertEqual(field.shape[1], metadata.angles_deg.shape[0])
        self.assertEqual(field.shape[2], 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "round-trip.ffs"
            write_ffs_sample(out_path, metadata, field)
            freq, angles, farfield = load_ffs(str(out_path))

        expected_farfield = np.stack(
            [
                field[..., 0] + 1j * field[..., 1],
                field[..., 2] + 1j * field[..., 3],
            ],
            axis=1,
        )

        np.testing.assert_allclose(freq.f, metadata.frequencies_hz, rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(angles, metadata.angles_deg, rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(farfield, expected_farfield, rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
