from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baseline.data import load_dataset, stack_records  # noqa: E402
from baseline.ffs_io import FfsMetadata, write_ffs_sample  # noqa: E402


FREQUENCIES_HZ = np.asarray([1.0e9, 1.5e9], dtype=np.float64)
ANGLES_DEG = np.asarray(
    [[0.0, 0.0], [0.0, 90.0], [180.0, 0.0], [180.0, 90.0]],
    dtype=np.float64,
)


def _sample_config() -> dict[str, object]:
    return {
        "mesh": {
            "verts": [
                [-1.0, -2.0, 0.0],
                [1.0, -2.0, 0.0],
                [1.0, 2.0, 0.5],
                [-1.0, 2.0, 0.5],
            ]
        },
        "ports": [
            {
                "positions": [
                    {
                        "from": {"x": -0.5, "y": -1.0, "z": 0.25},
                        "to": {"x": -0.5, "y": -1.2, "z": 0.25},
                    }
                ]
            },
            {
                "positions": [
                    {
                        "from": {"x": 0.5, "y": 1.0, "z": 0.25},
                        "to": {"x": 0.5, "y": 1.2, "z": 0.25},
                    }
                ]
            },
        ],
        "antennaConfig": {"frameWidth": 0.1, "gap": 0.2, "cuts": [], "nibs": []},
    }


def _write_s_parameter(path: Path) -> None:
    rows = [f"{freq:.1f} 0.1 0.0" for freq in FREQUENCIES_HZ]
    path.write_text("\n".join(rows), encoding="utf8")


def _write_ffs_export(sample_dir: Path, sample_index: int) -> None:
    for port in range(1, 3):
        for freq_index in reversed(range(len(FREQUENCIES_HZ))):
            base = sample_index * 100.0 + port * 10.0 + (freq_index + 1)
            metadata = FfsMetadata(
                frequencies_hz=np.asarray([FREQUENCIES_HZ[freq_index]], dtype=np.float64),
                angles_deg=ANGLES_DEG,
                radiated_power_w=np.asarray([base], dtype=np.float64),
                accepted_power_w=np.asarray([base + 0.25], dtype=np.float64),
                stimulated_power_w=np.asarray([base + 0.5], dtype=np.float64),
                position_m=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
                z_axis=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
                x_axis=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
                phi_count=2,
                theta_count=2,
            )
            field = np.full((1, len(ANGLES_DEG), 4), fill_value=base / 100.0, dtype=np.float64)
            write_ffs_sample(sample_dir / f"{port}-[f={int(FREQUENCIES_HZ[freq_index])}].ffs", metadata, field)


def _write_sample(root: Path, name: str, sample_index: int) -> None:
    (root / f"{name}.json").write_text(json.dumps(_sample_config()), encoding="utf8")
    sample_dir = root / name
    sample_dir.mkdir()
    for row in range(1, 3):
        for col in range(1, 3):
            _write_s_parameter(sample_dir / f"S{row},{col}.cst.txt")
    _write_ffs_export(sample_dir, sample_index)


class DataFfsTest(unittest.TestCase):
    def test_load_dataset_can_include_ffs_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_sample(root, "antenna_000", sample_index=0)
            _write_sample(root, "antenna_001", sample_index=1)

            bundle = load_dataset(root, n_points=4, freq_bins=2, include_ffs=True)
            stacked = stack_records(bundle.records)

            self.assertEqual(len(bundle.records), 2)
            self.assertIsNotNone(bundle.ffs_metadata)
            self.assertEqual(bundle.port_count, 2)
            np.testing.assert_allclose(bundle.ffs_metadata.frequencies_hz, FREQUENCIES_HZ)
            self.assertIn("ffs", stacked)
            self.assertIn("ffs_radiated_power", stacked)
            self.assertIn("ffs_stimulated_power", stacked)
            self.assertNotIn("ffs_coeff", type(bundle.records[0]).__dataclass_fields__)
            self.assertNotIn("ffs_coeff", stacked)
            np.testing.assert_allclose(
                stacked["ffs_radiated_power"].numpy(),
                np.asarray(
                    [
                        [[11.0, 12.0], [21.0, 22.0]],
                        [[111.0, 112.0], [121.0, 122.0]],
                    ],
                    dtype=np.float32,
                ),
            )
            np.testing.assert_allclose(
                stacked["ffs_stimulated_power"].numpy(),
                np.asarray(
                    [
                        [[11.5, 12.5], [21.5, 22.5]],
                        [[111.5, 112.5], [121.5, 122.5]],
                    ],
                    dtype=np.float32,
                ),
            )

    def test_load_dataset_default_path_stays_s_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_sample(root, "antenna_000", sample_index=0)

            bundle = load_dataset(root, n_points=4, freq_bins=2)
            stacked = stack_records(bundle.records)

            self.assertEqual(len(bundle.records), 1)
            self.assertIsNone(bundle.ffs_metadata)
            self.assertNotIn("ffs", stacked)


if __name__ == "__main__":
    unittest.main()
