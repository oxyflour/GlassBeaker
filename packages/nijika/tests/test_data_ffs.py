from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baseline.data import load_dataset, stack_records  # noqa: E402

DATASET_ROOT = Path("C:/Projects/GlassBeaker/tmp/dataset-v3-ffs")


def _copy_fixture(root: Path, names: list[str]) -> None:
    for name in names:
        shutil.copy2(DATASET_ROOT / f"{name}.json", root / f"{name}.json")
        shutil.copytree(DATASET_ROOT / name, root / name)


class DataFfsTest(unittest.TestCase):
    def test_load_dataset_can_include_ffs_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_fixture(root, ["antenna_000", "antenna_001"])

            bundle = load_dataset(root, n_points=8, freq_bins=11, include_ffs=True)
            stacked = stack_records(bundle.records)

            self.assertEqual(len(bundle.records), 2)
            self.assertIsNotNone(bundle.ffs_metadata)
            self.assertEqual(bundle.port_count, 3)
            self.assertEqual(bundle.records[0].ffs.shape[0], bundle.port_count)
            self.assertEqual(bundle.records[0].ffs.shape[1], len(bundle.ffs_metadata.frequencies_hz))
            self.assertNotIn("ffs_coeff", type(bundle.records[0]).__dataclass_fields__)
            self.assertIn("ffs", stacked)
            self.assertIn("ffs_radiated_power", stacked)
            self.assertIn("ffs_stimulated_power", stacked)
            self.assertNotIn("ffs_coeff", stacked)
            self.assertEqual(stacked["ffs"].shape[0], 2)
            self.assertEqual(
                tuple(stacked["ffs_radiated_power"].shape[:3]),
                (2, bundle.port_count, len(bundle.ffs_metadata.frequencies_hz)),
            )
            self.assertEqual(
                tuple(stacked["ffs_stimulated_power"].shape[:3]),
                (2, bundle.port_count, len(bundle.ffs_metadata.frequencies_hz)),
            )

    def test_load_dataset_default_path_stays_s_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_fixture(root, ["antenna_000"])

            bundle = load_dataset(root, n_points=8, freq_bins=11)
            stacked = stack_records(bundle.records)

            self.assertEqual(len(bundle.records), 1)
            self.assertIsNone(bundle.ffs_metadata)
            self.assertNotIn("ffs", stacked)


if __name__ == "__main__":
    unittest.main()
