from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from optimizer_geometry import bound_distance, regenerate_ports  # noqa: E402


class OptimizerGeometryTest(unittest.TestCase):
    def test_bounded_distance_stays_within_cross_range(self):
        raw = torch.tensor([-20.0, 0.0, 20.0], dtype=torch.float32)
        bounded = bound_distance(raw, cross_size=80.0, span_width=20.0)
        self.assertTrue(torch.all(bounded <= 30.0 + 1e-5))
        self.assertTrue(torch.all(bounded >= -30.0 - 1e-5))
        self.assertAlmostEqual(float(bounded[1]), 0.0, places=6)

    def test_regenerate_ports_matches_feed_placement_reference(self):
        config = {
            "mesh": {
                "verts": [
                    [-40.0, -80.0, -4.0],
                    [40.0, 80.0, 4.0],
                ]
            },
            "antennaConfig": {
                "frameWidth": 4.0,
                "gap": 10.0,
                "cuts": [],
                "nibs": [
                    {"position": "top", "distance": 12.0, "width": 20.0, "thickness": 8.0},
                    {"position": "left", "distance": -18.0, "width": 24.0, "thickness": 8.0},
                    {"position": "bottom", "distance": 0.0, "width": 16.0, "thickness": 8.0},
                ],
            },
        }

        ports = regenerate_ports(config)

        self.assertEqual(len(ports), 3)
        first = ports[0]["positions"][0]
        self.assertAlmostEqual(first["from"]["x"], 12.0, places=6)
        self.assertAlmostEqual(first["to"]["x"], 12.0, places=6)
        self.assertAlmostEqual(first["from"]["y"], 73.5, places=6)
        self.assertAlmostEqual(first["to"]["y"], 76.5, places=6)
        self.assertAlmostEqual(first["from"]["z"], 4.0, places=6)

        second = ports[1]["positions"][0]
        self.assertAlmostEqual(second["from"]["x"], -33.5, places=6)
        self.assertAlmostEqual(second["to"]["x"], -36.5, places=6)
        self.assertAlmostEqual(second["from"]["y"], -18.0, places=6)
        self.assertAlmostEqual(second["to"]["y"], -18.0, places=6)


if __name__ == "__main__":
    unittest.main()
