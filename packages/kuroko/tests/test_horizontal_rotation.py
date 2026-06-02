import sys
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

KUROKO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KUROKO_DIR))

from app import (
    SESSIONS,
    FarFieldPattern,
    app,
    build_realizations,
    channel_cluster_points,
    compute_mi_distribution_and_gradient,
)


class HorizontalRotationGradientTest(unittest.TestCase):
    def make_pattern(self) -> FarFieldPattern:
        gains = np.asarray([1.0, 2.0, 3.0, 4.0])
        return FarFieldPattern(
            name="rx0",
            frequency_hz=None,
            radiated_power=None,
            accepted_power=None,
            stimulated_power=None,
            theta_deg=np.asarray([90.0, 90.0, 90.0, 90.0]),
            phi_deg=np.asarray([0.0, 90.0, 180.0, 270.0]),
            etheta=gains.astype(np.complex128),
            ephi=np.zeros((4,), dtype=np.complex128),
            theta_unique=np.asarray([90.0]),
            phi_unique=np.asarray([0.0, 90.0, 180.0, 270.0]),
            sample_to_grid=np.asarray([[0, 0], [0, 1], [0, 2], [0, 3]]),
        )

    def test_sums_mi_and_gradient_over_horizontal_pattern_rotations(self):
        pattern = self.make_pattern()
        gains = np.abs(pattern.etheta)
        channel = {
            "paths": [
                {
                    "aoa_theta_deg": 90.0,
                    "aoa_phi_deg": 0.0,
                    "gain": [1.0, 0.0],
                    "pol": [[1.0, 0.0], [0.0, 0.0]],
                }
            ]
        }

        realizations = build_realizations([pattern], channel, num_snapshots=1)
        result = compute_mi_distribution_and_gradient([pattern], realizations, 0, snr_db=0.0)

        expected_mi = float(np.sum(np.log2(1.0 + gains * gains)))
        expected_grad = 2.0 * gains / ((1.0 + gains * gains) * np.log(2.0))

        self.assertAlmostEqual(float(result["mi_values"][0]), expected_mi)
        np.testing.assert_allclose(result["grad_abs"], expected_grad, rtol=1e-12, atol=1e-12)
        self.assertEqual(result["rotation_count"], 4)

    def test_uses_fixed_terminal_angle_when_provided(self):
        pattern = self.make_pattern()
        channel = {
            "paths": [
                {
                    "aoa_theta_deg": 90.0,
                    "aoa_phi_deg": 0.0,
                    "gain": [1.0, 0.0],
                    "pol": [[1.0, 0.0], [0.0, 0.0]],
                }
            ]
        }

        realizations = build_realizations([pattern], channel, num_snapshots=1)
        result = compute_mi_distribution_and_gradient(
            [pattern],
            realizations,
            0,
            snr_db=0.0,
            terminal_yaw_deg=90.0,
        )

        expected_grad = np.asarray([0.0, 0.0, 0.0, 8.0 / (17.0 * np.log(2.0))])

        self.assertAlmostEqual(float(result["mi_values"][0]), np.log2(17.0))
        np.testing.assert_allclose(result["grad_abs"], expected_grad, rtol=1e-12, atol=1e-12)
        self.assertEqual(result["rotation_count"], 1)

    def test_records_directional_mi_extrema_over_realizations(self):
        pattern = self.make_pattern()
        gains = np.abs(pattern.etheta)
        channel = {
            "snapshots": [
                {
                    "paths": [
                        {
                            "aoa_theta_deg": 90.0,
                            "aoa_phi_deg": 0.0,
                            "gain": [1.0, 0.0],
                            "pol": [[1.0, 0.0], [0.0, 0.0]],
                        }
                    ]
                },
                {
                    "paths": [
                        {
                            "aoa_theta_deg": 90.0,
                            "aoa_phi_deg": 0.0,
                            "gain": [2.0, 0.0],
                            "pol": [[1.0, 0.0], [0.0, 0.0]],
                        }
                    ]
                },
            ]
        }

        realizations = build_realizations([pattern], channel, num_snapshots=1)
        result = compute_mi_distribution_and_gradient([pattern], realizations, 0, snr_db=0.0)

        expected_min = np.log2(1.0 + gains * gains)
        expected_max = np.log2(1.0 + 4.0 * gains * gains)

        np.testing.assert_allclose(result["mi_min"], expected_min, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result["mi_max"], expected_max, rtol=1e-12, atol=1e-12)

    def test_rotates_cdl_clusters_into_fixed_terminal_local_phi(self):
        channel = {
            "delays": [0.0, 1e-9],
            "powers": [0.0, -3.0],
            "aoa": [30.0, 350.0],
            "zoa": [70.0, 100.0],
        }

        points = channel_cluster_points(channel, terminal_yaw_deg=60.0)

        self.assertEqual(
            [{k: point[k] for k in ("theta", "phi", "local_phi", "label")} for point in points],
            [
                {"theta": 70.0, "phi": 30.0, "local_phi": 330.0, "label": "cluster 0"},
                {"theta": 100.0, "phi": 350.0, "local_phi": 290.0, "label": "cluster 1"},
            ],
        )
        self.assertGreater(points[0]["energy"], points[1]["energy"])

    def test_heatmap_response_includes_fixed_terminal_channel_clusters(self):
        sid = "cluster_overlay_test"
        channel = {
            "delays": [0.0],
            "powers": [1.0],
            "aoa": [30.0],
            "zoa": [70.0],
        }
        SESSIONS[sid] = {
            "patterns": [self.make_pattern()],
            "channel": channel,
            "channel_type": "sionna_cdl",
        }
        try:
            response = TestClient(app).get(
                f"/api/heatmap/{sid}",
                params={
                    "port_index": 0,
                    "polarization_mode": "cross",
                    "terminal_pose_mode": "fixed_angle",
                    "terminal_pose_angle_deg": 60.0,
                },
            )
        finally:
            SESSIONS.pop(sid, None)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["channel_clusters"],
            [{"theta": 70.0, "phi": 30.0, "local_phi": 330.0, "energy": 1.0, "label": "cluster 0"}],
        )


if __name__ == "__main__":
    unittest.main()
