import sys
import unittest
from pathlib import Path

import numpy as np

KUROKO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KUROKO_DIR))

from app import FarFieldPattern, build_realizations, compute_mi_distribution_and_gradient


class HorizontalRotationGradientTest(unittest.TestCase):
    def test_sums_mi_and_gradient_over_horizontal_pattern_rotations(self):
        gains = np.asarray([1.0, 2.0, 3.0, 4.0])
        pattern = FarFieldPattern(
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


if __name__ == "__main__":
    unittest.main()
