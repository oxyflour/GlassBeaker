from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baseline.ffs_codec import TorchFfsCodec, decode_ffs, encode_ffs, fit_ffs_codec


def _synthetic_fields(samples: int = 12) -> np.ndarray:
    rng = np.random.default_rng(7)
    theta = np.linspace(0.0, np.pi, 5, dtype=np.float64)
    phi = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False, dtype=np.float64)
    base_a = np.stack(np.meshgrid(theta, phi, indexing="ij"), axis=-1)
    base_b = np.stack(np.meshgrid(np.sin(theta), np.cos(phi), indexing="ij"), axis=-1)
    base = np.stack(
        [
            base_a[..., 0],
            base_a[..., 1],
            base_b[..., 0],
            base_b[..., 1],
        ],
        axis=-1,
    )
    fields = []
    for index in range(samples):
        weight_a = 0.5 + 0.1 * index
        weight_b = -0.3 + 0.05 * index
        noise = rng.normal(scale=1e-4, size=base.shape)
        fields.append(weight_a * base + weight_b * base[::-1] + noise)
    return np.asarray(fields, dtype=np.float64)


class FfsCodecTest(unittest.TestCase):
    def test_fit_is_repeatable_for_identical_inputs(self):
        fields = _synthetic_fields()

        first = fit_ffs_codec(fields, rank=3)
        second = fit_ffs_codec(fields, rank=3)

        np.testing.assert_allclose(first.mean, second.mean)
        np.testing.assert_allclose(first.basis, second.basis)
        np.testing.assert_allclose(encode_ffs(fields, first), encode_ffs(fields, second))

    def test_fit_encode_decode_round_trip_stays_below_threshold(self):
        fields = _synthetic_fields()

        state = fit_ffs_codec(fields, rank=3)
        coeffs = encode_ffs(fields, state)
        decoded = decode_ffs(coeffs, state)

        self.assertEqual(coeffs.shape, (12, 3))
        self.assertEqual(decoded.shape, fields.shape)
        rel_error = np.linalg.norm(decoded - fields) / np.linalg.norm(fields)
        self.assertLess(rel_error, 1e-3)

    def test_encode_decode_preserves_batch_prefix_shapes(self):
        train_fields = _synthetic_fields(samples=10)
        eval_fields = _synthetic_fields(samples=4)[:2]

        state = fit_ffs_codec(train_fields, rank=2)
        coeffs = encode_ffs(eval_fields, state)
        decoded = decode_ffs(coeffs, state)

        self.assertEqual(coeffs.shape, (2, 2))
        self.assertEqual(decoded.shape, eval_fields.shape)
        self.assertEqual(tuple(state.config.field_shape), eval_fields.shape[1:])

    def test_torch_decode_matches_numpy_decode(self):
        fields = _synthetic_fields()
        state = fit_ffs_codec(fields, rank=3)
        coeffs = encode_ffs(fields[:4], state)

        codec = TorchFfsCodec.from_state(state, dtype=torch.float64)
        decoded = codec.decode(torch.tensor(coeffs, dtype=torch.float64)).detach().cpu().numpy()

        np.testing.assert_allclose(decoded, decode_ffs(coeffs, state), rtol=0.0, atol=1e-12)

    def test_torch_decode_backpropagates_to_coefficients(self):
        fields = _synthetic_fields()
        state = fit_ffs_codec(fields, rank=3)
        coeffs = torch.tensor(encode_ffs(fields[:2], state), dtype=torch.float64, requires_grad=True)

        codec = TorchFfsCodec.from_state(state, dtype=torch.float64)
        loss = codec.decode(coeffs).square().sum()
        loss.backward()

        self.assertIsNotNone(coeffs.grad)
        assert coeffs.grad is not None
        self.assertEqual(tuple(coeffs.grad.shape), tuple(coeffs.shape))
        self.assertTrue(torch.isfinite(coeffs.grad).all().item())
        self.assertGreater(coeffs.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
