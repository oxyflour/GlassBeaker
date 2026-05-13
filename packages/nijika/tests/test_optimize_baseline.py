from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baseline.ffs_codec import encode_ffs, fit_ffs_codec  # noqa: E402
from optimize_baseline import optimize_model  # noqa: E402


class _ToySurrogate(torch.nn.Module):
    def forward(
        self,
        points: torch.Tensor,
        ports: torch.Tensor,
        geom: torch.Tensor,
        frame: torch.Tensor,
        cuts: torch.Tensor,
        nibs: torch.Tensor,
    ) -> torch.Tensor:
        del points, geom, frame, ports
        batch = cuts.size(0)
        freq_bins = 3
        port_count = 3
        nib_penalty = nibs[:, :port_count, 5].abs() * 0.25
        cut_penalty = cuts[:, 0, 5].abs().unsqueeze(-1) * 0.15
        diag = 0.15 + nib_penalty + cut_penalty
        offdiag = 0.02 + cut_penalty * 0.5
        matrix = torch.zeros((batch, freq_bins, port_count, port_count, 2), dtype=cuts.dtype, device=cuts.device)
        for idx in range(port_count):
            matrix[:, :, idx, idx, 0] = diag[:, idx].unsqueeze(-1).expand(-1, freq_bins)
        for row in range(port_count):
            for col in range(port_count):
                if row != col:
                    matrix[:, :, row, col, 0] = offdiag[:, 0].unsqueeze(-1).expand(-1, freq_bins)
        return matrix.view(batch, freq_bins, port_count * port_count * 2)


def _toy_ffs_fields() -> np.ndarray:
    angles = 4
    fields = np.zeros((3, 3, angles, 4), dtype=np.float32)
    base = np.asarray(
        [
            [1.0, 0.0, 0.4, 0.0],
            [0.8, 0.0, 0.2, 0.0],
            [0.6, 0.0, 0.1, 0.0],
            [0.4, 0.0, 0.05, 0.0],
        ],
        dtype=np.float32,
    )
    for port in range(3):
        for freq in range(3):
            fields[port, freq] = base * (1.0 + 0.25 * port + 0.1 * freq)
    return fields


def _toy_ffs_metadata() -> dict[str, object]:
    return {
        "frequencies_hz": [1.0e9, 1.5e9, 2.0e9],
        "angles_deg": [[0.0, 0.0], [0.0, 90.0], [180.0, 0.0], [180.0, 90.0]],
        "position_m": [0.0, 0.0, 0.0],
        "z_axis": [0.0, 0.0, 1.0],
        "x_axis": [1.0, 0.0, 0.0],
        "phi_count": 2,
        "theta_count": 2,
    }


def _checkpoint_codec_payload(fields: np.ndarray) -> tuple[dict[str, object], torch.Tensor]:
    samples = np.stack(
        [
            fields,
            fields * 1.05 + 0.02,
            fields * 0.9 - 0.01,
        ],
        axis=0,
    )
    state = fit_ffs_codec(samples, rank=2)
    coeff = torch.tensor(encode_ffs(fields[np.newaxis, ...], state)[0], dtype=torch.float32)
    return {
        "field_shape": list(state.config.field_shape),
        "flat_dim": state.config.flat_dim,
        "rank": state.config.rank,
        "mean": state.mean.tolist(),
        "basis": state.basis.tolist(),
    }, coeff


class _ToyFfsSurrogate(_ToySurrogate):
    def __init__(self, coeff: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer(
            "ffs_coeff",
            coeff,
            persistent=False,
        )

    def forward_with_aux(
        self,
        points: torch.Tensor,
        ports: torch.Tensor,
        geom: torch.Tensor,
        frame: torch.Tensor,
        cuts: torch.Tensor,
        nibs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        s_pred = self.forward(points, ports, geom, frame, cuts, nibs)
        coeff = self.ffs_coeff.unsqueeze(0).expand(cuts.size(0), -1)
        return {"s_pred": s_pred, "ffs_coeff_pred": coeff}


def _toy_config() -> dict[str, object]:
    return {
        "mesh": {
            "verts": [
                [-40.0, -80.0, -4.0],
                [40.0, 80.0, 4.0],
            ]
        },
        "ports": [],
        "antennaConfig": {
            "frameWidth": 4.0,
            "gap": 10.0,
            "cuts": [{"position": "top", "distance": 18.0, "width": 20.0}],
            "nibs": [
                {"position": "top", "distance": 15.0, "width": 20.0, "thickness": 8.0},
                {"position": "left", "distance": -18.0, "width": 24.0, "thickness": 8.0},
                {"position": "bottom", "distance": 12.0, "width": 16.0, "thickness": 8.0},
            ],
        },
    }


class OptimizeBaselineTest(unittest.TestCase):
    def test_optimizer_emits_artifacts_and_keeps_distances_bounded(self):
        checkpoint = {
            "freq_grid": [1.0e9, 1.5e9, 2.0e9],
            "port_count": 3,
            "sample_points": 8,
        }
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            result = optimize_model(
                model=_ToySurrogate(),
                checkpoint=checkpoint,
                config=_toy_config(),
                output_dir=output_dir,
                steps=12,
                lr=0.2,
                top_k=2,
            )

            self.assertGreater(result["trace"][0]["loss"], result["trace"][-1]["loss"])
            self.assertTrue((output_dir / "optimization_trace.json").exists())
            self.assertTrue((output_dir / "optimized_soft_solution.json").exists())
            self.assertTrue((output_dir / "candidate_ranking.json").exists())
            self.assertTrue((output_dir / "candidate_01.json").exists())
            self.assertTrue((output_dir / "candidate_02.json").exists())

            soft = json.loads((output_dir / "optimized_soft_solution.json").read_text())
            self.assertLessEqual(abs(soft["cut_distances"][0]), 30.0)
            self.assertEqual(len(soft["nib_distances"]), 3)
            self.assertNotAlmostEqual(soft["cut_distances"][0], 18.0, places=3)
            self.assertTrue(any(abs(value - start) > 1e-3 for value, start in zip(soft["nib_distances"], [15.0, -18.0, 12.0], strict=False)))

    def test_farfield_mode_uses_predicted_ffs_and_emits_artifacts(self):
        fields = _toy_ffs_fields()
        codec_payload, coeff = _checkpoint_codec_payload(fields)
        checkpoint = {
            "freq_grid": [1.0e9, 1.5e9, 2.0e9],
            "port_count": 3,
            "sample_points": 8,
            "ffs_codec": codec_payload,
            "ffs_metadata": _toy_ffs_metadata(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            result = optimize_model(
                model=_ToyFfsSurrogate(coeff),
                checkpoint=checkpoint,
                config=_toy_config(),
                output_dir=output_dir,
                steps=10,
                lr=0.15,
                top_k=2,
                efficiency_mode="farfield",
            )

            self.assertGreater(result["trace"][0]["loss"], result["trace"][-1]["loss"])
            self.assertTrue((output_dir / "optimization_trace.json").exists())
            self.assertTrue((output_dir / "optimized_soft_solution.json").exists())
            self.assertTrue((output_dir / "candidate_ranking.json").exists())
            self.assertTrue((output_dir / "candidate_01.json").exists())
            self.assertTrue((output_dir / "candidate_02.json").exists())
            self.assertLess(checkpoint["ffs_codec"]["rank"], checkpoint["ffs_codec"]["flat_dim"])
            self.assertGreater(len(result["trace"]), 0)

    def test_farfield_mode_rejects_empty_frequency_overlap(self):
        fields = _toy_ffs_fields()
        codec_payload, coeff = _checkpoint_codec_payload(fields)
        checkpoint = {
            "freq_grid": [1.0e9, 1.5e9, 2.0e9],
            "port_count": 3,
            "sample_points": 8,
            "ffs_codec": codec_payload,
            "ffs_metadata": _toy_ffs_metadata(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                optimize_model(
                    model=_ToyFfsSurrogate(coeff),
                    checkpoint=checkpoint,
                    config=_toy_config(),
                    output_dir=Path(tmp),
                    steps=2,
                    lr=0.1,
                    top_k=1,
                    band_min=2.5e9,
                    band_max=2.6e9,
                    efficiency_mode="farfield",
                )


if __name__ == "__main__":
    unittest.main()
