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
from optimizer_geometry import mesh_geom  # noqa: E402
from optimizer_inputs import build_optimizer_inputs  # noqa: E402
from optimizer_runner import (  # noqa: E402
    _decode_ffs_coefficients,
    _decoded_ffs_basis,
    _farfield_efficiency,
    _farfield_grid,
    _farfield_objective,
    _interpolate_frequency_axis,
    _match_farfield_indices,
    _sample_points,
)


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


def _toy_ffs_fields(port_scales: tuple[float, float, float] = (1.0, 1.25, 1.5)) -> np.ndarray:
    fields = np.zeros((3, 3, 4, 4), dtype=np.float32)
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
            fields[port, freq] = base * (port_scales[port] + 0.1 * freq)
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


def _checkpoint_codec_payload(
    fields: np.ndarray,
    *,
    sample_fields: list[np.ndarray] | None = None,
) -> tuple[dict[str, object], torch.Tensor]:
    samples = np.stack(
        sample_fields
        if sample_fields is not None
        else [fields, fields * 1.05 + 0.02, fields * 0.9 - 0.01],
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


def _pair_spectrum(
    scale: torch.Tensor,
    *,
    diag: tuple[float, float, float] = (0.15, 0.35, 0.35),
    coupling_01: float = 0.0,
    coupling_02: float = 0.0,
    coupling_12: float = 0.0,
) -> torch.Tensor:
    pair = torch.zeros((scale.size(0), 3, 3, 3, 2), dtype=scale.dtype, device=scale.device)
    for idx, value in enumerate(diag):
        pair[:, :, idx, idx, 0] = value
    pair[:, :, 0, 1, 0] = coupling_01
    pair[:, :, 1, 0, 0] = coupling_01
    pair[:, :, 0, 2, 0] = coupling_02 + 0.6 * scale.view(-1, 1)
    pair[:, :, 2, 0, 0] = coupling_02 + 0.6 * scale.view(-1, 1)
    pair[:, :, 1, 2, 0] = coupling_12
    pair[:, :, 2, 1, 0] = coupling_12
    return pair.view(scale.size(0), 3, 18)


class _ToyFfsSurrogate(_ToySurrogate):
    def __init__(self, coeff: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("ffs_coeff", coeff, persistent=False)

    def forward_with_aux(
        self,
        points: torch.Tensor,
        ports: torch.Tensor,
        geom: torch.Tensor,
        frame: torch.Tensor,
        cuts: torch.Tensor,
        nibs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "s_pred": self.forward(points, ports, geom, frame, cuts, nibs),
            "ffs_coeff_pred": self.ffs_coeff.unsqueeze(0).expand(cuts.size(0), -1),
        }


class _GeometryDrivenToyFfsSurrogate(_ToySurrogate):
    def __init__(self, base_coeff: torch.Tensor, delta_coeff: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("base_coeff", base_coeff, persistent=False)
        self.register_buffer("delta_coeff", delta_coeff, persistent=False)

    def forward_with_aux(
        self,
        points: torch.Tensor,
        ports: torch.Tensor,
        geom: torch.Tensor,
        frame: torch.Tensor,
        cuts: torch.Tensor,
        nibs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        scale = torch.sigmoid(cuts[:, 0, 5] * 3.0 + nibs[:, 2, 5] * 1.5)
        return {
            "s_pred": self.forward(points, ports, geom, frame, cuts, nibs),
            "ffs_coeff_pred": self.base_coeff.unsqueeze(0) + scale.unsqueeze(-1) * self.delta_coeff.unsqueeze(0),
        }


class _InconsistentToyFfsSurrogate(torch.nn.Module):
    def __init__(self, base_coeff: torch.Tensor, delta_coeff: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("base_coeff", base_coeff, persistent=False)
        self.register_buffer("delta_coeff", delta_coeff, persistent=False)

    def forward(
        self,
        points: torch.Tensor,
        ports: torch.Tensor,
        geom: torch.Tensor,
        frame: torch.Tensor,
        cuts: torch.Tensor,
        nibs: torch.Tensor,
    ) -> torch.Tensor:
        del points, ports, geom, frame, nibs
        return _pair_spectrum(torch.sigmoid(-cuts[:, 0, 5] * 4.0))

    def forward_with_aux(
        self,
        points: torch.Tensor,
        ports: torch.Tensor,
        geom: torch.Tensor,
        frame: torch.Tensor,
        cuts: torch.Tensor,
        nibs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del points, ports, geom, frame, nibs
        scale = torch.sigmoid(cuts[:, 0, 5] * 4.0)
        return {
            "s_pred": _pair_spectrum(scale),
            "ffs_coeff_pred": self.base_coeff.unsqueeze(0) + scale.unsqueeze(-1) * self.delta_coeff.unsqueeze(0),
        }


class _RankingToyFfsSurrogate(torch.nn.Module):
    def __init__(self, coeff: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("ffs_coeff", coeff, persistent=False)

    def forward(
        self,
        points: torch.Tensor,
        ports: torch.Tensor,
        geom: torch.Tensor,
        frame: torch.Tensor,
        cuts: torch.Tensor,
        nibs: torch.Tensor,
    ) -> torch.Tensor:
        del points, ports, geom, frame, nibs
        zeros = torch.zeros((cuts.size(0),), dtype=cuts.dtype, device=cuts.device)
        return _pair_spectrum(zeros, diag=(0.2, 0.3, 0.4), coupling_01=0.8, coupling_02=0.2, coupling_12=0.1)

    def forward_with_aux(
        self,
        points: torch.Tensor,
        ports: torch.Tensor,
        geom: torch.Tensor,
        frame: torch.Tensor,
        cuts: torch.Tensor,
        nibs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "s_pred": self.forward(points, ports, geom, frame, cuts, nibs),
            "ffs_coeff_pred": self.ffs_coeff.unsqueeze(0),
        }


def _toy_config() -> dict[str, object]:
    return {
        "mesh": {"verts": [[-40.0, -80.0, -4.0], [40.0, 80.0, 4.0]]},
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
    def _farfield_checkpoint(
        self,
        fields: np.ndarray,
        *,
        sample_fields: list[np.ndarray] | None = None,
    ) -> tuple[dict[str, object], torch.Tensor]:
        codec_payload, coeff = _checkpoint_codec_payload(fields, sample_fields=sample_fields)
        return {
            "freq_grid": [1.0e9, 1.5e9, 2.0e9],
            "port_count": 3,
            "sample_points": 8,
            "ffs_codec": codec_payload,
            "ffs_metadata": _toy_ffs_metadata(),
        }, coeff

    def _coherent_farfield_loss(self, model: torch.nn.Module, checkpoint: dict[str, object]) -> float:
        config = _toy_config()
        tensor_ref = next(iter(model.buffers()), None)
        device = tensor_ref.device if tensor_ref is not None else torch.device("cpu")
        geom = mesh_geom(config)
        antenna = config["antennaConfig"]
        model_inputs = build_optimizer_inputs(
            config,
            points=_sample_points(config, int(checkpoint["sample_points"])),
            geom=geom,
            cut_distances=torch.tensor([item["distance"] for item in antenna["cuts"]], dtype=torch.float32, device=device),
            nib_distances=torch.tensor([item["distance"] for item in antenna["nibs"]], dtype=torch.float32, device=device),
            device=device,
            include_graph=False,
        )
        aux = model.forward_with_aux(
            model_inputs["points"],
            model_inputs["ports"],
            model_inputs["geom"],
            model_inputs["frame"],
            model_inputs["cuts"],
            model_inputs["nibs"],
        )
        lower, upper, alpha, _ = _match_farfield_indices(checkpoint, device)
        phi, theta, _ = _farfield_grid(checkpoint["ffs_metadata"], device)
        pair = aux["s_pred"][0].view(-1, 3, 3, 2)
        s_pred = torch.complex(
            _interpolate_frequency_axis(pair[..., 0], lower, upper, alpha),
            _interpolate_frequency_axis(pair[..., 1], lower, upper, alpha),
        )
        decoded = _decode_ffs_coefficients(aux["ffs_coeff_pred"], checkpoint)
        basis, _, _ = _decoded_ffs_basis(decoded, checkpoint["ffs_metadata"])
        eff_loss, _ = _farfield_objective(
            s_pred,
            ffs_basis=basis[0],
            phi=phi,
            theta=theta,
            feed_logits=torch.zeros(3, dtype=torch.float32, device=device),
            termination_logits=torch.zeros(3, dtype=torch.float32, device=device),
        )
        return float(eff_loss.item())

    def test_optimizer_emits_artifacts_and_keeps_distances_bounded(self):
        checkpoint = {"freq_grid": [1.0e9, 1.5e9, 2.0e9], "port_count": 3, "sample_points": 8}
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

    def test_farfield_mode_optimizes_geometry_dependent_ffs_and_emits_artifacts(self):
        low_fields = _toy_ffs_fields((1.0, 1.2, 0.6))
        mid_fields = _toy_ffs_fields((1.0, 1.2, 1.2))
        high_fields = _toy_ffs_fields((1.0, 1.2, 2.0))
        checkpoint, low_coeff = self._farfield_checkpoint(low_fields, sample_fields=[low_fields, mid_fields, high_fields])
        high_coeff = _checkpoint_codec_payload(high_fields, sample_fields=[low_fields, mid_fields, high_fields])[1]
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            result = optimize_model(
                model=_GeometryDrivenToyFfsSurrogate(low_coeff, high_coeff - low_coeff),
                checkpoint=checkpoint,
                config=_toy_config(),
                output_dir=output_dir,
                steps=10,
                lr=0.15,
                top_k=2,
                efficiency_mode="farfield",
            )

            self.assertGreater(result["trace"][0]["loss"], result["trace"][-1]["loss"])
            self.assertNotAlmostEqual(result["trace"][0]["cut_distances"][0], result["trace"][-1]["cut_distances"][0], places=3)
            self.assertTrue((output_dir / "optimization_trace.json").exists())
            self.assertTrue((output_dir / "optimized_soft_solution.json").exists())
            self.assertTrue((output_dir / "candidate_ranking.json").exists())
            self.assertTrue((output_dir / "candidate_01.json").exists())
            self.assertTrue((output_dir / "candidate_02.json").exists())

    def test_farfield_mode_uses_aux_s_predictions_for_loss(self):
        low_fields = _toy_ffs_fields((1.4, 0.4, 0.2))
        mid_fields = _toy_ffs_fields((1.4, 0.4, 1.2))
        high_fields = _toy_ffs_fields((1.4, 0.4, 2.2))
        checkpoint, low_coeff = self._farfield_checkpoint(low_fields, sample_fields=[low_fields, mid_fields, high_fields])
        high_coeff = _checkpoint_codec_payload(high_fields, sample_fields=[low_fields, mid_fields, high_fields])[1]
        model = _InconsistentToyFfsSurrogate(low_coeff, high_coeff - low_coeff)
        with tempfile.TemporaryDirectory() as tmp:
            result = optimize_model(
                model=model,
                checkpoint=checkpoint,
                config=_toy_config(),
                output_dir=Path(tmp),
                steps=1,
                lr=0.0,
                top_k=1,
                efficiency_mode="farfield",
            )
        self.assertAlmostEqual(result["trace"][0]["eff_loss"], self._coherent_farfield_loss(model, checkpoint), places=9)

    def test_farfield_candidate_ranking_uses_physical_port_ids(self):
        fields = _toy_ffs_fields((1.5, 1.2, 0.8))
        checkpoint, coeff = self._farfield_checkpoint(
            fields,
            sample_fields=[fields, _toy_ffs_fields((1.4, 1.2, 0.8)), _toy_ffs_fields((1.6, 1.2, 0.8))],
        )
        model = _RankingToyFfsSurrogate(coeff)
        with tempfile.TemporaryDirectory() as tmp:
            result = optimize_model(
                model=model,
                checkpoint=checkpoint,
                config=_toy_config(),
                output_dir=Path(tmp),
                steps=1,
                lr=0.0,
                top_k=1,
                efficiency_mode="farfield",
            )
        candidate = next(
            item
            for item in result["ranking"]
            if item["feed_index"] == 0 and item["roles"][1]["termination"] == "open" and item["roles"][2]["termination"] == "open"
        )
        decoded = _decode_ffs_coefficients(coeff.unsqueeze(0), checkpoint)
        basis, phi, theta = _decoded_ffs_basis(decoded, checkpoint["ffs_metadata"])
        s_pair = model.forward_with_aux(
            torch.zeros((1, 2, 3), dtype=torch.float32),
            torch.zeros((1, 3, 6), dtype=torch.float32),
            torch.zeros((1, 6), dtype=torch.float32),
            torch.zeros((1, 6), dtype=torch.float32),
            torch.zeros((1, 4, 7), dtype=torch.float32),
            torch.zeros((1, 8, 8), dtype=torch.float32),
        )["s_pred"][0].view(-1, 3, 3, 2)
        s_matrix = torch.complex(s_pair[..., 0], s_pair[..., 1])
        expected = float(
            (-_farfield_efficiency(s_matrix, ffs_basis=basis[0], phi=phi, theta=theta, feed_index=0, terminations={1: "open", 2: "open"}).mean()).item()
        )
        self.assertAlmostEqual(candidate["score"], expected, places=8)

    def test_farfield_mode_rejects_empty_frequency_overlap(self):
        checkpoint, coeff = self._farfield_checkpoint(_toy_ffs_fields())
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
