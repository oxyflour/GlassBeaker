from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import json
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baseline import predict as predict_module  # noqa: E402
from baseline import train as train_module  # noqa: E402
from baseline.ffs_codec import TorchFfsCodec, encode_ffs, fit_ffs_codec  # noqa: E402
from baseline.ffs_io import load_ffs_sample  # noqa: E402
from baseline.training_utils import ffs_aux_loss  # noqa: E402
from optimize_baseline import load_checkpoint_model, optimize_model  # noqa: E402
from optimizer_torch_farfield import integrate_decoded_ffs_power  # noqa: E402

DATASET_ROOT = Path("C:/Projects/GlassBeaker/tmp/dataset-v3-ffs")


def _copy_fixture(root: Path, names: list[str]) -> None:
    for name in names:
        shutil.copy2(DATASET_ROOT / f"{name}.json", root / f"{name}.json")
        shutil.copytree(DATASET_ROOT / name, root / name)


def _synthetic_ffs_samples(samples: int = 6) -> torch.Tensor:
    phi = torch.tensor([0.0, 180.0, 360.0], dtype=torch.float64)
    theta = torch.tensor([0.0, 90.0], dtype=torch.float64)
    phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing="ij")
    basis_a = torch.stack(
        [
            phi_grid / 360.0,
            theta_grid / 90.0,
            (phi_grid + theta_grid) / 450.0,
            (phi_grid - theta_grid) / 450.0,
        ],
        dim=-1,
    )
    basis_b = torch.stack(
        [
            torch.cos(torch.deg2rad(phi_grid)),
            torch.sin(torch.deg2rad(theta_grid)),
            torch.cos(torch.deg2rad(phi_grid + theta_grid)),
            torch.sin(torch.deg2rad(phi_grid - theta_grid)),
        ],
        dim=-1,
    )
    fields = []
    for index in range(samples):
        sample = []
        for port in range(2):
            port_fields = []
            for freq in range(2):
                weight_a = 0.2 * (index + 1) + 0.1 * port + 0.05 * freq
                weight_b = -0.15 * (index + 1) + 0.03 * port - 0.02 * freq
                port_fields.append((weight_a * basis_a + weight_b * basis_b).reshape(-1, 4))
            sample.append(torch.stack(port_fields, dim=0))
        fields.append(torch.stack(sample, dim=0))
    return torch.stack(fields, dim=0)


class FfsTrainPredictTest(unittest.TestCase):
    def test_ffs_aux_loss_backpropagates_field_and_power_terms(self) -> None:
        fields = _synthetic_ffs_samples()
        state = fit_ffs_codec(fields.numpy(), rank=2)
        codec = TorchFfsCodec.from_state(state, dtype=torch.float32)
        target_field = fields[:1].to(dtype=torch.float32)
        target_coeff = torch.tensor(encode_ffs(target_field.numpy(), state), dtype=torch.float32)
        pred_coeff = (target_coeff + torch.tensor([[0.2, -0.15]], dtype=torch.float32)).requires_grad_(True)
        phi = torch.tensor([0.0, torch.pi], dtype=torch.float64)
        theta = torch.tensor([0.0, torch.pi / 2.0], dtype=torch.float64)
        target_radiated_power = integrate_decoded_ffs_power(
            target_field,
            phi=phi,
            theta=theta,
            phi_count=3,
            theta_count=2,
            has_phi_closure=True,
        )

        loss, parts = ffs_aux_loss(
            pred_coeff=pred_coeff,
            target_coeff=target_coeff,
            target_field=target_field,
            target_radiated_power=target_radiated_power,
            codec=codec,
            phi=phi,
            theta=theta,
            phi_count=3,
            theta_count=2,
            has_phi_closure=True,
            loss_weights={"coeff": 0.0, "field": 1.0, "power": 1.0},
        )
        loss.backward()

        self.assertIn("ffs_field_loss", parts)
        self.assertIn("ffs_power_loss", parts)
        self.assertGreater(parts["ffs_field_loss"].item(), 0.0)
        self.assertGreater(parts["ffs_power_loss"].item(), 0.0)
        self.assertEqual(parts["ffs_power_loss"].dtype, pred_coeff.dtype)
        self.assertEqual(loss.dtype, pred_coeff.dtype)
        self.assertIsNotNone(pred_coeff.grad)
        assert pred_coeff.grad is not None
        self.assertTrue(torch.isfinite(pred_coeff.grad).all().item())
        self.assertGreater(pred_coeff.grad.abs().sum().item(), 0.0)

    def _run_train(self, dataset_root: Path, train_output: Path) -> Path:
        with mock.patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(train_output),
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--points",
                "8",
                "--freq-bins",
                "11",
                "--hidden-dim",
                "16",
                "--lr",
                "1e-3",
                "--model-kind",
                "structured_pair_spectral_ffs_head",
                "--ffs-rank",
                "8",
                "--ffs-loss-weight",
                "1.0",
            ],
        ):
            train_module.main()
        return train_output / "baseline_model.pt"

    def test_train_checkpoint_and_predict_export_ffs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            train_output = root / "train-output"
            predict_output = root / "predict-output"
            dataset_root.mkdir()
            _copy_fixture(dataset_root, ["antenna_000", "antenna_001"])
            checkpoint_path = self._run_train(dataset_root, train_output)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.assertIn("ffs_codec", checkpoint)
            self.assertIn("ffs_metadata", checkpoint)

            with mock.patch.object(
                sys,
                "argv",
                [
                    "predict.py",
                    "--dataset-root",
                    str(dataset_root),
                    "--model-path",
                    str(checkpoint_path),
                    "--sample-name",
                    "antenna_000",
                    "--output-dir",
                    str(predict_output),
                ],
            ):
                predict_module.main()

            ffs_dir = predict_output / "antenna_000_predicted_ffs"
            exported = sorted(ffs_dir.glob("*.ffs"))
            self.assertEqual(
                len(exported),
                int(checkpoint["port_count"]) * len(checkpoint["ffs_metadata"]["frequencies_hz"]),
            )
            metadata, field = load_ffs_sample(exported[0])
            self.assertEqual(field.shape[0], len(metadata.frequencies_hz))
            self.assertTrue((metadata.radiated_power_w > 0.0).all())

    def test_trained_ffs_checkpoint_runs_farfield_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            train_output = root / "train-output"
            optimize_output = root / "optimize-output"
            dataset_root.mkdir()
            _copy_fixture(dataset_root, ["antenna_000", "antenna_001"])
            checkpoint_path = self._run_train(dataset_root, train_output)
            checkpoint, model = load_checkpoint_model(checkpoint_path)
            config = json.loads((dataset_root / "antenna_000.json").read_text())

            result = optimize_model(
                model=model,
                checkpoint=checkpoint,
                config=config,
                output_dir=optimize_output,
                steps=2,
                lr=0.05,
                top_k=1,
                efficiency_mode="farfield",
            )

            self.assertTrue((optimize_output / "optimization_trace.json").exists())
            self.assertTrue((optimize_output / "optimized_soft_solution.json").exists())
            self.assertTrue((optimize_output / "candidate_ranking.json").exists())
            self.assertGreater(len(result["trace"]), 0)
            self.assertGreater(len(result["ranking"]), 0)


if __name__ == "__main__":
    unittest.main()
