from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baseline.model import create_model
from baseline.training_utils import composite_loss, evaluate, forward_model
from torch.utils.data import DataLoader, TensorDataset


def _synthetic_data(seed: int = 42, samples: int = 16, freq_bins: int = 51, port_count: int = 3):
    gen = torch.Generator()
    gen.manual_seed(seed)
    b = samples
    f = freq_bins
    p = port_count

    def _rand(*shape, lo=0.0, hi=1.0):
        return (lo + (hi - lo) * torch.rand(*shape, generator=gen)).float()

    # S-parameters: smooth damped sinusoids per pair, varying with frequency
    t = torch.linspace(0, 1, f).view(1, f, 1, 1, 1)  # (1, f, 1, 1, 1)
    amp = torch.rand(b, 1, p, p, 3, generator=gen).float() * 0.08  # (b, 1, p, p, 3)
    phase = torch.arange(3).float().view(1, 1, 1, 1, 3)  # (1, 1, 1, 1, 3)
    real = (amp * torch.sin(t * math.pi * (1.0 + 2.0 * phase))).sum(dim=-1)  # (b, f, p, p)
    imag = (amp * torch.cos(t * math.pi * (3.0 + 1.5 * phase))).sum(dim=-1) * 0.3  # (b, f, p, p)
    target = torch.stack([real, imag], dim=-1)  # (b, f, p, p, 2)
    # Enforce reciprocity and passivity
    real_sym = 0.5 * (real + real.transpose(2, 3))
    imag_sym = 0.5 * (imag + imag.transpose(2, 3))
    target = torch.stack([real_sym, imag_sym], dim=-1)
    target_norm = torch.linalg.vector_norm(target, dim=-1, keepdim=True).clamp_min(1e-6)
    too_large = target_norm > 0.99
    target = torch.where(too_large, target * (0.99 / target_norm), target)
    target_flat = target.reshape(b, f, p * p * 2)

    points = _rand(b, 128, 3, lo=-40, hi=40)
    ports = _rand(b, p, 6, lo=-40, hi=40)
    geom = _rand(b, 6, lo=-40, hi=40)
    frame = _rand(b, 6)
    cuts = _rand(b, 4, 7)
    nibs = _rand(b, 4, 8)
    return {
        "points": points,
        "ports": ports,
        "geom": geom,
        "frame": frame,
        "cuts": cuts,
        "nibs": nibs,
    }, target_flat, port_count


class TrainRegressionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.device = torch.device("cpu")
        self.tensors, self.target, self.port_count = _synthetic_data(seed=42)
        self.freq_grid = torch.linspace(1e9, 3e9, 51)
        split = int(len(self.target) * 0.75)
        self.train_tensors = {k: v[:split] for k, v in self.tensors.items()}
        self.val_tensors = {k: v[split:] for k, v in self.tensors.items()}
        self.train_target = self.target[:split]
        self.val_target = self.target[split:]
        self.target_mean = self.train_target.mean(dim=(0, 1), keepdim=True)
        self.target_std = self.train_target.std(dim=(0, 1), keepdim=True).clamp_min(1e-4)

    def _make_loader(self, tensors, target, shuffle):
        ds = TensorDataset(
            tensors["points"], tensors["ports"], tensors["geom"],
            tensors["frame"], tensors["cuts"], tensors["nibs"], target,
        )
        return DataLoader(ds, batch_size=4, shuffle=shuffle)

    def test_training_loss_decreases_with_physics_constraints(self):
        """Train for 50 epochs; loss must decrease and physics violations stay bounded."""
        model = create_model(
            freq_grid=self.freq_grid,
            port_count=self.port_count,
            model_kind="structured_pair_spectral_head",
            model_config={"hidden_dim": 32, "dropout": 0.0},
        ).to(self.device)

        train_loader = self._make_loader(self.train_tensors, self.train_target, shuffle=True)
        val_loader = self._make_loader(self.val_tensors, self.val_target, shuffle=False)

        opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
        target_mean_dev = self.target_mean.to(self.device)
        target_std_dev = self.target_std.to(self.device)

        loss_config = {
            "mag_weight": 0.2,
            "smooth_weight": 0.05,
            "db_weight": 0.1,
            "coupling_weight": 1.0,
            "notch_weight": 0.0,
            "notch_threshold_db": -20.0,
            "reciprocity_weight": 0.5,
            "passivity_weight": 0.5,
        }

        initial_loss = None
        final_loss = None

        for epoch in range(1, 51):
            model.train()
            for batch in train_loader:
                points, ports, geom, frame, cuts, nibs, target = batch
                opt.zero_grad(set_to_none=True)
                pred = forward_model(
                    model, points=points, ports=ports, geom=geom,
                    frame=frame, cuts=cuts, nibs=nibs, device=self.device,
                )
                loss = composite_loss(
                    pred, target.to(self.device),
                    port_count=self.port_count,
                    target_mean=target_mean_dev,
                    target_std=target_std_dev,
                    loss_config=loss_config,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                if initial_loss is None:
                    initial_loss = loss.item()

            if epoch % 10 == 0:
                val_metrics, _, _ = evaluate(
                    model=model, loader=val_loader, device=self.device,
                    target_mean=self.target_mean, target_std=self.target_std,
                    port_count=self.port_count, loss_config=loss_config,
                )
                print(f"  epoch={epoch:02d} val_db_mae={val_metrics['db_mae']:.4f}")

            final_loss = loss.item()

        self.assertLess(final_loss, initial_loss * 0.8,
                        f"Training loss did not decrease enough: {initial_loss:.4f} -> {final_loss:.4f}")

        # Evaluate final physics violations
        from baseline.training_utils import reciprocity_loss, passivity_loss
        model.eval()
        val_metrics, val_pred, val_truth = evaluate(
            model=model, loader=val_loader, device=self.device,
            target_mean=self.target_mean, target_std=self.target_std,
            port_count=self.port_count, loss_config=loss_config,
        )
        recip = reciprocity_loss(val_pred, self.port_count).item()
        passiv = passivity_loss(val_pred, self.port_count).item()
        self.assertLess(recip, 2e-4, f"Reciprocity violation too high: {recip:.6f}")
        self.assertLess(passiv, 2e-4, f"Passivity violation too high: {passiv:.6f}")
        print(f"  reciprocity_mse={recip:.6f}  passivity_mse={passiv:.6f}")

    def test_mc_dropout_produces_meaningful_uncertainty(self):
        """MC dropout with dropout>0 must produce non-zero variance."""
        model = create_model(
            freq_grid=self.freq_grid,
            port_count=self.port_count,
            model_kind="structured_pair_spectral_head",
            model_config={"hidden_dim": 32, "dropout": 0.1},
        ).to(self.device)

        # Train briefly so model isn't at random init
        train_loader = self._make_loader(self.train_tensors, self.train_target, shuffle=True)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
        loss_config = {
            "mag_weight": 0.2, "smooth_weight": 0.05, "db_weight": 0.1,
            "coupling_weight": 1.0, "notch_weight": 0.0,
            "notch_threshold_db": -20.0, "reciprocity_weight": 0.5, "passivity_weight": 0.5,
        }
        target_mean_dev = self.target_mean.to(self.device)
        target_std_dev = self.target_std.to(self.device)
        model.train()
        for _ in range(20):
            for batch in train_loader:
                points, ports, geom, frame, cuts, nibs, target = batch
                opt.zero_grad(set_to_none=True)
                pred = forward_model(
                    model, points=points, ports=ports, geom=geom,
                    frame=frame, cuts=cuts, nibs=nibs, device=self.device,
                )
                loss = composite_loss(
                    pred, target.to(self.device), port_count=self.port_count,
                    target_mean=target_mean_dev, target_std=target_std_dev, loss_config=loss_config,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

        from baseline.uncertainty import mc_predict, uncertainty_summary
        model.eval()
        mean, std = mc_predict(
            model,
            points=self.val_tensors["points"][:1],
            ports=self.val_tensors["ports"][:1],
            geom=self.val_tensors["geom"][:1],
            frame=self.val_tensors["frame"][:1],
            cuts=self.val_tensors["cuts"][:1],
            nibs=self.val_tensors["nibs"][:1],
            device=self.device,
            n_samples=30,
        )
        self.assertEqual(mean.shape, (1, 51, self.port_count * self.port_count * 2))
        self.assertEqual(std.shape, (1, 51, self.port_count * self.port_count * 2))
        self.assertGreater(std.mean().item(), 1e-6, "MC dropout std should be > 0")

        summary = uncertainty_summary(mean, std, self.port_count)
        self.assertGreater(summary["mean_abs_std"], 1e-6)
        self.assertGreater(summary["mean_rel_std"], 0.0)
        print(f"  mc_uncertainty: mean_abs_std={summary['mean_abs_std']:.6f} mean_rel_std={summary['mean_rel_std']:.6f}")


if __name__ == "__main__":
    unittest.main()
