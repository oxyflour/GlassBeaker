from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from baseline.data import load_dataset, split_records, stack_records
from baseline.model import SpectrumPredictor
from baseline.plotting import save_matrix_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the minimal Nijika S-parameter baseline.")
    parser.add_argument("--dataset-root", type=Path, default=Path("tmp/antenna-dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/nijika-baseline"))
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--freq-bins", type=int, default=201)
    parser.add_argument("--points", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def composite_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = nn.functional.mse_loss(pred, target)
    pred_pair = pred.view(pred.size(0), pred.size(1), -1, 2)
    target_pair = target.view(target.size(0), target.size(1), -1, 2)
    pred_mag = torch.linalg.vector_norm(pred_pair, dim=-1)
    target_mag = torch.linalg.vector_norm(target_pair, dim=-1)
    return mse + 0.1 * nn.functional.mse_loss(pred_mag, target_mag)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for points, ports, geom, target in loader:
            pred = model(points.to(device), ports.to(device), geom.to(device))
            losses.append(composite_loss(pred, target.to(device)).item())
    return float(np.mean(losses))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_dataset(args.dataset_root, n_points=args.points, freq_bins=args.freq_bins)
    train_records, val_records = split_records(bundle.records, seed=args.seed)
    train_tensors = stack_records(train_records)
    val_tensors = stack_records(val_records)
    target_mean = train_tensors["target"].mean(dim=(0, 1), keepdim=True)
    target_std = train_tensors["target"].std(dim=(0, 1), keepdim=True).clamp_min(1e-4)
    train_target = (train_tensors["target"] - target_mean) / target_std
    val_target = (val_tensors["target"] - target_mean) / target_std
    train_loader = DataLoader(
        TensorDataset(train_tensors["points"], train_tensors["ports"], train_tensors["geom"], train_target),
        batch_size=min(args.batch_size, len(train_records)),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_tensors["points"], val_tensors["ports"], val_tensors["geom"], val_target),
        batch_size=min(args.batch_size, len(val_records)),
        shuffle=False,
    )
    model = SpectrumPredictor(freq_bins=args.freq_bins, port_count=bundle.port_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = {"epoch": 0, "loss": float("inf"), "state": None}
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for points, ports, geom, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(points.to(device), ports.to(device), geom.to(device))
            loss = composite_loss(pred, target.to(device))
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        val_loss = evaluate(model, val_loader, device)
        if val_loss < best["loss"]:
            best = {"epoch": epoch, "loss": val_loss, "state": copy.deepcopy(model.state_dict())}
        if epoch == 1 or epoch % 50 == 0:
            print(
                f"epoch={epoch:03d} train_loss={np.mean(batch_losses):.4f} "
                f"val_loss={val_loss:.4f} device={device.type}"
            )
    model.load_state_dict(best["state"])
    model.eval()
    example = val_records[0]
    with torch.no_grad():
        example_pred = model(
            torch.tensor(example.points, dtype=torch.float32, device=device).unsqueeze(0),
            torch.tensor(example.ports, dtype=torch.float32, device=device).unsqueeze(0),
            torch.tensor(example.geom, dtype=torch.float32, device=device).unsqueeze(0),
        ).cpu()
    example_pred = example_pred * target_std + target_mean
    example_truth = torch.tensor(example.target).unsqueeze(0)
    rmse = torch.sqrt(torch.mean((example_pred - example_truth) ** 2)).item()
    plot_path = args.output_dir / f"{example.name}_matrix_db.png"
    save_matrix_plot(
        path=plot_path,
        freq_grid=bundle.freq_grid,
        truth=example.target,
        pred=example_pred.squeeze(0).numpy(),
        title=f"{example.name} magnitude comparison",
        port_count=bundle.port_count,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "baseline_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "freq_grid": bundle.freq_grid.tolist(),
            "port_count": bundle.port_count,
            "target_mean": target_mean.squeeze(0).squeeze(0).tolist(),
            "target_std": target_std.squeeze(0).squeeze(0).tolist(),
            "sample_points": args.points,
        },
        model_path,
    )
    metrics = {
        "device": device.type,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "best_epoch": best["epoch"],
        "best_val_loss": best["loss"],
        "example_sample": example.name,
        "example_rmse": rmse,
        "plot_path": str(plot_path),
        "model_path": str(model_path),
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
