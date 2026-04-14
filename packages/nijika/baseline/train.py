from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from baseline.data import load_dataset, split_records, stack_records
from baseline.model import create_model
from baseline.plotting import save_matrix_plot
from baseline.training_utils import composite_loss, evaluate

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Nijika S-parameter baseline.")
    parser.add_argument("--dataset-root", type=Path, default=Path("tmp/antenna-dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/nijika-baseline"))
    parser.add_argument(
        "--model-kind",
        choices=["structured_pair_pole_residue_head", "structured_pair_spectral_head", "structured_pair_split_decoder", "structured_token_decoder", "symmetric_freq_decoder", "legacy_global_head"],
        default="structured_pair_spectral_head",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--freq-bins", type=int, default=201)
    parser.add_argument("--points", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--num-poles", type=int, default=12)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mag-weight", type=float, default=0.2)
    parser.add_argument("--smooth-weight", type=float, default=0.05)
    parser.add_argument("--db-weight", type=float, default=0.0)
    parser.add_argument("--coupling-weight", type=float, default=1.0)
    parser.add_argument("--notch-weight", type=float, default=0.0)
    parser.add_argument("--notch-threshold-db", type=float, default=-20.0)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    return parser.parse_args()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def build_loss_config(args: argparse.Namespace) -> dict[str, float]:
    return {
        "mag_weight": args.mag_weight,
        "smooth_weight": args.smooth_weight,
        "db_weight": args.db_weight,
        "coupling_weight": args.coupling_weight,
        "notch_weight": args.notch_weight,
        "notch_threshold_db": args.notch_threshold_db,
    }

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_config = build_loss_config(args)
    bundle = load_dataset(args.dataset_root, n_points=args.points, freq_bins=args.freq_bins)
    train_records, val_records = split_records(bundle.records, seed=args.seed)
    train_tensors = stack_records(train_records)
    val_tensors = stack_records(val_records)
    target_mean = train_tensors["target"].mean(dim=(0, 1), keepdim=True)
    target_std = train_tensors["target"].std(dim=(0, 1), keepdim=True).clamp_min(1e-4)
    target_mean_device = target_mean.to(device)
    target_std_device = target_std.to(device)
    train_target = (train_tensors["target"] - target_mean) / target_std
    val_target = (val_tensors["target"] - target_mean) / target_std
    train_loader = DataLoader(
        TensorDataset(
            train_tensors["points"],
            train_tensors["ports"],
            train_tensors["geom"],
            train_tensors["frame"],
            train_tensors["cuts"],
            train_tensors["nibs"],
            train_target,
        ),
        batch_size=min(args.batch_size, len(train_records)),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            val_tensors["points"],
            val_tensors["ports"],
            val_tensors["geom"],
            val_tensors["frame"],
            val_tensors["cuts"],
            val_tensors["nibs"],
            val_target,
        ),
        batch_size=min(args.batch_size, len(val_records)),
        shuffle=False,
    )
    model_config = {"hidden_dim": args.hidden_dim, "dropout": 0.1, "freq_bands": 8, "num_poles": args.num_poles}
    model = create_model(
        freq_grid=bundle.freq_grid,
        port_count=bundle.port_count,
        model_kind=args.model_kind,
        model_config=model_config,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = min(args.warmup_epochs, args.epochs)
    best = {"epoch": 0, "score": float("inf"), "state": None, "metrics": None}
    for epoch in range(1, args.epochs + 1):
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            lr_scale = epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
            lr_scale = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr * lr_scale
        model.train()
        batch_losses = []
        for points, ports, geom, frame, cuts, nibs, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(
                points.to(device),
                ports.to(device),
                geom.to(device),
                frame.to(device),
                cuts.to(device),
                nibs.to(device),
            )
            loss = composite_loss(
                pred,
                target.to(device),
                port_count=bundle.port_count,
                target_mean=target_mean_device,
                target_std=target_std_device,
                loss_config=loss_config,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(loss.item())
        val_metrics, _, _ = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
            port_count=bundle.port_count,
            loss_config=loss_config,
        )
        score = float(val_metrics["db_mae"])
        if score < best["score"]:
            best = {"epoch": epoch, "score": score, "state": copy.deepcopy(model.state_dict()), "metrics": val_metrics}
        if epoch == 1 or epoch % 50 == 0:
            print(
                f"epoch={epoch:03d} train_loss={np.mean(batch_losses):.4f} "
                f"val_rmse={val_metrics['rmse']:.4f} val_db_mae={val_metrics['db_mae']:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} device={device.type}"
            )
    model.load_state_dict(best["state"])
    final_metrics, val_pred, val_truth = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
        port_count=bundle.port_count,
        loss_config=loss_config,
    )
    example = val_records[0]
    plot_path = args.output_dir / f"{example.name}_matrix_db.png"
    save_matrix_plot(
        path=plot_path,
        freq_grid=bundle.freq_grid,
        truth=val_truth[0].numpy(),
        pred=val_pred[0].numpy(),
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
            "model_kind": args.model_kind,
            "model_config": model_config,
            "loss_config": loss_config,
        },
        model_path,
    )
    metrics = {
        "device": device.type,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "best_epoch": best["epoch"],
        "best_val_db_mae": best["score"],
        "val_rmse": final_metrics["rmse"],
        "val_db_mae": final_metrics["db_mae"],
        "val_db_rmse": final_metrics["db_rmse"],
        "example_sample": example.name,
        "example_rmse": final_metrics["sample_rmse"][0],
        "example_db_mae": final_metrics["sample_db_mae"][0],
        "plot_path": str(plot_path),
        "model_path": str(model_path),
        "model_kind": args.model_kind,
        "model_config": model_config,
        "loss_config": loss_config,
        "warmup_epochs": warmup_epochs,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
