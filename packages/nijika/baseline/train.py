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
from baseline.ffs_codec import TorchFfsCodec, encode_ffs, fit_ffs_codec
from baseline.model import create_model
from baseline.plotting import save_matrix_plot
from baseline.training_utils import GRAPH_KEYS, composite_loss, evaluate, ffs_aux_loss, forward_model, uses_graph_features

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Nijika S-parameter baseline.")
    parser.add_argument("--dataset-root", type=Path, default=Path("tmp/antenna-dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/nijika-baseline"))
    parser.add_argument(
        "--model-kind",
        choices=["graph_topology_spectral_head", "structured_pair_coupling_freq_head", "structured_pair_pole_offset_residue_head", "structured_pair_pole_residue_head", "structured_pair_spectral_head", "structured_pair_spectral_ffs_head", "structured_pair_split_decoder", "structured_pair_topology_spectral_head", "structured_token_decoder", "symmetric_freq_decoder", "temporal_pair_spectral_head", "transolver_pair_spectral_head", "legacy_global_head"],
        default="structured_pair_spectral_head",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--freq-bins", type=int, default=201)
    parser.add_argument("--points", type=int, default=128)
    parser.add_argument("--max-temporal-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--num-poles", type=int, default=12)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mag-weight", type=float, default=0.2)
    parser.add_argument("--smooth-weight", type=float, default=0.05)
    parser.add_argument("--db-weight", type=float, default=0.0)
    parser.add_argument("--db-weight-final", type=float)
    parser.add_argument("--coupling-weight", type=float, default=1.0)
    parser.add_argument("--coupling-weight-final", type=float)
    parser.add_argument("--notch-weight", type=float, default=0.0)
    parser.add_argument("--notch-weight-final", type=float)
    parser.add_argument("--notch-threshold-db", type=float, default=-20.0)
    parser.add_argument("--loss-ramp-start-epoch", type=int, default=0)
    parser.add_argument("--loss-ramp-end-epoch", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--ffs-rank", type=int, default=16)
    parser.add_argument("--ffs-loss-weight", type=float, default=1.0)
    parser.add_argument("--ffs-field-loss-weight", type=float, default=1.0)
    parser.add_argument("--ffs-power-loss-weight", type=float, default=0.25)
    parser.add_argument("--reciprocity-weight", type=float, default=0.0, help="Weight for reciprocity constraint loss (S_ij = S_ji)")
    parser.add_argument("--passivity-weight", type=float, default=0.0, help="Weight for passivity constraint loss (|S| <= 1)")
    parser.add_argument("--physics", action="store_true", default=False, help="Enable physics-aware training: sets reciprocity/passivity weights to 0.1 and uses physics-combined model scoring")
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
        "reciprocity_weight": getattr(args, "reciprocity_weight", 0.0),
        "passivity_weight": getattr(args, "passivity_weight", 0.0),
    }


def build_final_loss_config(args: argparse.Namespace, base_config: dict[str, float]) -> dict[str, float] | None:
    final_config = dict(base_config)
    changed = False
    for arg_name, key in (
        ("db_weight_final", "db_weight"),
        ("coupling_weight_final", "coupling_weight"),
        ("notch_weight_final", "notch_weight"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            final_config[key] = float(value)
            changed = True
    return final_config if changed else None


def scheduled_loss_config(
    base_config: dict[str, float],
    final_config: dict[str, float] | None,
    *,
    epoch: int,
    start_epoch: int,
    end_epoch: int,
) -> dict[str, float]:
    if final_config is None or end_epoch <= start_epoch:
        return dict(base_config)
    if epoch <= start_epoch:
        alpha = 0.0
    elif epoch >= end_epoch:
        alpha = 1.0
    else:
        alpha = (epoch - start_epoch) / max(1, end_epoch - start_epoch)
    return {key: base_config[key] + alpha * (final_config[key] - base_config[key]) for key in base_config}


def build_dataset(
    tensors: dict[str, torch.Tensor],
    target: torch.Tensor,
    use_graph: bool,
    has_temporal: bool = False,
    ffs_coeff: torch.Tensor | None = None,
    ffs_field: torch.Tensor | None = None,
    ffs_radiated_power: torch.Tensor | None = None,
) -> TensorDataset:
    items = [tensors["points"], tensors["ports"], tensors["geom"], tensors["frame"], tensors["cuts"], tensors["nibs"], target]
    if ffs_coeff is not None:
        items.append(ffs_coeff)
    if ffs_field is not None:
        items.append(ffs_field)
    if ffs_radiated_power is not None:
        items.append(ffs_radiated_power)
    if has_temporal:
        items.append(tensors["temporal"])
    if use_graph:
        items.extend(tensors[key] for key in GRAPH_KEYS)
    return TensorDataset(*items)

def main() -> None:
    args = parse_args()
    if args.physics:
        if args.reciprocity_weight == 0.0:
            args.reciprocity_weight = 0.1
        if args.passivity_weight == 0.0:
            args.passivity_weight = 0.1
    set_seed(args.seed)
    # Disable memory-efficient SDP to avoid NaN with fully-masked tokens in transformer
    if torch.cuda.is_available():
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_config = build_loss_config(args)
    final_loss_config = build_final_loss_config(args, loss_config)
    use_ffs = args.model_kind == "structured_pair_spectral_ffs_head"
    bundle = load_dataset(args.dataset_root, n_points=args.points, freq_bins=args.freq_bins, max_temporal_steps=args.max_temporal_steps, include_ffs=use_ffs)
    train_records, val_records = split_records(bundle.records, seed=args.seed)
    train_tensors = stack_records(train_records)
    val_tensors = stack_records(val_records)
    target_mean = train_tensors["target"].mean(dim=(0, 1), keepdim=True)
    target_std = train_tensors["target"].std(dim=(0, 1), keepdim=True).clamp_min(1e-4)
    target_mean_device = target_mean.to(device)
    target_std_device = target_std.to(device)
    train_target = (train_tensors["target"] - target_mean) / target_std
    val_target = (val_tensors["target"] - target_mean) / target_std
    ffs_codec_state = None
    ffs_codec = None
    train_ffs_coeff = None
    val_ffs_coeff = None
    phi = None
    theta = None
    phi_count = 0
    theta_count = 0
    has_phi_closure = False
    model_config = {"hidden_dim": args.hidden_dim, "dropout": 0.1, "freq_bands": 8, "num_poles": args.num_poles}
    if use_ffs:
        if "ffs" not in train_tensors or bundle.ffs_metadata is None:
            raise ValueError("FFS model kind requires dataset FFS tensors and metadata")
        ffs_codec_state = fit_ffs_codec(train_tensors["ffs"].numpy(), rank=args.ffs_rank)
        ffs_codec = TorchFfsCodec.from_state(ffs_codec_state, dtype=torch.float32, device=device)
        train_ffs_coeff = torch.tensor(encode_ffs(train_tensors["ffs"].numpy(), ffs_codec_state), dtype=torch.float32)
        val_ffs_coeff = torch.tensor(encode_ffs(val_tensors["ffs"].numpy(), ffs_codec_state), dtype=torch.float32)
        model_config["ffs_coeff_dim"] = int(ffs_codec_state.config.rank)
        phi_count = int(bundle.ffs_metadata.phi_count)
        theta_count = int(bundle.ffs_metadata.theta_count)
        angle_grid = torch.tensor(bundle.ffs_metadata.angles_deg, dtype=torch.float64, device=device).view(
            phi_count,
            theta_count,
            2,
        )
        phi = torch.deg2rad(angle_grid[:, 0, 0])
        theta = torch.deg2rad(angle_grid[0, :, 1])
        has_phi_closure = bool(
            phi_count > 1 and torch.isclose(phi[-1], phi[0] + 2.0 * torch.pi, atol=1e-9, rtol=0.0)
        )
        if has_phi_closure:
            phi = phi[:-1]
    model = create_model(
        freq_grid=bundle.freq_grid,
        port_count=bundle.port_count,
        model_kind=args.model_kind,
        model_config=model_config,
    ).to(device)
    use_graph = uses_graph_features(model)
    has_temporal = "temporal" in train_tensors
    train_loader = DataLoader(
        build_dataset(
            train_tensors,
            train_target,
            use_graph,
            has_temporal,
            train_ffs_coeff,
            train_tensors.get("ffs"),
            train_tensors.get("ffs_radiated_power"),
        ),
        batch_size=min(args.batch_size, len(train_records)),
        shuffle=True,
    )
    val_loader = DataLoader(
        build_dataset(val_tensors, val_target, use_graph, has_temporal),
        batch_size=min(args.batch_size, len(val_records)),
        shuffle=False,
    )
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
        epoch_loss_config = scheduled_loss_config(
            loss_config,
            final_loss_config,
            epoch=epoch,
            start_epoch=args.loss_ramp_start_epoch,
            end_epoch=args.loss_ramp_end_epoch,
        )
        model.train()
        batch_losses = []
        for batch in train_loader:
            points, ports, geom, frame, cuts, nibs, target, *rest = batch
            ffs_coeff_target = None
            ffs_field_target = None
            ffs_radiated_power_target = None
            if use_ffs:
                ffs_coeff_target = rest[0]
                ffs_field_target = rest[1]
                ffs_radiated_power_target = rest[2]
                rest = rest[3:]
            temporal_batch = None
            graph_extra = rest
            if has_temporal:
                temporal_batch = rest[0]
                graph_extra = rest[1:]
            graph_tensors = None
            if graph_extra:
                graph_tensors = {key: tensor for key, tensor in zip(GRAPH_KEYS, graph_extra, strict=False)}
            optimizer.zero_grad(set_to_none=True)
            output = forward_model(
                model,
                points=points,
                ports=ports,
                geom=geom,
                frame=frame,
                cuts=cuts,
                nibs=nibs,
                device=device,
                graph_tensors=graph_tensors,
                temporal=temporal_batch,
                return_aux=use_ffs,
            )
            if use_ffs:
                assert isinstance(output, dict)
                pred = output["s_pred"]
            else:
                assert isinstance(output, torch.Tensor)
                pred = output
            loss = composite_loss(
                pred,
                target.to(device),
                port_count=bundle.port_count,
                target_mean=target_mean_device,
                target_std=target_std_device,
                loss_config=epoch_loss_config,
            )
            if use_ffs:
                assert ffs_codec is not None and phi is not None and theta is not None
                ffs_loss, _ = ffs_aux_loss(
                    pred_coeff=output["ffs_coeff_pred"],
                    target_coeff=ffs_coeff_target,
                    target_field=ffs_field_target,
                    target_radiated_power=ffs_radiated_power_target,
                    codec=ffs_codec,
                    phi=phi,
                    theta=theta,
                    phi_count=phi_count,
                    theta_count=theta_count,
                    has_phi_closure=has_phi_closure,
                    loss_weights={
                        "coeff": args.ffs_loss_weight,
                        "field": args.ffs_field_loss_weight,
                        "power": args.ffs_power_loss_weight,
                    },
                )
                loss = loss + ffs_loss
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
            loss_config=epoch_loss_config,
            has_temporal=has_temporal,
        )
        score = float(val_metrics["db_mae"])
        if args.physics:
            score = score + 0.5 * (float(val_metrics["reciprocity_mse"]) * 100.0 + float(val_metrics["passivity_mse"]) * 100.0)
        if score < best["score"]:
            best = {"epoch": epoch, "score": score, "state": copy.deepcopy(model.state_dict()), "opt_state": copy.deepcopy(optimizer.state_dict()), "metrics": val_metrics}
        if epoch == 1 or epoch % 50 == 0:
            parts = [
                f"epoch={epoch:03d} train_loss={np.mean(batch_losses):.4f}",
                f"val_rmse={val_metrics['rmse']:.4f} val_db_mae={val_metrics['db_mae']:.4f}",
                f"lr={optimizer.param_groups[0]['lr']:.2e} device={device.type}",
            ]
            if args.physics:
                parts.insert(-1, f"recip={val_metrics['reciprocity_mse']:.2e} passiv={val_metrics['passivity_mse']:.2e}")
            print(" ".join(parts))
    model.load_state_dict(best["state"])
    final_metrics, val_pred, val_truth = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
        port_count=bundle.port_count,
        loss_config=scheduled_loss_config(
            loss_config,
            final_loss_config,
            epoch=args.epochs,
            start_epoch=args.loss_ramp_start_epoch,
            end_epoch=args.loss_ramp_end_epoch,
        ),
        has_temporal=has_temporal,
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
            "opt_state_dict": optimizer.state_dict(),
            "epoch": best["epoch"],
            "freq_grid": bundle.freq_grid.tolist(),
            "port_count": bundle.port_count,
            "target_mean": target_mean.squeeze(0).squeeze(0).tolist(),
            "target_std": target_std.squeeze(0).squeeze(0).tolist(),
            "sample_points": args.points,
            "model_kind": args.model_kind,
            "model_config": model_config,
            "loss_config": loss_config,
            "loss_schedule": {
                "final_loss_config": final_loss_config,
                "start_epoch": args.loss_ramp_start_epoch,
                "end_epoch": args.loss_ramp_end_epoch,
            },
            "ffs_codec": None if ffs_codec_state is None else {
                "field_shape": list(ffs_codec_state.config.field_shape),
                "flat_dim": int(ffs_codec_state.config.flat_dim),
                "rank": int(ffs_codec_state.config.rank),
                "mean": ffs_codec_state.mean.tolist(),
                "basis": ffs_codec_state.basis.tolist(),
            },
            "ffs_metadata": None if bundle.ffs_metadata is None else {
                "frequencies_hz": bundle.ffs_metadata.frequencies_hz.tolist(),
                "angles_deg": bundle.ffs_metadata.angles_deg.tolist(),
                "radiated_power_w": bundle.ffs_metadata.radiated_power_w.tolist(),
                "accepted_power_w": bundle.ffs_metadata.accepted_power_w.tolist(),
                "stimulated_power_w": bundle.ffs_metadata.stimulated_power_w.tolist(),
                "position_m": bundle.ffs_metadata.position_m.tolist(),
                "z_axis": bundle.ffs_metadata.z_axis.tolist(),
                "x_axis": bundle.ffs_metadata.x_axis.tolist(),
                "phi_count": int(bundle.ffs_metadata.phi_count),
                "theta_count": int(bundle.ffs_metadata.theta_count),
            },
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
        "val_reciprocity_mse": final_metrics["reciprocity_mse"],
        "val_passivity_mse": final_metrics["passivity_mse"],
        "example_sample": example.name,
        "example_rmse": final_metrics["sample_rmse"][0],
        "example_db_mae": final_metrics["sample_db_mae"][0],
        "plot_path": str(plot_path),
        "model_path": str(model_path),
        "model_kind": args.model_kind,
        "model_config": model_config,
        "loss_config": loss_config,
        "loss_schedule": {
            "final_loss_config": final_loss_config,
            "start_epoch": args.loss_ramp_start_epoch,
            "end_epoch": args.loss_ramp_end_epoch,
        },
        "warmup_epochs": warmup_epochs,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
