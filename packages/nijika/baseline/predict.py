from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from baseline.data import load_dataset, load_inference_input, load_truth_target, split_records, stack_records
from baseline.ffs_codec import TorchFfsCodec, codec_state_from_payload
from baseline.ffs_io import FfsMetadata, write_ffs_sample
from baseline.metrics import summarize_prediction_metrics
from baseline.model import create_model
from baseline.plotting import save_matrix_plot
from baseline.training_utils import forward_model, uses_graph_features
from optimizer_torch_farfield import integrate_decoded_ffs_power


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline inference for one antenna sample or a dataset split.")
    parser.add_argument("--dataset-root", type=Path, default=Path("tmp/antenna-dataset"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sample-name", type=str)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--split", choices=["train", "val", "all"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/nijika-baseline-predict"))
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> tuple[dict[str, object], torch.nn.Module, np.ndarray]:
    checkpoint = torch.load(args.model_path, map_location="cpu")
    freq_grid = np.asarray(checkpoint["freq_grid"], dtype=np.float32)
    model = create_model(
        freq_grid=freq_grid,
        port_count=int(checkpoint["port_count"]),
        model_kind=checkpoint.get("model_kind", "legacy_global_head"),
        model_config=checkpoint.get("model_config"),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return checkpoint, model, freq_grid


def denormalize(pred: torch.Tensor, checkpoint: dict[str, object]) -> np.ndarray:
    mean = torch.tensor(checkpoint["target_mean"], dtype=torch.float32).view(1, 1, -1)
    std = torch.tensor(checkpoint["target_std"], dtype=torch.float32).view(1, 1, -1)
    return (pred.cpu() * std + mean).numpy()


def _ffs_metadata(checkpoint: dict[str, object]) -> FfsMetadata | None:
    payload = checkpoint.get("ffs_metadata")
    if payload is None:
        return None
    return FfsMetadata(
        frequencies_hz=np.asarray(payload["frequencies_hz"], dtype=np.float64),
        angles_deg=np.asarray(payload["angles_deg"], dtype=np.float64),
        radiated_power_w=np.asarray(payload.get("radiated_power_w", np.zeros(len(payload["frequencies_hz"]))), dtype=np.float64),
        accepted_power_w=np.asarray(payload.get("accepted_power_w", np.zeros(len(payload["frequencies_hz"]))), dtype=np.float64),
        stimulated_power_w=np.asarray(payload.get("stimulated_power_w", np.full(len(payload["frequencies_hz"]), 0.5)), dtype=np.float64),
        position_m=np.asarray(payload["position_m"], dtype=np.float64),
        z_axis=np.asarray(payload["z_axis"], dtype=np.float64),
        x_axis=np.asarray(payload["x_axis"], dtype=np.float64),
        phi_count=int(payload["phi_count"]),
        theta_count=int(payload["theta_count"]),
    )


def _farfield_grid_from_metadata(
    metadata: FfsMetadata,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    angle_grid = torch.as_tensor(metadata.angles_deg, dtype=dtype, device=device).view(
        metadata.phi_count,
        metadata.theta_count,
        2,
    )
    phi = torch.deg2rad(angle_grid[:, 0, 0])
    theta = torch.deg2rad(angle_grid[0, :, 1])
    has_phi_closure = bool(
        metadata.phi_count > 1 and torch.isclose(phi[-1], phi[0] + 2.0 * torch.pi, atol=1e-9, rtol=0.0)
    )
    if has_phi_closure:
        phi = phi[:-1]
    return phi, theta, has_phi_closure


def _export_predicted_ffs(
    *,
    output_dir: Path,
    sample_name: str,
    checkpoint: dict[str, object],
    coeff_pred: torch.Tensor,
) -> Path | None:
    payload = checkpoint.get("ffs_codec")
    metadata = _ffs_metadata(checkpoint)
    if payload is None or metadata is None:
        return None
    codec = TorchFfsCodec.from_state(
        codec_state_from_payload(payload),
        dtype=coeff_pred.dtype,
        device=coeff_pred.device,
    )
    decoded = codec.decode(coeff_pred)
    phi, theta, has_phi_closure = _farfield_grid_from_metadata(
        metadata,
        device=decoded.device,
        dtype=decoded.real.dtype,
    )
    radiated_power = integrate_decoded_ffs_power(
        decoded,
        phi=phi,
        theta=theta,
        phi_count=metadata.phi_count,
        theta_count=metadata.theta_count,
        has_phi_closure=has_phi_closure,
    )[0]
    ffs_dir = output_dir / f"{sample_name}_predicted_ffs"
    stimulated_power = np.maximum(metadata.stimulated_power_w, 1e-9)
    decoded_sample = decoded[0]
    for port_idx in range(decoded_sample.shape[0]):
        for freq_idx, freq_hz in enumerate(metadata.frequencies_hz):
            header = FfsMetadata(
                frequencies_hz=np.asarray([freq_hz], dtype=np.float64),
                angles_deg=metadata.angles_deg.copy(),
                radiated_power_w=radiated_power[port_idx, freq_idx : freq_idx + 1].detach().cpu().numpy(),
                accepted_power_w=np.zeros((1,), dtype=np.float64),
                stimulated_power_w=np.asarray([stimulated_power[freq_idx]], dtype=np.float64),
                position_m=metadata.position_m.copy(),
                z_axis=metadata.z_axis.copy(),
                x_axis=metadata.x_axis.copy(),
                phi_count=metadata.phi_count,
                theta_count=metadata.theta_count,
            )
            write_ffs_sample(
                ffs_dir / f"{port_idx + 1}-[f={int(round(float(freq_hz)))}].ffs",
                header,
                decoded_sample[port_idx, freq_idx : freq_idx + 1].detach().cpu().numpy(),
            )
    return ffs_dir


def save_prediction_artifact(
    *,
    output_dir: Path,
    sample_name: str,
    freq_grid: np.ndarray,
    truth: np.ndarray | None,
    pred: np.ndarray,
    port_count: int,
    predicted_ffs_dir: Path | None = None,
) -> dict[str, object]:
    plot_path = output_dir / f"{sample_name}_matrix_db.png"
    save_matrix_plot(
        path=plot_path,
        freq_grid=freq_grid,
        truth=truth,
        pred=pred,
        title=f"{sample_name} baseline prediction",
        port_count=port_count,
    )
    npz_path = output_dir / f"{sample_name}_prediction.npz"
    np.savez(npz_path, frequency=freq_grid, truth=truth, pred=pred)
    result: dict[str, object] = {
        "sample_name": sample_name,
        "has_truth": truth is not None,
        "plot_path": str(plot_path),
        "npz_path": str(npz_path),
    }
    if predicted_ffs_dir is not None:
        result["predicted_ffs_dir"] = str(predicted_ffs_dir)
    if truth is not None:
        metrics = summarize_prediction_metrics(pred[np.newaxis, ...], truth[np.newaxis, ...], port_count=port_count)
        result["rmse"] = metrics["rmse"]
        result["db_mae"] = metrics["db_mae"]
    result_path = output_dir / f"{sample_name}_prediction.json"
    result_path.write_text(json.dumps(result, indent=2))
    return result


def predict_split(args: argparse.Namespace, checkpoint: dict[str, object], model: torch.nn.Module, freq_grid: np.ndarray) -> None:
    bundle = load_dataset(args.dataset_root, n_points=int(checkpoint["sample_points"]), freq_bins=len(freq_grid))
    train_records, val_records = split_records(bundle.records, seed=args.seed, val_ratio=args.val_ratio)
    selected = {"train": train_records, "val": val_records, "all": bundle.records}[str(args.split)]
    tensors = stack_records(selected)
    graph_tensors = None
    if uses_graph_features(model):
        graph_tensors = {key: tensors[key] for key in ("graph_inner", "graph_segment", "graph_port", "graph_mask", "graph_adj", "graph_edge_attr", "pair_topology")}
    has_ffs = checkpoint.get("ffs_codec") is not None and _ffs_metadata(checkpoint) is not None
    with torch.no_grad():
        output = forward_model(
            model,
            points=tensors["points"],
            ports=tensors["ports"],
            geom=tensors["geom"],
            frame=tensors["frame"],
            cuts=tensors["cuts"],
            nibs=tensors["nibs"],
            device=torch.device("cpu"),
            graph_tensors=graph_tensors,
            return_aux=has_ffs,
        )
    if has_ffs:
        assert isinstance(output, dict)
        pred = output["s_pred"]
        coeff_pred = output.get("ffs_coeff_pred")
    else:
        assert isinstance(output, torch.Tensor)
        pred = output
        coeff_pred = None
    pred_np = denormalize(pred, checkpoint)
    truth_np = tensors["target"].numpy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, record in enumerate(selected):
        ffs_dir = None
        if coeff_pred is not None:
            ffs_dir = _export_predicted_ffs(
                output_dir=args.output_dir,
                sample_name=record.name,
                checkpoint=checkpoint,
                coeff_pred=coeff_pred[idx : idx + 1],
            )
        results.append(
            save_prediction_artifact(
                output_dir=args.output_dir,
                sample_name=record.name,
                freq_grid=freq_grid,
                truth=truth_np[idx],
                pred=pred_np[idx],
                port_count=bundle.port_count,
                predicted_ffs_dir=ffs_dir,
            )
        )
    summary = summarize_prediction_metrics(pred_np, truth_np, port_count=bundle.port_count)
    summary["split"] = args.split
    summary["seed"] = args.seed
    summary["val_ratio"] = args.val_ratio
    summary["sample_names"] = [record.name for record in selected]
    summary["predictions"] = results
    (args.output_dir / f"{args.split}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def predict_single(args: argparse.Namespace, checkpoint: dict[str, object], model: torch.nn.Module, freq_grid: np.ndarray) -> None:
    config_path = args.config_path or (args.dataset_root / f"{args.sample_name}.json")
    sample = load_inference_input(config_path, n_points=int(checkpoint["sample_points"]))
    graph_tensors = None
    if uses_graph_features(model):
        graph_tensors = {key: torch.tensor(sample["graph"][key], dtype=torch.float32).unsqueeze(0) for key in ("graph_inner", "graph_segment", "graph_port", "graph_mask", "graph_adj", "graph_edge_attr", "pair_topology")}
    has_ffs = checkpoint.get("ffs_codec") is not None and _ffs_metadata(checkpoint) is not None
    with torch.no_grad():
        output = forward_model(
            model,
            points=torch.tensor(sample["points"], dtype=torch.float32).unsqueeze(0),
            ports=torch.tensor(sample["ports"], dtype=torch.float32).unsqueeze(0),
            geom=torch.tensor(sample["geom"], dtype=torch.float32).unsqueeze(0),
            frame=torch.tensor(sample["frame"], dtype=torch.float32).unsqueeze(0),
            cuts=torch.tensor(sample["cuts"], dtype=torch.float32).unsqueeze(0),
            nibs=torch.tensor(sample["nibs"], dtype=torch.float32).unsqueeze(0),
            device=torch.device("cpu"),
            graph_tensors=graph_tensors,
            return_aux=has_ffs,
        )
    if has_ffs:
        assert isinstance(output, dict)
        pred = output["s_pred"]
        coeff_pred = output.get("ffs_coeff_pred")
    else:
        assert isinstance(output, torch.Tensor)
        pred = output
        coeff_pred = None
    pred_np = denormalize(pred, checkpoint)[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_name = str(sample["name"])
    truth = load_truth_target(args.dataset_root / sample_name, port_count=int(checkpoint["port_count"]), freq_grid=freq_grid)
    ffs_dir = None
    if coeff_pred is not None:
        ffs_dir = _export_predicted_ffs(
            output_dir=args.output_dir,
            sample_name=sample_name,
            checkpoint=checkpoint,
            coeff_pred=coeff_pred,
        )
    result = save_prediction_artifact(
        output_dir=args.output_dir,
        sample_name=sample_name,
        freq_grid=freq_grid,
        truth=truth,
        pred=pred_np,
        port_count=int(checkpoint["port_count"]),
        predicted_ffs_dir=ffs_dir,
    )
    result["config_path"] = str(config_path)
    (args.output_dir / f"{sample_name}_prediction.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main() -> None:
    args = parse_args()
    if not args.split and not args.sample_name and not args.config_path:
        raise ValueError("Provide either --split or one of --sample-name/--config-path")
    checkpoint, model, freq_grid = load_model(args)
    if args.split:
        predict_split(args, checkpoint, model, freq_grid)
        return
    predict_single(args, checkpoint, model, freq_grid)


if __name__ == "__main__":
    main()
