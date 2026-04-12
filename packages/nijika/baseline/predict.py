from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from baseline.data import load_dataset, load_inference_input, load_truth_target, split_records, stack_records
from baseline.metrics import summarize_prediction_metrics
from baseline.model import create_model
from baseline.plotting import save_matrix_plot


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


def save_prediction_artifact(
    *,
    output_dir: Path,
    sample_name: str,
    freq_grid: np.ndarray,
    truth: np.ndarray | None,
    pred: np.ndarray,
    port_count: int,
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
    with torch.no_grad():
        pred = model(
            tensors["points"],
            tensors["ports"],
            tensors["geom"],
            tensors["frame"],
            tensors["cuts"],
            tensors["nibs"],
        )
    pred_np = denormalize(pred, checkpoint)
    truth_np = tensors["target"].numpy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, record in enumerate(selected):
        results.append(
            save_prediction_artifact(
                output_dir=args.output_dir,
                sample_name=record.name,
                freq_grid=freq_grid,
                truth=truth_np[idx],
                pred=pred_np[idx],
                port_count=bundle.port_count,
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
    with torch.no_grad():
        pred = model(
            torch.tensor(sample["points"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(sample["ports"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(sample["geom"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(sample["frame"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(sample["cuts"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(sample["nibs"], dtype=torch.float32).unsqueeze(0),
        )
    pred_np = denormalize(pred, checkpoint)[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_name = str(sample["name"])
    truth = load_truth_target(args.dataset_root / sample_name, port_count=int(checkpoint["port_count"]), freq_grid=freq_grid)
    result = save_prediction_artifact(
        output_dir=args.output_dir,
        sample_name=sample_name,
        freq_grid=freq_grid,
        truth=truth,
        pred=pred_np,
        port_count=int(checkpoint["port_count"]),
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
