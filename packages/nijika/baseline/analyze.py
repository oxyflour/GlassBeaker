from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from baseline.data import load_dataset, split_records, stack_records
from baseline.metrics import magnitude_db, summarize_prediction_metrics
from baseline.predict import denormalize, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze baseline predictions on a dataset split.")
    parser.add_argument("--dataset-root", type=Path, default=Path("tmp/antenna-dataset"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/nijika-baseline-analysis"))
    return parser.parse_args()


def _pair_names(port_count: int) -> list[str]:
    return [f"S{row + 1}{col + 1}" for row in range(port_count) for col in range(port_count)]


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.allclose(x.std(), 0.0) or np.allclose(y.std(), 0.0):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _quartiles(values: np.ndarray, metric: np.ndarray) -> dict[str, dict[str, float | int]]:
    bins = np.searchsorted(np.quantile(values, [0.25, 0.5, 0.75]), values, side="right") + 1
    return {
        str(idx): {"count": int((bins == idx).sum()), "mean_db_mae": float(metric[bins == idx].mean())}
        for idx in range(1, 5)
    }


def _sample_meta(config_path: Path) -> dict[str, object]:
    antenna = json.loads(config_path.read_text()).get("antennaConfig") or {}
    cut_sides = [str(item["position"]) for item in antenna.get("cuts", [])]
    nib_sides = [str(item["position"]) for item in antenna.get("nibs", [])]
    return {
        "frameWidth": float(antenna.get("frameWidth", 0.0)),
        "gap": float(antenna.get("gap", 0.0)),
        "cut_count": len(cut_sides),
        "cut_sides": cut_sides,
        "nib_sides": nib_sides,
    }


def _predict(
    model: torch.nn.Module,
    checkpoint: dict[str, object],
    tensors: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tensors["points"]), batch_size):
            end = start + batch_size
            pred = model(
                tensors["points"][start:end].to(device),
                tensors["ports"][start:end].to(device),
                tensors["geom"][start:end].to(device),
                tensors["frame"][start:end].to(device),
                tensors["cuts"][start:end].to(device),
                tensors["nibs"][start:end].to(device),
            )
            preds.append(denormalize(pred.cpu(), checkpoint))
    return np.concatenate(preds, axis=0)


def _analyze(
    *,
    dataset_root: Path,
    records: list,
    pred_np: np.ndarray,
    truth_np: np.ndarray,
    port_count: int,
) -> tuple[dict[str, object], dict[str, object]]:
    metrics = summarize_prediction_metrics(pred_np, truth_np, port_count=port_count)
    pred_matrix = pred_np.reshape(pred_np.shape[0], pred_np.shape[1], port_count, port_count, 2)
    truth_matrix = truth_np.reshape(truth_np.shape[0], truth_np.shape[1], port_count, port_count, 2)
    diff_db = np.abs(magnitude_db(pred_matrix) - magnitude_db(truth_matrix))
    truth_db = magnitude_db(truth_matrix)
    pair_db_mae = {
        name: float(diff_db[:, :, row, col].mean())
        for name, (row, col) in zip(_pair_names(port_count), np.ndindex(port_count, port_count), strict=False)
    }
    thresholds = {}
    for threshold in (-10, -15, -20):
        below = truth_db < threshold
        above = truth_db >= threshold
        key = f"neg{abs(threshold)}"
        thresholds[f"truth_db_lt_{key}_mae"] = float(diff_db[below].mean())
        thresholds[f"truth_db_lt_{key}_fraction"] = float(below.mean())
        thresholds[f"truth_db_ge_{key}_mae"] = float(diff_db[above].mean())
    sample_rows = []
    for idx, record in enumerate(records):
        meta = _sample_meta(dataset_root / f"{record.name}.json")
        sample_rows.append(
            {
                "sample_name": record.name,
                "rmse": float(metrics["sample_rmse"][idx]),
                "db_mae": float(metrics["sample_db_mae"][idx]),
                **meta,
                "min_truth_db": float(truth_db[idx].min()),
                "p10_truth_db": float(np.quantile(truth_db[idx], 0.1)),
                "notch_fraction_db_lt_neg20": float((truth_db[idx] < -20).mean()),
                "notch_fraction_db_lt_neg15": float((truth_db[idx] < -15).mean()),
            }
        )
    sorted_rows = sorted(sample_rows, key=lambda row: float(row["db_mae"]))
    sample_db_mae = np.asarray(metrics["sample_db_mae"], dtype=np.float32)
    sample_rmse = np.asarray(metrics["sample_rmse"], dtype=np.float32)
    frame_values = np.asarray([float(row["frameWidth"]) for row in sample_rows], dtype=np.float32)
    gap_values = np.asarray([float(row["gap"]) for row in sample_rows], dtype=np.float32)
    cut_counts = np.asarray([int(row["cut_count"]) for row in sample_rows], dtype=np.int32)
    min_truth_db = np.asarray([float(row["min_truth_db"]) for row in sample_rows], dtype=np.float32)
    notch_fraction = np.asarray([float(row["notch_fraction_db_lt_neg20"]) for row in sample_rows], dtype=np.float32)
    by_cut_count = {}
    for cut_count in sorted(set(cut_counts.tolist())):
        mask = cut_counts == cut_count
        by_cut_count[str(cut_count)] = {
            "count": int(mask.sum()),
            "mean_db_mae": float(sample_db_mae[mask].mean()),
            "median_db_mae": float(np.median(sample_db_mae[mask])),
        }
    analysis = {
        "summary": {
            "n": len(sample_rows),
            "mean_db_mae": float(sample_db_mae.mean()),
            "median_db_mae": float(np.median(sample_db_mae)),
            "std_db_mae": float(sample_db_mae.std()),
            "p10_db_mae": float(np.quantile(sample_db_mae, 0.1)),
            "p90_db_mae": float(np.quantile(sample_db_mae, 0.9)),
            "p95_db_mae": float(np.quantile(sample_db_mae, 0.95)),
            "max_db_mae": float(sample_db_mae.max()),
            "min_db_mae": float(sample_db_mae.min()),
            "mean_rmse": float(sample_rmse.mean()),
            "median_rmse": float(np.median(sample_rmse)),
        },
        "pair_db_mae": pair_db_mae,
        "threshold_metrics": thresholds,
        "correlation": {
            "db_mae_vs_min_truth_db": _corr(sample_db_mae, min_truth_db),
            "db_mae_vs_notch_fraction_lt_neg20": _corr(sample_db_mae, notch_fraction),
            "db_mae_vs_rmse": _corr(sample_db_mae, sample_rmse),
        },
        "by_cut_count": by_cut_count,
        "by_frame_quartile": _quartiles(frame_values, sample_db_mae),
        "by_gap_quartile": _quartiles(gap_values, sample_db_mae),
        "unique_nib_side_patterns": sorted({str(row["nib_sides"]) for row in sample_rows}),
        "unique_cut_side_patterns": len({tuple(row["cut_sides"]) for row in sample_rows}),
        "best5": sorted_rows[:5],
        "worst10": list(reversed(sorted_rows[-10:])),
    }
    summary = {
        "rmse": metrics["rmse"],
        "db_mae": metrics["db_mae"],
        "db_rmse": metrics["db_rmse"],
        "sample_rmse": metrics["sample_rmse"],
        "sample_db_mae": metrics["sample_db_mae"],
        "sample_names": [record.name for record in records],
        "predictions": [{"sample_name": row["sample_name"], "rmse": row["rmse"], "db_mae": row["db_mae"]} for row in sample_rows],
    }
    return summary, analysis


def main() -> None:
    args = parse_args()
    checkpoint, model, _ = load_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    bundle = load_dataset(args.dataset_root, n_points=int(checkpoint["sample_points"]), freq_bins=len(checkpoint["freq_grid"]))
    train_records, val_records = split_records(bundle.records, seed=args.seed, val_ratio=args.val_ratio)
    records = {"train": train_records, "val": val_records, "all": bundle.records}[args.split]
    tensors = stack_records(records)
    pred_np = _predict(model, checkpoint, tensors, device, batch_size=args.batch_size)
    truth_np = tensors["target"].numpy()
    summary, analysis = _analyze(
        dataset_root=args.dataset_root,
        records=records,
        pred_np=pred_np,
        truth_np=truth_np,
        port_count=bundle.port_count,
    )
    summary.update({"split": args.split, "seed": args.seed, "val_ratio": args.val_ratio})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.split}_summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
    print(json.dumps({"summary_path": str(args.output_dir / f'{args.split}_summary.json'), "analysis_path": str(args.output_dir / 'analysis.json')}, indent=2))


if __name__ == "__main__":
    main()
