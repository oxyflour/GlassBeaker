from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from baseline.data import load_inference_input, load_truth_target
from baseline.model import SpectrumPredictor
from baseline.plotting import save_matrix_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline inference for one antenna sample.")
    parser.add_argument("--dataset-root", type=Path, default=Path("tmp/antenna-dataset"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sample-name", type=str)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/nijika-baseline-predict"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.sample_name and not args.config_path:
        raise ValueError("Provide either --sample-name or --config-path")
    checkpoint = torch.load(args.model_path, map_location="cpu")
    freq_bins = len(checkpoint["freq_grid"])
    freq_grid = np.asarray(checkpoint["freq_grid"], dtype=np.float32)
    config_path = args.config_path or (args.dataset_root / f"{args.sample_name}.json")
    sample = load_inference_input(config_path, n_points=int(checkpoint["sample_points"]))
    model = SpectrumPredictor(freq_bins=freq_bins, port_count=int(checkpoint["port_count"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        pred = model(
            torch.tensor(sample["points"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(sample["ports"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(sample["geom"], dtype=torch.float32).unsqueeze(0),
        )
    mean = torch.tensor(checkpoint["target_mean"], dtype=torch.float32).view(1, 1, -1)
    std = torch.tensor(checkpoint["target_std"], dtype=torch.float32).view(1, 1, -1)
    pred = (pred * std + mean).squeeze(0).numpy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_name = str(sample["name"])
    truth = load_truth_target(
        args.dataset_root / sample_name,
        port_count=int(checkpoint["port_count"]),
        freq_grid=freq_grid,
    )
    plot_path = args.output_dir / f"{sample_name}_matrix_db.png"
    save_matrix_plot(
        path=plot_path,
        freq_grid=freq_grid,
        truth=truth,
        pred=pred,
        title=f"{sample_name} baseline prediction",
        port_count=int(checkpoint["port_count"]),
    )
    npz_path = args.output_dir / f"{sample_name}_prediction.npz"
    np.savez(
        npz_path,
        frequency=freq_grid,
        truth=truth,
        pred=pred,
    )
    result = {
        "sample_name": sample_name,
        "config_path": str(config_path),
        "has_truth": truth is not None,
        "plot_path": str(plot_path),
        "npz_path": str(npz_path),
    }
    result_path = args.output_dir / f"{sample_name}_prediction.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
