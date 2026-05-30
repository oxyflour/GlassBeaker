from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from baseline.model import create_model
from optimizer_runner import optimize_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize cut/nib distances and soft port roles against a Nijika surrogate.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--band-min", type=float)
    parser.add_argument("--band-max", type=float)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--match-weight", type=float, default=5.0)
    parser.add_argument("--isolation-weight", type=float, default=3.0)
    parser.add_argument("--bandwidth-weight", type=float, default=2.0)
    parser.add_argument("--match-threshold-db", type=float, default=-10.0)
    parser.add_argument("--ground-admittance", type=float, default=1e6)
    parser.add_argument("--open-admittance", type=float, default=1e-6)
    return parser.parse_args()


def load_checkpoint_model(model_path: Path) -> tuple[dict[str, object], torch.nn.Module]:
    checkpoint = torch.load(model_path, map_location="cpu")
    freq_grid = np.asarray(checkpoint["freq_grid"], dtype=np.float32)
    model = create_model(freq_grid=freq_grid, port_count=int(checkpoint["port_count"]), model_kind=checkpoint.get("model_kind", "structured_pair_spectral_head"), model_config=checkpoint.get("model_config"))
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint, model


def main() -> None:
    args = parse_args()
    checkpoint, model = load_checkpoint_model(args.model_path)
    config = json.loads(args.config_path.read_text())
    result = optimize_model(model=model, checkpoint=checkpoint, config=config, output_dir=args.output_dir, steps=args.steps, lr=args.lr, top_k=args.top_k, band_min=args.band_min, band_max=args.band_max, match_weight=args.match_weight, isolation_weight=args.isolation_weight, bandwidth_weight=args.bandwidth_weight, match_threshold_db=args.match_threshold_db, ground_admittance=args.ground_admittance, open_admittance=args.open_admittance)
    print(json.dumps({"trace_steps": len(result["trace"]), "best_score": result["ranking"][0]["score"], "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
