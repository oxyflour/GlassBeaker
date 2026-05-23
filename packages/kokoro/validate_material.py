from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from kokoro.brdf import BrdfTrainingConfig, build_brdf_dataset, export_surrogate_npz, train_brdf_surrogate
from kokoro.height_field import compile_height_program
from kokoro.mitsuba_neural_bsdf import register_kokoro_bsdf
from kokoro.mitsuba_scene import prepare_mitsuba_scene_dict
from kokoro.validation import (
    ValidationArtifacts,
    angular_error_degrees,
    image_metrics,
    mirror_plane_scene,
    neural_plane_scene,
    pyramid_ply_scene,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path("packages/kokoro/tmp/validation")
DEFAULT_HDR = Path("apps/web/public/studio_small_03_1k.hdr")
DEFAULT_SAMPLES = 4096
DEFAULT_EPOCHS = 200
DEFAULT_HIDDEN_DIM = 96
DEFAULT_LOBE_KAPPA = 4096.0
PYRAMID_5CM_PERIOD_M = 0.05
FLAT_HEIGHT = """
def height(x, y):
    return x * 0.0
"""
PYRAMID_5CM_HEIGHT = """
def height(x, y):
    return pyramid_height(x, y, period_m=0.05, amplitude_m=0.008)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate kokoro neural BSDF against analytic and PLY baselines.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hdr-path", type=Path, default=DEFAULT_HDR)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--film-width", type=int, default=320)
    parser.add_argument("--film-height", type=int, default=240)
    parser.add_argument("--spp", type=int, default=64)
    parser.add_argument("--ply-grid", type=int, default=96)
    parser.add_argument("--lobe-kappa", type=float, default=DEFAULT_LOBE_KAPPA)
    parser.add_argument("--variant", default="cuda_ad_rgb")
    args = parser.parse_args()

    artifacts = ValidationArtifacts(args.output_dir)
    artifacts.root.mkdir(parents=True, exist_ok=True)
    import mitsuba as mi

    mi.set_variant(args.variant)
    register_kokoro_bsdf(mi)
    results = {
        "flat_z0": _validate_case(
            mi, artifacts, "flat", compile_height_program(FLAT_HEIGHT),
            neural_path=artifacts.flat_neural, reference_path=artifacts.flat_mirror,
            diff_path=artifacts.flat_diff, args=args, feature_period_m=None,
        ),
        "pyramid_period_5cm": _validate_case(
            mi, artifacts, "pyramid", compile_height_program(PYRAMID_5CM_HEIGHT),
            neural_path=artifacts.pyramid_neural, reference_path=artifacts.pyramid_ply,
            diff_path=artifacts.pyramid_diff, args=args, feature_period_m=PYRAMID_5CM_PERIOD_M,
        ),
    }
    write_json(artifacts.metrics, results)
    print(json.dumps({"metrics": str(artifacts.metrics), **results}, indent=2))


def _validate_case(
    mi,
    artifacts,
    name,
    program,
    *,
    neural_path: Path,
    reference_path: Path,
    diff_path: Path,
    args,
    feature_period_m: float | None,
):
    dataset = build_brdf_dataset(
        program,
        sample_count=args.samples,
        width_m=0.10,
        depth_m=0.10,
        seed=19,
        feature_period_m=feature_period_m,
    )
    result = train_brdf_surrogate(
        dataset,
        BrdfTrainingConfig(hidden_dim=args.hidden_dim, epochs=args.epochs, batch_size=128, lr=0.004, seed=23),
    )
    checkpoint = artifacts.root / f"{name}_brdf.npz"
    export_surrogate_npz(result.model, checkpoint, width_m=0.10, depth_m=0.10, feature_period_m=feature_period_m)
    scene = neural_plane_scene(
        checkpoint, args.hdr_path, width=args.film_width, height=args.film_height,
        spp=args.spp, lobe_kappa=args.lobe_kappa,
    )
    _render(mi, scene, neural_path)
    if name == "flat":
        _render(mi, mirror_plane_scene(scene), reference_path)
    else:
        _render(
            mi,
            pyramid_ply_scene(
                program, args.hdr_path, artifacts.pyramid_ply_mesh,
                width=args.film_width, height=args.film_height, grid_size=args.ply_grid, spp=args.spp,
            ),
            reference_path,
        )
    return {
        "checkpoint": str(checkpoint),
        "neural": str(neural_path),
        "reference": str(reference_path),
        "diff": str(diff_path),
        "training_initial_loss": result.loss_history[0],
        "training_final_loss": result.loss_history[-1],
        "angular_error": angular_error_degrees(result.model, dataset.features, dataset.targets),
        "image_error": image_metrics(reference_path, neural_path, diff_path),
    }


def _render(mi, scene, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = mi.render(mi.load_dict(prepare_mitsuba_scene_dict(scene, mi)))
    mi.util.write_bitmap(str(path), image)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        raise
