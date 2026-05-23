from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kokoro.brdf import BrdfTrainingConfig, build_brdf_dataset, export_surrogate_npz, train_brdf_surrogate
from kokoro.height_field import compile_height_program
from kokoro.mitsuba_neural_bsdf import register_kokoro_bsdf
from kokoro.mitsuba_scene import build_kokoro_scene_dict, orbit_scene_dicts, prepare_mitsuba_scene_dict
from kokoro.video import write_mjpeg_avi

DEFAULT_HEIGHT_SOURCE = """
def height(x, y):
    return pyramid_height(x, y, period_m=500e-6, amplitude_m=150e-6)
"""
DEFAULT_OUTPUT_DIR = Path("packages/kokoro/tmp")
DEFAULT_FEATURE_PERIOD_M = 500e-6
DEFAULT_VIDEO_FRAMES = 120
DEFAULT_VIDEO_FPS = 24
DEFAULT_FOV = 65.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a kokoro neural BRDF and write a Mitsuba scene.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--height-source", type=Path, default=None)
    parser.add_argument("--hdr-path", type=Path, default=Path("apps/web/public/studio_small_03_1k.hdr"))
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--film-width", type=int, default=512)
    parser.add_argument("--film-height", type=int, default=384)
    parser.add_argument("--fov", type=float, default=DEFAULT_FOV)
    parser.add_argument("--spp", type=int, default=64)
    parser.add_argument("--width-m", type=float, default=0.10)
    parser.add_argument("--depth-m", type=float, default=0.10)
    parser.add_argument("--feature-period-m", type=float, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-frames", type=int, default=DEFAULT_VIDEO_FRAMES)
    parser.add_argument("--video-fps", type=int, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--orbit-radius", type=float, default=0.18)
    parser.add_argument("--camera-height", type=float, default=0.10)
    parser.add_argument("--variant", default="cuda_ad_rgb")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.height_source.read_text(encoding="utf-8") if args.height_source else DEFAULT_HEIGHT_SOURCE
    feature_period_m = args.feature_period_m
    if feature_period_m is None and args.height_source is None:
        feature_period_m = DEFAULT_FEATURE_PERIOD_M
    program = compile_height_program(source)
    dataset = build_brdf_dataset(
        program,
        sample_count=args.samples,
        width_m=args.width_m,
        depth_m=args.depth_m,
        seed=13,
        feature_period_m=feature_period_m,
    )
    result = train_brdf_surrogate(
        dataset,
        BrdfTrainingConfig(
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=17,
        ),
    )
    checkpoint = args.output_dir / "kokoro_brdf.npz"
    export_surrogate_npz(
        result.model,
        checkpoint,
        width_m=args.width_m,
        depth_m=args.depth_m,
        feature_period_m=feature_period_m,
    )
    stale_mesh = args.output_dir / "kokoro_surface.ply"
    if stale_mesh.exists():
        stale_mesh.unlink()
    scene = build_kokoro_scene_dict(
        checkpoint_path=checkpoint,
        hdr_path=args.hdr_path,
        width=args.film_width,
        height=args.film_height,
        fov=args.fov,
        width_m=args.width_m,
        depth_m=args.depth_m,
        spp=args.spp,
    )
    scene_path = args.output_dir / "kokoro_scene.json"
    scene_path.write_text(json.dumps(scene, indent=2), encoding="utf-8")
    metrics = {"initial_loss": result.loss_history[0], "final_loss": result.loss_history[-1]}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.render:
        _render(scene, render_output_path(args.output_dir), args.variant)
    if args.video:
        _render_video(
            scene,
            video_output_path(args.output_dir),
            args.variant,
            frame_count=args.video_frames,
            fps=args.video_fps,
            radius_m=args.orbit_radius,
            height_m=args.camera_height,
        )
    print(json.dumps({"checkpoint": str(checkpoint), "scene": str(scene_path), **metrics}, indent=2))


def render_output_path(output_dir: Path) -> Path:
    return output_dir / "kokoro_render.png"


def video_output_path(output_dir: Path) -> Path:
    return output_dir / "kokoro_orbit.avi"


def _render(scene: dict[str, Any], output_path: Path, variant: str) -> None:
    mi = _load_mitsuba(variant)
    image = mi.render(mi.load_dict(prepare_mitsuba_scene_dict(scene, mi)))
    mi.util.write_bitmap(str(output_path), image)


def _render_video(
    scene: dict[str, Any],
    output_path: Path,
    variant: str,
    *,
    frame_count: int,
    fps: int,
    radius_m: float,
    height_m: float,
) -> None:
    import numpy as np

    mi = _load_mitsuba(variant)
    frames = []
    for frame_scene in orbit_scene_dicts(scene, frame_count=frame_count, radius_m=radius_m, height_m=height_m):
        image = mi.render(mi.load_dict(prepare_mitsuba_scene_dict(frame_scene, mi)))
        frames.append(np.asarray(image))
    write_mjpeg_avi(output_path, frames, fps=fps)


def _load_mitsuba(variant: str):
    import mitsuba as mi
    mi.set_variant(variant)
    register_kokoro_bsdf(mi)
    return mi


if __name__ == "__main__":
    main()
