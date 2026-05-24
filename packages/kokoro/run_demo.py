from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kokoro.brdf import BrdfTrainingConfig, build_brdf_dataset, export_surrogate_npz, train_brdf_surrogate
from kokoro.height_field import compile_height_program, write_height_map_png
from kokoro.mitsuba_height_field_bsdf import register_height_field_bsdf
from kokoro.mitsuba_neural_bsdf import register_kokoro_bsdf
from kokoro.mitsuba_scene import (
    build_height_field_reference_scene_dict,
    build_kokoro_ring_diagnostic_scene_dict,
    build_kokoro_scene_dict,
    orbit_scene_dicts,
    prepare_mitsuba_scene_dict,
)
from kokoro.video import write_mp4

DEFAULT_HEIGHT_SOURCE = """
def height(x, y):
    return radial_rotated_pyramid_height(
        x,
        y,
        period_m=500e-6,
        amplitude_m=150e-6,
        max_rotation_rad=2.0 * math.pi / 4.0,
    )
"""
DEFAULT_OUTPUT_DIR = Path("packages/kokoro/tmp")
DEFAULT_FEATURE_PERIOD_M = None
DEFAULT_LOCAL_FEATURE_PERIOD_M = None
DEFAULT_POSITION_FREQUENCY_COUNT = 9
DEFAULT_SAMPLES = 8192
DEFAULT_EPOCHS = 240
DEFAULT_HIDDEN_DIM = 128
DEFAULT_HIDDEN_LAYERS = 5
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 2e-3
DEFAULT_ACTIVATION = "sine"
DEFAULT_OMEGA_0 = 4.0
DEFAULT_AVERAGE_PATCH_RADIUS_M = 0.0
DEFAULT_AVERAGE_PATCH_SAMPLES = 1
DEFAULT_INSPECTION_LIGHT_SCALE = 0.0
DEFAULT_LOBE_KAPPA = 4096.0
DEFAULT_VIDEO_FRAMES = 120
DEFAULT_VIDEO_FPS = 24
DEFAULT_FOV = 65.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a kokoro neural BRDF and write a Mitsuba scene.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--height-source", type=Path, default=None)
    parser.add_argument("--hdr-path", type=Path, default=Path("apps/web/public/studio_small_03_1k.hdr"))
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--hidden-layers", type=int, choices=range(1, 6), default=DEFAULT_HIDDEN_LAYERS)
    parser.add_argument("--activation", choices=["sine", "tanh"], default=DEFAULT_ACTIVATION)
    parser.add_argument("--omega-0", type=float, default=DEFAULT_OMEGA_0)
    parser.add_argument("--local-feature-period-m", type=float, default=DEFAULT_LOCAL_FEATURE_PERIOD_M)
    parser.add_argument("--position-frequency-count", type=int, default=DEFAULT_POSITION_FREQUENCY_COUNT)
    parser.add_argument("--average-patch-radius-m", type=float, default=DEFAULT_AVERAGE_PATCH_RADIUS_M)
    parser.add_argument("--average-patch-samples", type=int, default=DEFAULT_AVERAGE_PATCH_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--film-width", type=int, default=512)
    parser.add_argument("--film-height", type=int, default=384)
    parser.add_argument("--height-map-size", type=int, default=4096)
    parser.add_argument("--fov", type=float, default=DEFAULT_FOV)
    parser.add_argument("--spp", type=int, default=64)
    parser.add_argument("--width-m", type=float, default=0.10)
    parser.add_argument("--depth-m", type=float, default=0.10)
    parser.add_argument("--inspection-light-scale", type=float, default=DEFAULT_INSPECTION_LIGHT_SCALE)
    parser.add_argument("--lobe-kappa", type=float, default=DEFAULT_LOBE_KAPPA)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--reference-render", action="store_true")
    parser.add_argument("--reference-lobe-kappa", type=float, default=4096.0)
    parser.add_argument("--reference-normal-step-m", type=float, default=25e-6)
    parser.add_argument("--ring-diagnostic", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-frames", type=int, default=DEFAULT_VIDEO_FRAMES)
    parser.add_argument("--video-fps", type=int, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--orbit-radius", type=float, default=0.18)
    parser.add_argument("--camera-height", type=float, default=0.10)
    parser.add_argument("--variant", default="cuda_ad_rgb")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.height_source.read_text(encoding="utf-8") if args.height_source else DEFAULT_HEIGHT_SOURCE
    program = compile_height_program(source)
    height_path = height_output_path(args.output_dir)
    write_height_map_png(
        program,
        height_path,
        width_m=args.width_m,
        depth_m=args.depth_m,
        image_size=args.height_map_size,
    )
    dataset = build_brdf_dataset(
        program,
        sample_count=args.samples,
        width_m=args.width_m,
        depth_m=args.depth_m,
        seed=13,
        local_feature_period_m=args.local_feature_period_m,
        position_frequency_count=args.position_frequency_count,
        average_patch_radius_m=args.average_patch_radius_m,
        average_patch_sample_count=args.average_patch_samples,
    )
    result = train_brdf_surrogate(
        dataset,
        BrdfTrainingConfig(
            hidden_dim=args.hidden_dim,
            hidden_layer_count=args.hidden_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=17,
            activation=args.activation,
            omega_0=args.omega_0,
        ),
    )
    checkpoint = args.output_dir / "kokoro_brdf.npz"
    export_surrogate_npz(
        result.model,
        checkpoint,
        width_m=args.width_m,
        depth_m=args.depth_m,
        local_feature_period_m=args.local_feature_period_m,
        position_frequency_count=args.position_frequency_count,
        average_patch_radius_m=args.average_patch_radius_m,
        average_patch_sample_count=args.average_patch_samples,
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
        inspection_light_scale=args.inspection_light_scale,
        lobe_kappa=args.lobe_kappa,
    )
    scene_path = args.output_dir / "kokoro_scene.json"
    scene_path.write_text(json.dumps(scene, indent=2), encoding="utf-8")
    metrics = {"initial_loss": result.loss_history[0], "final_loss": result.loss_history[-1]}
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    saved_files = {
        "height_map": height_path,
        "checkpoint": checkpoint,
        "scene": scene_path,
        "metrics": metrics_path,
    }
    reference_scene: dict[str, Any] | None = None
    if args.render:
        render_path = render_output_path(args.output_dir)
        _render(scene, render_path, args.variant)
        saved_files["render"] = render_path
    if args.reference_render:
        reference_scene = _build_reference_scene(source, args)
        reference_path = reference_render_output_path(args.output_dir)
        _render(reference_scene, reference_path, args.variant)
        saved_files["height_reference"] = reference_path
    if args.ring_diagnostic:
        ring_scene = build_kokoro_ring_diagnostic_scene_dict(
            checkpoint_path=checkpoint,
            width=args.film_width,
            height=args.film_height,
            width_m=args.width_m,
            depth_m=args.depth_m,
            spp=args.spp,
        )
        ring_path = ring_diagnostic_output_path(args.output_dir)
        _render(ring_scene, ring_path, args.variant)
        saved_files["ring_diagnostic"] = ring_path
    if args.video:
        video_path = video_output_path(args.output_dir)
        _render_video(
            scene,
            video_path,
            args.variant,
            frame_count=args.video_frames,
            fps=args.video_fps,
            radius_m=args.orbit_radius,
            height_m=args.camera_height,
        )
        saved_files["video"] = video_path
        if args.reference_render:
            if reference_scene is None:
                reference_scene = _build_reference_scene(source, args)
            reference_video_path = video_reference_output_path(args.output_dir)
            _render_video(
                reference_scene,
                reference_video_path,
                args.variant,
                frame_count=args.video_frames,
                fps=args.video_fps,
                radius_m=args.orbit_radius,
                height_m=args.camera_height,
            )
            saved_files["video_reference"] = reference_video_path
    print("Saved files:")
    for label, path in saved_files.items():
        print(f"  {label}: {path.resolve()}")
    print("Metrics:")
    print(f"  initial_loss: {metrics['initial_loss']}")
    print(f"  final_loss: {metrics['final_loss']}")


def _build_reference_scene(source: str, args: argparse.Namespace) -> dict[str, Any]:
    return build_height_field_reference_scene_dict(
        height_source=source,
        width=args.film_width,
        height=args.film_height,
        fov=args.fov,
        width_m=args.width_m,
        depth_m=args.depth_m,
        spp=args.spp,
        inspection_light_scale=args.inspection_light_scale,
        normal_step_m=args.reference_normal_step_m,
        lobe_kappa=args.reference_lobe_kappa,
    )


def render_output_path(output_dir: Path) -> Path:
    return output_dir / "kokoro_render.png"


def reference_render_output_path(output_dir: Path) -> Path:
    return output_dir / "kokoro_height_reference.png"


def height_output_path(output_dir: Path) -> Path:
    return output_dir / "kokoro_height.png"


def ring_diagnostic_output_path(output_dir: Path) -> Path:
    return output_dir / "kokoro_ring_diagnostic.png"


def video_output_path(output_dir: Path) -> Path:
    return output_dir / "kokoro_orbit.mp4"


def video_reference_output_path(output_dir: Path) -> Path:
    return output_dir / "kokoro_orbit_reference.mp4"


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
    write_mp4(output_path, frames, fps=fps)


def _load_mitsuba(variant: str):
    import mitsuba as mi
    mi.set_variant(variant)
    register_kokoro_bsdf(mi)
    register_height_field_bsdf(mi)
    return mi


if __name__ == "__main__":
    main()
