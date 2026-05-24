from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from kokoro.brdf import (
    BrdfTrainingConfig,
    build_brdf_dataset,
    estimate_periodic_phase_vectors,
    export_surrogate_npz,
    train_brdf_surrogate,
)
from kokoro.direction_metrics import (
    DirectionHoldoutConfig,
    angular_error_degrees,
    build_direction_holdout_dataset,
)
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

WAVE_BLOCK_HEIGHT_SOURCE = """
def _sin(value):
    if "torch" in globals():
        return torch.sin(value)
    return dr.sin(value)


def _abs(value):
    if "torch" in globals():
        return torch.abs(value)
    return dr.abs(value)


def _maximum(a, b):
    if "torch" in globals():
        return torch.maximum(a, b)
    return dr.maximum(a, b)


def _where(condition, when_true, when_false):
    if "torch" in globals():
        return torch.where(condition, when_true, when_false)
    return dr.select(condition, when_true, when_false)


def _remainder(value, period):
    if "torch" in globals():
        return torch.remainder(value, period)
    return value - period * dr.floor(value / period)


def wave(x, y):
    phase = (2.0 * math.pi / 250e-6) * (x + 0.01 * y)
    return 2e-6 * (0.5 + 0.5 * _sin(phase))


def block(x, y):
    period_m = 500e-6
    gap_m = 80e-6
    block_width_m = 40e-6
    slope_width_m = 80e-6
    block_pitch_m = block_width_m + 2.0 * slope_width_m + gap_m
    local_x = _remainder(x + 0.5 * period_m, period_m) - 0.5 * period_m
    local_y = _remainder(y + 0.5 * period_m, period_m) - 0.5 * period_m
    cell_x = _remainder(local_x + 0.5 * block_pitch_m, block_pitch_m) - 0.5 * block_pitch_m
    cell_y = _remainder(local_y + 0.5 * block_pitch_m, block_pitch_m) - 0.5 * block_pitch_m
    edge_distance = _maximum(_abs(cell_x), _abs(cell_y))
    top_half_width_m = 0.5 * block_width_m
    ramp = 1.0 - _maximum(edge_distance - top_half_width_m, x * 0.0) / slope_width_m
    height = 10e-6 * _maximum(ramp, x * 0.0)
    return _where(height > 0.0, height, x * 0.0)


def height(x, y):
    return _maximum(wave(x, y), block(x, y))
"""
RADIAL_PYRAMID_HEIGHT_SOURCE = """
def height(x, y):
    return radial_rotated_pyramid_height(
        x,
        y,
        period_m=500e-6,
        amplitude_m=150e-6,
        max_rotation_rad=2.0 * math.pi / 4.0,
    )
"""
DEFAULT_HEIGHT_PRESET = "wave-block"
HEIGHT_PRESET_SOURCES = {
    "wave-block": WAVE_BLOCK_HEIGHT_SOURCE,
    "radial-pyramid": RADIAL_PYRAMID_HEIGHT_SOURCE,
}
DEFAULT_HEIGHT_SOURCE = HEIGHT_PRESET_SOURCES[DEFAULT_HEIGHT_PRESET]
DEFAULT_OUTPUT_DIR = Path("packages/kokoro/tmp")
DEFAULT_HDR_PATH = Path("apps/web/public/studio_small_03_1k.hdr")
DEFAULT_FEATURE_PERIOD_M = None
DEFAULT_LOCAL_FEATURE_PERIOD_M = None
DEFAULT_DFT_PHASE_VECTOR_COUNT = 4
DEFAULT_DFT_PHASE_GRID_SIZE = 256
DEFAULT_DFT_PHASE_WINDOW_M = 2.0e-3
DEFAULT_POSITION_FREQUENCY_COUNT = 0
DEFAULT_RADIAL_CELL_FEATURE_PERIOD_M = 500e-6
DEFAULT_RADIAL_CELL_FEATURE_MAX_ROTATION_RAD = math.pi / 2.0
DEFAULT_RADIAL_CELL_FEATURE_RADIAL_POWER = 1.0
DEFAULT_RADIAL_CELL_FACET_FEATURES = True
DEFAULT_SAMPLES = 16384
DEFAULT_EPOCHS = 240
DEFAULT_HIDDEN_DIM = 128
DEFAULT_HIDDEN_LAYERS = 3
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 2e-3
DEFAULT_HOLDOUT_GRID_SIZE = 32
DEFAULT_HOLDOUT_THETA_COUNT = 5
DEFAULT_HOLDOUT_PHI_COUNT = 8
DEFAULT_TARGET_MODE = "normal"
DEFAULT_NORMAL_STEP_M = 0.5e-6
DEFAULT_ACTIVATION = "tanh"
DEFAULT_OMEGA_0 = 4.0
DEFAULT_AVERAGE_PATCH_RADIUS_M = 0.0
DEFAULT_AVERAGE_PATCH_SAMPLES = 1
DEFAULT_LIGHT_SOURCE = "point"
DEFAULT_ENV_SCALE = 1.0
DEFAULT_INSPECTION_LIGHT_SCALE = 0.0
DEFAULT_LOBE_KAPPA = 2048.0
DEFAULT_RENDER_SAMPLER = "ldsampler"
DEFAULT_RECONSTRUCTION_FILTER = "box"
DEFAULT_FILM_WIDTH = 384
DEFAULT_FILM_HEIGHT = 288
DEFAULT_SPP = 1024
DEFAULT_VIDEO_FRAMES = 120
DEFAULT_VIDEO_FPS = 24
DEFAULT_FOV = 65.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a kokoro neural BRDF and write a Mitsuba scene.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--height-source", type=Path, default=None)
    parser.add_argument("--height-preset", choices=sorted(HEIGHT_PRESET_SOURCES), default=DEFAULT_HEIGHT_PRESET)
    parser.add_argument("--hdr-path", type=Path, default=DEFAULT_HDR_PATH)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--hidden-layers", type=int, choices=range(1, 6), default=DEFAULT_HIDDEN_LAYERS)
    parser.add_argument("--activation", choices=["sine", "tanh"], default=DEFAULT_ACTIVATION)
    parser.add_argument("--omega-0", type=float, default=DEFAULT_OMEGA_0)
    parser.add_argument("--local-feature-period-m", type=float, default=DEFAULT_LOCAL_FEATURE_PERIOD_M)
    parser.add_argument("--dft-phase-vector-count", type=int, default=DEFAULT_DFT_PHASE_VECTOR_COUNT)
    parser.add_argument("--dft-phase-grid-size", type=int, default=DEFAULT_DFT_PHASE_GRID_SIZE)
    parser.add_argument("--dft-phase-window-m", type=float, default=DEFAULT_DFT_PHASE_WINDOW_M)
    parser.add_argument("--disable-dft-phase-features", action="store_true")
    parser.add_argument("--position-frequency-count", type=int, default=DEFAULT_POSITION_FREQUENCY_COUNT)
    parser.add_argument("--radial-cell-feature-period-m", type=float, default=None)
    parser.add_argument("--radial-cell-feature-max-rotation-rad", type=float, default=DEFAULT_RADIAL_CELL_FEATURE_MAX_ROTATION_RAD)
    parser.add_argument("--radial-cell-feature-radial-power", type=float, default=DEFAULT_RADIAL_CELL_FEATURE_RADIAL_POWER)
    parser.add_argument("--radial-cell-facet-features", action="store_true")
    parser.add_argument("--disable-radial-cell-facet-features", action="store_true")
    parser.add_argument("--average-patch-radius-m", type=float, default=DEFAULT_AVERAGE_PATCH_RADIUS_M)
    parser.add_argument("--average-patch-samples", type=int, default=DEFAULT_AVERAGE_PATCH_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--target-mode", choices=["reflection", "normal"], default=DEFAULT_TARGET_MODE)
    parser.add_argument("--normal-step-m", type=float, default=DEFAULT_NORMAL_STEP_M)
    parser.add_argument("--holdout-grid-size", type=int, default=DEFAULT_HOLDOUT_GRID_SIZE)
    parser.add_argument("--holdout-theta-count", type=int, default=DEFAULT_HOLDOUT_THETA_COUNT)
    parser.add_argument("--holdout-phi-count", type=int, default=DEFAULT_HOLDOUT_PHI_COUNT)
    parser.add_argument("--film-width", type=int, default=DEFAULT_FILM_WIDTH)
    parser.add_argument("--film-height", type=int, default=DEFAULT_FILM_HEIGHT)
    parser.add_argument("--height-map-size", type=int, default=4096)
    parser.add_argument("--fov", type=float, default=DEFAULT_FOV)
    parser.add_argument("--spp", type=int, default=DEFAULT_SPP)
    parser.add_argument("--width-m", type=float, default=0.10)
    parser.add_argument("--depth-m", type=float, default=0.10)
    parser.add_argument("--light-source", choices=["point", "hdr"], default=DEFAULT_LIGHT_SOURCE)
    parser.add_argument("--env-scale", type=float, default=DEFAULT_ENV_SCALE)
    parser.add_argument("--inspection-light-scale", type=float, default=DEFAULT_INSPECTION_LIGHT_SCALE)
    parser.add_argument("--lobe-kappa", type=float, default=DEFAULT_LOBE_KAPPA)
    parser.add_argument(
        "--sampler-type",
        choices=["independent", "ldsampler", "multijitter", "stratified"],
        default=DEFAULT_RENDER_SAMPLER,
    )
    parser.add_argument(
        "--reconstruction-filter",
        choices=["box", "tent", "gaussian"],
        default=DEFAULT_RECONSTRUCTION_FILTER,
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--reference-render", action="store_true")
    parser.add_argument("--reference-lobe-kappa", type=float, default=4096.0)
    parser.add_argument("--reference-normal-step-m", type=float, default=DEFAULT_NORMAL_STEP_M)
    parser.add_argument("--ring-diagnostic", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-frames", type=int, default=DEFAULT_VIDEO_FRAMES)
    parser.add_argument("--video-fps", type=int, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--orbit-radius", type=float, default=0.1)
    parser.add_argument("--camera-height", type=float, default=0.10)
    parser.add_argument("--variant", default="cuda_ad_rgb")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.height_source.read_text(encoding="utf-8") if args.height_source else HEIGHT_PRESET_SOURCES[args.height_preset]
    radial_cell_feature_period_m = args.radial_cell_feature_period_m
    uses_radial_default = args.height_source is None and args.height_preset == "radial-pyramid"
    if radial_cell_feature_period_m is None and uses_radial_default:
        radial_cell_feature_period_m = DEFAULT_RADIAL_CELL_FEATURE_PERIOD_M
    if radial_cell_feature_period_m is not None and radial_cell_feature_period_m <= 0.0:
        radial_cell_feature_period_m = None
    radial_cell_facet_features = (
        radial_cell_feature_period_m is not None
        and not args.disable_radial_cell_facet_features
        and (args.radial_cell_facet_features or (uses_radial_default and DEFAULT_RADIAL_CELL_FACET_FEATURES))
    )
    program = compile_height_program(source)
    dft_phase_vectors = [] if args.disable_dft_phase_features else default_dft_phase_vectors(
        program,
        width_m=args.width_m,
        depth_m=args.depth_m,
        window_m=args.dft_phase_window_m,
        grid_size=args.dft_phase_grid_size,
        max_vectors=args.dft_phase_vector_count,
    )
    height_path = height_output_path(args.output_dir)
    write_height_map_png(
        program,
        height_path,
        width_m=args.width_m,
        depth_m=args.depth_m,
        image_size=args.height_map_size,
    )
    sample_generation_start = time.perf_counter()
    dataset = build_brdf_dataset(
        program,
        sample_count=args.samples,
        width_m=args.width_m,
        depth_m=args.depth_m,
        seed=13,
        local_feature_period_m=args.local_feature_period_m,
        dft_phase_vectors=dft_phase_vectors,
        position_frequency_count=args.position_frequency_count,
        radial_cell_feature_period_m=radial_cell_feature_period_m,
        radial_cell_feature_max_rotation_rad=args.radial_cell_feature_max_rotation_rad,
        radial_cell_feature_radial_power=args.radial_cell_feature_radial_power,
        radial_cell_facet_features=radial_cell_facet_features,
        average_patch_radius_m=args.average_patch_radius_m,
        average_patch_sample_count=args.average_patch_samples,
        normal_step_m=args.normal_step_m,
        target_mode=args.target_mode,
    )
    sample_generation_time_seconds = time.perf_counter() - sample_generation_start
    training_start = time.perf_counter()
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
    training_time_seconds = time.perf_counter() - training_start
    checkpoint = args.output_dir / "kokoro_brdf.npz"
    export_surrogate_npz(
        result.model,
        checkpoint,
        width_m=args.width_m,
        depth_m=args.depth_m,
        local_feature_period_m=args.local_feature_period_m,
        dft_phase_vectors=dft_phase_vectors,
        position_frequency_count=args.position_frequency_count,
        radial_cell_feature_period_m=radial_cell_feature_period_m,
        radial_cell_feature_max_rotation_rad=args.radial_cell_feature_max_rotation_rad,
        radial_cell_feature_radial_power=args.radial_cell_feature_radial_power,
        radial_cell_facet_features=radial_cell_facet_features,
        average_patch_radius_m=args.average_patch_radius_m,
        average_patch_sample_count=args.average_patch_samples,
        normal_step_m=args.normal_step_m,
        include_incident_features=args.target_mode == "reflection",
        target_mode=args.target_mode,
    )
    stale_mesh = args.output_dir / "kokoro_surface.ply"
    if stale_mesh.exists():
        stale_mesh.unlink()
    holdout = build_direction_holdout_dataset(
        program,
        DirectionHoldoutConfig(
            x_count=args.holdout_grid_size,
            y_count=args.holdout_grid_size,
            theta_count=args.holdout_theta_count,
            phi_count=args.holdout_phi_count,
        ),
        width_m=args.width_m,
        depth_m=args.depth_m,
        local_feature_period_m=args.local_feature_period_m,
        dft_phase_vectors=dft_phase_vectors,
        position_frequency_count=args.position_frequency_count,
        radial_cell_feature_period_m=radial_cell_feature_period_m,
        radial_cell_feature_max_rotation_rad=args.radial_cell_feature_max_rotation_rad,
        radial_cell_feature_radial_power=args.radial_cell_feature_radial_power,
        radial_cell_facet_features=radial_cell_facet_features,
        normal_step_m=args.normal_step_m,
        include_incident_features=args.target_mode == "reflection",
        target_mode=args.target_mode,
    )
    scene = build_kokoro_scene_dict(
        checkpoint_path=checkpoint,
        hdr_path=args.hdr_path,
        width=args.film_width,
        height=args.film_height,
        fov=args.fov,
        width_m=args.width_m,
        depth_m=args.depth_m,
        spp=args.spp,
        light_source=args.light_source,
        env_scale=args.env_scale,
        inspection_light_scale=args.inspection_light_scale,
        lobe_kappa=args.lobe_kappa,
        sampler_type=args.sampler_type,
        reconstruction_filter=args.reconstruction_filter,
    )
    scene_path = args.output_dir / "kokoro_scene.json"
    scene_path.write_text(json.dumps(scene, indent=2), encoding="utf-8")
    metrics_path = args.output_dir / "metrics.json"
    saved_files = {
        "height_map": height_path,
        "checkpoint": checkpoint,
        "scene": scene_path,
        "metrics": metrics_path,
    }
    render_times_seconds: dict[str, float] = {}
    reference_scene: dict[str, Any] | None = None
    if args.render:
        render_path = render_output_path(args.output_dir)
        render_start = time.perf_counter()
        _render(scene, render_path, args.variant)
        render_times_seconds["render"] = time.perf_counter() - render_start
        saved_files["render"] = render_path
    if args.reference_render:
        reference_scene = _build_reference_scene(source, args)
        reference_path = reference_render_output_path(args.output_dir)
        render_start = time.perf_counter()
        _render(reference_scene, reference_path, args.variant)
        render_times_seconds["height_reference"] = time.perf_counter() - render_start
        saved_files["height_reference"] = reference_path
    if args.ring_diagnostic:
        ring_scene = build_kokoro_ring_diagnostic_scene_dict(
            checkpoint_path=checkpoint,
            width=args.film_width,
            height=args.film_height,
            width_m=args.width_m,
            depth_m=args.depth_m,
            spp=args.spp,
            sampler_type=args.sampler_type,
            reconstruction_filter=args.reconstruction_filter,
        )
        ring_path = ring_diagnostic_output_path(args.output_dir)
        render_start = time.perf_counter()
        _render(ring_scene, ring_path, args.variant)
        render_times_seconds["ring_diagnostic"] = time.perf_counter() - render_start
        saved_files["ring_diagnostic"] = ring_path
    if args.video:
        video_path = video_output_path(args.output_dir)
        render_start = time.perf_counter()
        _render_video(
            scene,
            video_path,
            args.variant,
            frame_count=args.video_frames,
            fps=args.video_fps,
            radius_m=args.orbit_radius,
            height_m=args.camera_height,
        )
        render_times_seconds["video"] = time.perf_counter() - render_start
        saved_files["video"] = video_path
        if args.reference_render:
            if reference_scene is None:
                reference_scene = _build_reference_scene(source, args)
            reference_video_path = video_reference_output_path(args.output_dir)
            render_start = time.perf_counter()
            _render_video(
                reference_scene,
                reference_video_path,
                args.variant,
                frame_count=args.video_frames,
                fps=args.video_fps,
                radius_m=args.orbit_radius,
                height_m=args.camera_height,
            )
            render_times_seconds["video_reference"] = time.perf_counter() - render_start
            saved_files["video_reference"] = reference_video_path
    metrics = {
        "initial_loss": result.loss_history[0],
        "final_loss": result.loss_history[-1],
        "sample_count": int(dataset.features.shape[0]),
        "loss_every_100_steps": [
            {"step": step, "loss": result.loss_history[step - 1]}
            for step in range(100, len(result.loss_history) + 1, 100)
        ],
        "sample_generation_time_seconds": sample_generation_time_seconds,
        "training_time_seconds": training_time_seconds,
        "rendering_time_seconds": sum(render_times_seconds.values(), 0.0),
        "render_times_seconds": render_times_seconds,
        "holdout_sample_count": int(holdout.features.shape[0]),
        "dft_phase_vectors": [list(vector) for vector in dft_phase_vectors],
        "dft_phase_window_m": args.dft_phase_window_m,
        "holdout_angular_error": angular_error_degrees(result.model, holdout.features, holdout.targets),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved files:")
    for label, path in saved_files.items():
        print(f"  {label}: {path.resolve()}")
    print("Metrics:")
    print(f"  initial_loss: {metrics['initial_loss']}")
    print(f"  final_loss: {metrics['final_loss']}")
    print(f"  sample_count: {metrics['sample_count']}")
    print(f"  sample_generation_time_seconds: {metrics['sample_generation_time_seconds']}")
    print(f"  training_time_seconds: {metrics['training_time_seconds']}")
    print(f"  rendering_time_seconds: {metrics['rendering_time_seconds']}")
    if render_times_seconds:
        print("  render_times_seconds:")
        for label, duration_seconds in render_times_seconds.items():
            print(f"    {label}: {duration_seconds}")
    print("  loss_every_100_steps:")
    for loss_point in metrics["loss_every_100_steps"]:
        print(f"    step {loss_point['step']}: {loss_point['loss']}")


def default_dft_phase_vectors(
    program,
    *,
    width_m: float,
    depth_m: float,
    window_m: float = DEFAULT_DFT_PHASE_WINDOW_M,
    grid_size: int = DEFAULT_DFT_PHASE_GRID_SIZE,
    max_vectors: int = DEFAULT_DFT_PHASE_VECTOR_COUNT,
) -> list[tuple[float, float]]:
    return estimate_periodic_phase_vectors(
        program,
        width_m=width_m,
        depth_m=depth_m,
        window_width_m=window_m,
        window_depth_m=window_m,
        grid_size=grid_size,
        max_vectors=max_vectors,
    )


def _build_reference_scene(source: str, args: argparse.Namespace) -> dict[str, Any]:
    return build_height_field_reference_scene_dict(
        height_source=source,
        width=args.film_width,
        height=args.film_height,
        fov=args.fov,
        width_m=args.width_m,
        depth_m=args.depth_m,
        spp=args.spp,
        hdr_path=args.hdr_path,
        light_source=args.light_source,
        env_scale=args.env_scale,
        inspection_light_scale=args.inspection_light_scale,
        normal_step_m=args.reference_normal_step_m,
        lobe_kappa=args.reference_lobe_kappa,
        sampler_type=args.sampler_type,
        reconstruction_filter=args.reconstruction_filter,
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
