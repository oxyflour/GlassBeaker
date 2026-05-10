from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco  # type: ignore
import numpy as np
from pxr import Usd

from utils.camera_override import apply_camera_overrides
from utils.user_config import read_user_config
from utils.zapdos.physics.mujoco_tools import compile_urdf_to_mjcf, merge_mjcf_files

from .camera_specs import build_render_cameras
from .render_bundle import DEFAULT_SCENE_USD, RenderBundle
from .robot_assets import RobotAssetDescriptor, resolve_robot_assets
from .scene_catalog import collect_scene_objects
from .stage_builder import (
    build_render_scene,
    build_robot_wrapper,
    build_scene_render,
    build_sim_scene,
    compose_stage_metadata,
    robot_source_map,
)
from .usd_to_mjcf_adapter import USDToMJCFConverter

REPO_ROOT = Path(__file__).resolve().parents[5]
TMP_ROOT = REPO_ROOT / "apps" / "python" / "tmp" / "rl_bundles"
BUNDLE_VERSION = 7
BUNDLE_DEPENDENCY_FILES = (
    Path(__file__).resolve(),
    Path(__file__).with_name("render_bundle.py").resolve(),
    Path(__file__).with_name("robot_assets.py").resolve(),
    Path(__file__).with_name("stage_builder.py").resolve(),
    Path(__file__).with_name("camera_specs.py").resolve(),
    Path(__file__).with_name("scene_catalog.py").resolve(),
    Path(__file__).with_name("usd_to_mjcf_adapter.py").resolve(),
    REPO_ROOT / "apps" / "python" / "utils" / "zapdos" / "usd_to_mjcf.py",
)


def ensure_render_bundle(robot_usd: Path, scene_usd: Path) -> RenderBundle:
    robot_usd = robot_usd.resolve()
    scene_usd = scene_usd.resolve()
    probe_dir = TMP_ROOT / "_probe" / robot_usd.stem
    probe_descriptor = resolve_robot_assets(robot_usd, probe_dir)
    bundle_dir = TMP_ROOT / _bundle_key(robot_usd, scene_usd, probe_descriptor.dependency_paths)
    manifest_path = bundle_dir / "manifest-v1.json"
    if manifest_path.exists():
        bundle = RenderBundle.from_json(json.loads(manifest_path.read_text(encoding="utf-8")))
        if all(path.exists() for path in bundle.outputs()):
            return bundle

    bundle_dir.mkdir(parents=True, exist_ok=True)
    descriptor = resolve_robot_assets(robot_usd, bundle_dir)
    bundle = _initial_bundle(robot_usd, scene_usd, bundle_dir)
    scene_objects = collect_scene_objects(scene_usd)
    up_axis, meters_per_unit = compose_stage_metadata(scene_usd, descriptor.visual_usd)
    robot_stage = Usd.Stage.Open(str(descriptor.visual_usd))
    if robot_stage is None:
        raise RuntimeError(f"Failed to open robot stage: {descriptor.visual_usd}")
    source_map = _source_map(descriptor, robot_stage, bundle, scene_usd, scene_objects, up_axis, meters_per_unit)
    model = mujoco.MjModel.from_xml_path(str(bundle.mjcf))  # type: ignore
    robot_bodies = _robot_body_names(model, source_map)
    body_poses = _body_pose_map(model, robot_bodies)
    cameras = apply_camera_overrides(build_render_cameras(model, {body: f"MyRobot/{body}" for body in robot_bodies}))
    body_map = _write_render_stages(
        descriptor,
        scene_usd,
        bundle,
        robot_stage,
        source_map,
        robot_bodies,
        body_poses,
        cameras,
        up_axis,
        meters_per_unit,
    )
    missing = [name for name in robot_bodies if name not in body_map]
    if missing:
        raise RuntimeError(f"Missing robot wrapper prims: {missing}")
    for scene_object in scene_objects:
        body_map[scene_object.body_name] = scene_object.render_path
    bundle = RenderBundle(**{**bundle.__dict__, "cameras": cameras})
    build_render_scene(
        bundle.scene_render_usda,
        bundle.robot_wrapper_usda,
        bundle.render_scene_usda,
        up_axis,
        meters_per_unit,
    )
    payload = json.dumps(body_map, indent=2, sort_keys=True)
    bundle.body_map_json.write_text(payload, encoding="utf-8")
    bundle.body_map_jsona.write_text(payload, encoding="utf-8")
    manifest_path.write_text(json.dumps(bundle.to_json(), indent=2, sort_keys=True), encoding="utf-8")
    return bundle


def _initial_bundle(robot_usd: Path, scene_usd: Path, bundle_dir: Path) -> RenderBundle:
    return RenderBundle(
        robot_usd=robot_usd,
        scene_usd=scene_usd,
        bundle_dir=bundle_dir,
        mjcf=bundle_dir / "sim_scene.xml",
        sim_scene_usda=bundle_dir / "sim_scene.usda",
        scene_render_usda=bundle_dir / "scene_render.usda",
        robot_wrapper_usda=bundle_dir / "robot_wrapper.usda",
        render_scene_usda=bundle_dir / "render_scene.usda",
        body_map_json=bundle_dir / "render_scene_body_map.json",
        body_map_jsona=bundle_dir / "render_scene_body_map.jsona",
        cameras=[],
    )


def _source_map(
    descriptor: RobotAssetDescriptor,
    robot_stage: Usd.Stage,
    bundle: RenderBundle,
    scene_usd: Path,
    scene_objects,
    up_axis: str,
    meters_per_unit: float,
) -> dict[str, list[str]]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        sim_scene_future = executor.submit(
            _build_sim_scene_mjcf,
            descriptor,
            scene_usd,
            bundle.sim_scene_usda,
            bundle.mjcf,
            up_axis,
            meters_per_unit,
            DEFAULT_SCENE_USD,
            {spec.sim_path for spec in scene_objects},
        )
        source_map = descriptor.attachments_by_body or robot_source_map(descriptor.visual_usd, source_stage=robot_stage)
        sim_scene_future.result()
    return source_map


def _write_render_stages(
    descriptor: RobotAssetDescriptor,
    scene_usd: Path,
    bundle: RenderBundle,
    robot_stage: Usd.Stage,
    source_map: dict[str, list[str]],
    robot_bodies: list[str],
    body_poses: dict[str, tuple[list[float], list[float]]],
    cameras,
    up_axis: str,
    meters_per_unit: float,
) -> dict[str, str]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        scene_render_future = executor.submit(
            build_scene_render,
            scene_usd,
            bundle.scene_render_usda,
            up_axis,
            meters_per_unit,
            cameras,
            DEFAULT_SCENE_USD,
        )
        body_map = build_robot_wrapper(
            descriptor.visual_usd,
            robot_bodies,
            source_map,
            body_poses,
            cameras,
            bundle.robot_wrapper_usda,
            up_axis,
            meters_per_unit,
            source_stage=robot_stage,
            static_visual_paths=descriptor.static_visual_paths,
        )
        scene_render_future.result()
    return body_map


def _build_sim_scene_mjcf(
    descriptor: RobotAssetDescriptor,
    scene_usd: Path,
    sim_scene_usda: Path,
    mjcf: Path,
    up_axis: str,
    meters_per_unit: float,
    fallback_scene_usd: Path | None,
    force_body_paths: set[str],
) -> None:
    if descriptor.physics_input.suffix.lower() == ".urdf":
        robot_xml = mjcf.with_name("robot.xml")
        scene_xml = mjcf.with_name("scene.xml")
        compile_urdf_to_mjcf(descriptor.physics_input, robot_xml)
        sim_stage = build_sim_scene(None, scene_usd, sim_scene_usda, up_axis, meters_per_unit, fallback_scene_usd)
        USDToMJCFConverter(
            sim_scene_usda,
            scene_xml,
            "scene_bundle",
            force_body_paths=force_body_paths,
            stage=sim_stage,
        ).convert()
        merge_mjcf_files(robot_xml, scene_xml, mjcf)
        return
    sim_stage = build_sim_scene(descriptor.physics_input, scene_usd, sim_scene_usda, up_axis, meters_per_unit, fallback_scene_usd)
    USDToMJCFConverter(
        sim_scene_usda,
        mjcf,
        "r1pro_bundle",
        force_body_paths=force_body_paths,
        stage=sim_stage,
    ).convert()


def _robot_body_names(model, source_map: dict[str, list[str]]) -> list[str]:
    names: list[str] = []
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)  # type: ignore
        if name and name in source_map:
            names.append(name)
    if not names:
        raise RuntimeError("No robot bodies matched the source USD.")
    return names


def _bundle_key(robot_usd: Path, scene_usd: Path, extra_paths: list[Path] | tuple[Path, ...] = ()) -> str:
    digest = hashlib.sha1()
    digest.update(f"bundle-v{BUNDLE_VERSION}".encode("utf-8"))
    try:
        overrides = read_user_config().get("override", {})
    except RuntimeError:
        overrides = {}
    digest.update(json.dumps(overrides, sort_keys=True).encode("utf-8"))
    for path in dict.fromkeys((robot_usd, scene_usd, *extra_paths, *BUNDLE_DEPENDENCY_FILES)):
        stat = path.stat()
        digest.update(str(path).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
    return digest.hexdigest()[:16]


def _body_pose_map(model, body_names: list[str]) -> dict[str, tuple[list[float], list[float]]]:
    data = mujoco.MjData(model)  # type: ignore
    mujoco.mj_forward(model, data)  # type: ignore
    quat = np.empty(4, dtype=float)
    pose_map: dict[str, tuple[list[float], list[float]]] = {}
    for body_name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)  # type: ignore
        pos = np.array(data.xpos[body_id], dtype=float)
        mujoco.mju_mat2Quat(quat, np.array(data.xmat[body_id], dtype=float).reshape(-1))  # type: ignore
        pose_map[body_name] = (pos.tolist(), quat.tolist())
    return pose_map


__all__ = [
    "BUNDLE_DEPENDENCY_FILES",
    "BUNDLE_VERSION",
    "TMP_ROOT",
    "_body_pose_map",
    "_build_sim_scene_mjcf",
    "_bundle_key",
    "_robot_body_names",
    "ensure_render_bundle",
]
