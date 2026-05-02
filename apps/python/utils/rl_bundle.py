from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco  # type: ignore
import numpy as np

from utils.rl_bundle_stage import (
    build_render_scene,
    build_robot_wrapper,
    build_scene_render,
    build_sim_scene,
    compose_stage_metadata,
    robot_source_map,
)
from utils.usd_to_mjcf import USDToMJCFConverter

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENE_USD = REPO_ROOT / "apps" / "python" / "assets" / "default_scene.usda"
TMP_ROOT = REPO_ROOT / "apps" / "python" / "tmp" / "rl_bundles"
BUNDLE_VERSION = 4


@dataclass(frozen=True)
class RenderBundle:
    robot_usd: Path
    scene_usd: Path
    bundle_dir: Path
    mjcf: Path
    sim_scene_usda: Path
    scene_render_usda: Path
    robot_wrapper_usda: Path
    render_scene_usda: Path
    body_map_json: Path
    body_map_jsona: Path
    main_camera_prim: str

    def to_json(self) -> dict[str, str]:
        data = asdict(self)
        return {key: (str(value) if isinstance(value, Path) else value) for key, value in data.items()}

    @classmethod
    def from_json(cls, data: dict[str, str]) -> "RenderBundle":
        path_fields = {
            "robot_usd",
            "scene_usd",
            "bundle_dir",
            "mjcf",
            "sim_scene_usda",
            "scene_render_usda",
            "robot_wrapper_usda",
            "render_scene_usda",
            "body_map_json",
            "body_map_jsona",
        }
        kwargs = {key: (Path(value) if key in path_fields else value) for key, value in data.items()}
        return cls(**kwargs)

    def outputs(self) -> tuple[Path, ...]:
        return (
            self.mjcf,
            self.sim_scene_usda,
            self.scene_render_usda,
            self.robot_wrapper_usda,
            self.render_scene_usda,
            self.body_map_json,
            self.body_map_jsona,
        )


def ensure_render_bundle(robot_usd: Path, scene_usd: Path) -> RenderBundle:
    robot_usd = robot_usd.resolve()
    scene_usd = scene_usd.resolve()
    bundle_dir = TMP_ROOT / _bundle_key(robot_usd, scene_usd)
    manifest_path = bundle_dir / "manifest.json"
    if manifest_path.exists():
        bundle = RenderBundle.from_json(json.loads(manifest_path.read_text(encoding="utf-8")))
        if all(path.exists() for path in bundle.outputs()):
            return bundle

    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle = RenderBundle(
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
        main_camera_prim="/default_viz_camera",
    )
    source_map = robot_source_map(robot_usd)
    up_axis, meters_per_unit = compose_stage_metadata(scene_usd, robot_usd)
    build_sim_scene(robot_usd, scene_usd, bundle.sim_scene_usda, up_axis, meters_per_unit)
    USDToMJCFConverter(bundle.sim_scene_usda, bundle.mjcf, "r1pro_bundle").convert()
    model = mujoco.MjModel.from_xml_path(str(bundle.mjcf))  # type: ignore
    robot_bodies = _robot_body_names(model, source_map)
    body_poses = _body_pose_map(model, robot_bodies)
    body_map = build_robot_wrapper(
        robot_usd,
        robot_bodies,
        source_map,
        body_poses,
        bundle.robot_wrapper_usda,
        up_axis,
        meters_per_unit,
    )
    missing = [name for name in robot_bodies if name not in body_map]
    if missing:
        raise RuntimeError(f"Missing robot wrapper prims: {missing}")
    bundle = RenderBundle(
        **{
            **bundle.__dict__,
            "main_camera_prim": build_scene_render(scene_usd, bundle.scene_render_usda, up_axis, meters_per_unit),
        }
    )
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


def _robot_body_names(model, source_map: dict[str, str]) -> list[str]:
    names: list[str] = []
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)  # type: ignore
        if name and name in source_map:
            names.append(name)
    if not names:
        raise RuntimeError("No robot bodies matched the source USD.")
    return names


def _bundle_key(robot_usd: Path, scene_usd: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"bundle-v{BUNDLE_VERSION}".encode("utf-8"))
    for path in (robot_usd, scene_usd):
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
