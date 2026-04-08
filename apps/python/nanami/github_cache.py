from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import CACHE_ROOT, URDF_ROOT
from .models import RobotManifest, SourceConfig, manifest_from_dict
from .stl_obj import stl_to_obj
from .urdf import parse_robot_manifest


@dataclass(slots=True)
class AssetBundle:
    manifest: RobotManifest
    cache_hit: bool
    cache_root: Path
    runtime_key: str
    asset_key: str


def asset_files(robot_root: Path) -> list[Path]:
    files = [path for path in robot_root.rglob("*") if path.is_file()]
    return [
        path for path in sorted(files)
        if path.name == "package.xml" or "meshes" in path.parts or "urdf" in path.parts
    ]


def cache_key(robot_root: Path) -> str:
    digest = hashlib.sha1()
    for path in asset_files(robot_root):
        stat = path.stat()
        digest.update(str(path.relative_to(robot_root)).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(str(stat.st_size).encode("ascii"))
    return digest.hexdigest()[:12]


def cache_root_for(robot_root: Path, source: SourceConfig) -> Path:
    source_key = hashlib.sha1(str(robot_root.resolve()).encode("utf-8")).hexdigest()[:8]
    name = f"{robot_root.name}-{source_key}"
    return CACHE_ROOT / name / cache_key(robot_root)


def runtime_key_for(cache_root: Path) -> str:
    return str(cache_root.resolve())


def asset_key_for(cache_root: Path) -> str:
    rel = cache_root.resolve().relative_to(CACHE_ROOT.resolve())
    safe = "_".join(part.replace("-", "_") for part in rel.parts)
    return safe.lower()


def find_robot_root(source: SourceConfig) -> Path:
    robot_root = Path(source.robot_path)
    if not robot_root.is_absolute():
        robot_root = URDF_ROOT / robot_root
    if not robot_root.exists():
        raise FileNotFoundError(f"Nanami asset root not found: {robot_root}")
    return robot_root


def find_urdf_path(robot_root: Path) -> Path:
    candidates = sorted((robot_root / "urdf").glob("*.urdf"))
    if not candidates:
        candidates = sorted(robot_root.rglob("*.urdf"))
    if not candidates:
        raise FileNotFoundError(f"No URDF found under {robot_root}")
    return candidates[0]


def mesh_relative_path(mesh_source: str, package_name: str) -> str:
    prefix = f"package://{package_name}/"
    if mesh_source.startswith(prefix):
        return mesh_source.removeprefix(prefix)
    if mesh_source.startswith("package://"):
        return mesh_source.split("/", 3)[-1]
    return mesh_source


def obj_relative_path(mesh_rel: str) -> Path:
    return Path(mesh_rel).with_suffix(".obj")


def build_manifest(robot_root: Path, cache_root: Path, source: SourceConfig) -> RobotManifest:
    urdf_src = find_urdf_path(robot_root)
    urdf_dst = cache_root / "raw" / "urdf" / urdf_src.name
    urdf_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(urdf_src, urdf_dst)
    manifest = parse_robot_manifest(urdf_dst, source, f"local:{cache_root.name}")
    for link in manifest.links:
        if not link.mesh_source:
            continue
        mesh_rel = mesh_relative_path(link.mesh_source, manifest.package_name)
        stl_src = robot_root / mesh_rel
        if not stl_src.exists():
            raise FileNotFoundError(f"Mesh not found: {stl_src}")
        stl_dst = cache_root / "raw" / mesh_rel
        obj_path = cache_root / "converted" / "obj" / obj_relative_path(mesh_rel)
        stl_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stl_src, stl_dst)
        stl_to_obj(stl_dst, obj_path)
        link.mesh_obj_path = str(obj_path)
    return manifest


def ensure_r1_asset_cache(source: SourceConfig) -> AssetBundle:
    robot_root = find_robot_root(source)
    cache_root = cache_root_for(robot_root, source)
    runtime_key = runtime_key_for(cache_root)
    asset_key = asset_key_for(cache_root)
    manifest_path = cache_root / "robot_manifest.json"
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return AssetBundle(
            manifest=manifest_from_dict(data),
            cache_hit=True,
            cache_root=cache_root,
            runtime_key=runtime_key,
            asset_key=asset_key,
        )

    manifest = build_manifest(robot_root, cache_root, source)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return AssetBundle(
        manifest=manifest,
        cache_hit=False,
        cache_root=cache_root,
        runtime_key=runtime_key,
        asset_key=asset_key,
    )
