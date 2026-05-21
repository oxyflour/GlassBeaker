from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from pxr import Usd, UsdGeom

if TYPE_CHECKING:
    from utils.zapdos.bundle import RenderBundle
    from utils.zapdos.bundle.camera_specs import RenderCamera


def build_mitsuba_scene_dict(
    bundle: "RenderBundle",
    mesh_dir: Path,
    width: int,
    height: int,
    *,
    spp: int = 4,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stage = Usd.Stage.Open(str(bundle.render_scene_usda))
    if stage is None:
        raise RuntimeError(f"Failed to open render scene: {bundle.render_scene_usda}")
    mesh_dir.mkdir(parents=True, exist_ok=True)
    scene: dict[str, Any] = {
        "type": "scene",
        "integrator": {"type": "direct"},
        "default_light": {"type": "constant", "radiance": {"type": "rgb", "value": [0.8, 0.8, 0.8]}},
    }
    xforms = UsdGeom.XformCache()
    body_paths = _body_render_paths(bundle)
    snapshots = [_camera_snapshot(camera) for camera in bundle.cameras]
    for index, camera in enumerate(bundle.cameras):
        scene[f"sensor_{camera.name}"] = _camera_sensor(camera, width, height, spp, stage, xforms, body_paths)
    mesh_count = _add_meshes(scene, stage, mesh_dir, bundle)
    if mesh_count == 0:
        scene["fallback_sphere"] = {
            "type": "sphere",
            "center": [0.0, 0.0, 0.6],
            "radius": 0.5,
            "bsdf": _diffuse([0.7, 0.7, 0.72]),
        }
    return scene, snapshots


def apply_mitsuba_transforms(scene: dict[str, Any], mi) -> dict[str, Any]:
    converted = dict(scene)
    for key, value in list(converted.items()):
        if not isinstance(value, dict):
            continue
        value = dict(value)
        for meta_key in list(value):
            if meta_key.startswith("_zapdos_"):
                value.pop(meta_key, None)
        matrix = value.pop("to_world_matrix", None)
        if matrix is not None:
            value["to_world"] = mi.ScalarTransform4f(matrix)
        look_at = value.pop("to_world_look_at", None)
        if look_at is not None:
            value["to_world"] = mi.ScalarTransform4f.look_at(
                origin=look_at["origin"],
                target=look_at["target"],
                up=look_at["up"],
            )
        converted[key] = value
    return converted


def _add_meshes(scene: dict[str, Any], stage: Usd.Stage, mesh_dir: Path, bundle: "RenderBundle") -> int:
    xforms = UsdGeom.XformCache()
    body_paths = _body_render_paths(bundle)
    mesh_count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not prim.IsActive():
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() or []
        counts = mesh.GetFaceVertexCountsAttr().Get() or []
        indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        triangles = _triangulate(counts, indices)
        if not points or not triangles:
            continue
        world = xforms.GetLocalToWorldTransform(prim)
        vertices = [list(point) for point in points]
        anchor = _mesh_anchor(prim.GetPath().pathString, body_paths)
        matrix = _matrix_rows(world)
        body_name = None
        body_local_matrix = None
        if anchor is not None:
            body_name, body_path = anchor
            body_prim = stage.GetPrimAtPath(body_path)
            if body_prim.IsValid():
                body_world = np.asarray(_matrix_rows(xforms.GetLocalToWorldTransform(body_prim)), dtype=float)
                body_local_matrix = (np.linalg.inv(body_world) @ np.asarray(matrix, dtype=float)).tolist()
                matrix = (body_world @ np.asarray(body_local_matrix, dtype=float)).tolist()
        mesh_path = mesh_dir / f"mesh_{mesh_count}.ply"
        _write_ply(mesh_path, vertices, triangles)
        entry = {
            "type": "ply",
            "filename": str(mesh_path),
            "to_world_matrix": matrix,
            "bsdf": _diffuse([0.72, 0.72, 0.72]),
        }
        if body_name is not None and body_local_matrix is not None:
            entry["_zapdos_body"] = body_name
            entry["_zapdos_body_local_matrix"] = body_local_matrix
        scene[f"mesh_{mesh_count}"] = entry
        mesh_count += 1
    return mesh_count


def _camera_sensor(
    camera: "RenderCamera",
    width: int,
    height: int,
    spp: int,
    stage: Usd.Stage,
    xforms: UsdGeom.XformCache,
    body_paths: list[tuple[str, str]],
) -> dict[str, Any]:
    origin, target, up = _camera_vectors(camera, stage, xforms)
    sensor = {
        "type": "perspective",
        "fov": float(camera.fovy),
        "to_world_look_at": {"origin": origin, "target": target, "up": up},
        "sampler": {"type": "independent", "sample_count": int(spp)},
        "film": {
            "type": "hdrfilm",
            "width": int(width),
            "height": int(height),
            "rfilter": {"type": "box"},
        },
    }
    if camera.body is not None:
        body_path = next((path for path, body in body_paths if body == camera.body), None)
        body_prim = stage.GetPrimAtPath(body_path) if body_path is not None else None
        if body_prim is not None and body_prim.IsValid():
            body_world = np.asarray(_matrix_rows(xforms.GetLocalToWorldTransform(body_prim)), dtype=float)
            body_inv = np.linalg.inv(body_world)
            sensor["_zapdos_body"] = camera.body
            sensor["_zapdos_camera_local_origin"] = (body_inv @ np.asarray([*origin, 1.0], dtype=float)).tolist()
            sensor["_zapdos_camera_local_target"] = (body_inv @ np.asarray([*target, 1.0], dtype=float)).tolist()
            sensor["_zapdos_camera_local_up"] = (body_inv[:3, :3] @ np.asarray(up, dtype=float)).tolist()
    return sensor


def _camera_vectors(
    camera: "RenderCamera",
    stage: Usd.Stage,
    xforms: UsdGeom.XformCache,
) -> tuple[list[float], list[float], list[float]]:
    prim = _camera_prim(stage, camera.prim)
    if prim is not None:
        world = xforms.GetLocalToWorldTransform(prim)
        origin = np.asarray(world.ExtractTranslation(), dtype=float)
        forward = np.asarray(world.TransformDir((0.0, 0.0, -1.0)), dtype=float)
        up = np.asarray(world.TransformDir((0.0, 1.0, 0.0)), dtype=float)
        return origin.tolist(), (origin + forward).tolist(), up.tolist()
    rotation = _quat_to_matrix(camera.quat)
    origin = np.asarray(camera.pos, dtype=float)
    forward = rotation @ np.array([0.0, 0.0, -1.0])
    up = rotation @ np.array([0.0, 1.0, 0.0])
    return origin.tolist(), (origin + forward).tolist(), up.tolist()


def _camera_prim(stage: Usd.Stage, prim_path: str):
    candidates = [prim_path]
    if prim_path.startswith("/"):
        candidates.append(f"/RenderScene{prim_path}")
    for candidate in candidates:
        prim = stage.GetPrimAtPath(candidate)
        if prim.IsValid() and prim.IsA(UsdGeom.Camera):
            return prim
    return None


def _body_render_paths(bundle: "RenderBundle") -> list[tuple[str, str]]:
    body_map_path = getattr(bundle, "body_map_json", None)
    if body_map_path is None:
        return []
    try:
        body_map = json.loads(Path(body_map_path).read_text(encoding="utf-8"))
    except (OSError, RuntimeError, json.JSONDecodeError):
        return []
    paths: list[tuple[str, str]] = []
    for body, render_path in body_map.items():
        if not isinstance(body, str) or not isinstance(render_path, str):
            continue
        normalized = render_path if render_path.startswith("/") else f"/RenderScene/{render_path.lstrip('/')}"
        paths.append((normalized.rstrip("/"), body))
    return sorted(paths, key=lambda item: len(item[0]), reverse=True)


def _mesh_anchor(mesh_path: str, body_paths: list[tuple[str, str]]) -> tuple[str, str] | None:
    for render_path, body in body_paths:
        if mesh_path == render_path or mesh_path.startswith(f"{render_path}/"):
            return body, render_path
    return None


def _matrix_rows(matrix) -> list[list[float]]:
    return [[float(matrix[row][col]) for col in range(4)] for row in range(4)]


def _quat_to_matrix(quat: list[float]) -> np.ndarray:
    w, x, y, z = [float(value) for value in quat]
    norm = max((w * w + x * x + y * y + z * z) ** 0.5, 1e-12)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _triangulate(counts, indices) -> list[tuple[int, int, int]]:
    triangles: list[tuple[int, int, int]] = []
    offset = 0
    for raw_count in counts:
        count = int(raw_count)
        face = [int(value) for value in indices[offset: offset + count]]
        offset += count
        if len(face) < 3:
            continue
        for i in range(1, len(face) - 1):
            triangles.append((face[0], face[i], face[i + 1]))
    return triangles


def _write_ply(path: Path, vertices: list[list[float]], faces: list[tuple[int, int, int]]) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines.extend(f"{x:.9g} {y:.9g} {z:.9g}" for x, y, z in vertices)
    lines.extend(f"3 {a} {b} {c}" for a, b, c in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _diffuse(color: list[float]) -> dict[str, Any]:
    return {"type": "diffuse", "reflectance": {"type": "rgb", "value": color}}


def _camera_snapshot(camera: "RenderCamera") -> dict[str, Any]:
    return {
        "name": camera.name,
        "parent_prim": camera.prim.rsplit("/", 1)[0],
        "pos": list(camera.pos),
        "quat": list(camera.quat),
        "fovy": float(camera.fovy),
        "horizontal_aperture": float(camera.horizontal_aperture),
        "vertical_aperture": float(camera.vertical_aperture),
        "clipping_range": list(camera.clipping_range),
    }
