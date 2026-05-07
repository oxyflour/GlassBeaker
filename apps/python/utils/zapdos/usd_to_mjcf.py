#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A minimal USDA/USD -> MuJoCo MJCF converter.

Supported (minimal):
- Xform hierarchy
- Mesh -> OBJ asset + geom type="mesh"
- Cube / Sphere / Capsule / Cylinder
- Basic inertial attrs: physics:mass, physics:centerOfMass, physics:diagonalInertia
- Basic joints: Revolute / Prismatic / Fixed / Spherical
- USD stage metersPerUnit -> meters
- USD Y-up -> MuJoCo Z-up (wrapped with one extra root body)

Not fully supported:
- complex materials / shaders
- variants / payload / custom schemas
- non-uniform scale exact handling
- advanced collision filtering / drives / tendon / sensors
- exact joint frame semantics for every USD producer
"""

import argparse
import hashlib
import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

from pxr import Usd, UsdGeom, UsdShade
from utils.zapdos.usd_asset_cache import (
    file_digest,
    materialize_cached_file,
    mesh_cache_path,
    mesh_source_cache_path,
    texture_cache_path,
)


# ----------------------------
# Helpers
# ----------------------------

def fmt_f(x: float) -> str:
    return f"{float(x):.9g}"


def fmt_vec(v) -> str:
    return " ".join(fmt_f(x) for x in v)


def sanitize_name(path: str) -> str:
    s = path.strip("/").replace("/", "_")
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    if not s:
        s = "root"
    if s[0].isdigit():
        s = "_" + s
    return s


def gf_matrix_to_np(m) -> np.ndarray:
    arr = np.zeros((4, 4), dtype=float)
    for i in range(4):
        for j in range(4):
            arr[i, j] = float(m[i][j])
    # USD stores GfMatrix4d in row-major form. Transpose it so the rest of this
    # converter can use the usual column-vector convention with translation in
    # the last column.
    return arr.T


def np_identity() -> np.ndarray:
    return np.eye(4, dtype=float)


def orthonormalize_rotation(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    R2 = U @ Vt
    if np.linalg.det(R2) < 0:
        U[:, -1] *= -1
        R2 = U @ Vt
    return R2


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to MuJoCo quaternion order: w x y z
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q) + 1e-12
    return q


def matrix_to_pos_quat(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose rigid-ish transform.
    Note: scale/shear are not preserved exactly. Rotation is orthonormalized.
    """
    t = M[:3, 3].copy()
    R = orthonormalize_rotation(M[:3, :3])
    q = rotmat_to_quat_wxyz(R)
    return t, q


def quat_wxyz_from_axis_angle(axis, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    s = math.sin(angle_rad / 2.0)
    return np.array([math.cos(angle_rad / 2.0), axis[0] * s, axis[1] * s, axis[2] * s], dtype=float)


def gf_quat_to_wxyz(q) -> Optional[np.ndarray]:
    if q is None:
        return None

    real_getter = getattr(q, "GetReal", None)
    imag_getter = getattr(q, "GetImaginary", None)
    if callable(real_getter) and callable(imag_getter):
        imag = imag_getter()
        out = np.array([float(real_getter()), float(imag[0]), float(imag[1]), float(imag[2])], dtype=float) # type: ignore
    else:
        try:
            out = np.array([float(q[0]), float(q[1]), float(q[2]), float(q[3])], dtype=float)
        except Exception:
            return None

    n = np.linalg.norm(out)
    if n <= 1e-12:
        return None
    return out / n


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def pose_to_matrix(
    pos: Optional[np.ndarray] = None,
    quat: Optional[np.ndarray] = None,
) -> np.ndarray:
    M = np_identity()
    if quat is not None:
        M[:3, :3] = quat_wxyz_to_rotmat(quat)
    if pos is not None:
        M[:3, 3] = np.asarray(pos, dtype=float)
    return M


def canonicalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q[0] < 0.0:
        return -q
    if abs(q[0]) <= 1e-12:
        for v in q[1:]:
            if abs(v) <= 1e-12:
                continue
            if v < 0.0:
                return -q
            break
    return q


def lookat_to_quat_wxyz(pos: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    forward = np.asarray(target, dtype=float) - np.asarray(pos, dtype=float)
    forward_norm = np.linalg.norm(forward)
    if forward_norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    forward /= forward_norm

    up = np.asarray(up_hint, dtype=float)
    up_norm = np.linalg.norm(up)
    if up_norm <= 1e-12:
        up = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        up /= up_norm

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm <= 1e-12:
        fallback_up = np.array([0.0, 1.0, 0.0], dtype=float)
        if abs(float(np.dot(forward, fallback_up))) > 0.99:
            fallback_up = np.array([1.0, 0.0, 0.0], dtype=float)
        right = np.cross(forward, fallback_up)
        right_norm = np.linalg.norm(right)
    right /= right_norm + 1e-12

    camera_up = np.cross(right, forward)
    camera_up /= np.linalg.norm(camera_up) + 1e-12

    # MuJoCo and USD cameras both use a frame where -Z is the view direction.
    R = np.column_stack((right, camera_up, -forward))
    return rotmat_to_quat_wxyz(R)


def axis_token_to_vec3(token) -> np.ndarray:
    if token is None:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    if isinstance(token, str):
        t = token.upper()
        if t == "X":
            return np.array([1.0, 0.0, 0.0], dtype=float)
        if t == "Y":
            return np.array([0.0, 1.0, 0.0], dtype=float)
        if t == "Z":
            return np.array([0.0, 0.0, 1.0], dtype=float)
    # maybe already a vector-like value
    try:
        v = np.array(token, dtype=float).reshape(3)
        n = np.linalg.norm(v)
        if n > 1e-12:
            return v / n
    except Exception:
        pass
    return np.array([0.0, 0.0, 1.0], dtype=float)


def capsule_or_cylinder_axis_quat(axis_token) -> Optional[np.ndarray]:
    """
    MuJoCo cylinder/capsule default axis is Z.
    Return quaternion wxyz that rotates Z axis to requested axis.
    """
    if axis_token is None:
        return None
    if isinstance(axis_token, str):
        t = axis_token.upper()
        if t == "Z":
            return None
        if t == "X":
            # rotate +90 deg about Y: z -> x
            return quat_wxyz_from_axis_angle([0, 1, 0], math.pi / 2.0)
        if t == "Y":
            # rotate -90 deg about X: z -> y
            return quat_wxyz_from_axis_angle([1, 0, 0], -math.pi / 2.0)
    return None


def triangulate_faces(face_counts: List[int], face_indices: List[int]) -> np.ndarray:
    tris = []
    cursor = 0
    for n in face_counts:
        poly = face_indices[cursor:cursor + n]
        cursor += n
        if n < 3:
            continue
        for i in range(1, n - 1):
            tris.append([poly[0], poly[i], poly[i + 1]])
    if len(tris) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(tris, dtype=np.int64)


def fallback_diaginertia(mass: float, pos: Optional[np.ndarray]) -> np.ndarray:
    # When USD provides mass but no inertia tensor, synthesize a small stable
    # isotropic inertia so MuJoCo can compile the articulated body.
    extent = 0.01
    if pos is not None:
        extent = max(extent, 2.0 * float(np.linalg.norm(pos)))
    inertia = mass * extent * extent / 6.0
    return np.full(3, max(inertia, 1e-8), dtype=float)


# ----------------------------
# Data classes
# ----------------------------

@dataclass
class JointData:
    name: str
    kind: str                      # hinge | slide | ball
    axis: Optional[np.ndarray] = None
    range: Optional[Tuple[float, float]] = None
    pos: Optional[np.ndarray] = None
    damping: Optional[float] = None
    stiffness: Optional[float] = None
    springref: Optional[float] = None
    actuatorfrcrange: Optional[Tuple[float, float]] = None


@dataclass
class InertialData:
    mass: float
    pos: Optional[np.ndarray] = None
    diaginertia: Optional[np.ndarray] = None
    quat: Optional[np.ndarray] = None


@dataclass
class GeomData:
    name: str
    kind: str                      # mesh | box | sphere | capsule | cylinder
    rgba: Optional[np.ndarray] = None
    material_name: Optional[str] = None
    size: Optional[np.ndarray] = None
    mesh_name: Optional[str] = None
    file_rel: Optional[str] = None
    quat: Optional[np.ndarray] = None
    contype: Optional[int] = None
    conaffinity: Optional[int] = None


@dataclass
class TextureAssetData:
    name: str
    file_rel: str


@dataclass
class MeshAssetData:
    name: str
    file_rel: str


@dataclass
class MaterialAssetData:
    name: str
    texture_name: str
    rgba: Optional[np.ndarray] = None


@dataclass
class CameraData:
    name: str
    pos: np.ndarray
    quat: np.ndarray
    fovy: Optional[float] = None


@dataclass
class BodyNode:
    path: str
    name: str
    prim_type: str
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)

    world_matrix: Optional[np.ndarray] = None
    local_pos: Optional[np.ndarray] = None
    local_quat: Optional[np.ndarray] = None

    geoms: List[GeomData] = field(default_factory=list)
    cameras: List[CameraData] = field(default_factory=list)
    inertial: Optional[InertialData] = None
    joint: Optional[JointData] = None
    kinematic: bool = False


# ----------------------------
# Converter
# ----------------------------

class USDToMJCFConverter:
    MAX_INFERRED_SLIDE_POSITION_KP = 1e4
    WHEEL_NAME_HINTS = ("wheel", "tire", "caster", "rim")
    SENSOR_CAMERA_SPECS = {
        "zed_link": {
            "camera_name": "head_camera",
            "quat": quat_wxyz_from_axis_angle([0, 1, 0], math.pi),
            "fovy": 45.0,
        },
        "left_realsense_link": {
            "camera_name": "left_wrist_camera",
            "quat": quat_wxyz_from_axis_angle([0, 1, 0], math.pi / 2.0),
            "fovy": 45.0,
        },
        "right_realsense_link": {
            "camera_name": "right_wrist_camera",
            "quat": quat_wxyz_from_axis_angle([0, 1, 0], math.pi / 2.0),
            "fovy": 45.0,
        },
    }
    SKIP_BODY_TYPES = {
        "Scope",
        "Material",
        "Shader",
        "NodeGraph",
        "Camera",
        "DomeLight",
        "RectLight",
        "SphereLight",
        "DiskLight",
        "DistantLight",
    }

    def __init__(
        self,
        usd_path: Path,
        output_xml: Path,
        model_name: str = "converted_from_usd",
        force_body_paths: set[str] | None = None,
        stage=None,
    ):
        self.usd_path = Path(usd_path)
        self.output_xml = Path(output_xml)
        self.output_dir = self.output_xml.parent.resolve()
        self.mesh_dir = self.output_dir / "meshes"
        self.texture_dir = self.output_dir / "textures"
        self.mesh_dir.mkdir(parents=True, exist_ok=True)
        self.texture_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.stage = stage if stage is not None else Usd.Stage.Open(str(self.usd_path)) # type: ignore
        if self.stage is None:
            raise RuntimeError(f"Failed to open USD stage: {self.usd_path}")

        self.xform_cache = UsdGeom.XformCache() # type: ignore

        # stage unit conversion
        if self.stage.HasAuthoredMetadata("metersPerUnit"):
            mpu = self.stage.GetMetadata("metersPerUnit")
            self.meters_per_unit = float(mpu) if mpu is not None else 1.0
        else:
            # Some Blender-exported USDA files omit this metadata while already
            # authoring transforms and vertices in meters.
            self.meters_per_unit = 1.0

        up = UsdGeom.GetStageUpAxis(self.stage) # type: ignore
        self.stage_up_axis = str(up) if up is not None else "Z"
        self.force_body_paths = set(force_body_paths or ())

        self.nodes: Dict[str, BodyNode] = {}
        self.world_cameras: List[CameraData] = []
        self.mesh_assets: Dict[str, MeshAssetData] = {}
        self.mesh_source_signatures: Dict[str, str] = {}
        self.texture_assets: Dict[str, TextureAssetData] = {}
        self.material_assets: Dict[str, MaterialAssetData] = {}
        self.contact_excludes: set[Tuple[str, str]] = set()

    # ------------------------
    # USD low-level read utils
    # ------------------------

    def get_attr(self, prim, *names, default=None):
        for name in names:
            attr = prim.GetAttribute(name)
            if attr and attr.IsValid():
                val = attr.Get()
                if val is not None:
                    return val
        return default

    def get_authored_attr(self, prim, *names, default=None):
        for name in names:
            attr = prim.GetAttribute(name)
            if not attr or not attr.IsValid():
                continue
            if not attr.HasAuthoredValueOpinion():
                continue
            val = attr.Get()
            if val is not None:
                return val
        return default

    def physics_attr_prims(self, prim):
        yield prim
        entity = prim.GetChild("entity")
        if entity and entity.IsValid():
            yield entity

    def get_physics_attr(self, prim, *names, default=None):
        for physics_prim in self.physics_attr_prims(prim):
            value = self.get_attr(physics_prim, *names, default=None)
            if value is not None:
                return value
        return default

    def get_authored_physics_attr(self, prim, *names, default=None):
        for physics_prim in self.physics_attr_prims(prim):
            value = self.get_authored_attr(physics_prim, *names, default=None)
            if value is not None:
                return value
        return default

    def get_rel_target_path(self, prim, *names) -> Optional[str]:
        for name in names:
            rel = prim.GetRelationship(name)
            if rel and rel.IsValid():
                targets = rel.GetTargets()
                if targets:
                    return str(targets[0])
        return None

    def prim_is_body_candidate(self, prim) -> bool:
        if not prim or not prim.IsValid():
            return False
        t = prim.GetTypeName()
        if t in self.SKIP_BODY_TYPES:
            return False
        # Only keep xformable scene prims as body candidates
        return prim.IsA(UsdGeom.Xformable) # type: ignore

    def get_nearest_body_ancestor_path(self, prim) -> Optional[str]:
        p = prim.GetParent()
        while p and p.IsValid():
            if self.prim_is_body_candidate(p):
                return str(p.GetPath())
            p = p.GetParent()
        return None

    def path_looks_wheel_like(self, *paths: Optional[str]) -> bool:
        for path in paths:
            if path is None:
                continue
            lower = path.lower()
            if any(token in lower for token in self.WHEEL_NAME_HINTS):
                return True
        return False

    def add_contact_exclude(self, body_a: str, body_b: str):
        if body_a == body_b or body_a not in self.nodes or body_b not in self.nodes:
            return
        if not self.should_emit_body(self.nodes[body_a]) or not self.should_emit_body(self.nodes[body_b]):
            return
        a = self.nodes[body_a].name
        b = self.nodes[body_b].name
        self.contact_excludes.add(tuple(sorted((a, b))))

    def exclude_contacts_with_ancestors(self, child_path: str):
        current = self.nodes[child_path].parent
        while current and current in self.nodes:
            self.add_contact_exclude(child_path, current)
            current = self.nodes[current].parent

    def exclude_gripper_finger_siblings(self):
        for node in self.nodes.values():
            finger_children = [
                child_path
                for child_path in node.children
                if "gripper_finger_link" in child_path.lower()
            ]
            if len(finger_children) < 2:
                continue
            for index, body_a in enumerate(finger_children):
                for body_b in finger_children[index + 1:]:
                    self.add_contact_exclude(body_a, body_b)

    def get_world_matrix(self, prim) -> np.ndarray:
        return gf_matrix_to_np(self.xform_cache.GetLocalToWorldTransform(prim))

    def rgba_from_rgb(self, rgb, alpha: float = 1.0) -> np.ndarray:
        return np.array([float(rgb[0]), float(rgb[1]), float(rgb[2]), float(alpha)], dtype=float)

    def get_bound_material_path(self, prim) -> Optional[str]:
        current = prim
        while current and current.IsValid():
            material_path = self.get_rel_target_path(current, "material:binding")
            if material_path is not None:
                return material_path
            current = current.GetParent()
        return None

    def get_surface_shader(self, material_prim):
        material = UsdShade.Material(material_prim) # type: ignore
        for purpose in (None, "mdl"):
            output = material.GetSurfaceOutput() if purpose is None else material.GetSurfaceOutput(purpose)
            if not output:
                continue
            shader_sources, _ = output.GetConnectedSources()
            if shader_sources:
                return UsdShade.Shader(shader_sources[0].source.GetPrim()) # type: ignore
        return None

    def get_shader_opacity(self, shader) -> float:
        for input_name in ("opacity", "opacity_constant"):
            opacity_input = shader.GetInput(input_name)
            if not opacity_input:
                continue
            opacity_sources, _ = opacity_input.GetConnectedSources()
            if opacity_sources:
                continue
            opacity_value = opacity_input.Get()
            if opacity_value is not None:
                return float(opacity_value)
        return 1.0

    def get_shader_constant_rgb(self, shader, *input_names) -> Optional[np.ndarray]:
        for input_name in input_names:
            color_input = shader.GetInput(input_name)
            if not color_input:
                continue
            color_sources, _ = color_input.GetConnectedSources()
            if color_sources:
                continue
            color_value = color_input.Get()
            if color_value is not None:
                return np.array(
                    [float(color_value[0]), float(color_value[1]), float(color_value[2])],
                    dtype=float,
                )
        return None

    def get_shader_diffuse_texture_path(self, shader) -> Optional[Path]:
        for input_name in ("diffuse_texture", "base_color_texture", "albedo_texture"):
            texture_input = shader.GetInput(input_name)
            if not texture_input:
                continue
            texture_path = self.resolve_asset_path(texture_input.Get())
            if texture_path is not None and texture_path.exists():
                return texture_path

        for input_name in ("diffuseColor", "baseColor"):
            diffuse_input = shader.GetInput(input_name)
            if not diffuse_input:
                continue
            diffuse_sources, _ = diffuse_input.GetConnectedSources()
            for source in diffuse_sources:
                texture_shader = UsdShade.Shader(source.source.GetPrim()) # type: ignore
                texture_file_input = texture_shader.GetInput("file")
                if not texture_file_input:
                    continue
                texture_path = self.resolve_asset_path(texture_file_input.Get())
                if texture_path is not None and texture_path.exists():
                    return texture_path
        return None

    def get_material_diffuse_rgba(self, prim) -> Optional[np.ndarray]:
        material_path = self.get_bound_material_path(prim)
        if material_path is None:
            return None

        material_prim = self.stage.GetPrimAtPath(material_path)
        if not material_prim or not material_prim.IsValid():
            return None

        shader = self.get_surface_shader(material_prim)
        if not shader:
            return None

        diffuse = self.get_shader_constant_rgb(
            shader,
            "diffuseColor",
            "baseColor",
            "diffuse_color_constant",
            "base_color_constant",
        )
        if diffuse is not None:
            return self.rgba_from_rgb(diffuse, self.get_shader_opacity(shader))
        return None

    def resolve_asset_path(self, asset_path) -> Optional[Path]:
        if asset_path is None:
            return None
        resolved = getattr(asset_path, "resolvedPath", "")
        if resolved:
            return Path(resolved)
        raw_path = getattr(asset_path, "path", "")
        if not raw_path:
            return None
        return (self.usd_path.parent / raw_path).resolve()

    def unique_asset_name(self, base_name: str, used_names: set[str]) -> str:
        if base_name not in used_names:
            return base_name
        suffix = 2
        while True:
            candidate = sanitize_name(f"{base_name}_{suffix}")
            if candidate not in used_names:
                return candidate
            suffix += 1

    def export_texture_asset(self, texture_path: Path) -> Tuple[str, str]:
        texture_path = texture_path.resolve()
        cache_key = str(texture_path)
        if cache_key in self.texture_assets:
            asset = self.texture_assets[cache_key]
            return asset.name, asset.file_rel

        try:
            rel_path = texture_path.relative_to(self.usd_path.parent.resolve())
            name_source = str(rel_path.with_suffix(""))
        except ValueError:
            name_source = texture_path.stem

        texture_name = self.unique_asset_name(
            sanitize_name(name_source),
            {asset.name for asset in self.texture_assets.values()},
        )
        out_file = self.texture_dir / f"{texture_name}.png"
        cache_file = texture_cache_path(file_digest(texture_path))
        if not cache_file.exists():
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            if texture_path.suffix.lower() == ".png":
                shutil.copy2(texture_path, cache_file)
            else:
                Image.open(texture_path).convert("RGB").save(cache_file, format="PNG")
        materialize_cached_file(cache_file, out_file)

        file_rel = (Path("textures") / out_file.name).as_posix()
        self.texture_assets[cache_key] = TextureAssetData(name=texture_name, file_rel=file_rel)
        return texture_name, file_rel

    def register_textured_material(self, prim) -> Optional[str]:
        material_path = self.get_bound_material_path(prim)
        if material_path is None:
            return None
        if material_path in self.material_assets:
            return self.material_assets[material_path].name

        material_prim = self.stage.GetPrimAtPath(material_path)
        if not material_prim or not material_prim.IsValid():
            return None

        shader = self.get_surface_shader(material_prim)
        if not shader:
            return None

        texture_path = self.get_shader_diffuse_texture_path(shader)
        if texture_path is None or not texture_path.exists():
            return None

        texture_name, _ = self.export_texture_asset(texture_path)
        material_name = sanitize_name(material_path)
        self.material_assets[material_path] = MaterialAssetData(
            name=material_name,
            texture_name=texture_name,
            rgba=np.array([1.0, 1.0, 1.0, self.get_shader_opacity(shader)], dtype=float),
        )
        return material_name

    def get_display_rgba(self, prim) -> Optional[np.ndarray]:
        try:
            gprim = UsdGeom.Gprim(prim) # type: ignore
            attr = gprim.GetDisplayColorAttr()
            cols = attr.Get()
            if cols and len(cols) > 0:
                return self.rgba_from_rgb(cols[0])
        except Exception:
            pass
        return None

    def get_prim_rgba(self, prim) -> np.ndarray:
        rgba = self.get_material_diffuse_rgba(prim)
        if rgba is not None:
            return rgba
        rgba = self.get_display_rgba(prim)
        if rgba is not None:
            return rgba
        return np.array([0.7, 0.7, 0.7, 1.0], dtype=float)

    def get_collision_enabled(self, prim) -> Tuple[Optional[int], Optional[int]]:
        enabled = self.get_attr(prim, "physics:collisionEnabled", default=None)
        if enabled is not None:
            if bool(enabled):
                return None, None
            return 0, 0

        path_tokens = {tok.lower() for tok in str(prim.GetPath()).strip("/").split("/")}
        if "visuals" in path_tokens or "visual" in path_tokens:
            return 0, 0
        return None, None

    def prim_visibility(self, prim) -> Optional[str]:
        if not prim.IsA(UsdGeom.Imageable): # type: ignore
            return None
        visibility = UsdGeom.Imageable(prim).ComputeVisibility() # type: ignore
        return str(visibility) if visibility is not None else None

    def primitive_world_scale(self, prim) -> np.ndarray:
        if not prim.IsA(UsdGeom.Xformable): # type: ignore
            return np.ones(3, dtype=float)
        world_matrix = self.get_world_matrix(prim)
        scale = np.linalg.norm(world_matrix[:3, :3], axis=0)
        scale[scale <= 1e-12] = 1.0
        return scale

    def primitive_axis_scales(self, prim, axis_token) -> Tuple[float, float]:
        scale = self.primitive_world_scale(prim)
        axis_name = str(axis_token).upper()
        axis_index = {"X": 0, "Y": 1, "Z": 2}.get(axis_name, 2)
        radial_indices = [idx for idx in range(3) if idx != axis_index]
        radial_scale = max(float(scale[radial_indices[0]]), float(scale[radial_indices[1]]))
        axial_scale = float(scale[axis_index])
        return radial_scale, axial_scale

    def get_stage_up_vector(self) -> np.ndarray:
        if self.stage_up_axis.upper() == "Y":
            return np.array([0.0, 1.0, 0.0], dtype=float)
        return np.array([0.0, 0.0, 1.0], dtype=float)

    def camera_fovy_from_prim(self, prim) -> Optional[float]:
        projection = str(self.get_attr(prim, "projection", default="perspective")).lower()
        if projection != "perspective":
            return None

        focal_length = self.get_attr(prim, "focalLength", default=None)
        vertical_aperture = self.get_attr(prim, "verticalAperture", default=None)
        if focal_length is None or vertical_aperture is None:
            return None

        focal = float(focal_length)
        aperture = float(vertical_aperture)
        if focal <= 1e-12 or aperture <= 1e-12:
            return None

        fovy = math.degrees(2.0 * math.atan(aperture / (2.0 * focal)))
        return min(max(fovy, 1e-3), 179.0)

    def camera_pose_key(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        fovy: Optional[float],
    ) -> Tuple[float, ...]:
        quat = canonicalize_quat_wxyz(quat)
        fovy_value = -1.0 if fovy is None else float(fovy)
        values = np.concatenate((np.asarray(pos, dtype=float), quat, np.array([fovy_value], dtype=float)))
        return tuple(round(float(v), 6) for v in values)

    def should_export_camera_prim(self, prim) -> bool:
        path_tokens = {tok.lower() for tok in str(prim.GetPath()).strip("/").split("/")}
        if "collision" in path_tokens or "collisions" in path_tokens:
            return False
        return True

    # ------------------------
    # Geometry export
    # ------------------------

    def build_geom_for_prim(self, prim) -> Optional[GeomData]:
        t = prim.GetTypeName()
        name = sanitize_name(str(prim.GetPath())) + "_geom"
        contype, conaffinity = self.get_collision_enabled(prim)
        if self.prim_visibility(prim) == "invisible":
            return None

        if prim.IsA(UsdGeom.Mesh): # type: ignore
            material_name = self.register_textured_material(prim)
            rgba = None if material_name else self.get_prim_rgba(prim)
            mesh_name, file_rel = self.export_mesh_prim(prim)
            if mesh_name is None:
                return None
            return GeomData(
                name=name,
                kind="mesh",
                material_name=material_name,
                mesh_name=mesh_name,
                file_rel=file_rel,
                rgba=rgba,
                contype=contype,
                conaffinity=conaffinity,
            )

        rgba = self.get_prim_rgba(prim)
        local_scale = self.primitive_world_scale(prim)

        if t == "Cube":
            size = self.get_attr(prim, "size", default=2.0)
            half = 0.5 * float(size) * self.meters_per_unit # type: ignore
            return GeomData(
                name=name,
                kind="box",
                size=np.array([half, half, half], dtype=float) * local_scale,
                rgba=rgba,
                contype=contype,
                conaffinity=conaffinity,
            )

        if t == "Sphere":
            radius = self.get_attr(prim, "radius", default=1.0)
            r = float(radius) * self.meters_per_unit * float(np.max(local_scale)) # type: ignore
            return GeomData(
                name=name,
                kind="sphere",
                size=np.array([r], dtype=float),
                rgba=rgba,
                contype=contype,
                conaffinity=conaffinity,
            )

        if t == "Capsule":
            radius = self.get_attr(prim, "radius", default=1.0)
            height = self.get_attr(prim, "height", default=2.0)
            axis = self.get_attr(prim, "axis", default="Z")
            # MuJoCo capsule size = radius half_length_of_cylindrical_part
            radial_scale, axial_scale = self.primitive_axis_scales(prim, axis)
            r = float(radius) * self.meters_per_unit * radial_scale # type: ignore
            half_len = 0.5 * float(height) * self.meters_per_unit * axial_scale # type: ignore
            q = capsule_or_cylinder_axis_quat(axis)
            return GeomData(
                name=name,
                kind="capsule",
                size=np.array([r, half_len], dtype=float),
                rgba=rgba,
                quat=q,
                contype=contype,
                conaffinity=conaffinity,
            )

        if t == "Cylinder":
            radius = self.get_attr(prim, "radius", default=1.0)
            height = self.get_attr(prim, "height", default=2.0)
            axis = self.get_attr(prim, "axis", default="Z")
            radial_scale, axial_scale = self.primitive_axis_scales(prim, axis)
            r = float(radius) * self.meters_per_unit * radial_scale              # type: ignore
            half_len = 0.5 * float(height) * self.meters_per_unit * axial_scale # type: ignore
            q = capsule_or_cylinder_axis_quat(axis)
            return GeomData(
                name=name,
                kind="cylinder",
                size=np.array([r, half_len], dtype=float),
                rgba=rgba,
                quat=q,
                contype=contype,
                conaffinity=conaffinity,
            )

        return None

    def get_mesh_texcoords(
        self,
        mesh,
        vertex_count: int,
        face_counts: List[int],
        face_indices: List[int],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        primvar = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st") # type: ignore
        if not primvar:
            return None, None

        values = primvar.Get()
        if values is None or len(values) == 0:
            return None, None

        texcoords = np.array([[float(v[0]), float(v[1])] for v in values], dtype=float)
        indices = primvar.GetIndices()
        interpolation = primvar.GetInterpolation()

        if interpolation == UsdGeom.Tokens.faceVarying: # type: ignore
            corner_indices = list(indices) if len(indices) > 0 else list(range(len(texcoords)))
        elif interpolation in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying): # type: ignore
            vertex_texcoord_indices = list(indices) if len(indices) > 0 else list(range(len(texcoords)))
            if len(vertex_texcoord_indices) != vertex_count:
                return None, None
            corner_indices = [vertex_texcoord_indices[i] for i in face_indices]
        else:
            return None, None

        if len(corner_indices) != len(face_indices):
            return None, None

        face_texcoords = triangulate_faces(face_counts, corner_indices)
        if len(face_texcoords) == 0:
            return None, None
        return texcoords, face_texcoords

    def write_obj_mesh(
        self,
        out_file: Path,
        vertices: np.ndarray,
        faces: np.ndarray,
        texcoords: Optional[np.ndarray] = None,
        face_texcoords: Optional[np.ndarray] = None,
    ) -> None:
        has_texcoords = (
            texcoords is not None
            and face_texcoords is not None
            and len(texcoords) > 0
            and len(face_texcoords) == len(faces)
        )
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("w", encoding="utf-8", newline="\n") as f:
            f.write("# Generated by usd_to_mjcf.py\n")
            for vertex in vertices:
                f.write(f"v {fmt_vec(vertex)}\n")
            if has_texcoords and texcoords is not None:
                for uv in texcoords:
                    f.write(f"vt {fmt_f(uv[0])} {fmt_f(uv[1])}\n")
            for i, face in enumerate(faces):
                if has_texcoords and face_texcoords is not None:
                    uv_face = face_texcoords[i]
                    refs = [f"{int(v) + 1}/{int(vt) + 1}" for v, vt in zip(face, uv_face)]
                else:
                    refs = [str(int(v) + 1) for v in face]
                f.write(f"f {' '.join(refs)}\n")

    def update_signature(self, digest, values, dtype) -> None:
        array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())

    def mesh_signature(
        self,
        vertices: np.ndarray,
        face_counts: List[int],
        face_indices: List[int],
        texcoords: Optional[np.ndarray],
        face_texcoords: Optional[np.ndarray],
    ) -> str:
        digest = hashlib.sha1()
        self.update_signature(digest, vertices, np.float64)
        self.update_signature(digest, face_counts, np.int64)
        self.update_signature(digest, face_indices, np.int64)
        if texcoords is None:
            digest.update(b"no_texcoords")
        else:
            self.update_signature(digest, texcoords, np.float64)
        if face_texcoords is None:
            digest.update(b"no_face_texcoords")
        else:
            self.update_signature(digest, face_texcoords, np.int64)
        return digest.hexdigest()

    def export_mesh_prim(self, prim) -> Tuple[Optional[str], Optional[str]]:
        mesh_name = sanitize_name(str(prim.GetPath()))
        out_file = self.mesh_dir / f"{mesh_name}.obj"
        file_rel = (Path("meshes") / f"{mesh_name}.obj").as_posix()
        source_cache_key = self.mesh_source_cache_key(prim)
        fast_signature = self.lookup_mesh_source_signature(source_cache_key)
        if fast_signature is not None:
            cached_asset = self.mesh_assets.get(fast_signature)
            if cached_asset is not None:
                return cached_asset.name, cached_asset.file_rel

            cache_file = mesh_cache_path(fast_signature)
            if cache_file.exists():
                materialize_cached_file(cache_file, out_file)
                asset = MeshAssetData(name=mesh_name, file_rel=file_rel)
                self.mesh_assets[fast_signature] = asset
                return asset.name, asset.file_rel

        mesh = UsdGeom.Mesh(prim) # type: ignore
        pts = mesh.GetPointsAttr().Get()
        fvc = mesh.GetFaceVertexCountsAttr().Get()
        fvi = mesh.GetFaceVertexIndicesAttr().Get()

        if pts is None or fvc is None or fvi is None:
            return None, None

        vertices = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in pts], dtype=float)
        vertices *= self.meters_per_unit

        face_counts = list(fvc)
        face_indices = list(fvi)
        faces = triangulate_faces(face_counts, face_indices)
        if len(vertices) == 0 or len(faces) == 0:
            return None, None

        texcoords, face_texcoords = self.get_mesh_texcoords(mesh, len(vertices), face_counts, face_indices)
        signature = self.mesh_signature(vertices, face_counts, face_indices, texcoords, face_texcoords)
        cached_asset = self.mesh_assets.get(signature)
        if cached_asset is not None:
            self.store_mesh_source_signature(source_cache_key, signature)
            return cached_asset.name, cached_asset.file_rel

        cache_file = mesh_cache_path(signature)
        if not cache_file.exists():
            self.write_obj_mesh(cache_file, vertices, faces, texcoords, face_texcoords)
        materialize_cached_file(cache_file, out_file)

        asset = MeshAssetData(name=mesh_name, file_rel=file_rel)
        self.mesh_assets[signature] = asset
        self.store_mesh_source_signature(source_cache_key, signature)
        return asset.name, asset.file_rel

    def mesh_source_cache_key(self, prim) -> Optional[str]:
        source = self.prim_source_spec(prim)
        if source is None:
            return None

        source_layer, source_prim_path = source
        try:
            stat = source_layer.stat()
        except OSError:
            return None

        digest = hashlib.sha1()
        digest.update(str(source_layer).encode("utf-8"))
        digest.update(source_prim_path.encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(fmt_f(self.meters_per_unit).encode("ascii"))
        return digest.hexdigest()

    def prim_source_spec(self, prim) -> Optional[Tuple[Path, str]]:
        try:
            prim_stack = prim.GetPrimStack()
        except Exception:
            return None

        for spec in prim_stack:
            layer = getattr(spec, "layer", None)
            if layer is None:
                continue
            layer_path = getattr(layer, "realPath", "") or getattr(layer, "identifier", "")
            if not layer_path:
                continue
            return Path(layer_path).resolve(), str(spec.path)
        return None

    def lookup_mesh_source_signature(self, source_cache_key: Optional[str]) -> Optional[str]:
        if source_cache_key is None:
            return None
        if source_cache_key in self.mesh_source_signatures:
            return self.mesh_source_signatures[source_cache_key]

        cache_file = mesh_source_cache_path(source_cache_key)
        if not cache_file.exists():
            return None

        try:
            signature = cache_file.read_text(encoding="utf-8").strip()
        except OSError:
            return None

        if len(signature) != 40 or any(ch not in "0123456789abcdef" for ch in signature):
            return None
        self.mesh_source_signatures[source_cache_key] = signature
        return signature

    def store_mesh_source_signature(self, source_cache_key: Optional[str], signature: str) -> None:
        if source_cache_key is None:
            return
        self.mesh_source_signatures[source_cache_key] = signature
        cache_file = mesh_source_cache_path(source_cache_key)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(signature, encoding="utf-8")

    def build_camera_for_prim(self, prim, parent_path: Optional[str]) -> Optional[CameraData]:
        if not self.should_export_camera_prim(prim):
            return None

        world_matrix = self.get_world_matrix(prim)
        parent_world = np_identity()
        if parent_path and parent_path in self.nodes and self.nodes[parent_path].world_matrix is not None:
            parent_world = self.nodes[parent_path].world_matrix

        local_matrix = np.linalg.inv(parent_world) @ world_matrix # type: ignore
        local_pos, local_quat = matrix_to_pos_quat(local_matrix)
        local_pos *= self.meters_per_unit

        return CameraData(
            name=sanitize_name(str(prim.GetPath())),
            pos=local_pos,
            quat=local_quat,
            fovy=self.camera_fovy_from_prim(prim),
        )

    def build_stage_metadata_camera(self) -> Optional[CameraData]:
        custom_layer_data = self.stage.GetMetadata("customLayerData")
        if not isinstance(custom_layer_data, dict):
            return None

        camera_settings = custom_layer_data.get("cameraSettings")
        if not isinstance(camera_settings, dict):
            return None

        perspective = camera_settings.get("Perspective")
        if not isinstance(perspective, dict):
            return None

        position = perspective.get("position")
        target = perspective.get("target")
        if position is None or target is None:
            return None

        pos = np.array([float(position[0]), float(position[1]), float(position[2])], dtype=float)
        tgt = np.array([float(target[0]), float(target[1]), float(target[2])], dtype=float)
        pos *= self.meters_per_unit
        tgt *= self.meters_per_unit

        name_source = camera_settings.get("boundCamera", "/usd_perspective")
        return CameraData(
            name=sanitize_name(str(name_source)),
            pos=pos,
            quat=lookat_to_quat_wxyz(pos, tgt, self.get_stage_up_vector()),
            fovy=45.0,
        )

    def build_sensor_link_cameras(self) -> bool:
        sensor_nodes: Dict[str, BodyNode] = {}
        found_sensor_camera = False
        for path, node in self.nodes.items():
            if node.inertial is None:
                continue

            link_name = path.rsplit("/", maxsplit=1)[-1]
            if link_name in self.SENSOR_CAMERA_SPECS:
                sensor_nodes[link_name] = node

        for link_name, spec in self.SENSOR_CAMERA_SPECS.items():
            node = sensor_nodes.get(link_name)
            if node is None:
                continue

            node.cameras.append(CameraData(
                name=spec["camera_name"],
                pos=np.zeros(3, dtype=float),
                quat=np.array(spec["quat"], dtype=float),
                fovy=float(spec["fovy"]),
            ))
            found_sensor_camera = True
        return found_sensor_camera

    # ------------------------
    # Inertial and joints
    # ------------------------

    def build_inertial_for_prim(self, prim) -> Optional[InertialData]:
        mass = self.get_authored_physics_attr(prim, "physics:mass", default=None)
        if mass is None:
            return None
        mass = float(mass)
        if mass <= 0.0:
            return None

        com = self.get_authored_physics_attr(prim, "physics:centerOfMass", default=None)
        pos = np.zeros(3, dtype=float)
        if com is not None:
            com_vec = np.array([float(com[0]), float(com[1]), float(com[2])], dtype=float)
            if np.all(np.isfinite(com_vec)):
                pos = com_vec * self.meters_per_unit

        diagI = self.get_authored_physics_attr(prim, "physics:diagonalInertia", default=None)
        diaginertia = None
        if diagI is not None:
            diaginertia = np.array([float(diagI[0]), float(diagI[1]), float(diagI[2])], dtype=float)
            if not np.all(np.isfinite(diaginertia)) or np.any(diaginertia <= 0.0):
                diaginertia = None

        quat = gf_quat_to_wxyz(self.get_authored_physics_attr(prim, "physics:principalAxes", default=None))
        if diaginertia is None:
            diaginertia = fallback_diaginertia(mass, pos)
            quat = None

        return InertialData(mass=mass, pos=pos, diaginertia=diaginertia, quat=quat)

    def prim_is_kinematic(self, prim) -> bool:
        value = self.get_physics_attr(prim, "physics:kinematicEnabled", default=None)
        return bool(value) if value is not None else False

    def joint_kind_from_type(self, type_name: str) -> Optional[str]:
        tn = type_name.lower()
        if "revolute" in tn or "hinge" in tn:
            return "hinge"
        if "prismatic" in tn:
            return "slide"
        if "spherical" in tn or "ball" in tn:
            return "ball"
        if "fixed" in tn:
            return "fixed"
        return None

    def joint_value_to_mjcf(self, kind: str, value: float) -> float:
        v = float(value)
        if kind == "hinge":
            return math.radians(v)
        if kind == "slide":
            return v * self.meters_per_unit
        return v

    def joint_leaf_name_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for prim in self.stage.Traverse():
            t = prim.GetTypeName()
            if "Joint" not in t:
                continue
            kind = self.joint_kind_from_type(t)
            if kind in (None, "fixed"):
                continue
            body1 = self.get_rel_target_path(prim, "physics:body1", "body1")
            if not body1 or body1 not in self.nodes:
                continue
            leaf_name = sanitize_name(str(prim.GetPath()).split("/")[-1])
            counts[leaf_name] = counts.get(leaf_name, 0) + 1
        return counts

    def unique_joint_name(
        self,
        prim_path: str,
        body1: str,
        leaf_name_counts: Dict[str, int],
        used_names: set[str],
    ) -> str:
        leaf_name = sanitize_name(prim_path.split("/")[-1])
        context_name = sanitize_name(body1)
        if leaf_name_counts.get(leaf_name, 0) > 1:
            candidates = [
                sanitize_name(f"{leaf_name}_{context_name}"),
                sanitize_name(prim_path),
                leaf_name,
            ]
        else:
            candidates = [
                leaf_name,
                sanitize_name(f"{leaf_name}_{context_name}"),
                sanitize_name(prim_path),
            ]

        for candidate in candidates:
            if candidate not in used_names:
                return candidate

        suffix = 2
        while True:
            candidate = sanitize_name(f"{candidates[-1]}_{suffix}")
            if candidate not in used_names:
                return candidate
            suffix += 1

    def parse_and_apply_joints(self):
        """
        Parse USD joint prims and assign them to body1.
        If body0/body1 relation exists, reparent body1 under body0.
        """
        leaf_name_counts = self.joint_leaf_name_counts()
        used_names: set[str] = set()
        for prim in self.stage.Traverse():
            t = prim.GetTypeName()
            if "Joint" not in t:
                continue

            body0 = self.get_rel_target_path(prim, "physics:body0", "body0")
            body1 = self.get_rel_target_path(prim, "physics:body1", "body1")

            if not body1 or body1 not in self.nodes:
                continue

            kind = self.joint_kind_from_type(t)
            if kind is None:
                continue

            child = self.nodes[body1]
            if body0 and body0 in self.nodes and body0 != child.parent:
                child.parent = body0
            if body0 and body0 in self.nodes:
                self.add_contact_exclude(body0, body1)
            if self.path_looks_wheel_like(str(prim.GetPath()), body0, body1):
                self.exclude_contacts_with_ancestors(body1)

            if kind == "fixed":
                # fixed joint in MuJoCo means "no joint element"
                child.joint = None
                continue

            axis_token = self.get_attr(prim, "physics:axis", "axis", default="Z")
            axis = axis_token_to_vec3(axis_token)

            lower = self.get_attr(prim, "physics:lowerLimit", "lowerLimit", default=None)
            upper = self.get_attr(prim, "physics:upperLimit", "upperLimit", default=None)
            jrange = None
            if lower is not None and upper is not None:
                jrange = (
                    self.joint_value_to_mjcf(kind, float(lower)),
                    self.joint_value_to_mjcf(kind, float(upper)),
                )

            local_pos1 = self.get_attr(prim, "physics:localPos1", "localPos1", default=None)
            jpos = None
            if local_pos1 is not None:
                jpos = np.array([float(local_pos1[0]), float(local_pos1[1]), float(local_pos1[2])], dtype=float)
                jpos *= self.meters_per_unit

            drive_axis = None
            if kind == "hinge":
                drive_axis = "angular"
            elif kind == "slide":
                drive_axis = "linear"

            damping = None
            stiffness = None
            springref = None
            actuatorfrcrange = None
            if drive_axis is not None:
                damping = self.get_attr(prim, f"drive:{drive_axis}:physics:damping", default=None)
                if damping is not None:
                    damping = float(damping)
                    if damping <= 0.0:
                        damping = None

                stiffness = self.get_attr(prim, f"drive:{drive_axis}:physics:stiffness", default=None)
                if stiffness is not None:
                    stiffness = float(stiffness)
                    if stiffness <= 0.0:
                        stiffness = None

                authored_target_position = self.get_authored_attr(
                    prim,
                    f"drive:{drive_axis}:physics:targetPosition",
                    default=None,
                )
                target_position = self.get_attr(prim, f"drive:{drive_axis}:physics:targetPosition", default=None)
                if target_position is not None and (
                    stiffness is not None or authored_target_position is not None
                ):
                    springref = self.joint_value_to_mjcf(kind, float(target_position))

                max_force = self.get_attr(prim, f"drive:{drive_axis}:physics:maxForce", default=None)
                if max_force is not None:
                    max_force = float(max_force)
                    if math.isfinite(max_force) and 0.0 < max_force < 1e20:
                        actuatorfrcrange = (-max_force, max_force)

            child.joint = JointData(
                name=self.unique_joint_name(str(prim.GetPath()), body1, leaf_name_counts, used_names),
                kind=kind,
                axis=axis if kind in ("hinge", "slide") else None,
                range=jrange,
                pos=jpos,
                damping=damping,
                stiffness=stiffness,
                springref=springref,
                actuatorfrcrange=actuatorfrcrange,
            )
            used_names.add(child.joint.name)

    # ------------------------
    # Build node tree
    # ------------------------

    def build_nodes(self):
        for prim in self.stage.Traverse():
            if not self.prim_is_body_candidate(prim):
                continue

            path = str(prim.GetPath())
            t = prim.GetTypeName()
            name = sanitize_name(path)
            parent = self.get_nearest_body_ancestor_path(prim)
            world = self.get_world_matrix(prim)

            node = BodyNode(
                path=path,
                name=name,
                prim_type=t,
                parent=parent,
                world_matrix=world,
                kinematic=self.prim_is_kinematic(prim),
            )

            geom = self.build_geom_for_prim(prim)
            if geom is not None:
                node.geoms.append(geom)

            inertial = self.build_inertial_for_prim(prim)
            if inertial is not None:
                node.inertial = inertial

            self.nodes[path] = node

        self.parse_and_apply_joints()
        self.rebuild_children()
        self.exclude_gripper_finger_siblings()
        self.recompute_local_poses()
        self.collect_cameras()

    def collect_cameras(self):
        if self.build_sensor_link_cameras():
            return

        emitted_camera_keys = set()

        metadata_camera = self.build_stage_metadata_camera()
        if metadata_camera is not None:
            key = self.camera_pose_key(metadata_camera.pos, metadata_camera.quat, metadata_camera.fovy)
            emitted_camera_keys.add(key)
            self.world_cameras.append(metadata_camera)

        for prim in self.stage.Traverse():
            if not prim.IsA(UsdGeom.Camera): # type: ignore
                continue

            parent_path = self.get_nearest_body_ancestor_path(prim)
            camera = self.build_camera_for_prim(prim, parent_path)
            if camera is None:
                continue

            world_matrix = self.get_world_matrix(prim)
            world_pos, world_quat = matrix_to_pos_quat(world_matrix)
            world_pos *= self.meters_per_unit
            key = self.camera_pose_key(world_pos, world_quat, camera.fovy)
            if key in emitted_camera_keys:
                continue
            emitted_camera_keys.add(key)

            if parent_path and parent_path in self.nodes:
                self.nodes[parent_path].cameras.append(camera)
            else:
                self.world_cameras.append(camera)

    def rebuild_children(self):
        for node in self.nodes.values():
            node.children = []
        for path, node in self.nodes.items():
            if node.parent and node.parent in self.nodes:
                self.nodes[node.parent].children.append(path)

    def recompute_local_poses(self):
        for path, node in self.nodes.items():
            M_child = node.world_matrix
            if node.parent and node.parent in self.nodes:
                M_parent = self.nodes[node.parent].world_matrix
            else:
                M_parent = np_identity()
            M_local = np.linalg.inv(M_parent) @ M_child # type: ignore
            pos, quat = matrix_to_pos_quat(M_local)
            pos *= self.meters_per_unit
            node.local_pos = pos
            node.local_quat = quat

    def root_paths(self) -> List[str]:
        roots = []
        for p, n in self.nodes.items():
            if not n.parent or n.parent not in self.nodes:
                roots.append(p)
        return roots

    def should_emit_body(self, node: BodyNode) -> bool:
        return node.path in self.force_body_paths or node.inertial is not None or node.joint is not None

    def should_emit_freejoint(self, node: BodyNode) -> bool:
        if node.path not in self.force_body_paths or node.kinematic or node.joint is not None:
            return False
        return node.parent is None or node.parent not in self.nodes or not self.should_emit_body(self.nodes[node.parent])

    def node_local_matrix(self, node: BodyNode) -> np.ndarray:
        return pose_to_matrix(node.local_pos, node.local_quat)

    def joint_has_drive(self, joint: JointData) -> bool:
        return any((
            joint.damping is not None,
            joint.stiffness is not None,
            joint.springref is not None,
            joint.actuatorfrcrange is not None,
        ))

    def range_is_finite(self, value_range: Optional[Tuple[float, float]]) -> bool:
        if value_range is None:
            return False
        return math.isfinite(float(value_range[0])) and math.isfinite(float(value_range[1]))

    def estimate_position_kp(self, joint: JointData) -> float:
        if joint.stiffness is not None and joint.stiffness > 0.0:
            return float(joint.stiffness)

        if joint.actuatorfrcrange is not None:
            max_force = max(abs(float(joint.actuatorfrcrange[0])), abs(float(joint.actuatorfrcrange[1])))
            if max_force > 0.0:
                if joint.range is not None and self.range_is_finite(joint.range):
                    target = float(joint.springref) if joint.springref is not None else 0.0
                    span = max(abs(float(joint.range[0]) - target), abs(float(joint.range[1]) - target))
                else:
                    span = math.pi if joint.kind == "hinge" else 0.05
                # Make the servo hit its force limit after a very small tracking
                # error so zero-control models still hold their rest pose.
                inferred_kp = max_force / max(span * 1e-3, 1e-6)
                if joint.kind == "slide":
                    return min(inferred_kp, self.MAX_INFERRED_SLIDE_POSITION_KP)
                return inferred_kp

        return self.MAX_INFERRED_SLIDE_POSITION_KP if joint.kind == "slide" else 1e3

    def prefers_velocity_actuator(self, joint: JointData) -> bool:
        return (
            joint.kind in ("hinge", "slide")
            and joint.springref is None
            and not self.range_is_finite(joint.range)
            and joint.damping is not None
        )

    def estimate_velocity_kv(self, joint: JointData) -> float:
        base = float(joint.damping) if joint.damping is not None and joint.damping > 0.0 else 1.0
        if joint.actuatorfrcrange is not None:
            max_force = max(abs(float(joint.actuatorfrcrange[0])), abs(float(joint.actuatorfrcrange[1])))
            if max_force > 0.0:
                speed_span = 1.0 if joint.kind == "slide" else 10.0
                return max(base, max_force / speed_span)

        # Treat unbounded wheel-like joints as speed-controlled with a firm
        # zero-velocity target, so ctrl=0 acts as a brake at reset.
        return max(base * 50.0, 100.0 if joint.kind == "slide" else 1000.0)

    # ------------------------
    # MJCF XML emit
    # ------------------------

    def create_mjcf(self) -> ET.Element:
        mujoco = ET.Element("mujoco", attrib={"model": self.model_name})
        has_velocity_actuator = any(
            node.joint is not None and self.prefers_velocity_actuator(node.joint)
            for node in self.nodes.values()
        )

        ET.SubElement(mujoco, "compiler", attrib={
            "angle": "radian",
            "autolimits": "true",
        })
        option_attr = {"gravity": "0 0 -9.81"}
        if has_velocity_actuator:
            option_attr["integrator"] = "implicitfast"
        ET.SubElement(mujoco, "option", attrib=option_attr)

        asset = ET.SubElement(mujoco, "asset")

        # de-duplicate mesh assets
        emitted_meshes = set()
        for node in self.nodes.values():
            for g in node.geoms:
                if g.kind == "mesh" and g.mesh_name and g.file_rel:
                    if g.mesh_name not in emitted_meshes:
                        ET.SubElement(asset, "mesh", attrib={
                            "name": g.mesh_name,
                            "file": g.file_rel,
                        })
                        emitted_meshes.add(g.mesh_name)

        for texture_asset in self.texture_assets.values():
            ET.SubElement(asset, "texture", attrib={
                "name": texture_asset.name,
                "type": "2d",
                "file": texture_asset.file_rel,
            })

        for material_asset in self.material_assets.values():
            material_attr = {
                "name": material_asset.name,
                "texture": material_asset.texture_name,
            }
            if material_asset.rgba is not None:
                material_attr["rgba"] = fmt_vec(material_asset.rgba)
            ET.SubElement(asset, "material", attrib=material_attr)

        if self.contact_excludes:
            contact = ET.SubElement(mujoco, "contact")
            for body1, body2 in sorted(self.contact_excludes):
                ET.SubElement(contact, "exclude", attrib={
                    "body1": body1,
                    "body2": body2,
                })

        worldbody = ET.SubElement(mujoco, "worldbody")
        self.worldbody_xml = worldbody

        # If USD stage is Y-up, wrap everything in a rotated root body to become Z-up.
        emit_parent = worldbody
        forced_root_transform = np_identity()
        if self.stage_up_axis.upper() == "Y":
            q = quat_wxyz_from_axis_angle([1, 0, 0], math.pi / 2.0)
            forced_root_transform = pose_to_matrix(quat=q)
            wrapper = ET.SubElement(worldbody, "body", attrib={
                "name": "usd_stage_root",
                "quat": fmt_vec(q),
            })
            emit_parent = wrapper

        for camera in self.world_cameras:
            self.emit_camera(camera, emit_parent)

        for root_path in self.root_paths():
            parent_xml = worldbody if root_path in self.force_body_paths else emit_parent
            parent_transform = forced_root_transform if root_path in self.force_body_paths else np_identity()
            self.emit_node_recursive(root_path, parent_xml, parent_transform)

        actuator = ET.SubElement(mujoco, "actuator")
        for node in self.nodes.values():
            if node.joint is None:
                continue

            joint = node.joint
            if joint.kind not in ("hinge", "slide") or not self.joint_has_drive(joint):
                continue

            if joint.springref is not None:
                actuator_attr = {
                    "name": f"{joint.name}_position",
                    "joint": joint.name,
                    "kp": fmt_f(self.estimate_position_kp(joint)),
                }
                if self.range_is_finite(joint.range):
                    actuator_attr["ctrlrange"] = fmt_vec(joint.range)
                if joint.actuatorfrcrange is not None:
                    actuator_attr["forcerange"] = fmt_vec(joint.actuatorfrcrange)
                ET.SubElement(actuator, "position", attrib=actuator_attr)
                continue

            if self.prefers_velocity_actuator(joint):
                actuator_attr = {
                    "name": f"{joint.name}_motor",
                    "joint": joint.name,
                    "kv": fmt_f(self.estimate_velocity_kv(joint)),
                }
                if joint.actuatorfrcrange is not None:
                    actuator_attr["forcerange"] = fmt_vec(joint.actuatorfrcrange)
                ET.SubElement(actuator, "velocity", attrib=actuator_attr)
                continue

            actuator_attr = {
                "name": f"{joint.name}_motor",
                "joint": joint.name,
            }
            if joint.actuatorfrcrange is not None:
                actuator_attr["ctrlrange"] = fmt_vec(joint.actuatorfrcrange)
                actuator_attr["forcerange"] = fmt_vec(joint.actuatorfrcrange)
            ET.SubElement(actuator, "motor", attrib=actuator_attr)

        return mujoco

    def emit_geom(
        self,
        geom: GeomData,
        parent_xml: ET.Element,
        parent_transform: Optional[np.ndarray] = None,
    ):
        geom_attr = {
            "name": geom.name,
            "type": geom.kind,
        }

        if geom.kind == "mesh" and geom.mesh_name:
            geom_attr["mesh"] = geom.mesh_name
        elif geom.size is not None:
            geom_attr["size"] = fmt_vec(geom.size)

        if geom.material_name is not None:
            geom_attr["material"] = geom.material_name
        if geom.rgba is not None:
            geom_attr["rgba"] = fmt_vec(geom.rgba)

        pose = parent_transform.copy() if parent_transform is not None else np_identity()
        if geom.quat is not None:
            pose = pose @ pose_to_matrix(quat=geom.quat)
        pos, quat = matrix_to_pos_quat(pose)
        if np.linalg.norm(pos) > 1e-12:
            geom_attr["pos"] = fmt_vec(pos)
        if np.linalg.norm(quat - np.array([1, 0, 0, 0], dtype=float)) > 1e-12:
            geom_attr["quat"] = fmt_vec(quat)

        if geom.contype is not None:
            geom_attr["contype"] = str(geom.contype)
        if geom.conaffinity is not None:
            geom_attr["conaffinity"] = str(geom.conaffinity)

        ET.SubElement(parent_xml, "geom", attrib=geom_attr)

    def emit_camera(
        self,
        camera: CameraData,
        parent_xml: ET.Element,
        parent_transform: Optional[np.ndarray] = None,
    ):
        camera_attr = {"name": camera.name}
        if camera.fovy is not None:
            camera_attr["fovy"] = fmt_f(camera.fovy)

        pose = parent_transform.copy() if parent_transform is not None else np_identity()
        pose = pose @ pose_to_matrix(camera.pos, camera.quat)
        pos, quat = matrix_to_pos_quat(pose)
        if np.linalg.norm(pos) > 1e-12:
            camera_attr["pos"] = fmt_vec(pos)
        if np.linalg.norm(quat - np.array([1, 0, 0, 0], dtype=float)) > 1e-12:
            camera_attr["quat"] = fmt_vec(quat)

        ET.SubElement(parent_xml, "camera", attrib=camera_attr)

    def emit_node_recursive(self, path: str, parent_xml: ET.Element, parent_transform: np.ndarray):
        node = self.nodes[path]
        node_transform = parent_transform @ self.node_local_matrix(node)

        if not self.should_emit_body(node):
            for g in node.geoms:
                self.emit_geom(g, parent_xml, node_transform)
            for camera in node.cameras:
                self.emit_camera(camera, parent_xml, node_transform)
            for child_path in node.children:
                self.emit_node_recursive(child_path, parent_xml, node_transform)
            return

        body_attr = {"name": node.name}

        body_pos, body_quat = matrix_to_pos_quat(node_transform)
        if np.linalg.norm(body_pos) > 1e-12:
            body_attr["pos"] = fmt_vec(body_pos)
        if np.linalg.norm(body_quat - np.array([1, 0, 0, 0], dtype=float)) > 1e-12:
            body_attr["quat"] = fmt_vec(body_quat)

        body_parent = parent_xml
        if (
            node.path in self.force_body_paths
            and self.stage_up_axis.upper() == "Y"
            and (node.parent is None or not self.should_emit_body(self.nodes[node.parent]))
        ):
            body_parent = self.worldbody_xml

        body_xml = ET.SubElement(body_parent, "body", attrib=body_attr)

        if node.inertial is not None:
            iner_attr = {
                "mass": fmt_f(node.inertial.mass),
            }
            if node.inertial.pos is not None:
                iner_attr["pos"] = fmt_vec(node.inertial.pos)
            if node.inertial.diaginertia is not None:
                iner_attr["diaginertia"] = fmt_vec(node.inertial.diaginertia)
            if node.inertial.quat is not None:
                if np.linalg.norm(node.inertial.quat - np.array([1, 0, 0, 0], dtype=float)) > 1e-12:
                    iner_attr["quat"] = fmt_vec(node.inertial.quat)
            ET.SubElement(body_xml, "inertial", attrib=iner_attr)

        if node.joint is not None:
            j = node.joint
            joint_attr = {
                "name": j.name,
                "type": j.kind,
            }
            if j.pos is not None:
                joint_attr["pos"] = fmt_vec(j.pos)
            if j.kind in ("hinge", "slide") and j.axis is not None:
                joint_attr["axis"] = fmt_vec(j.axis)
            if j.range is not None and j.kind in ("hinge", "slide"):
                joint_attr["range"] = fmt_vec(j.range)
            if j.damping is not None:
                joint_attr["damping"] = fmt_f(j.damping)
            if j.stiffness is not None:
                joint_attr["stiffness"] = fmt_f(j.stiffness)
            if j.springref is not None:
                joint_attr["springref"] = fmt_f(j.springref)
            if j.actuatorfrcrange is not None:
                joint_attr["actuatorfrcrange"] = fmt_vec(j.actuatorfrcrange)
            ET.SubElement(body_xml, "joint", attrib=joint_attr)
        elif self.should_emit_freejoint(node):
            ET.SubElement(body_xml, "freejoint", attrib={"name": f"{node.name}_freejoint"})

        for g in node.geoms:
            self.emit_geom(g, body_xml)

        for camera in node.cameras:
            self.emit_camera(camera, body_xml)

        for child_path in node.children:
            self.emit_node_recursive(child_path, body_xml, np_identity())

    def write_xml(self):
        mjcf_root = self.create_mjcf()
        tree = ET.ElementTree(mjcf_root)
        try:
            ET.indent(tree, space="  ", level=0)
        except Exception:
            pass

        self.output_dir.mkdir(parents=True, exist_ok=True)
        tree.write(self.output_xml, encoding="utf-8", xml_declaration=True)

    def convert(self):
        self.build_nodes()
        self.write_xml()


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert USDA/USD to MuJoCo MJCF.")
    parser.add_argument("input_usd", type=str, help="Input .usd/.usda/.usdc file")
    parser.add_argument("output_xml", type=str, help="Output MJCF .xml file")
    parser.add_argument("--model-name", type=str, default="converted_from_usd", help="MuJoCo model name")
    args = parser.parse_args()

    conv = USDToMJCFConverter(
        usd_path=Path(args.input_usd),
        output_xml=Path(args.output_xml),
        model_name=args.model_name,
    )
    conv.convert()
    import mujoco
    mujoco.MjModel.from_xml_path(str(conv.output_xml)) # type: ignore

    print(f"[OK] MJCF written to: {conv.output_xml}")
    print(f"[OK] Meshes written to: {conv.mesh_dir}")
    print(f"[INFO] stage metersPerUnit = {conv.meters_per_unit}")
    print(f"[INFO] stage upAxis = {conv.stage_up_axis}")


if __name__ == "__main__":
    main()

