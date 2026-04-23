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
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import xml.etree.ElementTree as ET

from pxr import Usd, UsdGeom


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


@dataclass
class InertialData:
    mass: float
    pos: Optional[np.ndarray] = None
    diaginertia: Optional[np.ndarray] = None


@dataclass
class GeomData:
    name: str
    kind: str                      # mesh | box | sphere | capsule | cylinder
    rgba: Optional[np.ndarray] = None
    size: Optional[np.ndarray] = None
    mesh_name: Optional[str] = None
    file_rel: Optional[str] = None
    quat: Optional[np.ndarray] = None
    contype: Optional[int] = None
    conaffinity: Optional[int] = None


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
    inertial: Optional[InertialData] = None
    joint: Optional[JointData] = None


# ----------------------------
# Converter
# ----------------------------

class USDToMJCFConverter:
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

    def __init__(self, usd_path: Path, output_xml: Path, model_name: str = "converted_from_usd"):
        self.usd_path = Path(usd_path)
        self.output_xml = Path(output_xml)
        self.output_dir = self.output_xml.parent.resolve()
        self.mesh_dir = self.output_dir / "meshes"
        self.mesh_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.stage = Usd.Stage.Open(str(self.usd_path))
        if self.stage is None:
            raise RuntimeError(f"Failed to open USD stage: {self.usd_path}")

        self.xform_cache = UsdGeom.XformCache()

        # stage unit conversion
        if self.stage.HasAuthoredMetadata("metersPerUnit"):
            mpu = self.stage.GetMetadata("metersPerUnit")
            self.meters_per_unit = float(mpu) if mpu is not None else 1.0
        else:
            # Some Blender-exported USDA files omit this metadata while already
            # authoring transforms and vertices in meters.
            self.meters_per_unit = 1.0

        up = UsdGeom.GetStageUpAxis(self.stage)
        self.stage_up_axis = str(up) if up is not None else "Z"

        self.nodes: Dict[str, BodyNode] = {}
        self.mesh_assets: Dict[str, GeomData] = {}

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
        return prim.IsA(UsdGeom.Xformable)

    def get_nearest_body_ancestor_path(self, prim) -> Optional[str]:
        p = prim.GetParent()
        while p and p.IsValid():
            if self.prim_is_body_candidate(p):
                return str(p.GetPath())
            p = p.GetParent()
        return None

    def get_world_matrix(self, prim) -> np.ndarray:
        return gf_matrix_to_np(self.xform_cache.GetLocalToWorldTransform(prim))

    def get_display_rgba(self, prim) -> Optional[np.ndarray]:
        try:
            gprim = UsdGeom.Gprim(prim)
            attr = gprim.GetDisplayColorAttr()
            cols = attr.Get()
            if cols and len(cols) > 0:
                c = cols[0]
                return np.array([float(c[0]), float(c[1]), float(c[2]), 1.0], dtype=float)
        except Exception:
            pass
        return np.array([0.7, 0.7, 0.7, 1.0], dtype=float)

    def get_collision_enabled(self, prim) -> Tuple[Optional[int], Optional[int]]:
        enabled = self.get_attr(prim, "physics:collisionEnabled", default=None)
        if enabled is False:
            return 0, 0
        return None, None

    # ------------------------
    # Geometry export
    # ------------------------

    def build_geom_for_prim(self, prim) -> Optional[GeomData]:
        t = prim.GetTypeName()
        name = sanitize_name(str(prim.GetPath())) + "_geom"
        rgba = self.get_display_rgba(prim)
        contype, conaffinity = self.get_collision_enabled(prim)

        if prim.IsA(UsdGeom.Mesh):
            mesh_name, file_rel = self.export_mesh_prim(prim)
            if mesh_name is None:
                return None
            return GeomData(
                name=name,
                kind="mesh",
                mesh_name=mesh_name,
                file_rel=file_rel,
                rgba=rgba,
                contype=contype,
                conaffinity=conaffinity,
            )

        if t == "Cube":
            size = self.get_attr(prim, "size", default=2.0)
            half = 0.5 * float(size) * self.meters_per_unit
            return GeomData(
                name=name,
                kind="box",
                size=np.array([half, half, half], dtype=float),
                rgba=rgba,
                contype=contype,
                conaffinity=conaffinity,
            )

        if t == "Sphere":
            radius = self.get_attr(prim, "radius", default=1.0)
            r = float(radius) * self.meters_per_unit
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
            r = float(radius) * self.meters_per_unit
            half_len = 0.5 * float(height) * self.meters_per_unit
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
            r = float(radius) * self.meters_per_unit              # type: ignore
            half_len = 0.5 * float(height) * self.meters_per_unit # type: ignore
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

    def export_mesh_prim(self, prim) -> Tuple[Optional[str], Optional[str]]:
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        fvc = mesh.GetFaceVertexCountsAttr().Get()
        fvi = mesh.GetFaceVertexIndicesAttr().Get()

        if pts is None or fvc is None or fvi is None:
            return None, None

        vertices = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in pts], dtype=float)
        vertices *= self.meters_per_unit

        faces = triangulate_faces(list(fvc), list(fvi))
        if len(vertices) == 0 or len(faces) == 0:
            return None, None

        mesh_name = sanitize_name(str(prim.GetPath()))
        out_file = self.mesh_dir / f"{mesh_name}.obj"

        tri = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        tri.export(out_file)

        file_rel = str(Path("meshes") / f"{mesh_name}.obj")
        return mesh_name, file_rel

    # ------------------------
    # Inertial and joints
    # ------------------------

    def build_inertial_for_prim(self, prim) -> Optional[InertialData]:
        mass = self.get_authored_attr(prim, "physics:mass", default=None)
        if mass is None:
            return None
        mass = float(mass)
        if mass <= 0.0:
            return None

        com = self.get_authored_attr(prim, "physics:centerOfMass", default=None)

        pos = None
        if com is not None:
            pos = np.array([float(com[0]), float(com[1]), float(com[2])], dtype=float) * self.meters_per_unit

        diagI = self.get_authored_attr(prim, "physics:diagonalInertia", default=None)
        diaginertia = None
        if diagI is not None:
            diaginertia = np.array([float(diagI[0]), float(diagI[1]), float(diagI[2])], dtype=float)
            if np.any(diaginertia <= 0.0):
                diaginertia = None
        if diaginertia is None:
            diaginertia = fallback_diaginertia(mass, pos)

        return InertialData(mass=mass, pos=pos, diaginertia=diaginertia)

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

    def parse_and_apply_joints(self):
        """
        Parse USD joint prims and assign them to body1.
        If body0/body1 relation exists, reparent body1 under body0.
        """
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
                jrange = (float(lower), float(upper))

            local_pos1 = self.get_attr(prim, "physics:localPos1", "localPos1", default=None)
            jpos = None
            if local_pos1 is not None:
                jpos = np.array([float(local_pos1[0]), float(local_pos1[1]), float(local_pos1[2])], dtype=float)
                jpos *= self.meters_per_unit

            child.joint = JointData(
                name=sanitize_name(str(prim.GetPath())),
                kind=kind,
                axis=axis if kind in ("hinge", "slide") else None,
                range=jrange,
                pos=jpos,
            )

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
        self.recompute_local_poses()

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

    # ------------------------
    # MJCF XML emit
    # ------------------------

    def create_mjcf(self) -> ET.Element:
        mujoco = ET.Element("mujoco", attrib={"model": self.model_name})

        ET.SubElement(mujoco, "compiler", attrib={
            "angle": "radian",
            "autolimits": "true",
        })
        ET.SubElement(mujoco, "option", attrib={
            "gravity": "0 0 -9.81",
        })

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

        worldbody = ET.SubElement(mujoco, "worldbody")

        # If USD stage is Y-up, wrap everything in a rotated root body to become Z-up.
        emit_parent = worldbody
        if self.stage_up_axis.upper() == "Y":
            q = quat_wxyz_from_axis_angle([1, 0, 0], math.pi / 2.0)
            wrapper = ET.SubElement(worldbody, "body", attrib={
                "name": "usd_stage_root",
                "quat": fmt_vec(q),
            })
            emit_parent = wrapper

        for root_path in self.root_paths():
            self.emit_body_recursive(root_path, emit_parent)

        return mujoco

    def emit_body_recursive(self, path: str, parent_xml: ET.Element):
        node = self.nodes[path]
        body_attr = {"name": node.name}

        if node.local_pos is not None and np.linalg.norm(node.local_pos) > 1e-12:
            body_attr["pos"] = fmt_vec(node.local_pos)
        if node.local_quat is not None:
            q = node.local_quat
            if np.linalg.norm(q - np.array([1, 0, 0, 0], dtype=float)) > 1e-12:
                body_attr["quat"] = fmt_vec(q)

        body_xml = ET.SubElement(parent_xml, "body", attrib=body_attr)

        if node.inertial is not None:
            iner_attr = {
                "mass": fmt_f(node.inertial.mass),
            }
            if node.inertial.pos is not None:
                iner_attr["pos"] = fmt_vec(node.inertial.pos)
            if node.inertial.diaginertia is not None:
                iner_attr["diaginertia"] = fmt_vec(node.inertial.diaginertia)
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
            ET.SubElement(body_xml, "joint", attrib=joint_attr)

        for g in node.geoms:
            geom_attr = {
                "name": g.name,
                "type": g.kind,
            }

            if g.kind == "mesh" and g.mesh_name:
                geom_attr["mesh"] = g.mesh_name
            elif g.size is not None:
                geom_attr["size"] = fmt_vec(g.size)

            if g.rgba is not None:
                geom_attr["rgba"] = fmt_vec(g.rgba)

            if g.quat is not None:
                geom_attr["quat"] = fmt_vec(g.quat)

            if g.contype is not None:
                geom_attr["contype"] = str(g.contype)
            if g.conaffinity is not None:
                geom_attr["conaffinity"] = str(g.conaffinity)

            ET.SubElement(body_xml, "geom", attrib=geom_attr)

        for child_path in node.children:
            self.emit_body_recursive(child_path, body_xml)

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
    mujoco.MjModel.from_xml_path(str(conv.output_xml))

    print(f"[OK] MJCF written to: {conv.output_xml}")
    print(f"[OK] Meshes written to: {conv.mesh_dir}")
    print(f"[INFO] stage metersPerUnit = {conv.meters_per_unit}")
    print(f"[INFO] stage upAxis = {conv.stage_up_axis}")


if __name__ == "__main__":
    main()
