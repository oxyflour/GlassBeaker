from __future__ import annotations

import asyncio
from copy import deepcopy
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import mujoco  # type: ignore
import numpy as np


def decode_mesh_path(model, mesh_id: int):
    start = int(model.mesh_pathadr[mesh_id])
    end = model.paths.find(b"\x00", start)
    if end < 0:
        end = len(model.paths)
    return Path(model.paths[start:end].decode("utf-8"))


def decode_texture_path(model, tex_id: int) -> Path | None:
    if tex_id < 0:
        return None
    start = int(model.tex_pathadr[tex_id])
    end = model.paths.find(b"\x00", start)
    if end < 0:
        end = len(model.paths)
    return Path(model.paths[start:end].decode("utf-8"))


def quat_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def pose_matrix(
    pos: np.ndarray,
    quat: np.ndarray,
    scale: np.ndarray | None = None,
) -> np.ndarray:
    matrix = quat_matrix(quat)
    if scale is not None:
        scaled = np.eye(4)
        scaled[0, 0], scaled[1, 1], scaled[2, 2] = [float(v) for v in scale]
        matrix = matrix @ scaled
    matrix[:3, 3] = [float(v) for v in pos]
    return matrix


def flatten_matrix(matrix: np.ndarray) -> list[float]:
    return [float(v) for v in matrix.T.reshape(-1)]


def mesh_world_pose(model, data, geom_id: int):
    geom_world = geom_world_pose(data, geom_id)
    mesh_id = int(model.geom_dataid[geom_id])
    mesh_local = pose_matrix(
        model.mesh_pos[mesh_id],
        model.mesh_quat[mesh_id],
        model.mesh_scale[mesh_id],
    )
    return geom_world @ np.linalg.inv(mesh_local)


def geom_world_pose(data, geom_id: int) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    matrix[:3, 3] = np.array(data.geom_xpos[geom_id], dtype=float)
    return matrix


def body_world_pose(data, body_id: int) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(data.xmat[body_id], dtype=float).reshape(3, 3)
    matrix[:3, 3] = np.array(data.xpos[body_id], dtype=float)
    return matrix


def geom_size(model, geom_id: int, kind: str) -> list[float]:
    size = np.array(model.geom_size[geom_id], dtype=float)
    if kind == "plane":
        return [float(max(size[0] * 2, 1e-3)), float(max(size[1] * 2, 1e-3))]
    if kind in {"box", "ellipsoid"}:
        return [
            float(max(size[0] * 2, 1e-3)),
            float(max(size[1] * 2, 1e-3)),
            float(max(size[2] * 2, 1e-3)),
        ]
    if kind == "sphere":
        return [float(max(size[0], 1e-3))]
    if kind in {"capsule", "cylinder"}:
        return [float(max(size[0], 1e-3)), float(max(size[1] * 2, 1e-3))]
    raise ValueError(f"Unsupported primitive kind: {kind}")


async def fix_urdf_path(urdf: Path) -> Path:
    xml = urdf.read_text(encoding="utf-8")
    converted = urdf.with_suffix(".converted.v5.urdf")
    if not os.path.exists(converted):
        script = str(Path(__file__).resolve().parents[2] / "scripts" / "convert_usd.py")
        blender = os.environ.get(
            "BLENDER_BINARY",
            r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
        )
        cmd = [blender, "--background", "--python", script, "--", str(urdf.parent / "usd")]
        print("RUN: ", " ".join([f'"{item}"' for item in cmd]))
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()
        xml = re.sub(r'filename="\./usd/(.*)\.usd"', r'filename="./usd/\1.stl"', xml)
        converted.write_text(xml, encoding="utf-8")
    return converted


async def create_xml(input: str):
    if input.endswith(".xml"):
        abs_xml = Path(input).resolve()
    elif input.endswith(".urdf"):
        abs_xml = Path(input).with_suffix(".xml").resolve()
        print(f"check {abs_xml} for {input}")
        if not abs_xml.exists():
            abs_urdf = await fix_urdf_path(Path(input))
            urdf_model = mujoco.MjModel.from_xml_path(str(abs_urdf))  # type: ignore
            mujoco.mj_saveLastXML(abs_xml, urdf_model)  # type: ignore
    elif input.endswith(".usda"):
        abs_xml = Path(input).with_suffix(".xml").resolve()
        print(f"check {abs_xml} for {input}")
        if not abs_xml.exists():
            script = str(Path(__file__).resolve().parents[1] / "usd_to_mjcf.py")
            cmd = [
                sys.executable,
                "-u",
                script,
                input,
                str(abs_xml),
                "--model-name",
                "r1pro",
            ]
            print("CMD: " + " ".join(cmd))
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()
    else:
        raise Exception(f"unknown input file format: {input}")

    xml_str = f"""
    <mujoco>
        <option timestep="0.001" />
        <include file="{abs_xml}"/>
        <worldbody>
            <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.9 0.8 1"/>
        </worldbody>
    </mujoco>
    """
    out_xml = abs_xml.parent / "out.xml"
    out_xml.write_text(xml_str)
    return out_xml


def compile_urdf_to_mjcf(urdf_path: Path, output_xml: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(urdf_path))  # type: ignore
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveLastXML(str(output_xml), model)  # type: ignore


def merge_mjcf_files(robot_xml: Path, scene_xml: Path, output_xml: Path) -> None:
    scene_tree = ET.parse(scene_xml)
    robot_tree = ET.parse(robot_xml)
    scene_root = scene_tree.getroot()
    robot_root = robot_tree.getroot()
    for tag in ("asset", "worldbody", "actuator", "sensor", "contact", "equality", "tendon", "default"):
        _merge_container(scene_root, robot_root, tag)
    for child in robot_root:
        if child.tag in {"compiler", "option", "asset", "worldbody", "actuator", "sensor", "contact", "equality", "tendon", "default"}:
            continue
        if scene_root.find(child.tag) is None:
            scene_root.append(deepcopy(child))
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    scene_tree.write(output_xml, encoding="utf-8", xml_declaration=True)


def _merge_container(scene_root, robot_root, tag: str) -> None:
    source = robot_root.find(tag)
    if source is None:
        return
    target = scene_root.find(tag)
    if target is None:
        scene_root.append(deepcopy(source))
        return
    for child in source:
        target.append(deepcopy(child))

__all__ = [
    "body_world_pose",
    "compile_urdf_to_mjcf",
    "create_xml",
    "decode_mesh_path",
    "decode_texture_path",
    "fix_urdf_path",
    "flatten_matrix",
    "geom_size",
    "geom_world_pose",
    "merge_mjcf_files",
    "mesh_world_pose",
    "pose_matrix",
    "quat_matrix",
]
