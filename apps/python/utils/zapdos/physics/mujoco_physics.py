from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco  # type: ignore
import mujoco.viewer
import numpy as np
from fastapi import HTTPException

from utils.zapdos.physics.body_capabilities import build_body_capabilities
from utils.zapdos.physics.mujoco_tools import (
    body_world_pose,
    decode_mesh_path,
    decode_texture_path,
    flatten_matrix,
    geom_size,
    geom_world_pose,
    mesh_world_pose,
)
from utils.zapdos.physics.visuals import SceneVisuals, serialize_body, serialize_mesh

RGB_TEXTURE_ROLE = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)  # type: ignore
PRIMITIVE_TYPES = {
    int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",  # type: ignore
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",  # type: ignore
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",  # type: ignore
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",  # type: ignore
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",  # type: ignore
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",  # type: ignore
    int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",  # type: ignore
}


@dataclass
class ZapdosGeometry:
    name: str = ""
    kind: str = ""
    geom_id: int = 0
    body: str = ""
    mesh: str = ""
    texture: str = ""
    color: list[float] | None = None
    size: list[float] | None = None


class MujocoPhysics:
    def __init__(self, sess: str, bundle: Any, body_map: dict[str, str]) -> None:
        self.sess = sess
        self.model = mujoco.MjModel.from_xml_path(str(bundle.mjcf))  # type: ignore
        self.data = mujoco.MjData(self.model)  # type: ignore
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data) if os.environ.get("DEBUG_MUJOCO_VIEWER") else None
        self.assets: dict[str, Path] = {}
        self.geoms = self._build_geometry(bundle.mjcf.parent)
        self.body_map = body_map
        self.body_labels = {name: path.rsplit("/", 1)[-1] for name, path in body_map.items()}
        capabilities = build_body_capabilities(self.model, body_map)
        self.editable_body_names = capabilities.editable_body_names
        self.robot_body_names = capabilities.robot_body_names
        self.robot_root_body_names = capabilities.robot_root_body_names
        self.movable_body_names = capabilities.movable_body_names
        self.selection_body_by_name = capabilities.selection_body_by_name
        self.actuator_name_to_id = self._actuator_map()
        self.joint_name_to_actuator = self._joint_command_map()
        self.data.ctrl[:] = 0
        for joint_name, actuator_id in self.joint_name_to_actuator.items():
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or ""  # type: ignore
            if actuator_name.endswith("_position"):
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)  # type: ignore
                qpos_adr = int(self.model.jnt_qposadr[joint_id])
                self.data.ctrl[actuator_id] = float(self.data.qpos[qpos_adr])
        mujoco.mj_forward(self.model, self.data)  # type: ignore

    def _build_geometry(self, asset_root: Path) -> dict[str, ZapdosGeometry]:
        geoms: dict[str, ZapdosGeometry] = {}
        for geom_id in range(self.model.ngeom):
            geom_model_name = (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""  # type: ignore
            )
            if "collision" in geom_model_name.lower():
                continue
            geom = ZapdosGeometry(geom_id=geom_id, kind=PRIMITIVE_TYPES.get(int(self.model.geom_type[geom_id])) or "")
            if geom.kind == "mesh":
                mesh_id = int(self.model.geom_dataid[geom_id])
                mesh_rel = decode_mesh_path(self.model, mesh_id)
                if "collisions" in mesh_rel.name:
                    continue
                geom.mesh = mesh_rel.name
                self.assets[geom.mesh] = (asset_root / mesh_rel).resolve()
                mat_id = int(self.model.geom_matid[geom_id])
                tex_id = int(self.model.mat_texid[mat_id, RGB_TEXTURE_ROLE]) if mat_id >= 0 else -1
                tex_rel = decode_texture_path(self.model, tex_id)
                if tex_rel is not None:
                    geom.texture = tex_rel.name
                    self.assets[geom.texture] = (asset_root / tex_rel).resolve()
            elif geom.kind:
                geom.size = geom_size(self.model, geom_id, geom.kind)
            else:
                continue
            body_id = int(self.model.geom_bodyid[geom_id])
            geom.body = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "world"  # type: ignore
            geom.color = [float(value) for value in self.model.geom_rgba[geom_id]]
            geom.name = f"geom-{geom_id}"
            geoms[geom.name] = geom
        return geoms

    def _actuator_map(self) -> dict[str, int]:
        return {
            name: actuator_id
            for actuator_id in range(self.model.nu)
            if (name := mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id))  # type: ignore
        }

    def _joint_command_map(self) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id][0])
            if joint_id >= 0 and (name := mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)):  # type: ignore
                mapping[name] = actuator_id
        return mapping

    def _body_matrices(self) -> dict[str, np.ndarray]:
        return {
            name: body_world_pose(self.data, body_id)
            for body_id in range(1, self.model.nbody)
            if (name := mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id))  # type: ignore
        }

    def _body_freejoint_id(self, body_id: int) -> int | None:
        joint_start = int(self.model.body_jntadr[body_id])
        for joint_id in range(joint_start, joint_start + int(self.model.body_jntnum[body_id])):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:  # type: ignore
                return joint_id
        return None

    def _set_freejoint_pose(self, body_id: int, pos: np.ndarray, quat: np.ndarray) -> None:
        freejoint_id = self._body_freejoint_id(body_id)
        if freejoint_id is None:
            return
        qpos_adr = int(self.model.jnt_qposadr[freejoint_id])
        qvel_adr = int(self.model.jnt_dofadr[freejoint_id])
        self.data.qpos[qpos_adr:qpos_adr + 3] = np.asarray(pos, dtype=float)
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = np.asarray(quat, dtype=float)
        self.data.qvel[qvel_adr:qvel_adr + 6] = 0.0

    def _mesh_anchor_body(self, body_name: str, body_matrices: dict[str, np.ndarray]) -> str | None:
        if body_name not in body_matrices:
            return None
        current_name: str | None = body_name
        while current_name is not None:
            if current_name in self.editable_body_names:
                return current_name
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, current_name)  # type: ignore
            parent_id = int(self.model.body_parentid[body_id])
            current_name = None if parent_id <= 0 else mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, parent_id)  # type: ignore
        return body_name

    def get_visual(self) -> SceneVisuals:
        body_matrices = self._body_matrices()
        bodies = [
            serialize_body(
                name,
                self.body_map.get(name, name),
                name in self.editable_body_names,
                flatten_matrix(matrix),
                selectable=name in self.selection_body_by_name,
                movable=name in self.movable_body_names,
                selection_body=self.selection_body_by_name.get(name),
            )
            for name, matrix in body_matrices.items()
        ]
        meshes = []
        for name, geom in self.geoms.items():
            world_matrix = mesh_world_pose(self.model, self.data, geom.geom_id) if geom.mesh else geom_world_pose(self.data, geom.geom_id)
            body_name = self._mesh_anchor_body(geom.body, body_matrices)
            meshes.append(serialize_mesh(
                name,
                body_name,
                geom.kind,
                geom.color,
                matrix=None if body_name else flatten_matrix(world_matrix),
                local_matrix=(flatten_matrix(np.linalg.inv(body_matrices[body_name]) @ world_matrix) if body_name else None),
                size=geom.size,
                mesh=(f"/python/zapdos/{self.sess}/asset/{geom.mesh}" if geom.mesh else ""),
                texture=(f"/python/zapdos/{self.sess}/asset/{geom.texture}" if geom.texture else ""),
            ))
        return {"bodies": bodies, "meshes": meshes}

    def get_pose(self) -> dict[str, list[float]]:
        return {name: flatten_matrix(matrix) for name, matrix in self._body_matrices().items()}

    def get_camera(self) -> dict[str, list[float]]:
        cameras: dict[str, list[float]] = {}
        for cam_id in range(self.model.ncam):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)  # type: ignore
            if name is None:
                continue
            mat4 = np.eye(4)
            mat4[:3, :3] = self.data.cam_xmat[cam_id].reshape(3, 3)
            mat4[:3, 3] = self.data.cam_xpos[cam_id]
            cameras[name] = flatten_matrix(mat4)
        return cameras

    def set_body_pose(self, body: str, pos: list[float], quat: list[float]) -> dict[str, object]:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)  # type: ignore
        if body_id < 0:
            raise HTTPException(status_code=404, detail=f"Body not found: {body}")
        if body not in self.movable_body_names:
            raise HTTPException(status_code=403, detail=f"Body is not movable: {body}")
        if len(pos) != 3 or len(quat) != 4:
            raise HTTPException(status_code=400, detail="Expected pos[3] and quat[4]")
        quat_vec = np.array(quat, dtype=float)
        quat_norm = np.linalg.norm(quat_vec)
        if quat_norm <= 1e-12:
            raise HTTPException(status_code=400, detail="Quaternion must be non-zero")
        normalized_quat = quat_vec / quat_norm
        self._set_freejoint_pose(body_id, np.array(pos, dtype=float), normalized_quat)
        self.model.body_pos[body_id] = np.array(pos, dtype=float)
        self.model.body_quat[body_id] = normalized_quat
        mujoco.mj_forward(self.model, self.data)  # type: ignore
        return {"ok": True}

    def joint_state_msg(self) -> dict[str, list[float] | list[str]]:
        names: list[str] = []
        positions: list[float] = []
        velocities: list[float] = []
        efforts: list[float] = []
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:  # type: ignore
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)  # type: ignore
            if not name:
                continue
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            qvel_adr = int(self.model.jnt_dofadr[joint_id])
            names.append(name)
            positions.append(float(self.data.qpos[qpos_adr]))
            velocities.append(float(self.data.qvel[qvel_adr]))
            efforts.append(float(self.data.qfrc_actuator[qvel_adr]) if qvel_adr < len(self.data.qfrc_actuator) else 0.0)
        return {"name": names, "position": positions, "velocity": velocities, "effort": efforts}

    def apply_joint_command(self, message: dict[str, Any] | None) -> None:
        if message is None:
            return
        ctrl = np.copy(self.data.ctrl)
        for name, pos in zip(message.get("name") or [], message.get("position") or []):
            actuator_id = self.actuator_name_to_id.get(name)
            if actuator_id is None:
                actuator_id = self.joint_name_to_actuator.get(name)
            if actuator_id is not None:
                ctrl[actuator_id] = float(pos)
        self.data.ctrl[:] = ctrl

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)  # type: ignore
        if self.viewer:
            self.viewer.sync()

    def close(self) -> None:
        if self.viewer:
            self.viewer.close()
            self.viewer = None


ZapdosPhysics = MujocoPhysics

__all__ = ["MujocoPhysics", "ZapdosGeometry", "ZapdosPhysics"]
