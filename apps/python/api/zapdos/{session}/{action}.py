import os
import re
import mujoco # type: ignore
import numpy as np
import subprocess
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from utils.session import Session


PRIMITIVE_TYPES = {
    int(mujoco.mjtGeom.mjGEOM_PLANE)    : 'plane',     # type: ignore
    int(mujoco.mjtGeom.mjGEOM_SPHERE)   : 'sphere',    # type: ignore
    int(mujoco.mjtGeom.mjGEOM_CAPSULE)  : 'capsule',   # type: ignore
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): 'ellipsoid', # type: ignore
    int(mujoco.mjtGeom.mjGEOM_CYLINDER) : 'cylinder',  # type: ignore
    int(mujoco.mjtGeom.mjGEOM_BOX)      : 'box',       # type: ignore
}


def fix_urdf_path(urdf: Path) -> Path:
    xml = urdf.read_text(encoding='utf-8')
    converted = urdf.with_suffix('.converted.v5.urdf')
    if not os.path.exists(converted):
        script = os.path.normpath(f"{__file__}/../../../../scripts/convert_usd.py")
        blender = os.environ.get('BLENDER_BINARY', rf"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe")
        cmd = [blender, '--background', '--python', script, '--', str(urdf.parent / 'usd')]
        print('RUN: ', ' '.join([f'"{item}"' for item in cmd]))
        #subprocess.run(cmd, check=True)
        xml = re.sub(r'filename="\./usd/(.*)\.usd"', r'filename="./usd/\1.stl"', xml)
        converted.write_text(xml, encoding='utf-8')
    return converted

def create_xml():
    urdf = Path(rf'..\..\deps\galaxea\object\r1pro\r1_pro_with_gripper.urdf').resolve()
    abs_urdf = fix_urdf_path(urdf)

    urdf_model = mujoco.MjModel.from_xml_path(str(abs_urdf))           # type: ignore
    abs_xml = str(abs_urdf.with_suffix('.xml'))
    mujoco.mj_saveLastXML(abs_xml, urdf_model) # type: ignore
    xml_str = f'''
    <mujoco>
        <option timestep="0.001" />
        <include file="{abs_xml}"/>
        <worldbody>
            <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.9 0.8 1"/>
        </worldbody>
        <actuator>
            <position name="servo1" joint="my_hinge" kp="100" ctrlrange="-1.57 1.57" ctrllimited="true"/>
        </actuator>
    </mujoco>
    '''
    asset_root = abs_urdf.parent
    return xml_str, asset_root

class ZapdosGeometry:
    def __init__(self, name: str, geom_id: int, body: str, color: list[float], size: list[float] | None = None, abs_path: Path | None = None):
        self.name = name
        self.geom_id = geom_id
        self.body = body
        self.color = color
        self.size = size
        self.abs_path = abs_path

class ZapdosSession(Session):
    def __init__(self, sess: str) -> None:
        self.sess = sess

        xml_str, asset_root = create_xml()
        self.model = mujoco.MjModel.from_xml_string(xml_str) # type: ignore
        self.data = mujoco.MjData(self.model)                # type: ignore
        mujoco.mj_step(self.model, self.data)                # type: ignore

        self.visuals: dict[str, ZapdosGeometry] = { }
        for geom_id in range(self.model.ngeom):
            geom_type = int(self.model.geom_type[geom_id])
            kind = PRIMITIVE_TYPES.get(geom_type)
            abs_path: Path | None = None
            size: list[float] | None = None
            if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH): # type: ignore
                mesh_id = int(self.model.geom_dataid[geom_id])
                rel_path = self.decode_path(int(self.model.mesh_pathadr[mesh_id]))
                kind = Path(rel_path).suffix.lower().lstrip('.')
                abs_path = (asset_root / rel_path).resolve()
            elif kind:
                size = self.geom_size(geom_id, kind)
            else:
                continue
            body_id = int(self.model.geom_bodyid[geom_id])
            body = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or 'world' # type: ignore
            name = f'geom-{geom_id}.{kind}'
            self.visuals[name] = ZapdosGeometry(
                name=name,
                geom_id=geom_id,
                body=body,
                color=[float(value) for value in self.model.geom_rgba[geom_id]],
                size=size,
                abs_path=abs_path,
            )

        super().__init__()

    def decode_path(self, start: int) -> str:
        end = self.model.paths.find(b'\x00', start)
        if end < 0:
            end = len(self.model.paths)
        return self.model.paths[start:end].decode('utf-8')

    def quat_matrix(self, quat: np.ndarray) -> np.ndarray:
        w, x, y, z = [float(v) for v in quat]
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

    def pose_matrix(self, pos: np.ndarray, quat: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
        matrix = self.quat_matrix(quat)
        if scale is not None:
            scaled = np.eye(4)
            scaled[0, 0], scaled[1, 1], scaled[2, 2] = [float(v) for v in scale]
            matrix = matrix @ scaled
        matrix[:3, 3] = [float(v) for v in pos]
        return matrix

    def flatten_matrix(self, matrix: np.ndarray) -> list[float]:
        return [float(v) for v in matrix.T.reshape(-1)]

    def geom_world_pose(self, geom_id: int) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, :3] = np.array(self.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        matrix[:3, 3] = np.array(self.data.geom_xpos[geom_id], dtype=float)
        return matrix

    def geom_source_matrix(self, geom_id: int) -> list[float]:
        mesh_id = int(self.model.geom_dataid[geom_id])
        geom_world = self.geom_world_pose(geom_id)
        mesh_local = self.pose_matrix(
            self.model.mesh_pos[mesh_id],
            self.model.mesh_quat[mesh_id],
            self.model.mesh_scale[mesh_id],
        )
        source_world = geom_world @ np.linalg.inv(mesh_local)
        return self.flatten_matrix(source_world)

    def geom_size(self, geom_id: int, kind: str) -> list[float]:
        size = np.array(self.model.geom_size[geom_id], dtype=float)
        if kind == 'plane':
            return [float(max(size[0] * 2, 1e-3)), float(max(size[1] * 2, 1e-3))]
        if kind in {'box', 'ellipsoid'}:
            return [float(max(size[0] * 2, 1e-3)), float(max(size[1] * 2, 1e-3)), float(max(size[2] * 2, 1e-3))]
        if kind == 'sphere':
            return [float(max(size[0], 1e-3))]
        if kind in {'capsule', 'cylinder'}:
            return [float(max(size[0], 1e-3)), float(max(size[1] * 2, 1e-3))]
        raise ValueError(f'Unsupported primitive kind: {kind}')

    def geom_primitive_matrix(self, geom_id: int) -> list[float]:
        return self.flatten_matrix(self.geom_world_pose(geom_id))

    def get_visual(self) -> list[dict]:
        poses = self.get_pose()
        return [{
            'name': name,
            'body': geom.body,
            'color': geom.color,
            'matrix': poses[name],
            **({ 'size': geom.size } if geom.size is not None else { }),
            **({ 'url': f'/python/zapdos/{self.sess}/asset/{name}' } if geom.abs_path is not None else { }),
        } for name, geom in self.visuals.items()]

    def get_pose(self) -> dict[str, list[float]]:
        # We serve source STL files, so remove MuJoCo's internal mesh normalization transform.
        return {
            name: self.geom_source_matrix(geom.geom_id) if geom.abs_path is not None else self.geom_primitive_matrix(geom.geom_id)
            for name, geom in self.visuals.items()
        }

    def on_call(self, method: str, args: tuple):
        if method == 'ping':
            return 'pong'
        elif method == 'get_visual':
            return self.get_visual()
        elif method == 'get_pose':
            return self.get_pose()
        return super().on_call(method, args)
    
    def step_once(self):
        if not self.msgs.full():
            self.msgs.put_nowait({ 'pose': self.get_pose() })
        mujoco.mj_step(self.model, self.data) # type: ignore
        return super().step_once()

sessions: dict[str, ZapdosSession] = { }
async def _name_(req: Request):
    sess = req.path_params['session']
    if sess not in sessions:
        sessions[sess] = ZapdosSession(sess)
    session = sessions[sess]

    action = req.path_params['action']
    name = req.path_params['name']
    if action == 'call':
        if name == 'start':
            return StreamingResponse(
                session.stream(),
                media_type="text/event-stream")
        else:
            args = await req.json()
            return await session.call(name, *args)

    elif action == 'asset':
        geom = session.visuals.get(name)
        if geom is None:
            raise HTTPException(status_code=404, detail='Geom not found')
        if geom.abs_path is None:
            raise HTTPException(status_code=400, detail='Geom has no asset file')
        path = geom.abs_path
        return FileResponse(path, media_type='model/stl')

    else:
        raise HTTPException(status_code=404, detail='Action not found')
