import mujoco # type: ignore
import numpy as np
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from utils.session import Session
from utils.mujoco_tools import create_xml, decode_path, flatten_matrix, geom_size, geom_world_pose, pose_matrix
from utils.ros_bridge import bridge

PRIMITIVE_TYPES = {
    int(mujoco.mjtGeom.mjGEOM_PLANE)    : 'plane',     # type: ignore
    int(mujoco.mjtGeom.mjGEOM_SPHERE)   : 'sphere',    # type: ignore
    int(mujoco.mjtGeom.mjGEOM_CAPSULE)  : 'capsule',   # type: ignore
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): 'ellipsoid', # type: ignore
    int(mujoco.mjtGeom.mjGEOM_CYLINDER) : 'cylinder',  # type: ignore
    int(mujoco.mjtGeom.mjGEOM_BOX)      : 'box',       # type: ignore
    int(mujoco.mjtGeom.mjGEOM_MESH)     : 'mesh',      # type: ignore
}

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
            abs_path: Path | None = None
            size: list[float] | None = None

            kind = PRIMITIVE_TYPES.get(int(self.model.geom_type[geom_id]))
            if kind == 'mesh':
                mesh_id = int(self.model.geom_dataid[geom_id])
                rel_path = decode_path(self.model, mesh_id)
                kind = Path(rel_path).suffix.lower().lstrip('.')
                abs_path = (asset_root / rel_path).resolve()
            elif kind:
                size = geom_size(self.model, geom_id, kind)
            else:
                continue

            body_id = int(self.model.geom_bodyid[geom_id])
            body = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or 'world' # type: ignore
            name = f'geom-{geom_id}.{kind}'
            color = [float(value) for value in self.model.geom_rgba[geom_id]]
            self.visuals[name] = ZapdosGeometry(name, geom_id, body, color, size, abs_path)

        super().__init__()

    def geom_source_matrix(self, geom_id: int):
        geom_world = geom_world_pose(self.data, geom_id)
        mesh_id = int(self.model.geom_dataid[geom_id])
        mesh_local = pose_matrix(
            self.model.mesh_pos[mesh_id],
            self.model.mesh_quat[mesh_id],
            self.model.mesh_scale[mesh_id],
        )
        return geom_world @ np.linalg.inv(mesh_local)

    def geom_primitive_matrix(self, geom_id: int):
        return geom_world_pose(self.data, geom_id)

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
        return {
            name: flatten_matrix( \
                  self.geom_source_matrix(geom.geom_id) if geom.abs_path is not None else \
                  self.geom_primitive_matrix(geom.geom_id))
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
    
    def on_message(self, topic: str, msg):
        print(topic)

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
        elif name == 'subscribe':
            topic, type = await req.json()
            await bridge.subscribe(topic, type, session.on_message)
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
