import mujoco # type: ignore
import mujoco.viewer
import numpy as np
import asyncio
import os

from pathlib import Path
from dataclasses import dataclass

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from utils.session import Session
from utils.mujoco_tools import create_xml, flatten_matrix, geom_size, geom_world_pose, mesh_world_pose
from utils.mujoco_tools import decode_mesh_path, decode_texture_path
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

@dataclass
class ZapdosGeometry:
    name = ''
    kind = ''
    geom_id = 0
    body = ''
    mesh = ''
    texture = ''
    color: list[float] | None = None
    size:  list[float] | None = None

class ZapdosSession(Session):
    @staticmethod
    async def create(sess: str):
        # TODO: load file
        path = Path('../../deps/galaxea/object/r1pro/r1pro.usda').resolve()
        xml = await create_xml(str(path))
        return ZapdosSession(sess, xml)

    def __init__(self, sess: str, xml: Path) -> None:
        self.sess = sess

        asset_root = xml.parent
        self.model = mujoco.MjModel.from_xml_path(str(xml)) # type: ignore
        self.data = mujoco.MjData(self.model)               # type: ignore
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data) \
            if os.environ.get('DEBUG_MUJOCO_VIEWER') else None
        mujoco.mj_step(self.model, self.data)               # type: ignore

        self.geoms: dict[str, ZapdosGeometry] = { }
        self.assets: dict[str, Path] = { }
        for geom_id in range(self.model.ngeom):
            geom = ZapdosGeometry()
            geom.geom_id = geom_id
            geom.kind = PRIMITIVE_TYPES.get(int(self.model.geom_type[geom_id])) or ''
            if geom.kind == 'mesh':
                mesh_id = int(self.model.geom_dataid[geom_id])
                mesh_rel = decode_mesh_path(self.model, mesh_id)
                geom.mesh = mesh_rel.name
                self.assets[geom.mesh] = (asset_root / mesh_rel).resolve()
                mat_id = int(self.model.geom_matid[geom_id])
                tex_id = int(self.model.mat_texid[mat_id, 0]) if mat_id >= 0 else -1
                tex_rel = decode_texture_path(self.model, tex_id)
                geom.texture = tex_rel.name
                print(geom.mesh, mat_id, tex_id, geom.texture)
                self.assets[geom.texture] = (asset_root / tex_rel).resolve()
            elif geom.kind:
                geom.size = geom_size(self.model, geom_id, geom.kind)
            else:
                continue

            body_id = int(self.model.geom_bodyid[geom_id])
            geom.body = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or 'world' # type: ignore
            geom.color = [float(value) for value in self.model.geom_rgba[geom_id]]
            geom.name = f'geom-{geom_id}'
            self.geoms[geom.name] = geom

        super().__init__()

    def get_visual(self) -> list[dict]:
        poses = self.get_pose()
        return [{
            'name': name,
            'kind': geom.kind,
            'color': geom.color,
            'matrix': poses[name],
            **({ 'size': geom.size } if geom.size is not None else { }),
            **({ 'mesh' : f'/python/zapdos/{self.sess}/asset/{geom.mesh}'  } if geom.mesh else { }),
            **({ 'texture': f'/python/zapdos/{self.sess}/asset/{geom.texture}' } if geom.texture else { }),
        } for name, geom in self.geoms.items()]

    def get_pose(self) -> dict[str, list[float]]:
        poses: dict[str, list[float]] = { }
        for name, geom in self.geoms.items():
            pose = \
                mesh_world_pose(self.model, self.data, geom.geom_id) if geom.mesh else \
                geom_world_pose(self.data, geom.geom_id)
            poses[name] = flatten_matrix(pose)
        return poses

    def get_camera(self) -> dict[str, list[float]]:
        cameras: dict[str, list[float]] = {}
        for cam_id in range(self.model.ncam):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id) # type: ignore
            if name is None:
                continue
            mat4 = np.eye(4)
            mat4[:3, :3] = self.data.cam_xmat[cam_id].reshape(3, 3)
            mat4[:3, 3] = self.data.cam_xpos[cam_id]
            cameras[name] = flatten_matrix(mat4)
        return cameras

    def call_once(self, method: str, args: tuple):
        if method == 'ping':
            return 'pong'
        elif method == 'get_visual':
            return self.get_visual()
        elif method == 'get_pose':
            return self.get_pose()
        elif method == 'get_camera':
            return self.get_camera()
        return super().call_once(method, args)
    
    def step_once(self):
        if not self.msgs.full():
            self.msgs.put_nowait({ 'pose': self.get_pose(), 'camera': self.get_camera() })
        mujoco.mj_step(self.model, self.data) # type: ignore
        if self.viewer:
            self.viewer.sync()
        return super().step_once()
    
    def on_message(self, topic: str, msg):
        self.msgs.put_nowait({ 'topic': topic, 'msg': msg })
    
    def destroy(self):
        for topic in bridge.subs:
            bridge.unsubscribe(topic, self.on_message)
        return super().destroy()

sessions: dict[str, asyncio.Future[ZapdosSession]] = { }
async def _name_(req: Request):
    sess = req.path_params['session']
    if sess not in sessions:
        sessions[sess] = asyncio.create_task(ZapdosSession.create(sess))
    session = await sessions[sess]

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
        asset = session.assets.get(name)
        if asset is None:
            raise HTTPException(status_code=404, detail='Asset not found')
        return FileResponse(asset)

    else:
        raise HTTPException(status_code=404, detail='Action not found')
