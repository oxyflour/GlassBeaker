import asyncio
import io
import os
import queue
import traceback
from dataclasses import dataclass
from pathlib import Path

import mujoco  # type: ignore
import mujoco.viewer
import numpy as np
from PIL import Image
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from utils.mujoco_tools import decode_mesh_path, decode_texture_path
from utils.mujoco_tools import flatten_matrix, geom_size, geom_world_pose, mesh_world_pose
from utils.rl_bundle import DEFAULT_SCENE_USD, ensure_render_bundle
from utils.ros_bridge import bridge
from utils.ros_worker import acquire_ros_worker, release_ros_worker, wait_for_ros_bridge
from utils.session import Session, Timer
from utils.sim_env import (
    IMAGE_TOPIC,
    IMAGE_TYPE,
    JOINT_COMMAND_TOPIC,
    JOINT_STATES_TOPIC,
    JOINT_STATE_TYPE,
    TF_RENDER_TOPIC,
    TF_RENDER_TYPE,
    IsaacRenderer,
    mjpeg_chunk,
    placeholder_jpeg,
    tf_message,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_ROBOT_USD = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"
RENDER_SIZE = (640, 480)
ROS_DT = 0.03

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
    name = ""
    kind = ""
    geom_id = 0
    body = ""
    mesh = ""
    texture = ""
    color: list[float] | None = None
    size: list[float] | None = None


class ZapdosSession(Session):
    @staticmethod
    async def create(sess: str, robot_usd: Path, scene_usd: Path):
        bundle = ensure_render_bundle(robot_usd, scene_usd)
        return ZapdosSession(sess, bundle)

    def __init__(self, sess: str, bundle) -> None:
        self.sess = sess
        self.bundle = bundle
        asset_root = bundle.mjcf.parent
        self.model = mujoco.MjModel.from_xml_path(str(bundle.mjcf))  # type: ignore
        self.data = mujoco.MjData(self.model)  # type: ignore
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data) if os.environ.get("DEBUG_MUJOCO_VIEWER") else None
        self.assets: dict[str, Path] = {}
        self.geoms = self._build_geometry(asset_root)
        self.command_msgs: queue.Queue[dict] = queue.Queue(maxsize=8)
        self.command_subscribed = False
        self.last_frame_index = -1
        self.actuator_name_to_id = self._actuator_map()
        self.joint_name_to_actuator = self._joint_command_map()
        self._seed_position_ctrl()
        mujoco.mj_forward(self.model, self.data)  # type: ignore
        super().__init__()
        acquire_ros_worker()
        self.timers.append(Timer(ROS_DT, self.send_sse))
        self.renderer = IsaacRenderer(sess, bundle, RENDER_SIZE[0], RENDER_SIZE[1], 30, True, 0)
        asyncio.run_coroutine_threadsafe(self.send_ros(), self.loop)

    def _build_geometry(self, asset_root: Path) -> dict[str, ZapdosGeometry]:
        geoms: dict[str, ZapdosGeometry] = {}
        for geom_id in range(self.model.ngeom):
            geom = ZapdosGeometry()
            geom.geom_id = geom_id
            geom.kind = PRIMITIVE_TYPES.get(int(self.model.geom_type[geom_id])) or ""
            if geom.kind == "mesh":
                mesh_id = int(self.model.geom_dataid[geom_id])
                mesh_rel = decode_mesh_path(self.model, mesh_id)
                geom.mesh = mesh_rel.name
                self.assets[geom.mesh] = (asset_root / mesh_rel).resolve()
                mat_id = int(self.model.geom_matid[geom_id])
                tex_id = int(self.model.mat_texid[mat_id, 0]) if mat_id >= 0 else -1
                tex_rel = decode_texture_path(self.model, tex_id)
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
        mapping: dict[str, int] = {}
        for actuator_id in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)  # type: ignore
            if name:
                mapping[name] = actuator_id
        return mapping

    def _joint_command_map(self) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id][0])
            if joint_id < 0:
                continue
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)  # type: ignore
            if joint_name:
                mapping[joint_name] = actuator_id
        return mapping

    def _seed_position_ctrl(self) -> None:
        self.data.ctrl[:] = 0
        for joint_name, actuator_id in self.joint_name_to_actuator.items():
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or "" # type: ignore
            if not actuator_name.endswith("_position"):
                continue
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)  # type: ignore
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            self.data.ctrl[actuator_id] = float(self.data.qpos[qpos_adr])

    def get_visual(self) -> list[dict]:
        poses = self.get_pose()
        return [{
            "name": name,
            "kind": geom.kind,
            "color": geom.color,
            "matrix": poses[name],
            **({"size": geom.size} if geom.size is not None else {}),
            **({"mesh": f"/python/zapdos/{self.sess}/asset/{geom.mesh}"} if geom.mesh else {}),
            **({"texture": f"/python/zapdos/{self.sess}/asset/{geom.texture}"} if geom.texture else {}),
        } for name, geom in self.geoms.items()]

    def get_pose(self) -> dict[str, list[float]]:
        poses: dict[str, list[float]] = {}
        for name, geom in self.geoms.items():
            pose = mesh_world_pose(self.model, self.data, geom.geom_id) if geom.mesh else geom_world_pose(self.data, geom.geom_id)
            poses[name] = flatten_matrix(pose)
        return poses

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

    def call_once(self, method: str, args: tuple):
        if method == "ping":
            return "pong"
        if method == "get_visual":
            return self.get_visual()
        if method == "get_pose":
            return self.get_pose()
        if method == "get_camera":
            return self.get_camera()
        return super().call_once(method, args)

    def send_sse(self):
        if not self.msgs.full():
            self.msgs.put_nowait({"pose": self.get_pose(), "camera": self.get_camera()})

    async def send_ros(self):
        await self.renderer.wait_ready()
        while self.is_active():
            try:
                if not bridge.conns:
                    await wait_for_ros_bridge()
                if not self.command_subscribed and bridge.conns:
                    await bridge.subscribe(JOINT_COMMAND_TOPIC, JOINT_STATE_TYPE, self.on_message)
                    self.command_subscribed = True
                if bridge.conns:
                    await bridge.call("publish", [JOINT_STATES_TOPIC, JOINT_STATE_TYPE, self._joint_state_msg()])
                    await bridge.call("publish", [TF_RENDER_TOPIC, TF_RENDER_TYPE, tf_message(self.model, self.data)])
                    image_msg = self._image_msg()
                    if image_msg is not None:
                        await bridge.call("publish", [IMAGE_TOPIC, IMAGE_TYPE, image_msg])
                await asyncio.sleep(ROS_DT)
            except Exception:
                traceback.print_exc()
                await asyncio.sleep(1)

    def _joint_state_msg(self) -> dict:
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

    def _image_msg(self) -> dict | None:
        frame_state = self.renderer.read()
        if frame_state is None:
            return None
        index, frame = frame_state
        if index == self.last_frame_index:
            return None
        self.last_frame_index = index
        return {
            "header": {"frame_id": "main_camera"},
            "height": int(frame.shape[0]),
            "width": int(frame.shape[1]),
            "encoding": "rgb8",
            "is_bigendian": 0,
            "step": int(frame.shape[1] * 3),
            "data": frame.tobytes(),
        }

    def _drain_joint_commands(self) -> None:
        latest = None
        while not self.command_msgs.empty():
            try:
                latest = self.command_msgs.get_nowait()
            except queue.Empty:
                break
        if latest is None:
            return
        ctrl = np.copy(self.data.ctrl)
        for name, pos in zip(latest.get("name") or [], latest.get("position") or []):
            actuator_id = self.actuator_name_to_id.get(name)
            if actuator_id is None:
                actuator_id = self.joint_name_to_actuator.get(name)
            if actuator_id is not None:
                ctrl[actuator_id] = float(pos)
        self.data.ctrl[:] = ctrl

    def step_once(self):
        self._drain_joint_commands()
        mujoco.mj_step(self.model, self.data)  # type: ignore
        if self.viewer:
            self.viewer.sync()
        return super().step_once()

    def on_message(self, topic: str, msg):
        if topic == JOINT_COMMAND_TOPIC:
            while self.command_msgs.full():
                try:
                    self.command_msgs.get_nowait()
                except queue.Empty:
                    break
            self.command_msgs.put_nowait(msg)
            return
        if not self.msgs.full():
            self.msgs.put_nowait({"topic": topic, "msg": msg})

    async def render(self):
        while self.is_active():
            while not self.renderer.ready:
                yield mjpeg_chunk(placeholder_jpeg(RENDER_SIZE[0], RENDER_SIZE[1], "Waiting"))
                await asyncio.sleep(1)
            try:
                frame_state = self.renderer.read()
                if frame_state is None:
                    await asyncio.sleep(ROS_DT)
                    continue
                _, frame = frame_state
                data = io.BytesIO()
                Image.fromarray(frame).save(data, format="JPEG", quality=80)
                yield mjpeg_chunk(data.getvalue())
                await asyncio.sleep(ROS_DT)
            except Exception:
                traceback.print_exc()
                await asyncio.sleep(1)

    def destroy(self):
        for topic in list(bridge.subs):
            bridge.unsubscribe(topic, self.on_message)
        self.renderer.close()
        if self.viewer:
            self.viewer.close()
        release_ros_worker()
        return super().destroy()


def _input_path(req: Request, key: str, default: Path) -> Path:
    raw = req.query_params.get(key, "").strip()
    path = default if not raw else (Path(raw) if Path(raw).is_absolute() else (REPO_ROOT / raw))
    path = path.resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{key} not found: {path}")
    return path


sessions: dict[str, asyncio.Future[ZapdosSession]] = {}


async def _name_(req: Request):
    sess = req.path_params["session"]
    if sess not in sessions:
        robot_usd = _input_path(req, "robot_usd", DEFAULT_ROBOT_USD)
        scene_usd = _input_path(req, "scene_usd", DEFAULT_SCENE_USD)
        sessions[sess] = asyncio.create_task(ZapdosSession.create(sess, robot_usd, scene_usd))
    session = await sessions[sess]
    action = req.path_params["action"]
    name = req.path_params["name"]
    if action == "call":
        if name == "start":
            return StreamingResponse(session.stream(), media_type="text/event-stream")
        if name == "subscribe":
            topic, type = await req.json()
            await bridge.subscribe(topic, type, session.on_message)
            return {"ok": True}
        args = await req.json()
        return await session.call(name, *args)
    if action == "render":
        return StreamingResponse(session.render(), media_type="multipart/x-mixed-replace; boundary=frame")
    if action == "asset":
        asset = session.assets.get(name)
        if asset is None:
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(asset)
    raise HTTPException(status_code=404, detail="Action not found")
