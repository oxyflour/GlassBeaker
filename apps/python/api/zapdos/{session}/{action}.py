import asyncio
import io
import json
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
from utils.mujoco_tools import body_world_pose, flatten_matrix, geom_size, geom_world_pose, mesh_world_pose
from utils.camera_override import save_camera_overrides
from utils.rl_cameras import camera_name_to_index, image_topic
from utils.rl_bundle import DEFAULT_SCENE_USD, ensure_render_bundle
from utils.ros_bridge import bridge
from utils.session import Session, Timer
from utils.sim_env import (
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
from utils.zapdos_scene_visuals import SceneVisuals, serialize_body, serialize_mesh

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_ROBOT_USD = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"
RENDER_SIZE = (640, 480)
ROS_DT = 0.03
INIT_STREAM_HEARTBEAT_SEC = 1.0
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
        bundle = await asyncio.to_thread(ensure_render_bundle, robot_usd, scene_usd)
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
        self.body_map = json.loads(bundle.body_map_json.read_text(encoding="utf-8"))
        self.body_labels = {name: path.rsplit("/", 1)[-1] for name, path in self.body_map.items()}
        self.editable_body_names = {
            name for name, path in self.body_map.items() if not str(path).startswith("MyRobot/")
        }
        self.command_msgs: queue.Queue[dict] = queue.Queue(maxsize=8)
        self.command_subscribed = False
        self.camera_index = camera_name_to_index(bundle.cameras)
        self.last_frame_index = {camera.name: -1 for camera in bundle.cameras}
        self.actuator_name_to_id = self._actuator_map()
        self.joint_name_to_actuator = self._joint_command_map()
        self._seed_position_ctrl()
        mujoco.mj_forward(self.model, self.data)  # type: ignore
        super().__init__()
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
                if 'collisions' in mesh_rel.name:
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

    def _body_matrices(self) -> dict[str, np.ndarray]:
        poses: dict[str, np.ndarray] = {}
        for body_id in range(1, self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)  # type: ignore
            if name:
                poses[name] = body_world_pose(self.data, body_id)
        return poses

    def _body_freejoint_id(self, body_id: int) -> int | None:
        joint_start = int(self.model.body_jntadr[body_id])
        joint_count = int(self.model.body_jntnum[body_id])
        for offset in range(joint_count):
            joint_id = joint_start + offset
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:  # type: ignore
                return joint_id
        return None

    def _set_freejoint_pose(
        self,
        body_id: int,
        pos: np.ndarray,
        quat: np.ndarray,
        zero_velocity: bool = True,
    ) -> bool:
        freejoint_id = self._body_freejoint_id(body_id)
        if freejoint_id is None:
            return False
        qpos_adr = int(self.model.jnt_qposadr[freejoint_id])
        self.data.qpos[qpos_adr:qpos_adr + 3] = np.asarray(pos, dtype=float)
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = np.asarray(quat, dtype=float)
        if zero_velocity:
            qvel_adr = int(self.model.jnt_dofadr[freejoint_id])
            self.data.qvel[qvel_adr:qvel_adr + 6] = 0.0
        return True

    def _mesh_anchor_body(self, body_name: str, body_matrices: dict[str, np.ndarray]) -> str | None:
        if body_name not in body_matrices:
            return None
        current_name: str | None = body_name
        while current_name is not None:
            if current_name in self.editable_body_names:
                return current_name
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, current_name)  # type: ignore
            parent_id = int(self.model.body_parentid[body_id])
            if parent_id <= 0:
                break
            current_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, parent_id)  # type: ignore
        return body_name

    def get_visual(self) -> SceneVisuals:
        body_matrices = self._body_matrices()
        bodies = [
            serialize_body(
                name,
                self.body_map.get(name, name),
                name in self.editable_body_names,
                flatten_matrix(matrix),
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
        return {
            name: flatten_matrix(matrix)
            for name, matrix in self._body_matrices().items()
        }

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

    def save_camera_override(self) -> dict[str, object]:
        snapshot = self.renderer.snapshot_cameras()
        path, saved = save_camera_overrides(snapshot)
        return {"ok": True, "saved": saved, "path": str(path)}

    def set_body_pose(self, body: str, pos: list[float], quat: list[float]) -> dict[str, object]:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)  # type: ignore
        if body_id < 0:
            raise HTTPException(status_code=404, detail=f"Body not found: {body}")
        if body not in self.editable_body_names:
            raise HTTPException(status_code=403, detail=f"Body is not editable: {body}")
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

    def call_once(self, method: str, args: tuple):
        if method == "ping":
            return "pong"
        if method == "get_visual":
            return self.get_visual()
        if method == "get_pose":
            return self.get_pose()
        if method == "get_camera":
            return self.get_camera()
        if method == "set_body_pose":
            return self.set_body_pose(*args)
        if method == "save_camera_override":
            return self.save_camera_override()
        return super().call_once(method, args)

    def send_sse(self):
        if not self.msgs.full():
            self.msgs.put_nowait({ "pose": self.get_pose() })

    async def send_ros(self):
        await self.renderer.wait_ready()
        while self.is_active():
            try:
                if not self.command_subscribed and bridge.conns:
                    await bridge.subscribe(JOINT_COMMAND_TOPIC, JOINT_STATE_TYPE, self.on_message)
                    self.command_subscribed = True
                if bridge.conns:
                    await bridge.call("publish", [JOINT_STATES_TOPIC, JOINT_STATE_TYPE, self._joint_state_msg()])
                    await bridge.call("publish", [TF_RENDER_TOPIC, TF_RENDER_TYPE, tf_message(self.model, self.data)])
                    for topic, image_msg in self._image_messages():
                        await bridge.call("publish", [topic, IMAGE_TYPE, image_msg])
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

    def _image_messages(self) -> list[tuple[str, dict]]:
        messages: list[tuple[str, dict]] = []
        for camera in self.bundle.cameras:
            frame_state = self.renderer.read(camera.name)
            if frame_state is None:
                continue
            index, frame = frame_state
            if index == self.last_frame_index[camera.name]:
                continue
            self.last_frame_index[camera.name] = index
            messages.append((image_topic(camera.name), {
                "header": {"frame_id": camera.frame_id},
                "height": int(frame.shape[0]),
                "width": int(frame.shape[1]),
                "encoding": "rgb8",
                "is_bigendian": 0,
                "step": int(frame.shape[1] * 3),
                "data": frame.tobytes(),
            }))
        return messages

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

    async def render(self, camera_name: str):
        while self.is_active():
            while not self.renderer.ready:
                yield mjpeg_chunk(placeholder_jpeg(RENDER_SIZE[0], RENDER_SIZE[1], "Waiting"))
                await asyncio.sleep(1)
            try:
                frame_state = self.renderer.read(camera_name)
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
        return super().destroy()


def _input_path(req: Request, key: str, default: Path) -> Path:
    raw = req.query_params.get(key, "").strip()
    path = default if not raw else (Path(raw) if Path(raw).is_absolute() else (REPO_ROOT / raw))
    path = path.resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{key} not found: {path}")
    return path


sessions: dict[str, asyncio.Future[ZapdosSession]] = {}


def _evict_session_future(sess: str, future: asyncio.Future[ZapdosSession]) -> None:
    if sessions.get(sess) is future:
        sessions.pop(sess, None)


def _session_future_state(sess: str) -> tuple[asyncio.Future[ZapdosSession] | None, str | None]:
    future = sessions.get(sess)
    if future is None:
        return None, None
    if future.cancelled():
        _evict_session_future(sess, future)
        return None, "missing"
    if not future.done():
        return future, None
    if future.exception() is not None:
        _evict_session_future(sess, future)
        return None, "missing"
    if not future.result().is_active():
        _evict_session_future(sess, future)
        return None, "expired"
    return future, None


def _get_or_create_session_future(req: Request, sess: str) -> asyncio.Future[ZapdosSession]:
    future, _ = _session_future_state(sess)
    if future is not None:
        return future

    robot_usd = _input_path(req, "robot_usd", DEFAULT_ROBOT_USD)
    scene_usd = _input_path(req, "scene_usd", DEFAULT_SCENE_USD)
    future = asyncio.create_task(ZapdosSession.create(sess, robot_usd, scene_usd))
    sessions[sess] = future
    return future


def _require_session_future(sess: str) -> asyncio.Future[ZapdosSession]:
    future, reason = _session_future_state(sess)
    if future is not None:
        return future
    detail = "Session expired" if reason == "expired" else "Session not initialized"
    raise HTTPException(status_code=409, detail=detail)


async def _await_session_future(sess: str, future: asyncio.Future[ZapdosSession]) -> ZapdosSession:
    try:
        return await future
    except Exception:
        _evict_session_future(sess, future)
        raise


def _bootstrap_error_message(exc: Exception) -> str:
    message = str(exc).strip()
    return message or "Session bootstrap failed"


async def _init_stream(sess: str, future: asyncio.Future[ZapdosSession]):
    yield "data: loading\n\n"
    while not future.done():
        done, _ = await asyncio.wait({future}, timeout=INIT_STREAM_HEARTBEAT_SEC)
        if done:
            break
        yield "data: loading: preparing render bundle\n\n"

    try:
        await _await_session_future(sess, future)
    except Exception as exc:
        yield f"data: error: {_bootstrap_error_message(exc)}\n\n"
        return

    yield "data: started\n\n"


def _require_camera_name(session: ZapdosSession, camera_name: str) -> str:
    if camera_name not in session.camera_index:
        raise HTTPException(status_code=404, detail=f"Camera not found: {camera_name}")
    return camera_name


def _require_active_session(sess: str, future: asyncio.Future[ZapdosSession], session: ZapdosSession) -> ZapdosSession:
    if session.is_active():
        return session
    _evict_session_future(sess, future)
    raise HTTPException(status_code=409, detail="Session expired")


async def _name_(req: Request):
    sess = req.path_params["session"]
    action = req.path_params["action"]
    name = req.path_params["name"]
    if action == "init":
        future = _get_or_create_session_future(req, sess)
        return StreamingResponse(
            _init_stream(sess, future),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    if action not in {"call", "render", "asset"}:
        raise HTTPException(status_code=404, detail="Action not found")

    future = _require_session_future(sess)
    session = _require_active_session(sess, future, await _await_session_future(sess, future))
    if action == "call":
        if name == "start":
            return StreamingResponse(
                session.stream(),
                media_type="text/event-stream; charset=utf-8",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        if name == "subscribe":
            topic, type = await req.json()
            await bridge.subscribe(topic, type, session.on_message)
            return {"ok": True}
        args = await req.json()
        return await session.call(name, *args)
    if action == "render":
        return StreamingResponse(
            session.render(_require_camera_name(session, name)),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )
    if action == "asset":
        asset = session.assets.get(name)
        if asset is None:
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(asset)
    raise HTTPException(status_code=404, detail="Action not found")
