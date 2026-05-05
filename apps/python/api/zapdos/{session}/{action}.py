import asyncio
import io
import json
import queue
import traceback
from pathlib import Path

from PIL import Image
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from utils.camera_override import save_camera_overrides
from utils.rl_cameras import camera_name_to_index, image_topic
from utils.rl_bundle import DEFAULT_SCENE_USD, RenderBundle, ensure_render_bundle
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
from utils.zapdos_physics import ZapdosPhysics

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_ROBOT_USD = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"
RENDER_SIZE = (640, 480)
ROS_DT = 0.03
INIT_STREAM_HEARTBEAT_SEC = 1.0


class ZapdosSession(Session):
    @staticmethod
    async def create(sess: str, robot_usd: Path, scene_usd: Path):
        bundle = await asyncio.to_thread(ensure_render_bundle, robot_usd, scene_usd)
        return ZapdosSession(sess, bundle)

    def __init__(self, sess: str, bundle: RenderBundle) -> None:
        self.sess = sess
        self.bundle = bundle
        self.physics = ZapdosPhysics(
            sess,
            bundle,
            json.loads(bundle.body_map_json.read_text(encoding="utf-8")),
        )
        self.command_msgs: queue.Queue[dict] = queue.Queue(maxsize=8)
        self.command_subscribed = False
        self.camera_index = camera_name_to_index(bundle.cameras)
        self.last_frame_index = {camera.name: -1 for camera in bundle.cameras}
        super().__init__()
        self.timers.append(Timer(ROS_DT, self.send_sse))
        self.renderer = IsaacRenderer(sess, bundle, RENDER_SIZE[0], RENDER_SIZE[1], 30, True, 0)
        asyncio.run_coroutine_threadsafe(self.send_ros(), self.loop)

    def save_camera_override(self) -> dict[str, object]:
        snapshot = self.renderer.snapshot_cameras()
        path, saved = save_camera_overrides(snapshot)
        return {"ok": True, "saved": saved, "path": str(path)}

    def call_once(self, method: str, args: tuple):
        if method == "ping":
            return "pong"
        if method == "get_visual":
            return self.physics.get_visual()
        if method == "get_pose":
            return self.physics.get_pose()
        if method == "get_camera":
            return self.physics.get_camera()
        if method == "set_body_pose":
            return self.physics.set_body_pose(*args)
        if method == "save_camera_override":
            return self.save_camera_override()
        return super().call_once(method, args)

    def send_sse(self):
        if not self.msgs.full():
            self.msgs.put_nowait({ "pose": self.physics.get_pose() })

    async def send_ros(self):
        await self.renderer.wait_ready()
        while self.is_active():
            try:
                if not self.command_subscribed and bridge.conns:
                    await bridge.subscribe(JOINT_COMMAND_TOPIC, JOINT_STATE_TYPE, self.on_message)
                    self.command_subscribed = True
                if bridge.conns:
                    await bridge.call("publish", [JOINT_STATES_TOPIC, JOINT_STATE_TYPE, self.physics.joint_state_msg()])
                    await bridge.call("publish", [TF_RENDER_TOPIC, TF_RENDER_TYPE, tf_message(self.physics.model, self.physics.data)])
                    for topic, image_msg in self._image_messages():
                        await bridge.call("publish", [topic, IMAGE_TYPE, image_msg])
                await asyncio.sleep(ROS_DT)
            except Exception:
                traceback.print_exc()
                await asyncio.sleep(1)

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

    def _latest_joint_command(self) -> dict | None:
        latest = None
        while not self.command_msgs.empty():
            try:
                latest = self.command_msgs.get_nowait()
            except queue.Empty:
                break
        return latest

    def step_once(self):
        self.physics.apply_joint_command(self._latest_joint_command())
        self.physics.step()
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
        self.physics.close()
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

    future = _require_session_future(sess)
    session = _require_active_session(sess, future, await _await_session_future(sess, future))
    if action == "stream":
        return StreamingResponse(
            session.stream(),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    elif action == "ros":
        if name == "subscribe":
            topic, type = await req.json()
            await bridge.subscribe(topic, type, session.on_message)
            return {"ok": True}

    elif action == "call":
        args = await req.json()
        return await session.call(name, *args)

    elif action == "render":
        return StreamingResponse(
            session.render(_require_camera_name(session, name)),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    elif action == "asset":
        asset = session.physics.assets.get(name)
        if asset is None:
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(asset)
    raise HTTPException(status_code=404, detail="Action not found")
