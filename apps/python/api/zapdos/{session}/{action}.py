import asyncio
from copy import deepcopy
import io
import json
import queue
import traceback
from pathlib import Path

import mujoco  # type: ignore
import numpy as np
from PIL import Image
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from utils.camera_override import save_camera_overrides
from utils.genie_sim_runtime import resolve_assets_root
from utils.mujoco_tools import body_world_pose, flatten_matrix
from utils.rl_cameras import camera_name_to_index, image_topic
from utils.rl_bundle import DEFAULT_SCENE_USD, RenderBundle, ensure_render_bundle
from utils.ros_bridge import bridge
from utils.session import Session, Timer
from utils.session_registry import AsyncSessionRegistry
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
from utils.zapdos_asset_library import asset_local_bounds, resolve_asset_record
from utils.zapdos_overlay import (
    default_overlay_state,
    load_overlay_state,
    overlay_body_name,
    save_overlay_state,
    scene_revision,
)
from utils.zapdos_overlay_scene import normalize_placement, write_overlay_scene
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
        self.robot_usd = getattr(bundle, "robot_usd", DEFAULT_ROBOT_USD)
        self.base_scene_usd = getattr(bundle, "scene_usd", DEFAULT_SCENE_USD)
        self.session_dir = REPO_ROOT / "apps" / "python" / "tmp" / "zapdos" / sess
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_path = self.session_dir / "overlay.json"
        self.composed_scene_usd = self.session_dir / "scene-overlay.usda"
        loaded_overlay = load_overlay_state(self.overlay_path)
        self.overlay_state = default_overlay_state(loaded_overlay.get("assets_root"))
        if loaded_overlay["instances"] or loaded_overlay["pose_overrides"]:
            save_overlay_state(self.overlay_path, self.overlay_state)
        self.scene_revision = scene_revision(self.base_scene_usd, self.overlay_state)
        self.rebuilding_scene = False
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

    def _next_instance_id(self, asset_id: str) -> str:
        existing = {item["id"] for item in self.overlay_state["instances"]}
        index = 1
        while True:
            instance_id = f"{asset_id}_{index:02d}"
            if instance_id not in existing:
                return instance_id
            index += 1

    def _build_support_infos(self) -> dict[str, dict[str, float]]:
        infos: dict[str, dict[str, float]] = {}
        assets_root = resolve_assets_root(self.overlay_state.get("assets_root"))
        instance_by_body = {
            overlay_body_name(item["id"]): item
            for item in self.overlay_state["instances"]
        }
        for body in self.physics.editable_body_names:
            body_id = mujoco.mj_name2id(self.physics.model, mujoco.mjtObj.mjOBJ_BODY, body)  # type: ignore
            top_z = float(body_world_pose(self.physics.data, body_id)[2, 3])
            instance = instance_by_body.get(body)
            if instance is not None:
                bounds = asset_local_bounds(assets_root / instance["url"])
                top_z += float(bounds["max"][2])
            infos[body] = {"top_z": top_z}
        return infos

    def _swap_runtime_bundle(self, bundle: RenderBundle, overlay_state) -> None:
        snapshot_qpos = np.copy(self.physics.data.qpos)
        snapshot_ctrl = np.copy(self.physics.data.ctrl)
        body_map = json.loads(bundle.body_map_json.read_text(encoding="utf-8"))
        new_physics = ZapdosPhysics(self.sess, bundle, body_map)
        count = min(len(snapshot_qpos), len(new_physics.data.qpos))
        if count:
            new_physics.data.qpos[:count] = snapshot_qpos[:count]
        ctrl_count = min(len(snapshot_ctrl), len(new_physics.data.ctrl))
        if ctrl_count:
            new_physics.data.ctrl[:ctrl_count] = snapshot_ctrl[:ctrl_count]
        mujoco.mj_forward(new_physics.model, new_physics.data)  # type: ignore
        for body, pose in overlay_state["pose_overrides"].items():
            if body in new_physics.editable_body_names:
                new_physics.set_body_pose(body, pose["pos"], pose["quat"])
        new_renderer = IsaacRenderer(self.sess, bundle, RENDER_SIZE[0], RENDER_SIZE[1], 30, True, 0)
        old_renderer = self.renderer
        self.bundle = bundle
        self.physics = new_physics
        self.camera_index = camera_name_to_index(bundle.cameras)
        self.last_frame_index = {camera.name: -1 for camera in bundle.cameras}
        self.renderer = new_renderer
        old_renderer.close()

    def list_scene_bodies(self) -> dict[str, object]:
        support_infos = self._build_support_infos()
        items = []
        for body in sorted(self.physics.editable_body_names):
            body_id = mujoco.mj_name2id(self.physics.model, mujoco.mjtObj.mjOBJ_BODY, body)  # type: ignore
            items.append({
                "body": body,
                "label": self.physics.body_labels.get(body, body),
                "matrix": flatten_matrix(body_world_pose(self.physics.data, body_id)),
                "support": support_infos.get(body),
            })
        return {"items": items, "scene_revision": self.scene_revision}

    def set_body_pose(self, body: str, pos: list[float], quat: list[float]) -> dict[str, object]:
        result = self.physics.set_body_pose(body, pos, quat)
        if hasattr(self, "overlay_state") and hasattr(self, "overlay_path"):
            quat_vec = np.array(quat, dtype=float)
            quat_norm = np.linalg.norm(quat_vec)
            self.overlay_state["pose_overrides"][body] = {
                "pos": list(pos),
                "quat": (quat_vec / quat_norm).tolist(),
            }
            save_overlay_state(self.overlay_path, self.overlay_state)
        return result

    def add_asset_to_scene(self, asset_id: str, motion: str, placement: dict[str, object]) -> dict[str, object]:
        if self.rebuilding_scene:
            raise HTTPException(status_code=409, detail="Scene rebuild already in progress")
        if motion not in {"static", "dynamic"}:
            raise HTTPException(status_code=400, detail=f"Unsupported motion: {motion}")
        try:
            normalized_placement = normalize_placement(placement)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        asset = resolve_asset_record(asset_id, self.overlay_state.get("assets_root"))
        instance_id = self._next_instance_id(asset_id)
        body = overlay_body_name(instance_id)

        def mutate(state):
            state["assets_root"] = state.get("assets_root") or str(resolve_assets_root(self.overlay_state.get("assets_root")))
            state["instances"].append({
                "id": instance_id,
                "asset_id": asset["asset_id"],
                "url": asset["url"],
                "motion": motion,
                "placement": normalized_placement,
            })

        revision = self._rebuild_overlay_runtime(mutate)
        return {"ok": True, "instance_id": instance_id, "body": body, "scene_revision": revision}

    def remove_asset_from_scene(self, instance_id: str) -> dict[str, object]:
        if self.rebuilding_scene:
            raise HTTPException(status_code=409, detail="Scene rebuild already in progress")
        if not any(item["id"] == instance_id for item in self.overlay_state["instances"]):
            raise HTTPException(status_code=404, detail=f"Overlay instance not found: {instance_id}")
        body = overlay_body_name(instance_id)

        def mutate(state):
            state["instances"] = [item for item in state["instances"] if item["id"] != instance_id]
            state["pose_overrides"].pop(body, None)

        revision = self._rebuild_overlay_runtime(mutate)
        return {"ok": True, "instance_id": instance_id, "scene_revision": revision}

    def _rebuild_overlay_runtime(self, mutate_overlay) -> str:
        previous_overlay = deepcopy(self.overlay_state)
        previous_revision = self.scene_revision
        self.rebuilding_scene = True
        try:
            next_overlay = deepcopy(previous_overlay)
            mutate_overlay(next_overlay)
            save_overlay_state(self.overlay_path, next_overlay)
            support_infos = self._build_support_infos()
            assets_root = resolve_assets_root(next_overlay.get("assets_root"))
            bounds_by_instance = {
                item["id"]: asset_local_bounds(assets_root / item["url"])
                for item in next_overlay["instances"]
            }
            write_overlay_scene(
                self.composed_scene_usd,
                self.base_scene_usd,
                assets_root,
                next_overlay,
                support_infos=support_infos,
                asset_bounds_by_instance=bounds_by_instance,
            )
            bundle = ensure_render_bundle(self.robot_usd, self.composed_scene_usd)
            self._swap_runtime_bundle(bundle, next_overlay)
            self.overlay_state = next_overlay
            self.scene_revision = scene_revision(self.base_scene_usd, next_overlay)
            if not self.msgs.full():
                self.msgs.put_nowait({"scene_revision": self.scene_revision})
            return self.scene_revision
        except Exception:
            self.overlay_state = previous_overlay
            self.scene_revision = previous_revision
            save_overlay_state(self.overlay_path, previous_overlay)
            raise
        finally:
            self.rebuilding_scene = False

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
        if method == "list_scene_bodies":
            return self.list_scene_bodies()
        if method == "add_asset_to_scene":
            return self.add_asset_to_scene(*args)
        if method == "remove_asset_from_scene":
            return self.remove_asset_from_scene(*args)
        if method == "set_body_pose":
            return self.set_body_pose(*args)
        if method == "save_camera_override":
            return self.save_camera_override()
        return super().call_once(method, args)

    def send_sse(self):
        if self.rebuilding_scene:
            return
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


session_registry = AsyncSessionRegistry[ZapdosSession]()
sessions = session_registry.sessions


def _get_or_create_session_future(req: Request, sess: str) -> asyncio.Future[ZapdosSession]:
    robot_usd = _input_path(req, "robot_usd", DEFAULT_ROBOT_USD)
    scene_usd = _input_path(req, "scene_usd", DEFAULT_SCENE_USD)
    return session_registry.get_or_create(
        sess,
        lambda: ZapdosSession.create(sess, robot_usd, scene_usd),
    )


def _require_session_future(sess: str) -> asyncio.Future[ZapdosSession]:
    future, reason = session_registry.resolve(sess)
    if future is not None:
        return future
    detail = "Session expired" if reason == "expired" else "Session not initialized"
    raise HTTPException(status_code=409, detail=detail)


async def _await_session_future(sess: str, future: asyncio.Future[ZapdosSession]) -> ZapdosSession:
    return await session_registry.await_ready(sess, future)


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
    session_registry.discard(sess, future)
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
