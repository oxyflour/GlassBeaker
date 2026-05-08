from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import json
import queue
import traceback
from pathlib import Path

import mujoco  # type: ignore
import numpy as np
from PIL import Image
from fastapi import HTTPException

from utils.camera_override import save_camera_overrides
from utils.genie_sim_runtime import resolve_assets_root
from utils.ros_bridge import bridge
from utils.session import Session, Timer
from utils.zapdos.mujoco_tools import body_world_pose, flatten_matrix
from utils.zapdos.rl_bundle import DEFAULT_SCENE_USD, RenderBundle, ensure_render_bundle
from utils.zapdos.rl_cameras import camera_name_to_index, image_topic
from utils.zapdos.sim_env import (
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
from utils.zapdos.zapdos_asset_library import asset_local_bounds
from utils.zapdos.zapdos_overlay import (
    default_overlay_state,
    load_overlay_state,
    overlay_body_name,
    save_overlay_state,
    scene_revision,
)
from utils.zapdos.zapdos_physics import ZapdosPhysics
from utils.zapdos import zapdos_scene_operations as scene_ops

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROBOT_USD = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"
RENDER_SIZE = (640, 480)
ROS_DT = 0.03333


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
        self.overlay_executor = ThreadPoolExecutor(max_workers=1)
        scene_ops.ensure_scene_operation_state(self)
        super().__init__()
        self.timers.append(Timer(ROS_DT, self.send_sse))
        self.renderer = IsaacRenderer(sess, bundle, RENDER_SIZE[0], RENDER_SIZE[1], 30, True, 0)
        asyncio.run_coroutine_threadsafe(self.send_ros(), self.loop)

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
        old_physics = self.physics
        old_renderer = self.renderer
        new_physics = None
        try:
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
            reload_scene = getattr(old_renderer, "reload_scene", None)
            if callable(reload_scene):
                try:
                    reload_scene(bundle)
                except Exception:
                    pass
                else:
                    self.bundle = bundle
                    self.physics = new_physics
                    self.camera_index = camera_name_to_index(bundle.cameras)
                    self.last_frame_index = {camera.name: -1 for camera in bundle.cameras}
                    old_physics.close()
                    return
            new_renderer = IsaacRenderer(self.sess, bundle, RENDER_SIZE[0], RENDER_SIZE[1], 30, True, 0)
        except Exception:
            if new_physics is not None:
                new_physics.close()
            raise
        self.bundle = bundle
        self.physics = new_physics
        self.camera_index = camera_name_to_index(bundle.cameras)
        self.last_frame_index = {camera.name: -1 for camera in bundle.cameras}
        self.renderer = new_renderer
        old_renderer.close(stop_remote=False)
        old_physics.close()

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
        if getattr(self, "rebuilding_scene", False):
            raise HTTPException(status_code=409, detail="Scene rebuild already in progress")
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

    def set_scene_assets(self, assets: list[dict[str, object]]) -> dict[str, object]:
        next_overlay, result_items = scene_ops.build_set_scene_assets_overlay(self, assets)
        return self._start_overlay_operation(next_overlay, {"ok": True, "items": result_items})

    def remove_asset_from_scene(self, instance_id: str) -> dict[str, object]:
        next_overlay, _ = scene_ops.build_remove_asset_overlay(self, instance_id)
        return self._start_overlay_operation(next_overlay, {"ok": True, "instance_id": instance_id})

    def _start_overlay_operation(self, next_overlay, success_payload: dict[str, object]) -> dict[str, object]:
        return scene_ops.start_overlay_operation(self, next_overlay, success_payload)

    def _prepare_overlay_rebuild(self, next_overlay, support_infos, previous_overlay, previous_revision):
        return scene_ops.prepare_overlay_rebuild(
            self,
            next_overlay,
            support_infos,
            previous_overlay,
            previous_revision,
        )

    def _run_overlay_rebuild_background(
        self,
        op_id: str,
        next_overlay,
        support_infos,
        previous_overlay,
        previous_revision: str,
    ) -> None:
        scene_ops.run_overlay_rebuild_background(
            self,
            op_id,
            next_overlay,
            support_infos,
            previous_overlay,
            previous_revision,
        )

    def _apply_prepared_overlay_rebuild(self, prepared) -> str:
        return scene_ops.apply_prepared_overlay_rebuild(self, prepared)

    def _drain_overlay_completions(self) -> None:
        scene_ops.drain_overlay_completions(self)

    def scene_operation_future(self, op_id: str):
        return scene_ops.scene_operation_future(self, op_id)

    def discard_scene_operation(self, op_id: str) -> None:
        scene_ops.discard_scene_operation(self, op_id)

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
        if method == "set_scene_assets":
            return self.set_scene_assets(*args)
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
            self.msgs.put_nowait({"pose": self.physics.get_pose()})

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
        self._drain_overlay_completions()
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
        yield mjpeg_chunk(placeholder_jpeg(RENDER_SIZE[0], RENDER_SIZE[1], "Closed"))

    def destroy(self):
        with self.scene_operations_lock:
            for operation in self.scene_operations.values():
                operation.future.cancel()
            self.scene_operations.clear()
        for topic in list(bridge.subs):
            bridge.unsubscribe(topic, self.on_message)
        self.overlay_executor.shutdown(wait=False)
        self.renderer.close()
        self.physics.close()
        return super().destroy()
