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
from utils.genie_sim import resolve_assets_root
from utils.ros_bridge import bridge
from utils.session import Session, Timer
from utils.zapdos.bundle import DEFAULT_SCENE_USD, RenderBundle, ensure_render_bundle
from utils.zapdos.bundle.camera_specs import camera_name_to_index, image_topic
from utils.zapdos.overlay.overlay_repository import load_overlay_state, save_overlay_state
from utils.zapdos.overlay.overlay_state import default_overlay_state, overlay_body_name, scene_revision
from utils.zapdos.physics.mujoco_physics import ZapdosPhysics
from utils.zapdos.physics.mujoco_tools import body_world_pose, flatten_matrix
from utils.zapdos.rebuild import scene_rebuild_manager as rebuild_manager
from utils.zapdos.rebuild.scene_rebuild_service import SceneRebuildService
from utils.zapdos.ros.topics import (
    IMAGE_TYPE,
    JOINT_COMMAND_TOPIC,
    JOINT_STATES_TOPIC,
    JOINT_STATE_TYPE,
    TF_RENDER_TOPIC,
    TF_RENDER_TYPE,
)
from utils.zapdos.renderer.isaac_renderer import IsaacRenderer, mjpeg_chunk, placeholder_jpeg, tf_message
from utils.zapdos.session.runtime_mixin import SessionRuntimeMixin
from utils.zapdos.session.streaming_mixin import SessionStreamingMixin
from utils.zapdos.zapdos_asset_library import asset_local_bounds

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_ROBOT_USD = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"
RENDER_SIZE = (640, 480)
ROS_DT = 0.03333


class ZapdosSession(SessionRuntimeMixin, SessionStreamingMixin, Session):
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
        rebuild_manager.ensure_scene_rebuild_state(self)
        self.scene_rebuild_service = SceneRebuildService(self)
        super().__init__()
        self.timers.append(Timer(ROS_DT, self.send_sse))
        self.renderer = IsaacRenderer(sess, bundle, RENDER_SIZE[0], RENDER_SIZE[1], 30, True, 0)
        asyncio.run_coroutine_threadsafe(self.send_ros(), self.loop)

    def _scene_rebuild_service(self) -> SceneRebuildService:
        service = getattr(self, "scene_rebuild_service", None)
        if service is None:
            service = SceneRebuildService(self)
            self.scene_rebuild_service = service
        return service

    def set_scene_assets(self, assets: list[dict[str, object]]) -> dict[str, object]:
        return self._scene_rebuild_service().submit_replace(assets)

    def remove_asset_from_scene(self, instance_id: str) -> dict[str, object]:
        return self._scene_rebuild_service().submit_remove(instance_id)

    def _start_overlay_operation(self, next_overlay, success_payload: dict[str, object]) -> dict[str, object]:
        return rebuild_manager.start_overlay_operation(self, next_overlay, success_payload)

    def _prepare_overlay_rebuild(self, next_overlay, support_infos, previous_overlay, previous_revision, op_id: str | None = None):
        return rebuild_manager.prepare_overlay_rebuild(
            self,
            next_overlay,
            support_infos,
            previous_overlay,
            previous_revision,
            op_id=op_id,
        )

    def _run_overlay_rebuild_background(
        self,
        op_id: str,
        next_overlay,
        support_infos,
        previous_overlay,
        previous_revision: str,
    ) -> None:
        rebuild_manager.run_overlay_rebuild_background(
            self,
            op_id,
            next_overlay,
            support_infos,
            previous_overlay,
            previous_revision,
        )

    def _apply_prepared_overlay_rebuild(self, prepared, op_id: str | None = None) -> str:
        return rebuild_manager.apply_prepared_overlay_rebuild(self, prepared, op_id)

    def _drain_overlay_completions(self) -> None:
        self._scene_rebuild_service().drain_completions()

    def scene_rebuild_future(self, op_id: str):
        return rebuild_manager.scene_rebuild_future(self, op_id)

    def discard_scene_rebuild_job(self, op_id: str) -> None:
        rebuild_manager.discard_scene_rebuild_job(self, op_id)

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
