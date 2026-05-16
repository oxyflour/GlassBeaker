from __future__ import annotations

import asyncio
import json
import queue
import traceback
from pathlib import Path

from utils.ros_bridge import BridgeUnavailable, bridge
from utils.session import Session, Timer
from utils.zapdos.bundle import DEFAULT_SCENE_USD, RenderBundle, ensure_render_bundle
from utils.zapdos.bundle.camera_specs import image_topic
from utils.zapdos.editor.zapdos_editor import ZapdosEditor
from utils.zapdos.manipulation.runtime import ManipulationRuntime
from utils.zapdos.physics.mujoco_physics import ZapdosPhysics
from utils.zapdos.renderer import ZapdosRenderer
from utils.zapdos.renderer.isaac_renderer import IsaacRenderer, tf_message
from utils.zapdos.ros.topics import IMAGE_TYPE, JOINT_COMMAND_TOPIC, JOINT_STATES_TOPIC, JOINT_STATE_TYPE, TF_RENDER_TOPIC, TF_RENDER_TYPE

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
        self.command_msgs: queue.Queue[dict] = queue.Queue(maxsize=8)
        self.command_subscribed = False
        self.physics = self._create_physics(bundle)
        self.renderer = self._create_renderer(bundle)
        self.editor = ZapdosEditor(
            self,
            repo_root=REPO_ROOT,
            default_robot_usd=DEFAULT_ROBOT_USD,
            default_scene_usd=DEFAULT_SCENE_USD,
        )
        self.runtime = ManipulationRuntime(self)
        super().__init__()
        self.timers.append(Timer(ROS_DT, self.send_sse))
        asyncio.run_coroutine_threadsafe(self.send_ros(), self.loop)

    def _create_physics(self, bundle: RenderBundle) -> ZapdosPhysics:
        return ZapdosPhysics(self.sess, bundle, json.loads(bundle.body_map_json.read_text(encoding="utf-8")))

    def _create_renderer(self, bundle: RenderBundle) -> ZapdosRenderer:
        return ZapdosRenderer(
            backend=IsaacRenderer(self.sess, bundle, RENDER_SIZE[0], RENDER_SIZE[1], 30, True, 0),
            bundle=bundle,
            render_size=RENDER_SIZE,
            is_active=self.is_active,
            image_topic=image_topic,
            image_subscriptions=bridge.subs,
            frame_delay=ROS_DT,
        )

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
            return self.editor.list_scene_bodies()
        if method == "list_scene_objects":
            return self.runtime.list_scene_objects()
        if method == "grab_apple":
            return self.runtime.grab_apple()
        if method == "place_apple":
            return self.runtime.place_apple()
        if method == "set_scene_assets":
            return self.editor.set_scene_assets(*args)
        if method == "remove_asset_from_scene":
            return self.editor.remove_asset_from_scene(*args)
        if method == "set_body_pose":
            return self.editor.set_body_pose(*args)
        if method == "pick_object":
            return self.runtime.pick_object(*args)
        if method == "save_camera_override":
            return self.save_camera_override()
        return super().call_once(method, args)

    def save_camera_override(self) -> dict[str, object]:
        return self.renderer.save_camera_override()

    def send_sse(self) -> None:
        if self.editor.rebuilding_scene or self.msgs.full():
            return
        self.msgs.put_nowait({"pose": self.physics.get_pose()})

    async def send_ros(self) -> None:
        await self.renderer.wait_ready()
        while self.is_active():
            try:
                if not self.command_subscribed and bridge.conns:
                    await bridge.subscribe(JOINT_COMMAND_TOPIC, JOINT_STATE_TYPE, self.on_message)
                    self.command_subscribed = True
                if bridge.conns:
                    await bridge.call("publish", [JOINT_STATES_TOPIC, JOINT_STATE_TYPE, self.physics.joint_state_msg()])
                    await bridge.call("publish", [TF_RENDER_TOPIC, TF_RENDER_TYPE, tf_message(self.physics.model, self.physics.data)])
                    if self.renderer.should_publish_camera_images():
                        for topic, image_msg in self.renderer.image_messages():
                            await bridge.call("publish", [topic, IMAGE_TYPE, image_msg])
                await asyncio.sleep(ROS_DT)
            except BridgeUnavailable:
                await asyncio.sleep(1)
            except Exception:
                traceback.print_exc()
                await asyncio.sleep(1)

    def _latest_joint_command(self) -> dict | None:
        latest = None
        while not self.command_msgs.empty():
            try:
                latest = self.command_msgs.get_nowait()
            except queue.Empty:
                break
        return latest

    def step_once(self):
        self.editor.drain_completions()
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

    def destroy(self):
        for topic in list(bridge.subs):
            bridge.unsubscribe(topic, self.on_message)
        self.editor.close()
        self.renderer.close()
        self.physics.close()
        return super().destroy()
