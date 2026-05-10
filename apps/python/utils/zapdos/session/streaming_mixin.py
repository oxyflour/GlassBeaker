from __future__ import annotations

import os

from utils.ros_bridge import BridgeUnavailable


def _session_module():
    from . import zapdos_session as session_module

    return session_module


class SessionStreamingMixin:
    def _encode_jpeg(self, frame) -> bytes:
        session_module = _session_module()
        data = session_module.io.BytesIO()
        session_module.Image.fromarray(frame).save(
            data,
            format="JPEG",
            quality=80,
        )
        return data.getvalue()

    def _placeholder_frame(self, text: str) -> bytes:
        session_module = _session_module()
        return session_module.placeholder_jpeg(
            session_module.RENDER_SIZE[0],
            session_module.RENDER_SIZE[1],
            text,
        )

    def send_sse(self):
        if self.rebuilding_scene:
            return
        if not self.msgs.full():
            self.msgs.put_nowait({"pose": self.physics.get_pose()})

    async def send_ros(self):
        session_module = _session_module()
        await self.renderer.wait_ready()
        while self.is_active():
            try:
                if not self.command_subscribed and session_module.bridge.conns:
                    await session_module.bridge.subscribe(
                        session_module.JOINT_COMMAND_TOPIC,
                        session_module.JOINT_STATE_TYPE,
                        self.on_message,
                    )
                    self.command_subscribed = True
                if session_module.bridge.conns:
                    await session_module.bridge.call(
                        "publish",
                        [
                            session_module.JOINT_STATES_TOPIC,
                            session_module.JOINT_STATE_TYPE,
                            self.physics.joint_state_msg(),
                        ],
                    )
                    await session_module.bridge.call(
                        "publish",
                        [
                            session_module.TF_RENDER_TOPIC,
                            session_module.TF_RENDER_TYPE,
                            session_module.tf_message(
                                self.physics.model,
                                self.physics.data,
                            ),
                        ],
                    )
                    if self._should_publish_camera_images():
                        for topic, image_msg in self._image_messages():
                            await session_module.bridge.call(
                                "publish",
                                [topic, session_module.IMAGE_TYPE, image_msg],
                            )
                await session_module.asyncio.sleep(session_module.ROS_DT)
            except BridgeUnavailable:
                await session_module.asyncio.sleep(1)
            except Exception:
                session_module.traceback.print_exc()
                await session_module.asyncio.sleep(1)

    def _image_messages(self) -> list[tuple[str, dict]]:
        session_module = _session_module()
        messages: list[tuple[str, dict]] = []
        for camera in self.bundle.cameras:
            frame_state = self.renderer.read(camera.name)
            if frame_state is None:
                continue
            index, frame = frame_state
            if index == self.last_frame_index[camera.name]:
                continue
            self.last_frame_index[camera.name] = index
            messages.append(
                (
                    session_module.image_topic(camera.name),
                    {
                        "header": {"frame_id": camera.frame_id},
                        "height": int(frame.shape[0]),
                        "width": int(frame.shape[1]),
                        "encoding": "rgb8",
                        "is_bigendian": 0,
                        "step": int(frame.shape[1] * 3),
                        "data": frame.tobytes(),
                    },
                )
            )
        return messages

    def _should_publish_camera_images(self) -> bool:
        mode = os.getenv("ZAPDOS_PUBLISH_CAMERA_IMAGES", "").strip().lower()
        if mode in {"1", "true", "yes", "always"}:
            return True
        if mode in {"0", "false", "no", "never"}:
            return False
        session_module = _session_module()
        for camera in self.bundle.cameras:
            if session_module.bridge.subs.get(session_module.image_topic(camera.name)):
                return True
        return False

    def _latest_joint_command(self) -> dict | None:
        session_module = _session_module()
        latest = None
        while not self.command_msgs.empty():
            try:
                latest = self.command_msgs.get_nowait()
            except session_module.queue.Empty:
                break
        return latest

    def step_once(self):
        self._drain_overlay_completions()
        self.physics.apply_joint_command(self._latest_joint_command())
        self.physics.step()
        return super().step_once()

    def on_message(self, topic: str, msg):
        session_module = _session_module()
        if topic == session_module.JOINT_COMMAND_TOPIC:
            while self.command_msgs.full():
                try:
                    self.command_msgs.get_nowait()
                except session_module.queue.Empty:
                    break
            self.command_msgs.put_nowait(msg)
            return
        if not self.msgs.full():
            self.msgs.put_nowait({"topic": topic, "msg": msg})

    def snapshot(self, camera_name: str) -> bytes:
        if not self.renderer.ready:
            return self._placeholder_frame("Waiting")
        frame_state = self.renderer.read(camera_name)
        if frame_state is None:
            return self._placeholder_frame("Waiting" if self.is_active() else "Closed")
        _, frame = frame_state
        return self._encode_jpeg(frame)

    def _composite_camera_names(self) -> list[str]:
        return [camera.name for camera in self.bundle.cameras[:3]]

    def _read_composite_frame(self, camera_names: list[str]):
        session_module = _session_module()
        frames = []
        frame_index = -1
        for camera_name in camera_names:
            frame_state = self.renderer.read(camera_name)
            if frame_state is None:
                return None
            index, frame = frame_state
            if frame_index < 0:
                frame_index = index
            frames.append(frame)
        if frame_index < 0:
            return None
        return frame_index, session_module.np.concatenate(frames, axis=1)

    async def render(self, camera_name: str):
        session_module = _session_module()
        last_frame_index = -1
        while self.is_active():
            while not self.renderer.ready:
                yield session_module.mjpeg_chunk(self._placeholder_frame("Waiting"))
                await session_module.asyncio.sleep(1)
            try:
                frame_state = self.renderer.read(camera_name)
                if frame_state is None:
                    await session_module.asyncio.sleep(session_module.ROS_DT)
                    continue
                index, frame = frame_state
                if index == last_frame_index:
                    await session_module.asyncio.sleep(session_module.ROS_DT)
                    continue
                last_frame_index = index
                yield session_module.mjpeg_chunk(self._encode_jpeg(frame))
                await session_module.asyncio.sleep(session_module.ROS_DT)
            except Exception:
                session_module.traceback.print_exc()
                await session_module.asyncio.sleep(1)
        yield session_module.mjpeg_chunk(self._placeholder_frame("Closed"))

    async def render_multi_camera(self):
        session_module = _session_module()
        last_frame_index = -1
        camera_names = self._composite_camera_names()
        if not camera_names:
            yield session_module.mjpeg_chunk(self._placeholder_frame("No Cameras"))
            return
        while self.is_active():
            while not self.renderer.ready:
                yield session_module.mjpeg_chunk(
                    session_module.placeholder_jpeg(
                        session_module.RENDER_SIZE[0] * len(camera_names),
                        session_module.RENDER_SIZE[1],
                        "Waiting",
                    )
                )
                await session_module.asyncio.sleep(1)
            try:
                frame_state = self._read_composite_frame(camera_names)
                if frame_state is None:
                    await session_module.asyncio.sleep(session_module.ROS_DT)
                    continue
                index, frame = frame_state
                if index == last_frame_index:
                    await session_module.asyncio.sleep(session_module.ROS_DT)
                    continue
                last_frame_index = index
                yield session_module.mjpeg_chunk(self._encode_jpeg(frame))
                await session_module.asyncio.sleep(session_module.ROS_DT)
            except Exception:
                session_module.traceback.print_exc()
                await session_module.asyncio.sleep(1)
        yield session_module.mjpeg_chunk(
            session_module.placeholder_jpeg(
                session_module.RENDER_SIZE[0] * len(camera_names),
                session_module.RENDER_SIZE[1],
                "Closed",
            )
        )

    def destroy(self):
        session_module = _session_module()
        with self.scene_rebuild_jobs_lock:
            for job in self.scene_rebuild_jobs.values():
                job.future.cancel()
            self.scene_rebuild_jobs.clear()
        for topic in list(session_module.bridge.subs):
            session_module.bridge.unsubscribe(topic, self.on_message)
        self.overlay_executor.shutdown(wait=False)
        self.renderer.close()
        self.physics.close()
        return super().destroy()
