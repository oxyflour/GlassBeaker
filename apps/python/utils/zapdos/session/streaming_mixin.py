from __future__ import annotations


def _session_module():
    from . import zapdos_session as session_module

    return session_module


class SessionStreamingMixin:
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
                    for topic, image_msg in self._image_messages():
                        await session_module.bridge.call(
                            "publish",
                            [topic, session_module.IMAGE_TYPE, image_msg],
                        )
                await session_module.asyncio.sleep(session_module.ROS_DT)
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

    async def render(self, camera_name: str):
        session_module = _session_module()
        while self.is_active():
            while not self.renderer.ready:
                yield session_module.mjpeg_chunk(
                    session_module.placeholder_jpeg(
                        session_module.RENDER_SIZE[0],
                        session_module.RENDER_SIZE[1],
                        "Waiting",
                    )
                )
                await session_module.asyncio.sleep(1)
            try:
                frame_state = self.renderer.read(camera_name)
                if frame_state is None:
                    await session_module.asyncio.sleep(session_module.ROS_DT)
                    continue
                _, frame = frame_state
                data = session_module.io.BytesIO()
                session_module.Image.fromarray(frame).save(
                    data,
                    format="JPEG",
                    quality=80,
                )
                yield session_module.mjpeg_chunk(data.getvalue())
                await session_module.asyncio.sleep(session_module.ROS_DT)
            except Exception:
                session_module.traceback.print_exc()
                await session_module.asyncio.sleep(1)
        yield session_module.mjpeg_chunk(
            session_module.placeholder_jpeg(
                session_module.RENDER_SIZE[0],
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
