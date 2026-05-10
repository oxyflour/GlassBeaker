from __future__ import annotations


def _session_module():
    from . import zapdos_session as session_module

    return session_module


class SessionRuntimeMixin:
    def _build_support_infos(self) -> dict[str, dict[str, float]]:
        session_module = _session_module()
        infos: dict[str, dict[str, float]] = {}
        assets_root = session_module.resolve_assets_root(
            self.overlay_state.get("assets_root")
        )
        instance_by_body = {
            session_module.overlay_body_name(item["id"]): item
            for item in self.overlay_state["instances"]
        }
        for body in self.physics.editable_body_names:
            body_id = session_module.mujoco.mj_name2id(
                self.physics.model,
                session_module.mujoco.mjtObj.mjOBJ_BODY,
                body,
            )  # type: ignore
            top_z = float(session_module.body_world_pose(self.physics.data, body_id)[2, 3])
            instance = instance_by_body.get(body)
            if instance is not None:
                try:
                    bounds = session_module.asset_local_bounds(assets_root / instance["url"])
                except (FileNotFoundError, OSError, RuntimeError) as exc:
                    raise session_module.HTTPException(
                        status_code=409,
                        detail=f"Existing overlay asset unavailable: {instance['id']}: {exc}",
                    ) from exc
                top_z += float(bounds["max"][2])
            infos[body] = {"top_z": top_z}
        return infos

    def _swap_runtime_bundle(
        self,
        bundle,
        overlay_state,
        op_id: str | None = None,
    ) -> None:
        session_module = _session_module()
        session_module.rebuild_manager.emit_scene_rebuild_progress(
            self, op_id, "swap_runtime_bundle.started"
        )
        snapshot_qpos = session_module.np.copy(self.physics.data.qpos)
        snapshot_ctrl = session_module.np.copy(self.physics.data.ctrl)
        body_map = session_module.json.loads(
            bundle.body_map_json.read_text(encoding="utf-8")
        )
        old_physics = self.physics
        old_renderer = self.renderer
        new_physics = None
        try:
            new_physics = session_module.ZapdosPhysics(self.sess, bundle, body_map)
            session_module.rebuild_manager.emit_scene_rebuild_progress(
                self, op_id, "swap_runtime_bundle.physics_loaded"
            )
            count = min(len(snapshot_qpos), len(new_physics.data.qpos))
            if count:
                new_physics.data.qpos[:count] = snapshot_qpos[:count]
            ctrl_count = min(len(snapshot_ctrl), len(new_physics.data.ctrl))
            if ctrl_count:
                new_physics.data.ctrl[:ctrl_count] = snapshot_ctrl[:ctrl_count]
            session_module.mujoco.mj_forward(new_physics.model, new_physics.data)  # type: ignore
            for body, pose in overlay_state["pose_overrides"].items():
                if body in new_physics.movable_body_names:
                    new_physics.set_body_pose(body, pose["pos"], pose["quat"])
            reload_scene = getattr(old_renderer, "reload_scene", None)
            if callable(reload_scene):
                session_module.rebuild_manager.emit_scene_rebuild_progress(
                    self, op_id, "swap_runtime_bundle.reload_scene.started"
                )
                try:
                    reload_scene(bundle)
                except Exception:
                    session_module.rebuild_manager.emit_scene_rebuild_progress(
                        self, op_id, "swap_runtime_bundle.reload_scene.failed"
                    )
                else:
                    session_module.rebuild_manager.emit_scene_rebuild_progress(
                        self, op_id, "swap_runtime_bundle.reload_scene.done"
                    )
                    self.bundle = bundle
                    self.physics = new_physics
                    self.camera_index = session_module.camera_name_to_index(bundle.cameras)
                    self.last_frame_index = {camera.name: -1 for camera in bundle.cameras}
                    old_physics.close()
                    session_module.rebuild_manager.emit_scene_rebuild_progress(
                        self, op_id, "swap_runtime_bundle.done"
                    )
                    return
            session_module.rebuild_manager.emit_scene_rebuild_progress(
                self, op_id, "swap_runtime_bundle.new_renderer.started"
            )
            new_renderer = session_module.IsaacRenderer(
                self.sess,
                bundle,
                session_module.RENDER_SIZE[0],
                session_module.RENDER_SIZE[1],
                30,
                True,
                0,
            )
            session_module.rebuild_manager.emit_scene_rebuild_progress(
                self, op_id, "swap_runtime_bundle.new_renderer.done"
            )
        except Exception:
            if new_physics is not None:
                new_physics.close()
            raise
        self.bundle = bundle
        self.physics = new_physics
        self.camera_index = session_module.camera_name_to_index(bundle.cameras)
        self.last_frame_index = {camera.name: -1 for camera in bundle.cameras}
        self.renderer = new_renderer
        old_renderer.close(stop_remote=False)
        old_physics.close()
        session_module.rebuild_manager.emit_scene_rebuild_progress(
            self, op_id, "swap_runtime_bundle.done"
        )

    def list_scene_bodies(self) -> dict[str, object]:
        session_module = _session_module()
        support_infos = self._build_support_infos()
        items = []
        for body in sorted(self.physics.editable_body_names):
            body_id = session_module.mujoco.mj_name2id(
                self.physics.model,
                session_module.mujoco.mjtObj.mjOBJ_BODY,
                body,
            )  # type: ignore
            items.append(
                {
                    "body": body,
                    "label": self.physics.body_labels.get(body, body),
                    "matrix": session_module.flatten_matrix(
                        session_module.body_world_pose(self.physics.data, body_id)
                    ),
                    "support": support_infos.get(body),
                }
            )
        return {"items": items, "scene_revision": self.scene_revision}

    def set_body_pose(
        self,
        body: str,
        pos: list[float],
        quat: list[float],
    ) -> dict[str, object]:
        session_module = _session_module()
        if getattr(self, "rebuilding_scene", False):
            raise session_module.HTTPException(
                status_code=409,
                detail="Scene rebuild already in progress",
            )
        result = self.physics.set_body_pose(body, pos, quat)
        if hasattr(self, "overlay_state") and hasattr(self, "overlay_path"):
            quat_vec = session_module.np.array(quat, dtype=float)
            quat_norm = session_module.np.linalg.norm(quat_vec)
            self.overlay_state["pose_overrides"][body] = {
                "pos": list(pos),
                "quat": (quat_vec / quat_norm).tolist(),
            }
            session_module.save_overlay_state(self.overlay_path, self.overlay_state)
        return result

    def save_camera_override(self) -> dict[str, object]:
        session_module = _session_module()
        path, saved = session_module.save_camera_overrides(
            self.renderer.snapshot_cameras()
        )
        return {"ok": True, "saved": saved, "path": str(path)}
