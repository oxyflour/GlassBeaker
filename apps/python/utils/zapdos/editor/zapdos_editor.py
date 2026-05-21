from __future__ import annotations

from pathlib import Path

import mujoco  # type: ignore
import numpy as np
from fastapi import HTTPException

from utils.genie_sim import resolve_assets_root
from utils.zapdos.bundle.camera_specs import camera_name_to_index
from utils.zapdos.editor.commands import build_remove_asset_overlay, build_set_scene_assets_overlay
from utils.zapdos.editor.rebuild_events import (
    SceneRebuildState,
    discard_scene_rebuild_job,
    emit_scene_rebuild_progress,
    scene_rebuild_future,
    stream_scene_rebuild_job,
)
from utils.zapdos.editor import rebuild_manager
from utils.zapdos.editor.repository import load_overlay_state, save_overlay_state
from utils.zapdos.editor.state import default_overlay_state, overlay_body_name, scene_revision
from utils.zapdos.physics.mujoco_tools import body_world_pose, flatten_matrix
from utils.zapdos.zapdos_asset_library import asset_local_bounds


class ZapdosEditor:
    def __init__(
        self,
        session,
        *,
        repo_root: Path,
        default_robot_usd: Path,
        default_scene_usd: Path,
    ) -> None:
        self.session = session
        self.robot_usd = getattr(session.bundle, "robot_usd", default_robot_usd)
        self.base_scene_usd = getattr(session.bundle, "scene_usd", default_scene_usd)
        self.session_dir = repo_root / "apps" / "python" / "tmp" / "zapdos" / session.sess
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_path = self.session_dir / "overlay.json"
        self.composed_scene_usd = self.session_dir / "scene-overlay.usda"
        loaded_overlay = load_overlay_state(self.overlay_path)
        self.overlay_state = default_overlay_state(loaded_overlay.get("assets_root"))
        if loaded_overlay["instances"] or loaded_overlay["pose_overrides"]:
            save_overlay_state(self.overlay_path, self.overlay_state)
        self.scene_revision = scene_revision(self.base_scene_usd, self.overlay_state)
        self.rebuilding_scene = False
        self.scene_rebuild_state = SceneRebuildState()

    @property
    def msgs(self):
        return self.session.msgs

    def close(self) -> None:
        with self.scene_rebuild_state.lock:
            tasks = list(self.scene_rebuild_state.tasks.values())
            jobs = list(self.scene_rebuild_state.jobs.values())
            self.scene_rebuild_state.jobs.clear()
            self.scene_rebuild_state.tasks.clear()
        for task in tasks:
            task.cancel()
        for job in jobs:
            job.future.cancel()

    def stream_rebuild_job(self, op_id: str):
        return stream_scene_rebuild_job(self, op_id)
    def list_scene_bodies(self) -> dict[str, object]:
        support_infos = self._build_support_infos()
        items = []
        for body in sorted(self.session.physics.editable_body_names):
            body_id = mujoco.mj_name2id(self.session.physics.model, mujoco.mjtObj.mjOBJ_BODY, body)  # type: ignore
            items.append(
                {
                    "body": body,
                    "label": self.session.physics.body_labels.get(body, body),
                    "matrix": flatten_matrix(body_world_pose(self.session.physics.data, body_id)),
                    "support": support_infos.get(body),
                    "world_aabb": self.session.physics.body_world_aabb(body),
                }
            )
        return {
            "items": items,
            "robot_bounds": self.session.physics.robot_bounds(),
            "scene_revision": self.scene_revision,
        }

    def set_scene_assets(self, assets: list[dict[str, object]]) -> dict[str, object]:
        next_overlay, items = build_set_scene_assets_overlay(self, assets)
        return self._start_overlay_operation(next_overlay, {"ok": True, "items": items})

    def remove_asset_from_scene(self, instance_id: str) -> dict[str, object]:
        next_overlay, _ = build_remove_asset_overlay(self, instance_id)
        return self._start_overlay_operation(next_overlay, {"ok": True, "instance_id": instance_id})

    def set_body_pose(self, body: str, pos: list[float], quat: list[float]) -> dict[str, object]:
        if self.rebuilding_scene:
            raise HTTPException(status_code=409, detail="Scene rebuild already in progress")
        result = self.session.physics.set_body_pose(body, pos, quat)
        quat_vec = np.array(quat, dtype=float)
        quat_norm = np.linalg.norm(quat_vec)
        self.overlay_state["pose_overrides"][body] = {
            "pos": list(pos),
            "quat": (quat_vec / quat_norm).tolist(),
        }
        save_overlay_state(self.overlay_path, self.overlay_state)
        return result

    def scene_rebuild_future(self, op_id: str):
        return scene_rebuild_future(self, op_id)

    def discard_scene_rebuild_job(self, op_id: str) -> None:
        discard_scene_rebuild_job(self, op_id)
    def _build_support_infos(self) -> dict[str, dict[str, float]]:
        infos: dict[str, dict[str, float]] = {}
        assets_root = resolve_assets_root(self.overlay_state.get("assets_root"))
        instance_by_body = {
            overlay_body_name(item["id"]): item
            for item in self.overlay_state["instances"]
        }
        for body in self.session.physics.editable_body_names:
            body_id = mujoco.mj_name2id(self.session.physics.model, mujoco.mjtObj.mjOBJ_BODY, body)  # type: ignore
            instance = instance_by_body.get(body)
            if instance is not None:
                try:
                    asset_local_bounds(assets_root / instance["url"])
                except (FileNotFoundError, OSError, RuntimeError) as exc:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Existing overlay asset unavailable: {instance['id']}: {exc}",
                    ) from exc
            world_aabb = self.session.physics.body_world_aabb(body)
            if world_aabb is not None:
                top_z = float(world_aabb["max"][2])
            else:
                top_z = float(body_world_pose(self.session.physics.data, body_id)[2, 3])
                if instance is not None:
                    bounds = asset_local_bounds(assets_root / instance["url"])
                    top_z += float(bounds["max"][2])
            infos[body] = {"top_z": top_z}
        return infos

    def _start_overlay_operation(self, next_overlay, success_payload: dict[str, object]) -> dict[str, object]:
        return rebuild_manager.start_overlay_operation(self, next_overlay, success_payload)
    def _prepare_overlay_rebuild(self, next_overlay, support_infos, previous_overlay, previous_revision, op_id: str | None = None):
        return rebuild_manager.prepare_overlay_rebuild(self, next_overlay, support_infos, previous_overlay, previous_revision, op_id=op_id)
    def _run_overlay_rebuild(self, op_id: str, next_overlay, previous_overlay, previous_revision: str):
        return rebuild_manager.run_overlay_rebuild(self, op_id, next_overlay, previous_overlay, previous_revision)
    def _apply_prepared_overlay_rebuild(self, prepared, op_id: str | None = None) -> str:
        return rebuild_manager.apply_prepared_overlay_rebuild(self, prepared, op_id)
    def _swap_runtime_bundle(self, bundle, overlay_state, op_id: str | None = None) -> None:
        emit_scene_rebuild_progress(self, op_id, "swap_runtime_bundle.started")
        old_physics = self.session.physics
        old_renderer = self.session.renderer
        new_physics = None
        joint_state = old_physics.joint_state_msg() if callable(getattr(old_physics, "joint_state_msg", None)) else {}
        old_actuators = getattr(old_physics, "actuator_name_to_id", {})
        ctrl_by_actuator = {
            name: float(old_physics.data.ctrl[actuator_id])
            for name, actuator_id in old_actuators.items()
            if 0 <= int(actuator_id) < len(old_physics.data.ctrl)
        } if isinstance(old_actuators, dict) else {}
        try:
            new_physics = self.session._create_physics(bundle)
            emit_scene_rebuild_progress(self, op_id, "swap_runtime_bundle.physics_loaded")
            for joint_name, position in zip(joint_state.get("name") or [], joint_state.get("position") or []):
                joint_id = mujoco.mj_name2id(new_physics.model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name))  # type: ignore
                if joint_id < 0:
                    continue
                qpos_adr = int(new_physics.model.jnt_qposadr[joint_id])
                next_qpos_adr = int(new_physics.model.nq if joint_id + 1 >= new_physics.model.njnt else new_physics.model.jnt_qposadr[joint_id + 1])
                if next_qpos_adr - qpos_adr == 1:
                    new_physics.data.qpos[qpos_adr] = float(position)
            new_actuators = getattr(new_physics, "actuator_name_to_id", {})
            if isinstance(new_actuators, dict):
                for actuator_name, ctrl in ctrl_by_actuator.items():
                    actuator_id = new_actuators.get(actuator_name)
                    if actuator_id is not None and 0 <= int(actuator_id) < len(new_physics.data.ctrl):
                        new_physics.data.ctrl[int(actuator_id)] = ctrl
            mujoco.mj_forward(new_physics.model, new_physics.data)  # type: ignore
            for body, pose in overlay_state["pose_overrides"].items():
                if body in new_physics.movable_body_names:
                    new_physics.set_body_pose(body, pose["pos"], pose["quat"])
            try:
                emit_scene_rebuild_progress(self, op_id, "swap_runtime_bundle.reload_scene.started")
                old_renderer.reload_scene(bundle)
            except Exception:
                emit_scene_rebuild_progress(self, op_id, "swap_runtime_bundle.reload_scene.failed")
                self.session.renderer = self.session._create_renderer(bundle)
            else:
                emit_scene_rebuild_progress(self, op_id, "swap_runtime_bundle.reload_scene.done")
                if hasattr(old_renderer, "set_bundle"):
                    old_renderer.set_bundle(bundle)
                else:
                    old_renderer.camera_index = camera_name_to_index(bundle.cameras)
                    old_renderer.last_frame_index = {camera.name: -1 for camera in bundle.cameras}
                self.session.renderer = old_renderer
            self.session.bundle = bundle
            self.session.physics = new_physics
        except Exception:
            if new_physics is not None:
                new_physics.close()
            raise
        if self.session.renderer is not old_renderer:
            old_renderer.close(stop_remote=False)
        old_physics.close()
        emit_scene_rebuild_progress(self, op_id, "swap_runtime_bundle.done")
