from __future__ import annotations

from collections import defaultdict

import unreal  # type: ignore

from nanami_assets import apply_material, ensure_color_material, import_static_mesh, import_texture_cube, log
from nanami_export import finalize_output, start_render
from nanami_math import identity, joint_matrix, mat_mul, matrix_to_pose
from nanami_scene import create_camera_rig, export_preview, new_blank_level, setup_lighting, update_camera


DEFAULT_CAMERA = {"target_xyz": [0.0, 0.0, 1.1], "yaw_deg": 35.0, "pitch_deg": -18.0, "distance": 3.2}


class RobotRuntime:
    def __init__(self, cfg: dict, send_event):
        self.cfg = cfg
        self.send_event = send_event
        self.asset_root = f'/Game/Nanami/{cfg["asset_key"]}'
        self.map_path = f"{self.asset_root}/Maps/NanamiAutoMap"
        self.sequence_path = f"{self.asset_root}/Sequences/NanamiPreview"
        self.import_root = self.asset_root
        self.material_root = f"{self.asset_root}/Materials"
        self.hdr_root = f"{self.asset_root}/HDRI"
        self.frame_index = 0
        self.manifest = None
        self.link_actors = {}
        self.children = defaultdict(list)
        self.cine_camera = None
        self.capture_actor = None
        self.render_target = None
        self.active_session_id = None
        self.active_preview_dir = ""

    def setup_scene(self) -> None:
        hdr = import_texture_cube(self.cfg["hdr_path"], self.asset_root) if self.cfg.get("hdr_path") else None
        new_blank_level(self.map_path)
        setup_lighting(hdr)
        self.cine_camera, self.capture_actor, self.render_target = create_camera_rig(DEFAULT_CAMERA["target_xyz"])
        unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)

    def attach_session(self, session_id: str, preview_dir: str) -> None:
        self.active_session_id = session_id
        self.active_preview_dir = preview_dir
        if self.manifest is not None:
            self.apply_state(session_id, {}, DEFAULT_CAMERA)

    def detach_session(self, session_id: str) -> None:
        if self.active_session_id == session_id:
            self.active_session_id = None
            self.active_preview_dir = ""

    def load_robot(self, manifest: dict) -> None:
        self.manifest = manifest
        self.children.clear()
        for actor in list(self.link_actors.values()):
            try:
                unreal.EditorLevelLibrary.destroy_actor(actor)
            except Exception:
                pass
        self.link_actors = {}
        for joint in manifest["joints"]:
            self.children[joint["parent_link"]].append(joint)
        for link in manifest["links"]:
            mesh_path = link.get("mesh_obj_path")
            if not mesh_path:
                continue
            mesh, _mesh_asset = import_static_mesh(mesh_path, self.import_root)
            actor = unreal.EditorLevelLibrary.spawn_actor_from_object(mesh, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
            actor.set_actor_label(link["name"])
            apply_material(actor, ensure_color_material(self.material_root, link["color_rgba"]))
            self.link_actors[link["name"]] = actor
        if self.active_session_id:
            self.apply_state(self.active_session_id, {}, DEFAULT_CAMERA)
        self.send_event(
            {
                "type": "robot_loaded",
                "link_count": manifest["link_count"],
                "movable_joint_count": manifest["movable_joint_count"],
                "controls": manifest["controls"],
            }
        )

    def apply_state(self, session_id: str, joints: dict[str, float], camera: dict | None) -> None:
        if self.manifest is None or self.active_session_id != session_id or not self.active_preview_dir:
            return
        root = next(link["name"] for link in self.manifest["links"] if not link.get("parent_joint"))
        matrices = {root: identity()}
        stack = [root]
        while stack:
            link_name = stack.pop()
            parent_matrix = matrices[link_name]
            for joint in self.children.get(link_name, []):
                value = joints.get(joint["name"], 0.0)
                matrices[joint["child_link"]] = mat_mul(parent_matrix, joint_matrix(joint, value))
                stack.append(joint["child_link"])
        for link_name, actor in self.link_actors.items():
            location, rotation = matrix_to_pose(matrices.get(link_name, identity()))
            actor.set_actor_location(unreal.Vector(*location), False, False)
            actor.set_actor_rotation(unreal.Rotator(*rotation), False)
        next_camera = camera or DEFAULT_CAMERA
        update_camera(
            self.cine_camera,
            self.capture_actor,
            next_camera["target_xyz"],
            next_camera["yaw_deg"],
            next_camera["pitch_deg"],
            next_camera["distance"],
        )
        frame_name = "frame_A.jpg" if self.frame_index % 2 == 0 else "frame_B.jpg"
        frame_path = f"{self.active_preview_dir}/{frame_name}"
        export_preview(self.render_target, self.capture_actor, frame_path)
        self.frame_index += 1
        self.send_event({"type": "frame_ready", "session_id": session_id, "path": frame_path})

    def export_final(self, session_id: str, job_id: str, output_dir: str) -> None:
        def on_done(success: bool):
            result_path = finalize_output(output_dir)
            self.send_event({"type": "export_done", "session_id": session_id, "job_id": job_id, "success": success, "result_path": result_path})

        def on_error(error_text: str):
            self.send_event({"type": "worker_error", "session_id": session_id, "message": error_text, "job_id": job_id})

        log(f"Starting final render {job_id}")
        start_render(self.map_path, self.sequence_path, self.cine_camera, output_dir, 1920, 1080, job_id, on_done, on_error)
