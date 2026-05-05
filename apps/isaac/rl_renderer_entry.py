from __future__ import annotations

import json
import os
import sys
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "apps" / "python"
SOURCE_ROOT = REPO_ROOT / "deps" / "genie_sim" / "source"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
os.environ.setdefault("SIM_REPO_ROOT", str(REPO_ROOT / "deps" / "genie_sim"))

import geniesim.rl.renderer.rl_renderer as upstream  # type: ignore

from utils.camera_math import fovy_from_focal_length
from utils.isaac_renderer_reload import (
    CameraBinding,
    rebuild_camera_bindings,
    reset_subscriber_caches,
    validate_camera_topology,
)
from utils.renderer_ipc import request_path, response_path


def _noop_spin(self) -> None:
    return None


upstream.rclpy.executors.SingleThreadedExecutor.spin = _noop_spin


class LocalRLRenderer(upstream.RLRenderer):
    def _create_default_viz_camera(self, env_path: str, cam_pos, cam_target) -> str:
        existing_path = env_path + "/default_viz_camera"
        if upstream.is_prim_path_valid(existing_path):
            return existing_path
        return super()._create_default_viz_camera(env_path, cam_pos, cam_target)

    def _camera_snapshot(self) -> list[dict[str, object]]:
        env_root = "/World/envs/env_0"
        cameras: list[dict[str, object]] = []
        for camera in getattr(self, "_camera_list", []):
            prim_path = env_root + str(camera["prim"])
            prim = self.stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise RuntimeError(f"Camera prim not found: {prim_path}")
            usd_camera = upstream.UsdGeom.Camera(prim)
            translate = prim.GetAttribute("xformOp:translate").Get()
            orient = prim.GetAttribute("xformOp:orient").Get()
            focal_length = float(usd_camera.GetFocalLengthAttr().Get())
            vertical_aperture = float(usd_camera.GetVerticalApertureAttr().Get())
            cameras.append({
                "name": str(camera["name"]),
                "prim": str(camera["prim"]),
                "parent_prim": PurePosixPath(str(camera["prim"])).parent.as_posix(),
                "pos": [float(translate[0]), float(translate[1]), float(translate[2])],
                "quat": [
                    float(orient.GetReal()),
                    float(orient.GetImaginary()[0]),
                    float(orient.GetImaginary()[1]),
                    float(orient.GetImaginary()[2]),
                ],
                "focal_length": focal_length,
                "horizontal_aperture": float(usd_camera.GetHorizontalApertureAttr().Get()),
                "vertical_aperture": vertical_aperture,
                "clipping_range": [float(value) for value in usd_camera.GetClippingRangeAttr().Get()],
                "fovy": fovy_from_focal_length(focal_length, vertical_aperture),
            })
        return cameras

    def _setup_camera(self, cam_prim_path: str) -> CameraBinding:
        resolved_path = cam_prim_path
        if not upstream.is_prim_path_valid(resolved_path):
            parts = resolved_path.strip("/").split("/")
            env_root = "/" + "/".join(parts[:3]) if len(parts) >= 3 else resolved_path
            fallback = self._find_first_camera_under_env(env_root)
            if not fallback:
                print(f"[RLRenderer] Camera prim not found: {cam_prim_path}")
                return CameraBinding(annotator=None, render_product=None)
            resolved_path = fallback
        render_product = upstream.rep.create.render_product(
            resolved_path,
            (self.args.cam_width, self.args.cam_height),
        )
        annotator = upstream.rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach(render_product)
        return CameraBinding(annotator=annotator, render_product=render_product)

    def _destroy_render_product(self, render_product: object | None) -> None:
        if render_product is None:
            return
        destroy = getattr(render_product, "destroy", None)
        if callable(destroy):
            destroy()
            return
        if self.stage is None or not isinstance(render_product, str):
            return
        for layer in self.stage.GetLayerStack():
            with upstream.Usd.EditContext(self.stage, layer):
                prim = self.stage.GetPrimAtPath(render_product)
                if prim.IsValid():
                    self.stage.RemovePrim(render_product)

    def _release_camera_binding(self, binding: CameraBinding) -> None:
        if binding.annotator is not None:
            binding.annotator.detach(binding.render_product)
        self._destroy_render_product(binding.render_product)

    def _env_paths(self) -> list[str]:
        return [sub.env_root for sub in self.env_subscribers]

    def _rebuild_scene_clones(self) -> list[str]:
        env_root = "/World/envs"
        if self.stage is None:
            raise RuntimeError("renderer stage not initialized")
        if self.stage.GetPrimAtPath(env_root).IsValid():
            self.stage.RemovePrim(env_root)
        cloner = upstream.GridCloner(spacing=self.args.clone_spacing)
        cloner.define_base_env(env_root)
        env_paths = cloner.generate_paths("/World/envs/env", self.num_envs)
        upstream.add_reference_to_stage(self.args.scene_usd, env_paths[0])
        if self.args.robot_usd:
            robot_prim_path = env_paths[0] + self.args.robot_prim
            upstream.add_reference_to_stage(self.args.robot_usd, robot_prim_path)
            if upstream._task_resolved is not None:
                self._set_robot_base_pose(
                    robot_prim_path,
                    upstream._task_resolved["robot_init_position"],
                    upstream._task_resolved["robot_init_quaternion"],
                )
        if self.args.main_cam_prim == "/default_viz_camera":
            self._create_default_viz_camera(
                env_paths[0],
                self.args.default_cam_pos,
                self.args.default_cam_target,
            )
        cloner.clone(
            source_prim_path=env_paths[0],
            prim_paths=env_paths,
            copy_from_source=True,
            replicate_physics=False,
        )
        self._add_ground_plane(env_paths[0])
        if self._count_lights_under_env(env_paths[0]) == 0:
            dome = upstream.UsdLux.DomeLight.Define(self.stage, env_paths[0] + "/AutoDomeLight")
            dome.CreateIntensityAttr(float(self.args.auto_dome_light_intensity))
        return env_paths

    def _reload_scene(self, scene_usd: str, cameras: list[dict[str, object]]) -> None:
        validate_camera_topology(self._camera_list, cameras)
        rebuild_camera_bindings(
            [],
            [],
            self.cam_annotators_all,
            lambda path: CameraBinding(annotator=None, render_product=None),
            self._release_camera_binding,
        )
        self.args.scene_usd = scene_usd
        env_paths = self._rebuild_scene_clones()
        self._camera_list = list(cameras)
        self._num_cams = len(self._camera_list)
        body_name_map = self._build_body_name_map(env_paths[0])
        reset_subscriber_caches(self.env_subscribers, body_name_map)
        self.world.reset()
        self.cam_annotators_all = rebuild_camera_bindings(
            env_paths,
            self._camera_list,
            [],
            self._setup_camera,
            self._release_camera_binding,
        )
        self.cam_annotators_main = [
            self.cam_annotators_all[0][index].annotator if self._num_cams > 0 else None
            for index in range(self.num_envs)
        ]
        self.cam_annotators_wrist = [
            self.cam_annotators_all[1][index].annotator if self._num_cams > 1 else None
            for index in range(self.num_envs)
        ]

    def _service_control_request(self) -> None:
        control_dir = os.environ.get("GB_RENDERER_CONTROL_DIR", "").strip()
        if not control_dir:
            return
        req_path = request_path(Path(control_dir))
        res_path = response_path(Path(control_dir))
        if not req_path.exists():
            return
        request = json.loads(req_path.read_text(encoding="utf-8"))
        try:
            operation = request.get("op")
            if operation == "snapshot_cameras":
                payload = {"id": request.get("id"), "ok": True, "cameras": self._camera_snapshot()}
            elif operation == "reload_scene":
                scene_usd = str(request.get("scene_usd") or "").strip()
                cameras = request.get("cameras")
                if not scene_usd:
                    raise RuntimeError("reload_scene requires scene_usd")
                if not isinstance(cameras, list):
                    raise RuntimeError("reload_scene requires cameras")
                self._reload_scene(scene_usd, cameras)
                payload = {"id": request.get("id"), "ok": True}
            else:
                raise RuntimeError(f"Unsupported renderer op: {operation}")
        except Exception as err:
            payload = {"id": request.get("id"), "ok": False, "error": str(err)}
        res_path.write_text(json.dumps(payload), encoding="utf-8")
        req_path.unlink(missing_ok=True)

    def _render_callback(self, step_size: float):
        if not any(sub._dirty for sub in self.env_subscribers):
            return
        with upstream.Sdf.ChangeBlock():
            for sub in self.env_subscribers:
                sub.apply_tf()
        h, w = self.args.cam_height, self.args.cam_width
        for env_index in range(self.num_envs):
            for camera_index in range(self._num_cams):
                binding = self.cam_annotators_all[camera_index][env_index]
                annotator = binding.annotator
                if annotator is None:
                    continue
                data = annotator.get_data()
                if data is not None and data.shape == (h, w, 4):
                    self.shm_array[env_index, camera_index, :, :, :] = data[:, :, :3]
        self.frame_counter[0] = (int(self.frame_counter[0]) + 1) % (2**32)

    def run(self) -> None:
        while upstream.simulation_app.is_running():
            self._service_control_request()
            self._ros_executor.spin_once(timeout_sec=0.0)
            self.world.step(render=True)
        self._ros_executor.shutdown(timeout_sec=2.0)
        for sub in self.env_subscribers:
            sub.destroy_node()
        upstream.rclpy.shutdown()
        if self.shm:
            self.shm.close()
            self.shm.unlink()
        self.world.stop()
        upstream.simulation_app.close()


if __name__ == "__main__":
    renderer = LocalRLRenderer(upstream._args)
    renderer.setup()
    renderer.run()
