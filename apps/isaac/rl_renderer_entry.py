from __future__ import annotations

import argparse
import json
import math
import os
import sys
from multiprocessing import shared_memory
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "apps" / "python"
SOURCE_ROOT = REPO_ROOT / "deps" / "genie_sim" / "source"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
os.environ.setdefault("SIM_REPO_ROOT", str(REPO_ROOT / "deps" / "genie_sim"))

from geniesim.rl.renderer.shm_layout import SHM_HEADER_BYTES as _SHM_HEADER_BYTES
from geniesim.rl.renderer.shm_layout import shm_total_bytes as _shm_total_bytes
from utils.camera_math import fovy_from_focal_length
from utils.zapdos.renderer.control_channel import request_path, response_path
from utils.zapdos.renderer.frame_policy import should_render_frame
from utils.zapdos.renderer.isaac_renderer_reload import (
    CameraBinding,
    rebuild_camera_bindings,
    reset_subscriber_caches,
    validate_camera_topology,
)
from utils.zapdos.ros.publish_config import configured_image_publish_specs

simulation_app: Any = None
rep: Any = None
GridCloner: Any = None
World: Any = None
is_prim_path_valid: Any = None
add_reference_to_stage: Any = None
get_current_stage: Any = None
Gf: Any = None
Sdf: Any = None
Usd: Any = None
UsdGeom: Any = None
UsdLux: Any = None
UsdShade: Any = None
rclpy: Any = None
QOS_BE: Any = None
RosNode: Any = None
ImageMsg: Any = None
EnvTFSubscriber: Any = None


def resolve_task_config(task_name: str, robot_type: str = "G2", instance_id: int = 0) -> dict[str, Any]:
    from geniesim.benchmark.config.robot_init_states import TASK_INFO_DICT
    from geniesim.benchmark.config.task_config_mapping import TASK_MAPPING
    from geniesim.utils import system_utils
    from geniesim.utils.infer_pre_process import TaskInfo

    if task_name not in TASK_MAPPING:
        raise ValueError(f"Unknown task '{task_name}'. Available: {list(TASK_MAPPING.keys())}")
    mapping = TASK_MAPPING[task_name]
    if robot_type not in mapping.get("background", {}):
        raise ValueError(
            f"Robot type '{robot_type}' not supported for task '{task_name}'. "
            f"Supported: {list(mapping['background'].keys())}"
        )

    bg_value = mapping["background"][robot_type]
    bg_name = bg_value[0] if isinstance(bg_value, list) else bg_value
    bg_json_path = Path(system_utils.benchmark_conf_path()) / "eval_tasks" / f"{bg_name}.json"
    bg_cfg = json.loads(bg_json_path.read_text(encoding="utf-8"))

    robot_cfg_name = bg_cfg["robot"].get("robot_cfg", "G1_120s.json")
    workspace = "workspace_00"
    init_pos = bg_cfg["robot"]["robot_init_pose"][workspace]["position"]
    init_quat = bg_cfg["robot"]["robot_init_pose"][workspace]["quaternion"]

    robot_cfg_path = Path(system_utils.app_root_path()) / "robot_cfg" / robot_cfg_name
    robot_cfg = json.loads(robot_cfg_path.read_text(encoding="utf-8"))["robot"]
    robot_usd = str(Path(system_utils.assets_path()) / robot_cfg["robot_usd"])
    robot_prim = robot_cfg.get("base_prim_path", "/robot")

    fg_usd = Path(system_utils.benchmark_conf_path()) / "llm_task" / task_name / str(instance_id) / "scene.usda"
    fg_usd = fg_usd.resolve()
    if not fg_usd.exists():
        raise FileNotFoundError(f"Foreground USD not found: {fg_usd}")

    robot_key = f"{robot_type}_omnipicker"
    task_states = TASK_INFO_DICT.get(task_name, {}).get(robot_key)
    init_joints = None
    if task_states is not None:
        ti = TaskInfo(task_states, robot_key)
        arm, head, waist, hand, gripper = ti.init_pose()
        init_joints = {"arm": arm, "head": head, "waist": waist, "gripper": gripper}

    print(
        f"[resolve_task_config] task={task_name} robot_type={robot_type} instance={instance_id}\n"
        f"  foreground_usd  = {fg_usd}\n"
        f"  robot_usd       = {robot_usd}\n"
        f"  robot_prim      = {robot_prim}\n"
        f"  init_pos        = {init_pos}\n"
        f"  init_quat       = {init_quat}\n"
        f"  init_joints     = {'present' if init_joints else 'none'}"
    )
    return {
        "foreground_usd": str(fg_usd),
        "robot_usd": robot_usd,
        "robot_prim_path": robot_prim,
        "robot_init_position": init_pos,
        "robot_init_quaternion": init_quat,
        "init_joint_positions": init_joints,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GlassBeaker IsaacSim camera renderer")
    parser.add_argument("--task-name", default="", help="Task name; auto-resolves scene/robot USDs")
    parser.add_argument("--robot-type", default="G2", choices=["G1", "G2"])
    parser.add_argument("--task-instance-id", type=int, default=0)
    parser.add_argument("--scene-usd", default="", help="Foreground scene USD path")
    parser.add_argument("--robot-usd", default="", help="Robot USDA path")
    parser.add_argument("--robot-prim", default="/robot", help="Robot prim path inside env")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--clone-spacing", type=float, default=3.0)
    parser.add_argument("--render-hz", type=float, default=30.0)
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--main-cam-prim", default="")
    parser.add_argument("--wrist-cam-prim", default="")
    parser.add_argument("--cameras-json", default="")
    parser.add_argument("--shm-name", default="geniesim_frames")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--ros-domain-id", type=int, default=0)
    parser.add_argument("--auto-dome-light", action="store_true")
    parser.add_argument("--auto-dome-light-intensity", type=float, default=600.0)
    parser.add_argument("--default-cam-pos", nargs=3, type=float, default=[-0.2, -1.8, 1.8])
    parser.add_argument("--default-cam-target", nargs=3, type=float, default=[-0.4, 0.0, 0.9])
    return parser


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, dict[str, Any] | None]:
    parser = build_parser()
    args = parser.parse_args(argv)
    task_resolved = None
    if args.task_name:
        task_resolved = resolve_task_config(args.task_name, args.robot_type, args.task_instance_id)
        args.scene_usd = task_resolved["foreground_usd"]
        args.robot_usd = task_resolved["robot_usd"]
        args.robot_prim = task_resolved["robot_prim_path"]
        if not args.main_cam_prim:
            args.main_cam_prim = "/default_viz_camera"
    else:
        if not args.scene_usd:
            parser.error("Either --task-name or --scene-usd must be provided.")
        if not args.main_cam_prim:
            args.main_cam_prim = "/camera_main"
    return args, task_resolved


def launch_simulation_app(args: argparse.Namespace):
    import isaacsim
    from isaacsim import SimulationApp

    launch_config: dict[str, Any] = {
        "headless": bool(args.headless),
        "multi_gpu": False,
        "width": int(args.cam_width),
        "height": int(args.cam_height),
    }
    experience = ""
    if args.headless:
        launch_config.update({
            "disable_viewport_updates": os.environ.get("GB_RENDERER_DISABLE_VIEWPORT_UPDATES", "0") == "1",
            "hide_ui": True,
            "create_new_stage": False,
            "limit_cpu_threads": 16,
        })
        isaac_root = Path(isaacsim.__file__).resolve().parent
        extra_args: list[str] = []
        for folder in ("extscache", "extsUser", "extsDeprecated"):
            extra_args.extend(["--ext-folder", str(isaac_root / folder)])
        extra_args.append("--/app/vulkan=false")
        extra_args.append("--/renderer/gpuEnumeration/glInterop/enabled=true")
        extra_args.append("--/app/extensions/registryEnabled=false")
        launch_config["extra_args"] = extra_args
        base_experience = isaac_root / "apps" / "isaacsim.exp.base.kit"
        experience = os.environ.get("GB_ISAAC_HEADLESS_EXPERIENCE", str(base_experience))
    return SimulationApp(launch_config, experience)


def load_isaac_runtime() -> None:
    global rep, GridCloner, World, is_prim_path_valid, add_reference_to_stage
    global get_current_stage, Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
    global rclpy, QOS_BE, RosNode, ImageMsg, EnvTFSubscriber

    import omni.replicator.core as _rep
    from omni.isaac.cloner import GridCloner as _GridCloner
    from omni.isaac.core import World as _World
    from omni.isaac.core.utils.prims import is_prim_path_valid as _is_prim_path_valid
    from omni.isaac.core.utils.stage import add_reference_to_stage as _add_reference_to_stage
    from omni.isaac.core.utils.stage import get_current_stage as _get_current_stage
    from pxr import Gf as _Gf
    from pxr import Sdf as _Sdf
    from pxr import Usd as _Usd
    from pxr import UsdGeom as _UsdGeom
    from pxr import UsdLux as _UsdLux
    from pxr import UsdShade as _UsdShade
    import rclpy as _rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import Image as _ImageMsg
    from tf2_msgs.msg import TFMessage

    rep = _rep
    GridCloner = _GridCloner
    World = _World
    is_prim_path_valid = _is_prim_path_valid
    add_reference_to_stage = _add_reference_to_stage
    get_current_stage = _get_current_stage
    Gf = _Gf
    Sdf = _Sdf
    Usd = _Usd
    UsdGeom = _UsdGeom
    UsdLux = _UsdLux
    UsdShade = _UsdShade
    rclpy = _rclpy
    QOS_BE = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
    )
    RosNode = Node
    ImageMsg = _ImageMsg
    EnvTFSubscriber = _build_env_tf_subscriber_class(Node, TFMessage)


def _build_env_tf_subscriber_class(NodeBase: Any, _TFMessage: Any):
    class _EnvTFSubscriber(NodeBase):
        def __init__(
            self,
            env_id: int,
            stage: Any,
            env_root_prim_path: str,
            body_name_map: dict[str, str],
            robot_prim: str,
        ):
            super().__init__(f"geniesim_renderer_env{env_id}")
            self.env_id = env_id
            self.stage = stage
            self.env_root = env_root_prim_path
            self.body_name_map = body_name_map
            self._tf_cache = None
            self._dirty = False
            self._robot_root_rel = robot_prim.lstrip("/") if robot_prim else "robot"
            self._attr_cache: dict[str, Any] = {}
            self._ordered_attrs: list[Any] | None = None
            self.create_subscription(_TFMessage, f"/env_{env_id}/tf_render", self._on_tf, QOS_BE)

        @staticmethod
        def _normalize_frame_id(frame_id: str) -> str:
            frame = (frame_id or "").strip().lstrip("/")
            if frame.startswith("World/"):
                frame = frame[len("World/"):]
            if frame.startswith("objects/"):
                frame = "Objects/" + frame[len("objects/"):]
            return frame

        def _on_tf(self, msg: Any) -> None:
            self._tf_cache = msg
            self._dirty = True

        def apply_tf(self) -> None:
            if not self._dirty or self._tf_cache is None:
                return
            self._dirty = False
            transforms = self._tf_cache.transforms
            if self._ordered_attrs is None:
                for tf in transforms:
                    body_name = tf.child_frame_id
                    if body_name in self._attr_cache:
                        continue
                    mapped = self._robot_root_rel if body_name in ("base_link", "chassis_site") else self.body_name_map.get(body_name, body_name)
                    prim_name = self._normalize_frame_id(mapped)
                    prim = self.stage.GetPrimAtPath(f"{self.env_root}/{prim_name}")
                    if not prim.IsValid():
                        self._attr_cache[body_name] = None
                        continue
                    translate_attr = prim.GetAttribute("xformOp:translate")
                    orient_attr = prim.GetAttribute("xformOp:orient")
                    is_quatf = orient_attr.IsValid() and str(orient_attr.GetTypeName()) == "quatf"
                    self._attr_cache[body_name] = (
                        translate_attr if translate_attr.IsValid() else None,
                        orient_attr if orient_attr.IsValid() else None,
                        is_quatf,
                    )
                self._ordered_attrs = [self._attr_cache.get(tf.child_frame_id) for tf in transforms]

            for tf, entry in zip(transforms, self._ordered_attrs):
                if entry is None:
                    continue
                translate_attr, orient_attr, is_quatf = entry
                t = tf.transform.translation
                r = tf.transform.rotation
                if translate_attr is not None:
                    translate_attr.Set(Gf.Vec3d(t.x, t.y, t.z))
                if orient_attr is not None:
                    quat = Gf.Quatf(r.w, r.x, r.y, r.z) if is_quatf else Gf.Quatd(r.w, r.x, r.y, r.z)
                    orient_attr.Set(quat)

    return _EnvTFSubscriber


class RLRenderer:
    def __init__(self, args: argparse.Namespace, task_resolved: dict[str, Any] | None = None):
        self.args = args
        self.task_resolved = task_resolved
        self.num_envs = args.num_envs
        self.stage = None
        self.world = None
        self.env_subscribers: list[Any] = []
        self.cam_annotators_main: list[Any] = []
        self.cam_annotators_wrist: list[Any] = []
        self._ros_publish_node = None
        self._ros_image_publishers: dict[str, Any] = {}
        self._ros_image_specs_by_camera: dict[int, list[Any]] = {}
        self.shm = None
        self.shm_array = None
        if getattr(args, "ros_domain_id", None) is not None:
            os.environ["ROS_DOMAIN_ID"] = str(int(args.ros_domain_id))
        rclpy.init()

    def setup(self) -> None:
        if os.environ.get("GB_RENDERER_SETUP_TRACE"):
            print("[RLRenderer] setup:start", flush=True)
        self.world = World(stage_units_in_meters=1.0, physics_dt=0.0, rendering_dt=1.0 / self.args.render_hz)
        self.stage = get_current_stage()
        env_paths = self._rebuild_scene_clones()
        if os.environ.get("GB_RENDERER_SETUP_TRACE"):
            print("[RLRenderer] setup:after_rebuild_scene_clones", flush=True)
        self.world.reset()
        if os.environ.get("GB_RENDERER_SETUP_TRACE"):
            print("[RLRenderer] setup:after_world_reset", flush=True)
        body_name_map = self._build_body_name_map(env_paths[0])
        print(f"[RLRenderer] body_name_map built: {len(body_name_map)} prims indexed under env_0")
        self._camera_list = self._parse_camera_list()
        self._num_cams = len(self._camera_list)
        self._setup_ros_image_publishers()
        self.cam_annotators_all = [[] for _ in range(self._num_cams)]
        for index, env_path in enumerate(env_paths):
            for camera_index, camera in enumerate(self._camera_list):
                binding = self._setup_camera(env_path + str(camera["prim"]))
                self.cam_annotators_all[camera_index].append(binding)
            self.cam_annotators_main.append(self.cam_annotators_all[0][index].annotator if self._num_cams > 0 else None)
            self.cam_annotators_wrist.append(self.cam_annotators_all[1][index].annotator if self._num_cams > 1 else None)

        for index, env_path in enumerate(env_paths):
            self.env_subscribers.append(EnvTFSubscriber(index, self.stage, env_path, body_name_map, self.args.robot_prim))
        self._ros_executor = rclpy.executors.SingleThreadedExecutor()
        for sub in self.env_subscribers:
            self._ros_executor.add_node(sub)
        if self._ros_publish_node is not None:
            self._ros_executor.add_node(self._ros_publish_node)

        height, width = self.args.cam_height, self.args.cam_width
        shm_bytes = _shm_total_bytes(self.num_envs, height, width, num_cams=self._num_cams)
        try:
            self.shm = shared_memory.SharedMemory(name=self.args.shm_name, create=True, size=shm_bytes)
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=self.args.shm_name, create=False, size=shm_bytes)
        self.shm_array = np.ndarray(
            (self.num_envs, self._num_cams, height, width, 3),
            dtype=np.uint8,
            buffer=self.shm.buf,
            offset=_SHM_HEADER_BYTES,
        )
        self.frame_counter = np.ndarray((1,), dtype=np.uint32, buffer=self.shm.buf, offset=0)
        self.frame_counter[0] = 0
        self.world.add_render_callback("geniesim_rl_render", self._render_callback)
        print(f"[RLRenderer] Ready | envs={self.num_envs} | shm={self.args.shm_name} | {shm_bytes // 1024}KB")

    def _parse_camera_list(self) -> list[dict[str, Any]]:
        if self.args.cameras_json:
            try:
                cameras = json.loads(self.args.cameras_json)
                if isinstance(cameras, list):
                    return cameras
            except Exception:
                pass
        cameras = [{"name": "main", "prim": self.args.main_cam_prim}]
        if self.args.wrist_cam_prim:
            cameras.append({"name": "wrist", "prim": self.args.wrist_cam_prim})
        return cameras

    def _setup_ros_image_publishers(self) -> None:
        camera_indices = {str(camera["name"]): index for index, camera in enumerate(self._camera_list)}
        specs = configured_image_publish_specs(camera_indices.keys())
        if not specs:
            return
        self._ros_publish_node = RosNode("geniesim_renderer_publish")
        for spec in specs:
            camera_index = camera_indices.get(spec.camera_name)
            if camera_index is None:
                continue
            self._ros_image_specs_by_camera.setdefault(camera_index, []).append(spec)
            self._ros_image_publishers[spec.topic] = self._ros_publish_node.create_publisher(
                ImageMsg,
                spec.topic,
                QOS_BE,
            )

    def _create_default_viz_camera(self, env_path: str, cam_pos: list[float], cam_target: list[float]) -> str:
        camera_path = env_path + "/default_viz_camera"
        if is_prim_path_valid(camera_path):
            return camera_path
        camera = UsdGeom.Camera.Define(self.stage, camera_path)
        xf = UsdGeom.Xformable(camera.GetPrim())
        w, x, y, z = self._lookat_quaternion(cam_pos, cam_target)
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*cam_pos))
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(w, x, y, z))
        camera.CreateFocalLengthAttr(18.0)
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(15.2908)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        print(f"[RLRenderer] Default viz camera created at {camera_path}")
        return camera_path

    @staticmethod
    def _lookat_quaternion(
        eye: list[float],
        target: list[float],
        world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> tuple[float, float, float, float]:
        def normalize(value):
            norm = math.sqrt(sum(component * component for component in value))
            return tuple(component / norm for component in value) if norm > 1e-9 else value

        def cross(left, right):
            return (
                left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0],
            )

        forward = normalize(tuple(t - e for t, e in zip(target, eye)))
        right = normalize(cross(forward, world_up))
        up = cross(right, forward)
        local_z = tuple(-component for component in forward)
        m00, m01, m02 = right[0], up[0], local_z[0]
        m10, m11, m12 = right[1], up[1], local_z[1]
        m20, m21, m22 = right[2], up[2], local_z[2]
        trace = m00 + m11 + m22
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w, x, y, z = 0.25 / s, (m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s
        elif m00 > m11 and m00 > m22:
            s = 2.0 * math.sqrt(max(0.0, 1.0 + m00 - m11 - m22))
            w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * math.sqrt(max(0.0, 1.0 + m11 - m00 - m22))
            w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
        else:
            s = 2.0 * math.sqrt(max(0.0, 1.0 + m22 - m00 - m11))
            w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        return (w / norm, x / norm, y / norm, z / norm)

    def _set_robot_base_pose(self, robot_prim_path: str, position: list[float], quaternion: list[float]) -> None:
        prim = self.stage.GetPrimAtPath(robot_prim_path)
        if not prim.IsValid():
            print(f"[RLRenderer] Warning: robot prim not found at {robot_prim_path}, skipping pose init")
            return
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*position))
        w, x, y, z = quaternion
        existing_attr = prim.GetAttribute("xformOp:orient")
        existing_type = str(existing_attr.GetTypeName()) if existing_attr.IsValid() else ""
        precision = UsdGeom.XformOp.PrecisionDouble if existing_type == "quatd" else UsdGeom.XformOp.PrecisionFloat
        quat = Gf.Quatd(w, x, y, z) if existing_type == "quatd" else Gf.Quatf(w, x, y, z)
        xf.AddOrientOp(precision).Set(quat)
        print(f"[RLRenderer] Robot base pose set: pos={position} quat(wxyz)={quaternion}")

    def _build_body_name_map(self, env0_path: str) -> dict[str, str]:
        map_path = self.args.scene_usd.replace(".usd", "_body_map.json").replace(".usda", "_body_map.json")
        if os.path.exists(map_path):
            return json.loads(Path(map_path).read_text(encoding="utf-8"))
        prefix = env0_path.rstrip("/") + "/"
        leaf_map: dict[str, str] = {}
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(prefix):
                continue
            rel = path[len(prefix):]
            leaf = rel.split("/")[-1]
            if leaf and leaf not in leaf_map:
                leaf_map[leaf] = rel
        return leaf_map

    def _setup_camera(self, cam_prim_path: str) -> CameraBinding:
        resolved_path = cam_prim_path
        if not is_prim_path_valid(resolved_path):
            parts = resolved_path.strip("/").split("/")
            env_root = "/" + "/".join(parts[:3]) if len(parts) >= 3 else resolved_path
            fallback = self._find_first_camera_under_env(env_root)
            if not fallback:
                print(f"[RLRenderer] Camera prim not found: {cam_prim_path}")
                return CameraBinding(annotator=None, render_product=None)
            resolved_path = fallback
        render_product = rep.create.render_product(resolved_path, (self.args.cam_width, self.args.cam_height))
        if os.environ.get("GB_RENDERER_DEBUG_FRAMES"):
            print(f"[RLRenderer] render_product camera={resolved_path} rp={getattr(render_product, 'path', render_product)}", flush=True)
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([render_product])
        return CameraBinding(annotator=annotator, render_product=render_product)

    def _find_first_camera_under_env(self, env_root: str) -> str | None:
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith(env_root + "/") and prim.IsA(UsdGeom.Camera):
                return path
        return None

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
            with Usd.EditContext(self.stage, layer):
                prim = self.stage.GetPrimAtPath(render_product)
                if prim.IsValid():
                    self.stage.RemovePrim(render_product)

    def _release_camera_binding(self, binding: CameraBinding) -> None:
        if binding.annotator is not None and binding.render_product is not None:
            binding.annotator.detach([binding.render_product])
        self._destroy_render_product(binding.render_product)

    def _rebuild_scene_clones(self) -> list[str]:
        if os.environ.get("GB_RENDERER_SETUP_TRACE"):
            print("[RLRenderer] rebuild_scene:start", flush=True)
        env_root = "/World/envs"
        if self.stage is None:
            raise RuntimeError("renderer stage not initialized")
        if self.stage.GetPrimAtPath(env_root).IsValid():
            self.stage.RemovePrim(env_root)
        cloner = GridCloner(spacing=self.args.clone_spacing)
        cloner.define_base_env(env_root)
        env_paths = cloner.generate_paths("/World/envs/env", self.num_envs)
        if os.environ.get("GB_RENDERER_SETUP_TRACE"):
            print("[RLRenderer] rebuild_scene:before_scene_reference", flush=True)
        add_reference_to_stage(self.args.scene_usd, env_paths[0])
        if os.environ.get("GB_RENDERER_SETUP_TRACE"):
            print("[RLRenderer] rebuild_scene:after_scene_reference", flush=True)
        if self.args.robot_usd:
            robot_prim_path = env_paths[0] + self.args.robot_prim
            if os.environ.get("GB_RENDERER_SETUP_TRACE"):
                print("[RLRenderer] rebuild_scene:before_robot_reference", flush=True)
            add_reference_to_stage(self.args.robot_usd, robot_prim_path)
            if os.environ.get("GB_RENDERER_SETUP_TRACE"):
                print("[RLRenderer] rebuild_scene:after_robot_reference", flush=True)
            if self.task_resolved is not None:
                if os.environ.get("GB_RENDERER_SETUP_TRACE"):
                    print("[RLRenderer] rebuild_scene:before_set_robot_pose", flush=True)
                self._set_robot_base_pose(
                    robot_prim_path,
                    self.task_resolved["robot_init_position"],
                    self.task_resolved["robot_init_quaternion"],
                )
                if os.environ.get("GB_RENDERER_SETUP_TRACE"):
                    print("[RLRenderer] rebuild_scene:after_set_robot_pose", flush=True)
        if self.args.main_cam_prim == "/default_viz_camera":
            self._create_default_viz_camera(env_paths[0], self.args.default_cam_pos, self.args.default_cam_target)
        if os.environ.get("GB_RENDERER_SETUP_TRACE"):
            print("[RLRenderer] rebuild_scene:before_clone", flush=True)
        cloner.clone(source_prim_path=env_paths[0], prim_paths=env_paths, copy_from_source=True, replicate_physics=False)
        if os.environ.get("GB_RENDERER_SETUP_TRACE"):
            print("[RLRenderer] rebuild_scene:after_clone", flush=True)
        self._add_ground_plane(env_paths[0])
        if self._count_lights_under_env(env_paths[0]) == 0:
            dome = UsdLux.DomeLight.Define(self.stage, env_paths[0] + "/AutoDomeLight")
            dome.CreateIntensityAttr(float(self.args.auto_dome_light_intensity))
        if os.environ.get("GB_RENDERER_SETUP_TRACE"):
            print("[RLRenderer] rebuild_scene:done", flush=True)
        return env_paths

    def _add_ground_plane(self, env_path: str, size: float = 100.0) -> None:
        floor_path = env_path + "/GroundPlane"
        mat_path = env_path + "/GroundPlaneMat"
        half = size / 2.0
        mesh = UsdGeom.Mesh.Define(self.stage, floor_path)
        mesh.CreatePointsAttr([
            Gf.Vec3f(-half, -half, 0.0),
            Gf.Vec3f(half, -half, 0.0),
            Gf.Vec3f(half, half, 0.0),
            Gf.Vec3f(-half, half, 0.0),
        ])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)] * 4)
        mesh.SetNormalsInterpolation("vertex")
        material = UsdShade.Material.Define(self.stage, mat_path)
        shader = UsdShade.Shader.Define(self.stage, mat_path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.459, 0.636, 0.922))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(mesh).Bind(material)

    def _count_lights_under_env(self, env_root: str) -> int:
        count = 0
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            type_name = prim.GetTypeName()
            if path.startswith(env_root + "/") and isinstance(type_name, str) and "Light" in type_name:
                count += 1
        return count

    def _camera_snapshot(self) -> list[dict[str, object]]:
        env_root = "/World/envs/env_0"
        cameras: list[dict[str, object]] = []
        for camera in getattr(self, "_camera_list", []):
            prim_path = env_root + str(camera["prim"])
            prim = self.stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise RuntimeError(f"Camera prim not found: {prim_path}")
            usd_camera = UsdGeom.Camera(prim)
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

    def _reload_scene(self, scene_usd: str, cameras: list[dict[str, object]]) -> None:
        validate_camera_topology(self._camera_list, cameras)
        rebuild_camera_bindings([], [], self.cam_annotators_all, lambda path: CameraBinding(None, None), self._release_camera_binding)
        self.args.scene_usd = scene_usd
        env_paths = self._rebuild_scene_clones()
        self._camera_list = list(cameras)
        self._num_cams = len(self._camera_list)
        body_name_map = self._build_body_name_map(env_paths[0])
        reset_subscriber_caches(self.env_subscribers, body_name_map)
        self.world.reset()
        self.cam_annotators_all = rebuild_camera_bindings(env_paths, self._camera_list, [], self._setup_camera, self._release_camera_binding)
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

    def _publish_image_frame(self, env_index: int, camera_index: int, frame: np.ndarray) -> None:
        if env_index != 0:
            return
        specs = getattr(self, "_ros_image_specs_by_camera", {}).get(camera_index, [])
        if not specs:
            return
        publish_node = getattr(self, "_ros_publish_node", None)
        if publish_node is None:
            return
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        height, width = int(frame.shape[0]), int(frame.shape[1])
        payload = frame.tobytes()
        stamp = publish_node.get_clock().now().to_msg()
        for spec in specs:
            publisher = getattr(self, "_ros_image_publishers", {}).get(spec.topic)
            if publisher is None:
                continue
            msg = ImageMsg()
            msg.header.stamp = stamp
            msg.header.frame_id = spec.camera_name
            msg.height = height
            msg.width = width
            msg.encoding = "rgb8"
            msg.is_bigendian = 0
            msg.step = width * 3
            msg.data = payload
            publisher.publish(msg)

    def _render_callback(self, step_size: float) -> None:
        del step_size
        force_render = os.environ.get("GB_RENDERER_FORCE_RENDER") == "1"
        if not force_render and not should_render_frame(self.env_subscribers, int(self.frame_counter[0])):
            return
        with Sdf.ChangeBlock():
            for sub in self.env_subscribers:
                sub.apply_tf()
        height, width = self.args.cam_height, self.args.cam_width
        copied_frame = False
        for env_index in range(self.num_envs):
            for camera_index in range(self._num_cams):
                binding = self.cam_annotators_all[camera_index][env_index]
                annotator = binding.annotator
                if annotator is None:
                    continue
                data = annotator.get_data()
                if isinstance(data, dict):
                    data = data.get("data")
                if data is None:
                    continue
                if os.environ.get("GB_RENDERER_DEBUG_FRAMES") and int(self.frame_counter[0]) < 5:
                    shape = getattr(data, "shape", None)
                    size = getattr(data, "size", None)
                    data_sum = int(data.sum()) if size else 0
                    print(f"[RLRenderer] annotator env={env_index} camera={camera_index} shape={shape} size={size} sum={data_sum}", flush=True)
                rgb_frame = None
                if data.shape == (height, width, 4):
                    rgb_frame = data[:, :, :3]
                elif data.shape == (height, width, 3):
                    rgb_frame = data
                elif data.size == height * width * 4:
                    rgb_frame = data.reshape(height, width, 4)[:, :, :3]
                elif data.size == height * width * 3:
                    rgb_frame = data.reshape(height, width, 3)
                if rgb_frame is None:
                    continue
                self.shm_array[env_index, camera_index, :, :, :] = rgb_frame
                self._publish_image_frame(env_index, camera_index, rgb_frame)
                copied_frame = True
        if copied_frame:
            self.frame_counter[0] = (int(self.frame_counter[0]) + 1) % (2**32)

    def run(self) -> None:
        while simulation_app.is_running():
            self._service_control_request()
            self._ros_executor.spin_once(timeout_sec=0.0)
            force_render = os.environ.get("GB_RENDERER_FORCE_RENDER") == "1"
            if force_render or should_render_frame(self.env_subscribers, int(self.frame_counter[0])):
                rep.orchestrator.step(wait_for_render=True)
            self.world.step(render=True)
        self._ros_executor.shutdown(timeout_sec=2.0)
        for sub in self.env_subscribers:
            sub.destroy_node()
        publish_node = getattr(self, "_ros_publish_node", None)
        if publish_node is not None:
            publish_node.destroy_node()
        rclpy.shutdown()
        if self.shm:
            self.shm.close()
            self.shm.unlink()
        self.world.stop()
        simulation_app.close()


def main(argv: list[str] | None = None) -> None:
    args, task_resolved = parse_args(argv)
    global simulation_app
    simulation_app = launch_simulation_app(args)
    load_isaac_runtime()
    renderer = RLRenderer(args, task_resolved)
    renderer.setup()
    renderer.run()


if __name__ == "__main__":
    main()
