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

from utils.renderer_ipc import request_path, response_path
from utils.rl_cameras import fovy_from_focal_length


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
            if request.get("op") != "snapshot_cameras":
                raise RuntimeError(f"Unsupported renderer op: {request.get('op')}")
            payload = {"id": request.get("id"), "ok": True, "cameras": self._camera_snapshot()}
        except Exception as err:
            payload = {"id": request.get("id"), "ok": False, "error": str(err)}
        res_path.write_text(json.dumps(payload), encoding="utf-8")
        req_path.unlink(missing_ok=True)

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
