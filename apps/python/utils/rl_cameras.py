from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import mujoco  # type: ignore

SCENE_CAMERA_ROOT = "/SceneRender"
CAMERA_CLIPPING_RANGE = [0.01, 100.0]
CAMERA_HORIZONTAL_APERTURE = 32.0
CAMERA_VERTICAL_APERTURE = 24.0


@dataclass(frozen=True)
class RenderCamera:
    name: str
    prim: str
    topic: str
    frame_id: str
    body: str | None
    pos: list[float]
    quat: list[float]
    fovy: float

    def to_json(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "RenderCamera":
        return cls(
            name=str(data["name"]),
            prim=str(data["prim"]),
            topic=str(data["topic"]),
            frame_id=str(data["frame_id"]),
            body=(str(data["body"]) if data.get("body") is not None else None),
            pos=[float(value) for value in data["pos"]],  # type: ignore[index]
            quat=[float(value) for value in data["quat"]],  # type: ignore[index]
            fovy=float(data["fovy"]),
        )


def image_topic(camera_name: str, env_name: str = "env_0") -> str:
    return f"/{env_name}/{camera_name}/image_raw"


def camera_name_to_index(cameras: list[RenderCamera]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for index, camera in enumerate(cameras):
        if camera.name in mapping:
            raise RuntimeError(f"Duplicate camera name: {camera.name}")
        mapping[camera.name] = index
    return mapping


def cameras_json(cameras: list[RenderCamera]) -> str:
    payload = [{"name": camera.name, "prim": camera.prim} for camera in cameras]
    return json.dumps(payload, separators=(",", ":"))


def focal_length_from_fovy(fovy: float) -> float:
    radians = math.radians(float(fovy))
    return 0.5 * CAMERA_VERTICAL_APERTURE / math.tan(radians * 0.5)


def build_render_cameras(model, body_paths: dict[str, str]) -> list[RenderCamera]:
    cameras: list[RenderCamera] = []
    for cam_id in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)  # type: ignore
        if not name:
            raise RuntimeError(f"MuJoCo camera {cam_id} has no name.")
        body_id = int(model.cam_bodyid[cam_id])
        body = None if body_id <= 0 else mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)  # type: ignore
        if body is None:
            prim = f"{SCENE_CAMERA_ROOT}/{name}"
        else:
            parent = body_paths.get(body)
            if parent is None:
                raise RuntimeError(f"Camera '{name}' references unmapped body '{body}'.")
            prim = f"/{parent}/{name}"
        cameras.append(RenderCamera(
            name=name,
            prim=prim,
            topic=image_topic(name),
            frame_id=name,
            body=body,
            pos=[float(value) for value in model.cam_pos[cam_id]],
            quat=[float(value) for value in model.cam_quat[cam_id]],
            fovy=float(model.cam_fovy[cam_id]),
        ))
    if not cameras:
        raise RuntimeError("MuJoCo model has no cameras.")
    camera_name_to_index(cameras)
    return cameras
