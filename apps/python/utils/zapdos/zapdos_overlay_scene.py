from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TypedDict

from pxr import Gf, Sdf, Usd, UsdGeom

from utils.zapdos.zapdos_overlay import OverlayInstance, OverlayPoseOverride, OverlayState, overlay_body_name


class SupportInfo(TypedDict):
    top_z: float


def normalize_placement(placement: object):
    if not isinstance(placement, dict):
        raise ValueError("placement must be an object")
    normalized = dict(placement)
    if "position" in normalized and "pos" not in normalized:
        normalized["pos"] = normalized["position"]
    if "orientation" in normalized and "quat" not in normalized:
        normalized["quat"] = normalized["orientation"]
    kind = normalized.get("kind")
    if kind is None:
        kind = _infer_placement_kind(normalized)
        if kind is None:
            raise ValueError("placement.kind is required")
        normalized["kind"] = kind
    if kind == "world_pose":
        normalized["pos"] = _require_float_list(normalized, "pos", 3)
        normalized["quat"] = _require_float_list(normalized, "quat", 4)
        if "payload_quat" in normalized:
            normalized["payload_quat"] = _require_float_list(normalized, "payload_quat", 4)
        return normalized
    if kind == "floor_at_xy":
        normalized["xy"] = _require_float_list(normalized, "xy", 2)
        normalized["z_offset"] = float(normalized.get("z_offset", 0.0))
        if "quat" in normalized:
            normalized["quat"] = _require_float_list(normalized, "quat", 4)
        else:
            normalized["yaw"] = float(normalized.get("yaw", 0.0))
        if "payload_quat" in normalized:
            normalized["payload_quat"] = _require_float_list(normalized, "payload_quat", 4)
        return normalized
    if kind == "on_top_of_body":
        normalized["body"] = _require_nonempty_string(normalized, "body")
        normalized["xy"] = _require_float_list(normalized, "xy", 2)
        normalized["gap"] = float(normalized.get("gap", 0.0))
        if "quat" in normalized:
            normalized["quat"] = _require_float_list(normalized, "quat", 4)
        else:
            normalized["yaw"] = float(normalized.get("yaw", 0.0))
        if "payload_quat" in normalized:
            normalized["payload_quat"] = _require_float_list(normalized, "payload_quat", 4)
        return normalized
    raise ValueError(f"Unsupported placement.kind: {kind}")


def resolve_instance_pose(
    instance: OverlayInstance,
    *,
    asset_bounds: dict[str, list[float]],
    support_infos: dict[str, SupportInfo],
    pose_overrides: dict[str, OverlayPoseOverride],
) -> dict[str, list[float]]:
    body_name = overlay_body_name(instance["id"])
    placement = normalize_placement(instance["placement"])
    if body_name in pose_overrides:
        override = pose_overrides[body_name]
        return {
            "pos": list(override["pos"]),
            "quat": list(override["quat"]),
            "payload_quat": _payload_quat(placement),
        }
    if placement["kind"] == "world_pose":
        return {
            "pos": list(placement["pos"]),
            "quat": list(placement["quat"]),
            "payload_quat": _payload_quat(placement),
        }
    quat = _placement_quat(placement)
    payload_quat = _payload_quat(placement)
    min_z = _rotated_asset_min_z(asset_bounds, _quat_mul(quat, payload_quat))
    if placement["kind"] == "floor_at_xy":
        return {
            "pos": [
                float(placement["xy"][0]),
                float(placement["xy"][1]),
                float(placement.get("z_offset", 0.0)) - min_z,
            ],
            "quat": quat,
            "payload_quat": payload_quat,
        }
    support = support_infos[placement["body"]]
    return {
        "pos": [
            float(placement["xy"][0]),
            float(placement["xy"][1]),
            float(support["top_z"]) + float(placement.get("gap", 0.0)) - min_z,
        ],
        "quat": quat,
        "payload_quat": payload_quat,
    }


def write_overlay_scene(
    output_path: Path,
    base_scene_usd: Path,
    assets_root: Path,
    overlay_state: OverlayState,
    *,
    support_infos: dict[str, SupportInfo],
    asset_bounds_by_instance: dict[str, dict[str, list[float]]],
) -> Path:
    base_stage = Usd.Stage.Open(base_scene_usd.as_posix())
    up_axis = UsdGeom.GetStageUpAxis(base_stage) if base_stage is not None else UsdGeom.Tokens.z
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(base_stage) if base_stage is not None else 1.0
    stage = Usd.Stage.CreateNew(output_path.as_posix())
    UsdGeom.SetStageUpAxis(stage, up_axis)
    UsdGeom.SetStageMetersPerUnit(stage, float(meters_per_unit))
    world = UsdGeom.Xform.Define(stage, "/World")
    world.GetPrim().GetReferences().AddReference(str(base_scene_usd.resolve()))
    for instance in overlay_state["instances"]:
        pose = resolve_instance_pose(
            instance,
            asset_bounds=asset_bounds_by_instance[instance["id"]],
            support_infos=support_infos,
            pose_overrides=overlay_state["pose_overrides"],
        )
        object_path = Sdf.Path(f"/World/{instance['id']}")
        payload_path = object_path.AppendChild("Payload")
        asset_path = payload_path.AppendChild("Asset")
        object_prim = UsdGeom.Xform.Define(stage, object_path).GetPrim()
        payload_prim = UsdGeom.Xform.Define(stage, payload_path).GetPrim()
        asset_prim = UsdGeom.Xform.Define(stage, asset_path).GetPrim()
        rel_asset_path = Path(
            os.path.relpath((assets_root / instance["url"]).resolve(), output_path.parent.resolve())
        ).as_posix()
        asset_prim.GetPayloads().AddPayload(rel_asset_path)
        xform = UsdGeom.Xformable(object_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(*pose["pos"]))
        quat = pose["quat"]
        xform.AddOrientOp().Set(Gf.Quatf(quat[0], quat[1], quat[2], quat[3]))
        xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        payload_xform = UsdGeom.Xformable(payload_prim)
        payload_quat = pose["payload_quat"]
        payload_xform.AddOrientOp().Set(Gf.Quatf(
            payload_quat[0],
            payload_quat[1],
            payload_quat[2],
            payload_quat[3],
        ))
        object_prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(
            instance["motion"] == "static"
        )
        if instance["motion"] == "dynamic":
            object_prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Double).Set(1.0)
    stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().Save()
    return output_path


def _infer_placement_kind(placement: dict[str, object]) -> str | None:
    if "pos" in placement or "quat" in placement:
        if "pos" in placement and "quat" not in placement:
            placement["quat"] = [1.0, 0.0, 0.0, 0.0]
        return "world_pose" if "pos" in placement and "quat" in placement else None
    if "body" in placement:
        return "on_top_of_body" if "xy" in placement else None
    if "xy" in placement:
        return "floor_at_xy"
    return None


def _placement_quat(placement: dict[str, object]) -> list[float]:
    if "quat" in placement:
        return [float(value) for value in placement["quat"]]
    yaw = float(placement.get("yaw", 0.0))
    return [float(math.cos(yaw / 2.0)), 0.0, 0.0, float(math.sin(yaw / 2.0))]


def _payload_quat(placement: dict[str, object]) -> list[float]:
    if "payload_quat" in placement:
        return [float(value) for value in placement["payload_quat"]]
    return [1.0, 0.0, 0.0, 0.0]


def _rotated_asset_min_z(asset_bounds: dict[str, list[float]], quat: list[float]) -> float:
    w, x, y, z = _normalize_quat(quat)
    mins = asset_bounds["min"]
    maxs = asset_bounds["max"]
    min_z = math.inf
    for corner_x in (mins[0], maxs[0]):
        for corner_y in (mins[1], maxs[1]):
            for corner_z in (mins[2], maxs[2]):
                rot_z = (
                    (2.0 * (x * z - y * w)) * float(corner_x)
                    + (2.0 * (y * z + x * w)) * float(corner_y)
                    + (1.0 - 2.0 * (x * x + y * y)) * float(corner_z)
                )
                min_z = min(min_z, rot_z)
    return float(min_z)


def _normalize_quat(quat: list[float]) -> tuple[float, float, float, float]:
    w, x, y, z = (float(value) for value in quat)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("placement.quat must be non-zero")
    return w / norm, x / norm, y / norm, z / norm


def _quat_mul(lhs: list[float], rhs: list[float]) -> list[float]:
    lw, lx, ly, lz = _normalize_quat(lhs)
    rw, rx, ry, rz = _normalize_quat(rhs)
    return [
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]


def _require_float_list(placement: dict[str, object], key: str, length: int) -> list[float]:
    value = placement.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"placement.{key} must be a {length}-item list")
    return [float(item) for item in value]


def _require_nonempty_string(placement: dict[str, object], key: str) -> str:
    value = placement.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"placement.{key} must be a non-empty string")
    return value

