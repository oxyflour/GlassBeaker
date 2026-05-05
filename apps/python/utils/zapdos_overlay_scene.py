from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TypedDict

from pxr import Gf, Sdf, Usd, UsdGeom

from utils.zapdos_overlay import OverlayInstance, OverlayPoseOverride, OverlayState, overlay_body_name


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
        return normalized
    if kind == "floor_at_xy":
        normalized["xy"] = _require_float_list(normalized, "xy", 2)
        normalized["z_offset"] = float(normalized.get("z_offset", 0.0))
        normalized["yaw"] = float(normalized.get("yaw", 0.0))
        return normalized
    if kind == "on_top_of_body":
        normalized["body"] = _require_nonempty_string(normalized, "body")
        normalized["xy"] = _require_float_list(normalized, "xy", 2)
        normalized["gap"] = float(normalized.get("gap", 0.0))
        normalized["yaw"] = float(normalized.get("yaw", 0.0))
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
    if body_name in pose_overrides:
        return pose_overrides[body_name]
    placement = normalize_placement(instance["placement"])
    if placement["kind"] == "world_pose":
        return {"pos": list(placement["pos"]), "quat": list(placement["quat"])}
    yaw = float(placement.get("yaw", 0.0))
    quat = [float(math.cos(yaw / 2.0)), 0.0, 0.0, float(math.sin(yaw / 2.0))]
    if placement["kind"] == "floor_at_xy":
        return {
            "pos": [
                float(placement["xy"][0]),
                float(placement["xy"][1]),
                float(placement.get("z_offset", 0.0)) - float(asset_bounds["min"][2]),
            ],
            "quat": quat,
        }
    support = support_infos[placement["body"]]
    return {
        "pos": [
            float(placement["xy"][0]),
            float(placement["xy"][1]),
            float(support["top_z"]) + float(placement.get("gap", 0.0)) - float(asset_bounds["min"][2]),
        ],
        "quat": quat,
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
    stage = Usd.Stage.CreateNew(output_path.as_posix())
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
        object_prim = UsdGeom.Xform.Define(stage, object_path).GetPrim()
        payload_prim = UsdGeom.Xform.Define(stage, payload_path).GetPrim()
        rel_asset_path = Path(
            os.path.relpath((assets_root / instance["url"]).resolve(), output_path.parent.resolve())
        ).as_posix()
        payload_prim.GetPayloads().AddPayload(rel_asset_path)
        xform = UsdGeom.Xformable(object_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(*pose["pos"]))
        quat = pose["quat"]
        xform.AddOrientOp().Set(Gf.Quatf(quat[0], quat[1], quat[2], quat[3]))
        xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
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
