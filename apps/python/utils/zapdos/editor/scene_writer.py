from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TypedDict

from pxr import Gf, Sdf, Usd, UsdGeom

from utils.zapdos.editor.placement import normalize_placement
from utils.zapdos.editor.state import OverlayInstance, OverlayPoseOverride, OverlayState, overlay_body_name


class SupportInfo(TypedDict):
    top_z: float


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
        return {"pos": list(override["pos"]), "quat": list(override["quat"]), "payload_quat": _payload_quat(placement)}
    if placement["kind"] == "world_pose":
        return {"pos": list(placement["pos"]), "quat": list(placement["quat"]), "payload_quat": _payload_quat(placement)}
    quat = _placement_quat(placement)
    payload_quat = _payload_quat(placement)
    min_z, _ = _rotated_asset_z_bounds(asset_bounds, _quat_mul(quat, payload_quat))
    if placement["kind"] == "floor_at_xy":
        return {
            "pos": [float(placement["xy"][0]), float(placement["xy"][1]), float(placement.get("z_offset", 0.0)) - min_z],
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
    resolved_support_infos = {body: {"top_z": float(info["top_z"])} for body, info in support_infos.items()}
    stage = Usd.Stage.CreateNew(output_path.as_posix())
    UsdGeom.SetStageUpAxis(stage, up_axis)
    UsdGeom.SetStageMetersPerUnit(stage, float(meters_per_unit))
    world = UsdGeom.Xform.Define(stage, "/World")
    world.GetPrim().GetReferences().AddReference(str(base_scene_usd.resolve()))
    for instance in overlay_state["instances"]:
        pose = resolve_instance_pose(
            instance,
            asset_bounds=asset_bounds_by_instance[instance["id"]],
            support_infos=resolved_support_infos,
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
        payload_quat = pose["payload_quat"]
        payload_xform = UsdGeom.Xformable(payload_prim)
        payload_xform.AddOrientOp().Set(Gf.Quatf(payload_quat[0], payload_quat[1], payload_quat[2], payload_quat[3]))
        if instance["asset_id"] in {"benchmark_table_000", "benchmark_building_blocks_006"}:
            _disable_asset_body_visual_collision(stage, asset_path)
            bounds = asset_bounds_by_instance[instance["id"]]
            mins = [float(value) for value in bounds["min"]]
            maxs = [float(value) for value in bounds["max"]]
            center = [0.5 * (mins[index] + maxs[index]) for index in range(3)]
            scale = [maxs[index] - mins[index] for index in range(3)]
            collision_name = "CollisionProxy"
            if instance["asset_id"] == "benchmark_table_000":
                payload_prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
                thickness = 0.012
                center[2] = maxs[2] - 0.5 * thickness
                scale[2] = thickness
                collision_name = "CollisionSupport"
            _add_hidden_box_collision(stage, object_path.AppendChild(collision_name), center, scale)
        object_prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(instance["motion"] == "static")
        if instance["motion"] == "dynamic":
            object_prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Double).Set(1.0)
        _, max_z = _rotated_asset_z_bounds(
            asset_bounds_by_instance[instance["id"]],
            _quat_mul(pose["quat"], pose["payload_quat"]),
        )
        resolved_support_infos[overlay_body_name(instance["id"])] = {"top_z": float(pose["pos"][2]) + max_z}
    stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().Save()
    return output_path


def _disable_asset_body_visual_collision(stage, asset_path: Sdf.Path) -> None:
    visual_prim = stage.OverridePrim(asset_path.AppendChild("entity").AppendChild("body").AppendChild("visual"))
    visual_prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)


def _add_hidden_box_collision(stage, path: Sdf.Path, center: list[float], scale: list[float]) -> None:
    collision = UsdGeom.Cube.Define(stage, path)
    collision.CreateSizeAttr(1.0)
    collision_prim = collision.GetPrim()
    collision_prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    UsdGeom.Imageable(collision_prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    collision_xform = UsdGeom.Xformable(collision_prim)
    collision_xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    collision_xform.AddScaleOp().Set(Gf.Vec3f(*scale))


def _placement_quat(placement: dict[str, object]) -> list[float]:
    if "quat" in placement:
        return [float(value) for value in placement["quat"]]
    yaw = float(placement.get("yaw", 0.0))
    return [float(math.cos(yaw / 2.0)), 0.0, 0.0, float(math.sin(yaw / 2.0))]


def _payload_quat(placement: dict[str, object]) -> list[float]:
    if "payload_quat" in placement:
        return [float(value) for value in placement["payload_quat"]]
    return [1.0, 0.0, 0.0, 0.0]


def _rotated_asset_z_bounds(asset_bounds: dict[str, list[float]], quat: list[float]) -> tuple[float, float]:
    w, x, y, z = _normalize_quat(quat)
    mins = asset_bounds["min"]
    maxs = asset_bounds["max"]
    min_z = math.inf
    max_z = -math.inf
    for corner_x in (mins[0], maxs[0]):
        for corner_y in (mins[1], maxs[1]):
            for corner_z in (mins[2], maxs[2]):
                rot_z = (
                    (2.0 * (x * z - y * w)) * float(corner_x)
                    + (2.0 * (y * z + x * w)) * float(corner_y)
                    + (1.0 - 2.0 * (x * x + y * y)) * float(corner_z)
                )
                min_z = min(min_z, rot_z)
                max_z = max(max_z, rot_z)
    return float(min_z), float(max_z)


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


__all__ = ["SupportInfo", "resolve_instance_pose", "write_overlay_scene"]
