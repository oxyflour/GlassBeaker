from __future__ import annotations

from pathlib import Path

from pxr import Gf, Usd, UsdGeom

from utils.rl_cameras import (
    RenderCamera,
    SCENE_CAMERA_ROOT,
    focal_length_from_fovy,
)
from utils.usd_to_mjcf import sanitize_name

SKIP_TYPES = {"Scope", "Material", "Shader", "NodeGraph", "Camera"}


def compose_stage_metadata(scene_usd: Path, robot_usd: Path | None = None) -> tuple[str, float]:
    for candidate in (scene_usd, robot_usd):
        if candidate is None:
            continue
        stage = Usd.Stage.Open(str(candidate))
        if stage is None:
            continue
        up_axis = str(UsdGeom.GetStageUpAxis(stage) or "Z").upper()
        meters = stage.GetMetadata("metersPerUnit")
        if meters is not None:
            return up_axis, float(meters)
        return up_axis, 1.0
    return "Z", 1.0


def robot_source_map(robot_usd: Path) -> dict[str, str]:
    stage = Usd.Stage.Open(str(robot_usd))
    if stage is None:
        raise RuntimeError(f"Failed to open robot stage: {robot_usd}")
    source_map: dict[str, str] = {}
    for prim in stage.Traverse():
        if not prim.IsValid() or prim.GetTypeName() in SKIP_TYPES:
            continue
        if not prim.IsA(UsdGeom.Xformable):
            continue
        source_map[sanitize_name(str(prim.GetPath()))] = str(prim.GetPath())
    return source_map


def build_sim_scene(
    robot_usd: Path,
    scene_usd: Path,
    output_path: Path,
    up_axis: str,
    meters_per_unit: float,
) -> None:
    stage = Usd.Stage.CreateNew(str(output_path))
    _configure_stage(stage, up_axis, meters_per_unit)
    UsdGeom.Xform.Define(stage, "/Root").GetPrim().GetReferences().AddReference(str(robot_usd.resolve()))
    UsdGeom.Xform.Define(stage, "/Scene").GetPrim().GetReferences().AddReference(str(scene_usd.resolve()))
    stage.GetRootLayer().Save()


def build_robot_wrapper(
    robot_usd: Path,
    body_names: list[str],
    source_map: dict[str, str],
    initial_poses: dict[str, tuple[list[float], list[float]]] | None,
    cameras: list[RenderCamera],
    output_path: Path,
    up_axis: str,
    meters_per_unit: float,
) -> dict[str, str]:
    source_stage = Usd.Stage.Open(str(robot_usd))
    if source_stage is None:
        raise RuntimeError(f"Failed to open robot stage: {robot_usd}")
    stage = Usd.Stage.CreateNew(str(output_path))
    _configure_stage(stage, up_axis, meters_per_unit)
    root = UsdGeom.Xform.Define(stage, "/MyRobot")
    stage.SetDefaultPrim(root.GetPrim())
    body_map: dict[str, str] = {}
    body_cameras = _group_body_cameras(cameras)
    for body_name in body_names:
        source_path = source_map.get(body_name)
        if source_path is None:
            continue
        body = UsdGeom.Xform.Define(stage, f"/MyRobot/{body_name}")
        xform = UsdGeom.Xformable(body.GetPrim())
        pos, quat = (([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]) if initial_poses is None else initial_poses.get(body_name, ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])))
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(*quat))
        visuals_path = f"{source_path}/visuals"
        source_visuals = source_stage.GetPrimAtPath(visuals_path)
        if source_visuals.IsValid():
            visuals = stage.DefinePrim(f"/MyRobot/{body_name}/visuals", source_visuals.GetTypeName())
            visuals.GetReferences().AddReference(str(robot_usd.resolve()), visuals_path)
            _deactivate_embedded_cameras(visuals)
        for camera in body_cameras.get(body_name, []):
            _define_camera(stage, f"/MyRobot/{body_name}/{camera.name}", camera)
        body_map[body_name] = f"MyRobot/{body_name}"
    stage.GetRootLayer().Save()
    return body_map


def build_scene_render(
    scene_usd: Path,
    output_path: Path,
    up_axis: str,
    meters_per_unit: float,
    cameras: list[RenderCamera],
) -> None:
    stage = Usd.Stage.CreateNew(str(output_path))
    _configure_stage(stage, up_axis, meters_per_unit)
    root = UsdGeom.Xform.Define(stage, "/SceneRender")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().GetReferences().AddReference(str(scene_usd.resolve()))
    for camera in cameras:
        if camera.body is None:
            _define_camera(stage, f"{SCENE_CAMERA_ROOT}/{camera.name}", camera)
    stage.GetRootLayer().Save()


def build_render_scene(
    scene_render_usd: Path,
    robot_wrapper_usd: Path,
    output_path: Path,
    up_axis: str,
    meters_per_unit: float,
) -> None:
    stage = Usd.Stage.CreateNew(str(output_path))
    _configure_stage(stage, up_axis, meters_per_unit)
    root = UsdGeom.Xform.Define(stage, "/RenderScene")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().GetReferences().AddReference(str(scene_render_usd.resolve()))
    robot = UsdGeom.Xform.Define(stage, "/RenderScene/MyRobot")
    robot.GetPrim().GetReferences().AddReference(str(robot_wrapper_usd.resolve()))
    stage.GetRootLayer().Save()


def _deactivate_embedded_cameras(root_prim) -> None:
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Camera):
            prim.SetActive(False)


def _configure_stage(stage: Usd.Stage, up_axis: str, meters_per_unit: float) -> None:
    axis = (up_axis or "Z").upper()
    token = UsdGeom.Tokens.y if axis == "Y" else UsdGeom.Tokens.z
    UsdGeom.SetStageUpAxis(stage, token)
    UsdGeom.SetStageMetersPerUnit(stage, float(meters_per_unit))


def _group_body_cameras(cameras: list[RenderCamera]) -> dict[str, list[RenderCamera]]:
    grouped: dict[str, list[RenderCamera]] = {}
    for camera in cameras:
        if camera.body is None:
            continue
        grouped.setdefault(camera.body, []).append(camera)
    return grouped


def _define_camera(stage: Usd.Stage, path: str, spec: RenderCamera) -> None:
    camera = UsdGeom.Camera.Define(stage, path)
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*spec.pos))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(*spec.quat))
    camera.CreateFocalLengthAttr(float(focal_length_from_fovy(spec.fovy, spec.vertical_aperture)))
    camera.CreateHorizontalApertureAttr(float(spec.horizontal_aperture))
    camera.CreateVerticalApertureAttr(float(spec.vertical_aperture))
    camera.CreateClippingRangeAttr(Gf.Vec2f(*spec.clipping_range))
