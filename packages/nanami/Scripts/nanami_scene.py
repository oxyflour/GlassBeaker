from __future__ import annotations

import math

import unreal  # type: ignore


def new_blank_level(map_path: str) -> None:
    for path in ["/Game/Auto", "/Game/Auto/Maps"]:
        if not unreal.EditorAssetLibrary.does_directory_exist(path):
            unreal.EditorAssetLibrary.make_directory(path)
    if unreal.EditorAssetLibrary.does_asset_exist(map_path):
        unreal.EditorAssetLibrary.delete_asset(map_path)
    if not unreal.EditorLevelLibrary.new_level(map_path):
        raise RuntimeError(f"Failed to create level: {map_path}")
    if not unreal.EditorLevelLibrary.load_level(map_path):
        raise RuntimeError(f"Failed to load level: {map_path}")


def setup_lighting(hdr_cubemap=None) -> None:
    sky = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 200), unreal.Rotator(0, 0, 0))
    if sky:
        comp = sky.get_component_by_class(unreal.SkyLightComponent)
        comp.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
        if hdr_cubemap is not None:
            comp.set_editor_property("source_type", unreal.SkyLightSourceType.SLS_SPECIFIED_CUBEMAP)
            comp.set_editor_property("cubemap", hdr_cubemap)
            comp.set_editor_property("intensity", 4.0)
        else:
            comp.set_editor_property("real_time_capture", True)

    sun = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(-300, -200, 700), unreal.Rotator(-35, 45, 0))
    if sun:
        comp = sun.get_component_by_class(unreal.DirectionalLightComponent)
        comp.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
        comp.set_editor_property("intensity", 8.0)

    ppv = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.PostProcessVolume, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    if ppv:
        ppv.set_editor_property("b_unbound", True)
        settings = ppv.get_editor_property("settings")
        settings.set_editor_property("auto_exposure_method", unreal.AutoExposureMethod.AEM_MANUAL)
        settings.set_editor_property("camera_iso", 400.0)
        settings.set_editor_property("camera_shutter_speed", 50.0)
        settings.set_editor_property("camera_aperture", 8.0)
        ppv.set_editor_property("settings", settings)


def camera_pose(target_xyz, yaw_deg: float, pitch_deg: float, distance: float):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    location = unreal.Vector(
        target_xyz[0] - distance * math.cos(pitch) * math.cos(yaw),
        target_xyz[1] - distance * math.cos(pitch) * math.sin(yaw),
        target_xyz[2] - distance * math.sin(pitch),
    )
    target = unreal.Vector(*target_xyz)
    rotation = unreal.MathLibrary.find_look_at_rotation(location, target)
    return location, rotation


def create_camera_rig(target_xyz):
    world = unreal.EditorLevelLibrary.get_editor_world()
    location, rotation = camera_pose(target_xyz, 35.0, -18.0, 3.2)
    cine = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.CineCameraActor, location, rotation)
    capture = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, location, rotation)
    target = unreal.RenderingLibrary.create_render_target2d(
        world,
        960,
        540,
        unreal.TextureRenderTargetFormat.RTF_RGBA8,
        unreal.LinearColor(0, 0, 0, 1),
        False,
    )
    comp = capture.get_component_by_class(unreal.SceneCaptureComponent2D)
    comp.texture_target = target
    comp.capture_every_frame = False
    comp.capture_on_movement = False
    comp.capture_source = unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR
    return cine, capture, target


def update_camera(cine, capture, target_xyz, yaw_deg: float, pitch_deg: float, distance: float) -> None:
    location, rotation = camera_pose(target_xyz, yaw_deg, pitch_deg, distance)
    cine.set_actor_location(location, False, False)
    cine.set_actor_rotation(rotation, False)
    capture.set_actor_location(location, False, False)
    capture.set_actor_rotation(rotation, False)


def export_preview(render_target, capture, path: str) -> None:
    options = unreal.ImageWriteOptions()
    options.set_editor_property("format", unreal.DesiredImageFormat.JPG)
    options.set_editor_property("compression_quality", 85)
    options.set_editor_property("overwrite_file", True)
    try:
        options.set_editor_property("async", False)
    except Exception:
        try:
            options.set_editor_property("async_", False)
        except Exception:
            pass
    capture.get_component_by_class(unreal.SceneCaptureComponent2D).capture_scene()
    unreal.ImageWriteBlueprintLibrary.export_to_disk(render_target, path, options)
