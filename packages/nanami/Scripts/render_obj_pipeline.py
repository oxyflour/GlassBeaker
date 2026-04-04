import json
import os
import traceback
import unreal


CONFIG_ENV = "UE_HEADLESS_RENDER_CONFIG"


def log(msg: str) -> None:
    unreal.log(f"[HeadlessOBJ] {msg}")


def fail(msg: str) -> None:
    unreal.log_error(f"[HeadlessOBJ] {msg}")
    raise RuntimeError(msg)


def load_config() -> dict:
    cfg_path = os.environ.get(CONFIG_ENV, "").strip()
    if not cfg_path:
        fail(f"Environment variable {CONFIG_ENV} is missing")
    if not os.path.exists(cfg_path):
        fail(f"Config file does not exist: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_content_dir(path: str) -> None:
    if not unreal.EditorAssetLibrary.does_directory_exist(path):
        unreal.EditorAssetLibrary.make_directory(path)


def delete_asset_if_exists(path: str) -> None:
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        unreal.EditorAssetLibrary.delete_asset(path)


def import_obj(obj_path: str, dest_path: str) -> tuple[object, str]:
    ensure_content_dir(dest_path)

    task = unreal.AssetImportTask()
    task.filename = obj_path
    task.destination_path = dest_path
    task.automated = True
    task.save = True
    task.replace_existing = True
    task.replace_existing_settings = True

    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])

    imported_paths = list(task.get_editor_property("imported_object_paths") or [])
    if not imported_paths:
        fail(f"OBJ import returned no assets: {obj_path}")

    mesh_asset = None
    mesh_path = None
    for p in imported_paths:
        asset = unreal.EditorAssetLibrary.load_asset(p)
        if isinstance(asset, unreal.StaticMesh):
            mesh_asset = asset
            mesh_path = p
            break

    if mesh_asset is None:
        fail(
            "OBJ import completed, but no StaticMesh was found.\n"
            f"Imported assets: {imported_paths}"
        )

    log(f"Imported mesh: {mesh_path}")
    return mesh_asset, mesh_path


def new_blank_level(map_path: str) -> None:
    ensure_content_dir("/Game/Auto")
    ensure_content_dir("/Game/Auto/Maps")

    delete_asset_if_exists(map_path)

    ok = unreal.EditorLevelLibrary.new_level(map_path)
    if not ok:
        fail(f"Failed to create new level: {map_path}")

    # Load it explicitly
    if not unreal.EditorLevelLibrary.load_level(map_path):
        fail(f"Failed to load level: {map_path}")

    log(f"Created and loaded level: {map_path}")


def spawn_actor_from_mesh(mesh_asset, rotation_deg) -> unreal.Actor:
    rot = unreal.Rotator(rotation_deg[0], rotation_deg[1], rotation_deg[2])
    actor = unreal.EditorLevelLibrary.spawn_actor_from_object(
        mesh_asset,
        unreal.Vector(0.0, 0.0, 0.0),
        rot,
    )
    if actor is None:
        fail("Failed to spawn mesh actor from imported StaticMesh")
    actor.set_actor_label("ImportedOBJ")
    return actor


def setup_lighting(bounds_origin: unreal.Vector, bounds_extent: unreal.Vector) -> None:
    # Directional light
    sun = unreal.EditorLevelLibrary.spawn_actor_from_class(
        unreal.DirectionalLight,
        unreal.Vector(bounds_origin.x - 300.0, bounds_origin.y - 300.0, bounds_origin.z + 800.0),
        unreal.Rotator(-45.0, 45.0, 0.0),
    )
    if sun:
        sun.set_actor_label("AutoSun")
        comp = sun.get_component_by_class(unreal.DirectionalLightComponent)
        if comp:
            comp.set_editor_property("intensity", 10.0)

    # Sky light
    sky = unreal.EditorLevelLibrary.spawn_actor_from_class(
        unreal.SkyLight,
        unreal.Vector(bounds_origin.x, bounds_origin.y, bounds_origin.z + 200.0),
        unreal.Rotator(0.0, 0.0, 0.0),
    )
    if sky:
        sky.set_actor_label("AutoSky")
        comp = sky.get_component_by_class(unreal.SkyLightComponent)
        if comp:
            comp.set_editor_property("intensity", 1.0)
            try:
                comp.set_editor_property("real_time_capture", True)
            except Exception:
                pass

    # Sky atmosphere
    try:
        atm = unreal.EditorLevelLibrary.spawn_actor_from_class(
            unreal.SkyAtmosphere,
            unreal.Vector(0.0, 0.0, 0.0),
            unreal.Rotator(0.0, 0.0, 0.0),
        )
        if atm:
            atm.set_actor_label("AutoSkyAtmosphere")
    except Exception:
        pass

    # Post process volume to avoid aggressive auto exposure
    try:
        ppv = unreal.EditorLevelLibrary.spawn_actor_from_class(
            unreal.PostProcessVolume,
            unreal.Vector(bounds_origin.x, bounds_origin.y, bounds_origin.z),
            unreal.Rotator(0.0, 0.0, 0.0),
        )
        if ppv:
            ppv.set_actor_label("AutoPPV")
            ppv.set_editor_property("b_unbound", True)
            settings = ppv.get_editor_property("settings")
            settings.set_editor_property("auto_exposure_method", unreal.AutoExposureMethod.AEM_MANUAL)
            settings.set_editor_property("camera_iso", 100.0)
            settings.set_editor_property("camera_shutter_speed", 100.0)
            settings.set_editor_property("camera_aperture", 8.0)
            ppv.set_editor_property("settings", settings)
    except Exception:
        pass


def setup_camera(bounds_origin: unreal.Vector, bounds_extent: unreal.Vector) -> unreal.CineCameraActor:
    radius = max(bounds_extent.x, bounds_extent.y, bounds_extent.z, 1.0)
    distance = radius * 3.2

    cam_loc = unreal.Vector(
        bounds_origin.x - distance,
        bounds_origin.y + distance * 0.35,
        bounds_origin.z + radius * 0.35,
    )
    look_rot = unreal.MathLibrary.find_look_at_rotation(cam_loc, bounds_origin)

    cam = unreal.EditorLevelLibrary.spawn_actor_from_class(
        unreal.CineCameraActor,
        cam_loc,
        look_rot,
    )
    if cam is None:
        fail("Failed to spawn CineCameraActor")

    cam.set_actor_label("AutoCamera")

    try:
        cine_comp = cam.get_cine_camera_component()
        cine_comp.set_editor_property("current_focal_length", 50.0)
        cine_comp.set_editor_property("current_aperture", 8.0)
    except Exception:
        pass

    return cam


def create_level_sequence(sequence_path: str, camera_actor: unreal.Actor, sequence_frames: int):
    ensure_content_dir("/Game/Auto/Sequences")
    delete_asset_if_exists(sequence_path)

    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    seq_name = sequence_path.split("/")[-1]
    seq_dir = "/".join(sequence_path.split("/")[:-1])

    sequence = asset_tools.create_asset(
        asset_name=seq_name,
        package_path=seq_dir,
        asset_class=unreal.LevelSequence,
        factory=unreal.LevelSequenceFactoryNew(),
    )
    if sequence is None:
        fail(f"Failed to create LevelSequence: {sequence_path}")

    sequence.set_display_rate(unreal.FrameRate(24, 1))
    sequence.set_playback_start(0)
    sequence.set_playback_end(sequence_frames)

    cam_binding = sequence.add_possessable(camera_actor)
    if not cam_binding:
        fail("Failed to add camera possessable to sequence")

    cut_track = sequence.add_track(unreal.MovieSceneCameraCutTrack)
    if cut_track is None:
        fail("Failed to add camera cut track")

    cut_section = cut_track.add_section()
    if cut_section is None:
        fail("Failed to add camera cut section")

    cut_section.set_range(0, sequence_frames)

    cam_binding_id = sequence.get_binding_id(cam_binding)
    cut_section.set_camera_binding_id(cam_binding_id)

    unreal.EditorAssetLibrary.save_loaded_asset(sequence)
    log(f"Created sequence: {sequence_path}")
    return sequence


def save_everything() -> None:
    try:
        unreal.EditorLevelLibrary.save_current_level()
    except Exception:
        pass
    unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)


_render_done = False
_executor_ref = None


def release_keep_alive(success: bool) -> None:
    global _executor_ref
    global _render_done
    _render_done = True
    _executor_ref = None
    log(f"Releasing python keepalive. success={success}")
    unreal.EditorPythonScripting.set_keep_python_script_alive(False)


def on_executor_finished(executor, success):
    log(f"Render finished. success={success}")
    release_keep_alive(success)


def on_executor_errored(executor, pipeline, is_fatal, error_text):
    unreal.log_error(f"[HeadlessOBJ] Render error. fatal={is_fatal}, error={error_text}")
    if is_fatal:
        release_keep_alive(False)


def render_sequence(map_path: str, sequence_path: str, output_dir: str, width: int, height: int, job_name: str):
    global _executor_ref
    subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    queue = subsystem.get_queue()
    queue.delete_all_jobs()

    job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    job.job_name = job_name
    job.map = unreal.SoftObjectPath(map_path)
    job.sequence = unreal.SoftObjectPath(sequence_path)

    config = job.get_configuration()

    output = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    output.output_directory = unreal.DirectoryPath(output_dir)
    output.file_name_format = "{job_name}.{frame_number}"
    output.output_resolution = unreal.IntPoint(width, height)
    output.flush_disk_writes_per_shot = True

    render_pass = config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    try:
        render_pass.disable_multisample_effects = True
    except Exception:
        pass

    config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)

    aa = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    aa.spatial_sample_count = 1
    aa.temporal_sample_count = 8

    # Optional: some quality knobs
    cvars = config.find_or_add_setting_by_class(unreal.MoviePipelineConsoleVariableSetting)
    cvars.add_or_update_console_variable("r.Tonemapper.Quality", 5.0)
    cvars.add_or_update_console_variable("r.MotionBlurQuality", 4.0)
    cvars.add_or_update_console_variable("r.DepthOfFieldQuality", 4.0)

    _executor_ref = unreal.MoviePipelinePIEExecutor(subsystem)
    _executor_ref.on_executor_finished_delegate.add_callable_unique(on_executor_finished)
    _executor_ref.on_executor_errored_delegate.add_callable_unique(on_executor_errored)

    log("Starting MRQ render...")
    subsystem.render_queue_with_executor_instance(_executor_ref)


def main():
    cfg = load_config()

    obj_path = cfg["obj_path"]
    output_dir = cfg["output_dir"]
    width = int(cfg["width"])
    height = int(cfg["height"])
    rotation_deg = cfg.get("rotation_deg", [0.0, 0.0, 0.0])
    sequence_frames = int(cfg.get("sequence_frames", 2))
    import_dest = cfg["import_dest"]
    map_path = cfg["map_path"]
    sequence_path = cfg["sequence_path"]
    job_name = cfg.get("job_name", "obj_render")

    unreal.EditorPythonScripting.set_keep_python_script_alive(True)

    mesh_asset, _mesh_path = import_obj(obj_path, import_dest)

    new_blank_level(map_path)

    mesh_actor = spawn_actor_from_mesh(mesh_asset, rotation_deg)
    bounds_origin, bounds_extent = mesh_actor.get_actor_bounds(False)

    setup_lighting(bounds_origin, bounds_extent)
    camera_actor = setup_camera(bounds_origin, bounds_extent)

    create_level_sequence(sequence_path, camera_actor, sequence_frames)
    save_everything()

    render_sequence(
        map_path=map_path,
        sequence_path=sequence_path,
        output_dir=output_dir,
        width=width,
        height=height,
        job_name=job_name,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        unreal.log_error("[HeadlessOBJ] Fatal exception:\n" + "".join(traceback.format_exc()))
        try:
            unreal.EditorPythonScripting.set_keep_python_script_alive(False)
        except Exception:
            pass
        raise
