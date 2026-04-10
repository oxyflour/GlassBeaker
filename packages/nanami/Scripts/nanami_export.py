from __future__ import annotations

import shutil
from pathlib import Path

import unreal  # type: ignore

_callbacks = {"done": None, "error": None}
_executor_ref = None


def create_sequence(sequence_path: str, camera_actor):
    for path in ["/Game/Auto", "/Game/Auto/Sequences"]:
        if not unreal.EditorAssetLibrary.does_directory_exist(path):
            unreal.EditorAssetLibrary.make_directory(path)
    if unreal.EditorAssetLibrary.does_asset_exist(sequence_path):
        unreal.EditorAssetLibrary.delete_asset(sequence_path)

    tools = unreal.AssetToolsHelpers.get_asset_tools()
    sequence = tools.create_asset(
        asset_name=sequence_path.split("/")[-1],
        package_path="/".join(sequence_path.split("/")[:-1]),
        asset_class=unreal.LevelSequence,
        factory=unreal.LevelSequenceFactoryNew(),
    )
    sequence.set_display_rate(unreal.FrameRate(24, 1))
    sequence.set_playback_start(0)
    sequence.set_playback_end(2)
    binding = sequence.add_possessable(camera_actor)
    cut_track = sequence.add_track(unreal.MovieSceneCameraCutTrack)
    cut_section = cut_track.add_section()
    cut_section.set_range(0, 2)
    cut_section.set_camera_binding_id(sequence.get_binding_id(binding))
    unreal.EditorAssetLibrary.save_loaded_asset(sequence)
    return sequence


def _finished(_executor, success):
    if _callbacks["done"] is not None:
        _callbacks["done"](success)


def _errored(_executor, _pipeline, is_fatal, error_text):
    if is_fatal and _callbacks["error"] is not None:
        _callbacks["error"](error_text)


def start_render(map_path: str, sequence_path: str, camera_actor, output_dir: str, width: int, height: int, job_name: str, on_done, on_error):
    global _executor_ref
    _callbacks["done"] = on_done
    _callbacks["error"] = on_error
    create_sequence(sequence_path, camera_actor)
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
    config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
    aa = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    aa.spatial_sample_count = 1
    aa.temporal_sample_count = 8

    _executor_ref = unreal.MoviePipelinePIEExecutor(subsystem)
    _executor_ref.on_executor_finished_delegate.add_callable_unique(_finished)
    _executor_ref.on_executor_errored_delegate.add_callable_unique(_errored)
    subsystem.render_queue_with_executor_instance(_executor_ref)


def finalize_output(output_dir: str) -> str | None:
    pngs = sorted(Path(output_dir).glob("*.png"))
    if not pngs:
        return None
    result = Path(output_dir) / "result.png"
    shutil.copy2(pngs[0], result)
    return str(result)
