from __future__ import annotations

from pathlib import Path
from typing import Callable

from utils.genie_sim import resolve_assets_root
from utils.zapdos.bundle import ensure_render_bundle
from utils.zapdos.editor.scene_writer import write_overlay_scene
from utils.zapdos.editor.state import scene_revision
from utils.zapdos.zapdos_asset_library import asset_local_bounds

StageLogger = Callable[[str], None]


def prepare_overlay_rebuild_request(
    request: dict[str, object],
    stage_logger: StageLogger | None = None,
) -> dict[str, object]:
    _log_stage(stage_logger, "resolve_request")
    next_overlay = request["next_overlay"]
    if not isinstance(next_overlay, dict):
        raise TypeError("next_overlay must be an object")
    support_infos = request["support_infos"]
    if not isinstance(support_infos, dict):
        raise TypeError("support_infos must be an object")
    robot_usd = Path(str(request["robot_usd"]))
    base_scene_usd = Path(str(request["base_scene_usd"]))
    composed_scene_usd = Path(str(request["composed_scene_usd"]))
    assets_root = resolve_assets_root(next_overlay.get("assets_root"))
    instances = next_overlay.get("instances")
    if not isinstance(instances, list):
        raise TypeError("next_overlay.instances must be a list")
    bounds_by_instance = {str(item["id"]): asset_local_bounds(assets_root / str(item["url"])) for item in instances}
    _log_stage(stage_logger, "write_overlay_scene")
    write_overlay_scene(
        composed_scene_usd,
        base_scene_usd,
        assets_root,
        next_overlay,
        support_infos=support_infos,
        asset_bounds_by_instance=bounds_by_instance,
    )
    _log_stage(stage_logger, "ensure_render_bundle")
    bundle = ensure_render_bundle(robot_usd, composed_scene_usd)
    _log_stage(stage_logger, "scene_revision")
    return {"bundle": bundle.to_json(), "next_revision": scene_revision(base_scene_usd, next_overlay)}


def _log_stage(stage_logger: StageLogger | None, stage: str) -> None:
    if stage_logger is not None:
        stage_logger(stage)
