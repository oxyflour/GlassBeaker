from __future__ import annotations

from typing import Any

from utils.zapdos.overlay.overlay_commands import build_remove_asset_overlay, build_set_scene_assets_overlay

from .scene_rebuild_manager import drain_overlay_completions


class SceneRebuildService:
    def __init__(self, session: Any) -> None:
        self.session = session

    def submit_replace(self, assets: list[dict[str, object]]) -> dict[str, object]:
        next_overlay, items = build_set_scene_assets_overlay(self.session, assets)
        return self.session._start_overlay_operation(
            next_overlay,
            {"ok": True, "items": items},
        )

    def submit_remove(self, instance_id: str) -> dict[str, object]:
        next_overlay, _ = build_remove_asset_overlay(self.session, instance_id)
        return self.session._start_overlay_operation(
            next_overlay,
            {"ok": True, "instance_id": instance_id},
        )

    def drain_completions(self) -> None:
        drain_overlay_completions(self.session)
