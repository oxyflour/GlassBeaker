from __future__ import annotations

from dataclasses import dataclass

from utils.zapdos.bundle import RenderBundle


@dataclass(frozen=True)
class PreparedOverlayRebuild:
    bundle: RenderBundle
    next_overlay: dict[str, object]
    previous_overlay: dict[str, object]
    previous_revision: str
    next_revision: str


@dataclass(frozen=True)
class OverlayRebuildCompletion:
    op_id: str
    prepared: PreparedOverlayRebuild | None = None
    error: Exception | None = None
