from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SessionState:
    sess: str
    scene_revision: str
    rebuilding_scene: bool = False
    camera_index: dict[str, int] = field(default_factory=dict)
    last_frame_index: dict[str, int] = field(default_factory=dict)
