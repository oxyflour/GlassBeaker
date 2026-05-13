from __future__ import annotations

import json
from pathlib import Path

from utils.zapdos.editor.state import default_overlay_state


def load_overlay_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return default_overlay_state()
    data = json.loads(path.read_text(encoding="utf-8"))
    state = default_overlay_state(data.get("assets_root"))
    state["instances"] = list(data.get("instances") or [])
    state["pose_overrides"] = dict(data.get("pose_overrides") or {})
    return state


def save_overlay_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


__all__ = ["load_overlay_state", "save_overlay_state"]
