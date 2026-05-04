from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def user_config_path() -> Path:
    root = os.environ.get("USERPROFILE", "").strip()
    if not root:
        raise RuntimeError("USERPROFILE is not set.")
    return Path(root).resolve() / ".glass-beaker" / "config.json"


def read_user_config() -> dict[str, Any]:
    path = user_config_path()
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object.")
    return payload


def write_user_config(payload: dict[str, Any]) -> Path:
    path = user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
