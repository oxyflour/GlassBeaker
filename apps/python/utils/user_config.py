from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def default_config_path() -> Path:
    return REPO_ROOT / "apps" / "desktop" / "config.json"


def user_config_path() -> Path:
    root = os.environ.get("USERPROFILE", "").strip()
    if not root:
        raise RuntimeError("USERPROFILE is not set.")
    return Path(root).resolve() / ".glass-beaker" / "config.json"


def read_user_config() -> dict[str, Any]:
    payload = _read_json_object(default_config_path(), missing_ok=True)
    if not os.environ.get("USERPROFILE", "").strip():
        return payload
    return _deep_merge(payload, read_raw_user_config())


def read_raw_user_config() -> dict[str, Any]:
    return _read_json_object(user_config_path(), missing_ok=True)


def write_user_config(payload: dict[str, Any]) -> Path:
    path = user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _read_json_object(path: Path, *, missing_ok: bool) -> dict[str, Any]:
    if missing_ok and not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object.")
    return payload


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged
