from __future__ import annotations

from pathlib import Path


def request_path(control_dir: Path) -> Path:
    return control_dir / "request.json"


def response_path(control_dir: Path) -> Path:
    return control_dir / "response.json"
