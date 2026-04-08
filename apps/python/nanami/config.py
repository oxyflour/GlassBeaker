from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO_ROOT / "packages" / "nanami"
TEMP_ROOT = PROJECT_ROOT / "Temp"
URDF_ROOT = TEMP_ROOT / "URDF"
SCRIPTS_ROOT = PROJECT_ROOT / "Scripts"
SHOTS_ROOT = PROJECT_ROOT / "Shots"
SESSION_ROOT = TEMP_ROOT / "sessions"
CACHE_ROOT = TEMP_ROOT / "asset-cache"
RUNTIME_ROOT = TEMP_ROOT / "runtimes"
DEFAULT_ENGINE_ROOT = Path(
    os.getenv("NANAMI_ENGINE_ROOT", r"C:\Program Files\Epic Games\UE_5.7")
)
UPROJECT_PATH = PROJECT_ROOT / "HeadlessObjRender.uproject"
WORKER_SCRIPT = SCRIPTS_ROOT / "nanami_worker.py"
WORKER_BOOTSTRAP = SCRIPTS_ROOT / "nanami_worker_bootstrap.py"
HDR_PATH = PROJECT_ROOT / "Assets" / "HDRI" / "studio_kominka_01_1k.hdr"
SESSION_ENV = "UE_NANAMI_SESSION_CONFIG"
SESSION_DEFAULT_SOURCE = {"robot_path": "R1"}


def ensure_roots() -> None:
    for path in [TEMP_ROOT, URDF_ROOT, SESSION_ROOT, SHOTS_ROOT, CACHE_ROOT, RUNTIME_ROOT]:
        path.mkdir(parents=True, exist_ok=True)
