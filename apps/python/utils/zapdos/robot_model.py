from __future__ import annotations

from pathlib import Path
from typing import Literal

RobotModelKey = Literal["r1pro", "moz1"]

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROBOT_MODEL_KEY: RobotModelKey = "r1pro"
ROBOT_USD_BY_KEY: dict[RobotModelKey, Path] = {
    "moz1": (REPO_ROOT / "deps" / "spirit01_model" / "USD" / "Moz1_robot_only.usda").resolve(),
    "r1pro": (REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda").resolve(),
}


def get_robot_usd_for_model(key: RobotModelKey) -> Path:
    return ROBOT_USD_BY_KEY[key]


def get_robot_model_key_from_usd(robot_usd: str | Path | None) -> RobotModelKey | None:
    if robot_usd is None:
        return None
    resolved = Path(robot_usd).resolve()
    for key, value in ROBOT_USD_BY_KEY.items():
        if value == resolved:
            return key
    return None


__all__ = [
    "DEFAULT_ROBOT_MODEL_KEY",
    "ROBOT_USD_BY_KEY",
    "RobotModelKey",
    "get_robot_model_key_from_usd",
    "get_robot_usd_for_model",
]
