from __future__ import annotations

import traceback
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from teleop.manager import SpaceMouseManager

router = APIRouter()
manager = SpaceMouseManager()


class StartRequest(BaseModel):
    robot_usd: str | None = None
    scene_usd: str | None = None
    rate_hz: float = 60.0
    linear_scale: float = 0.15
    angular_scale: float = 0.8
    gripper_step: float = 0.005


class ActiveArmRequest(BaseModel):
    arm: Literal["left", "right"]


class ModeRequest(BaseModel):
    mode: Literal["off", "left", "right"]


@router.post("/start")
async def start(request: StartRequest):
    return manager.start(**request.model_dump(exclude_none=True))


@router.post("/stop")
async def stop():
    return manager.stop()


@router.get("/status")
async def status():
    return manager.status()


@router.post("/set_active_arm")
async def set_active_arm(request: ActiveArmRequest):
    return manager.set_active_arm(request.arm)


@router.post("/set_mode")
async def set_mode(request: ModeRequest):
    return manager.set_mode(request.mode)


@router.on_event("startup")
async def startup() -> None:
    try:
        manager.start(mode_override="off")
    except Exception:  # pragma: no cover - startup fallback
        traceback.print_exc()


@router.on_event("shutdown")
async def shutdown() -> None:
    manager.shutdown()
