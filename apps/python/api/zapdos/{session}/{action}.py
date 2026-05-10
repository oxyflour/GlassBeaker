import asyncio
from concurrent.futures import Future as ConcurrentFuture
from pathlib import Path

import mujoco  # type: ignore
from fastapi import HTTPException, Request

from utils.ros_bridge import bridge
from utils.session import Session
from utils.session_registry import AsyncSessionRegistry
from utils.zapdos.bundle import DEFAULT_SCENE_USD, RenderBundle
from utils.zapdos.overlay.overlay_state import default_overlay_state
from utils.zapdos.physics.mujoco_physics import MujocoPhysics as ZapdosPhysics
from utils.zapdos.rebuild.scene_rebuild_manager import (
    OverlayRebuildCompletion,
    PreparedOverlayRebuild,
    SceneRebuildJob,
    stream_scene_rebuild_job as _stream_scene_rebuild_job,
)
from utils.zapdos.session import request_router
from utils.zapdos.session.zapdos_session import DEFAULT_ROBOT_USD, ZapdosSession

REPO_ROOT = Path(__file__).resolve().parents[5]
INIT_STREAM_HEARTBEAT_SEC = 1.0


def _input_path(req: Request, key: str, default: Path) -> Path:
    return request_router.resolve_input_path(req, key, default, REPO_ROOT)


session_registry = AsyncSessionRegistry[ZapdosSession]()
sessions = session_registry.sessions


def _get_or_create_session_future(req: Request, sess: str) -> asyncio.Future[ZapdosSession]:
    robot_usd = _input_path(req, "robot_usd", DEFAULT_ROBOT_USD)
    scene_usd = _input_path(req, "scene_usd", DEFAULT_SCENE_USD)
    return session_registry.get_or_create(
        sess,
        lambda: ZapdosSession.create(sess, robot_usd, scene_usd),
    )


def _require_session_future(sess: str) -> asyncio.Future[ZapdosSession]:
    future, reason = session_registry.resolve(sess)
    if future is not None:
        return future
    detail = "Session expired" if reason == "expired" else "Session not initialized"
    raise HTTPException(status_code=409, detail=detail)


async def _await_session_future(sess: str, future: asyncio.Future[ZapdosSession]) -> ZapdosSession:
    return await session_registry.await_ready(sess, future)


def _bootstrap_error_message(exc: Exception) -> str:
    return request_router.bootstrap_error_message(exc)


async def _init_stream(sess: str, future: asyncio.Future[ZapdosSession]):
    async for chunk in request_router.init_stream(
        sess,
        future,
        _await_session_future,
        INIT_STREAM_HEARTBEAT_SEC,
    ):
        yield chunk


def _require_camera_name(session: ZapdosSession, camera_name: str) -> str:
    return request_router.require_camera_name(session, camera_name)


def _require_active_session(sess: str, future: asyncio.Future[ZapdosSession], session: ZapdosSession) -> ZapdosSession:
    return request_router.require_active_session(
        sess,
        future,
        session,
        session_registry.discard,
    )


_name_ = request_router.build_name_handler(
    bridge=bridge,
    heartbeat_sec_getter=lambda: INIT_STREAM_HEARTBEAT_SEC,
    get_or_create_session_future=_get_or_create_session_future,
    require_session_future=_require_session_future,
    await_session_future=_await_session_future,
    require_active_session=_require_active_session,
    require_camera_name=_require_camera_name,
    stream_scene_rebuild_job=_stream_scene_rebuild_job,
)
