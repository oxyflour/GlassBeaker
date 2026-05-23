import asyncio
from pathlib import Path

from fastapi import HTTPException, Request

from utils.ros_bridge import bridge
from utils.session_registry import AsyncSessionRegistry
from utils.sse import sse_json_event
from utils.zapdos.bundle import DEFAULT_SCENE_USD
from utils.zapdos import request_router
from utils.zapdos.zapdos_session import DEFAULT_ROBOT_USD, ZapdosSession

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


def _get_or_create_session_future_with_status(
    req: Request,
    sess: str,
) -> tuple[asyncio.Future[ZapdosSession], bool]:
    robot_usd = _input_path(req, "robot_usd", DEFAULT_ROBOT_USD)
    scene_usd = _input_path(req, "scene_usd", DEFAULT_SCENE_USD)
    return session_registry.get_or_create_with_status(
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
    async for event, payload in request_router.init_stream(
        sess,
        future,
        _await_session_future,
        INIT_STREAM_HEARTBEAT_SEC,
    ):
        yield sse_json_event(event, payload)


def _require_camera_name(camera_index: dict[str, int], camera_name: str) -> str:
    return request_router.require_camera_name(camera_index, camera_name)


def _require_active_session(sess: str, future: asyncio.Future[ZapdosSession], session: ZapdosSession) -> ZapdosSession:
    return request_router.require_active_session(
        sess,
        future,
        session,
        session_registry.discard,
    )


def _stream_scene_rebuild_job(session: ZapdosSession, op_id: str):
    return session.editor.stream_rebuild_job(op_id)


_name_ = request_router.build_name_handler(
    bridge=bridge,
    heartbeat_sec_getter=lambda: INIT_STREAM_HEARTBEAT_SEC,
    get_or_create_session_future_with_status=_get_or_create_session_future_with_status,
    require_session_future=_require_session_future,
    await_session_future=_await_session_future,
    discard_session_future=session_registry.discard,
    require_active_session=_require_active_session,
    require_camera_name=_require_camera_name,
)
