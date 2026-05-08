import asyncio
from concurrent.futures import Future as ConcurrentFuture
import json
from pathlib import Path

import mujoco  # type: ignore
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from utils.ros_bridge import bridge
from utils.session import Session
from utils.session_registry import AsyncSessionRegistry
from utils.zapdos.rl_bundle import DEFAULT_SCENE_USD, RenderBundle
from utils.zapdos.zapdos_overlay import default_overlay_state
from utils.zapdos.zapdos_physics import ZapdosPhysics
from utils.zapdos.zapdos_scene_operations import (
    OverlayRebuildCompletion,
    PreparedOverlayRebuild,
    SceneOperation,
    stream_scene_operation as _stream_scene_operation,
)
from utils.zapdos.zapdos_session import DEFAULT_ROBOT_USD, ZapdosSession

REPO_ROOT = Path(__file__).resolve().parents[5]
INIT_STREAM_HEARTBEAT_SEC = 1.0


def _input_path(req: Request, key: str, default: Path) -> Path:
    raw = req.query_params.get(key, "").strip()
    path = default if not raw else (Path(raw) if Path(raw).is_absolute() else (REPO_ROOT / raw))
    path = path.resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{key} not found: {path}")
    return path


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
    message = str(exc).strip()
    return message or "Session bootstrap failed"


async def _init_stream(sess: str, future: asyncio.Future[ZapdosSession]):
    yield "data: loading\n\n"
    while not future.done():
        done, _ = await asyncio.wait({future}, timeout=INIT_STREAM_HEARTBEAT_SEC)
        if done:
            break
        yield "data: loading: preparing render bundle\n\n"
    try:
        await _await_session_future(sess, future)
    except Exception as exc:
        yield f"data: error: {_bootstrap_error_message(exc)}\n\n"
        return
    yield "data: started\n\n"


def _require_camera_name(session: ZapdosSession, camera_name: str) -> str:
    if camera_name not in session.camera_index:
        raise HTTPException(status_code=404, detail=f"Camera not found: {camera_name}")
    return camera_name


def _require_active_session(sess: str, future: asyncio.Future[ZapdosSession], session: ZapdosSession) -> ZapdosSession:
    if session.is_active():
        return session
    session_registry.discard(sess, future)
    raise HTTPException(status_code=409, detail="Session expired")


async def _name_(req: Request):
    sess = req.path_params["session"]
    action = req.path_params["action"]
    name = req.path_params["name"]
    if action == "init":
        future = _get_or_create_session_future(req, sess)
        return StreamingResponse(
            _init_stream(sess, future),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    future = _require_session_future(sess)
    session = _require_active_session(sess, future, await _await_session_future(sess, future))
    if action == "stream":
        return StreamingResponse(
            session.stream(),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    if action == "op":
        return StreamingResponse(
            _stream_scene_operation(session, name),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    if action == "ros" and name == "subscribe":
        topic, type_name = await req.json()
        await bridge.subscribe(topic, type_name, session.on_message)
        return {"ok": True}
    if action == "call":
        args = await req.json()
        return await session.call(name, *args)
    if action == "render":
        return StreamingResponse(
            session.render(_require_camera_name(session, name)),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )
    if action == "asset":
        asset = session.physics.assets.get(name)
        if asset is None:
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(asset)
    raise HTTPException(status_code=404, detail="Action not found")
