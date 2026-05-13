from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Awaitable, Callable

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse


type SessionFutureResolver = Callable[[str], asyncio.Future]
type SessionFutureAwaiter = Callable[[str, asyncio.Future], Awaitable[object]]
type SessionFutureFactory = Callable[[Request, str], asyncio.Future]
type SessionValidator = Callable[[str, asyncio.Future, object], object]
type StreamFactory = Callable[[object, str], object]


def resolve_input_path(req: Request, key: str, default: Path, repo_root: Path) -> Path:
    raw = req.query_params.get(key, "").strip()
    path = default if not raw else (Path(raw) if Path(raw).is_absolute() else (repo_root / raw))
    path = path.resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{key} not found: {path}")
    return path


def bootstrap_error_message(exc: Exception) -> str:
    message = str(exc).strip()
    return message or "Session bootstrap failed"


async def init_stream(
    sess: str,
    future: asyncio.Future,
    await_session_future: SessionFutureAwaiter,
    heartbeat_sec: float,
):
    yield "data: loading\n\n"
    while not future.done():
        done, _ = await asyncio.wait({future}, timeout=heartbeat_sec)
        if done:
            break
        yield "data: loading: preparing render bundle\n\n"
    try:
        await await_session_future(sess, future)
    except Exception as exc:
        yield f"data: error: {bootstrap_error_message(exc)}\n\n"
        return
    yield "data: started\n\n"


def require_camera_name(camera_index: dict[str, int], camera_name: str) -> str:
    if camera_name not in camera_index:
        raise HTTPException(status_code=404, detail=f"Camera not found: {camera_name}")
    return camera_name


def require_active_session(
    sess: str,
    future: asyncio.Future,
    session,
    discard_future: Callable[[str, asyncio.Future], None],
):
    if session.is_active():
        return session
    discard_future(sess, future)
    raise HTTPException(status_code=409, detail="Session expired")


def build_name_handler(
    *,
    bridge,
    heartbeat_sec_getter: Callable[[], float],
    get_or_create_session_future: SessionFutureFactory,
    require_session_future: SessionFutureResolver,
    await_session_future: SessionFutureAwaiter,
    require_active_session: SessionValidator,
    require_camera_name: Callable[[dict[str, int], str], str],
    stream_scene_rebuild_job: StreamFactory,
):
    def _camera_index_for(session) -> dict[str, int]:
        camera_index = getattr(session.renderer, "camera_index", None)
        if isinstance(camera_index, dict):
            return camera_index
        return {
            camera.name: index
            for index, camera in enumerate(getattr(session.renderer.bundle, "cameras", ()))
        }

    async def _name_(req: Request):
        sess = req.path_params["session"]
        action = req.path_params["action"]
        name = req.path_params["name"]
        if action == "init":
            future = get_or_create_session_future(req, sess)
            return StreamingResponse(
                init_stream(
                    sess,
                    future,
                    await_session_future,
                    heartbeat_sec_getter(),
                ),
                media_type="text/event-stream; charset=utf-8",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        future = require_session_future(sess)
        session = require_active_session(
            sess,
            future,
            await await_session_future(sess, future),
        )
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
                stream_scene_rebuild_job(session, name),
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
                session.renderer.render(require_camera_name(_camera_index_for(session), name)),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )
        if action == "snapshot":
            return Response(
                content=session.renderer.snapshot(require_camera_name(_camera_index_for(session), name)),
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
            )
        if action == "multicam" and name == "stream":
            return StreamingResponse(
                session.renderer.render_multi_camera(),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )
        if action == "asset":
            asset = session.physics.assets.get(name)
            if asset is None:
                raise HTTPException(status_code=404, detail="Asset not found")
            return FileResponse(asset)
        raise HTTPException(status_code=404, detail="Action not found")

    return _name_
