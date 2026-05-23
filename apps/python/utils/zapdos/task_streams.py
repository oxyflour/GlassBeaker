from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

FINITE_TASK_NAMES = {
    "set_scene_assets",
    "remove_asset_from_scene",
    "reset_pose",
    "pick_apple",
    "place_apple",
    "pick_object",
}

type SessionFutureAwaiter = Callable[[str, asyncio.Future], Awaitable[object]]


def bootstrap_error_message(exc: Exception) -> str:
    message = str(exc).strip()
    return message or "Session bootstrap failed"


async def init_stream(
    sess: str,
    future: asyncio.Future,
    await_session_future: SessionFutureAwaiter,
    heartbeat_sec: float,
):
    yield "started", {"task": "init"}
    while not future.done():
        done, _ = await asyncio.wait({future}, timeout=heartbeat_sec)
        if done:
            break
        yield "progress", {"message": "preparing render bundle"}
    try:
        await await_session_future(sess, future)
    except Exception as exc:
        yield "failed", {"detail": bootstrap_error_message(exc)}
        return
    yield "done", {"ok": True}


async def init_stream_response_events(
    sess: str,
    future: asyncio.Future,
    await_session_future: SessionFutureAwaiter,
    heartbeat_sec: float,
    *,
    created: bool = False,
    discard_session_future: Callable[[str, asyncio.Future], None] | None = None,
):
    delivered = False
    try:
        async for event in init_stream(sess, future, await_session_future, heartbeat_sec):
            if event[0] in {"done", "failed"}:
                delivered = True
            yield event
    finally:
        if not delivered and created and not future.done():
            future.cancel()
            if discard_session_future is not None:
                discard_session_future(sess, future)


async def scene_task_events(session: Any, task_name: str, args: list[object]):
    from utils.zapdos.editor.rebuild_events import stream_scene_rebuild_events

    yield "started", {"task": task_name}
    try:
        result = await session.call(task_name, *args)
        op_id = result.get("op_id") if isinstance(result, dict) else None
        if not isinstance(op_id, str):
            yield "done", result
            return
        async for event, payload in stream_scene_rebuild_events(
            session.editor,
            op_id,
            cancel_on_disconnect=True,
        ):
            yield event, payload
    except Exception as exc:
        payload = {"detail": getattr(exc, "detail", None) or str(exc)}
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            payload["status_code"] = status_code
        yield "failed", payload
