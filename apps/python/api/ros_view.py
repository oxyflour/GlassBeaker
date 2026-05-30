from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from utils.ros_view.ros_view_store import store

router = APIRouter()


@router.get("/state")
async def state() -> dict[str, object]:
    return await store.state()


@router.get("/stream")
async def stream() -> StreamingResponse:
    return StreamingResponse(
        store.stream(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/render/{topic_id:path}")
async def render(topic_id: str) -> StreamingResponse:
    if not store.has_image_topic(topic_id):
        raise HTTPException(status_code=404, detail="Image topic not found")
    content = store.render(topic_id)
    return StreamingResponse(content, media_type="multipart/x-mixed-replace; boundary=frame")
