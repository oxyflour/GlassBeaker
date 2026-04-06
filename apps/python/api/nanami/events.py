from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from nanami.manager import MANAGER

router = APIRouter()


@router.get("/stream")
async def stream(session_id: str):
    try:
        session = MANAGER.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown session.") from exc

    async def content():
        async for event in session.stream_events():
            if event["type"] == "keepalive":
                yield ": keepalive\n\n"
                continue
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(content(), media_type="text/event-stream")
