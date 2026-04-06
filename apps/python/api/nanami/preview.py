from __future__ import annotations

import asyncio

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
        version = -1
        while not session.closed:
            version, frame = await asyncio.to_thread(session.wait_for_frame, version, 5.0)
            if session.closed:
                return
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

    return StreamingResponse(content(), media_type="multipart/x-mixed-replace; boundary=frame")
