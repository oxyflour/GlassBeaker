from __future__ import annotations

from fastapi import HTTPException, status
from pydantic import BaseModel

from nanami.manager import MANAGER
from nanami.runtime import RuntimeBusyError


class SourceRequest(BaseModel):
    robot_path: str = "R1"


class SessionRequest(BaseModel):
    source: SourceRequest | None = None


async def create(body: SessionRequest):
    try:
        source = body.source.model_dump(exclude_none=True) if body.source else None
        data = MANAGER.create_session(source)
    except RuntimeBusyError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Nanami runtime busy.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    session_id = data["session_id"]
    data["preview_url"] = f"/python/nanami/preview/stream?session_id={session_id}"
    data["events_url"] = f"/python/nanami/events/stream?session_id={session_id}"
    return data


class DestroyRequest(BaseModel):
    session_id: str


async def destroy(body: DestroyRequest):
    try:
        MANAGER.destroy_session(body.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown session.") from exc
    return {"ok": True}
