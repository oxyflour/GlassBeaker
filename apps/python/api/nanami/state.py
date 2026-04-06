from __future__ import annotations

from fastapi import HTTPException
from pydantic import BaseModel, Field

from nanami.manager import MANAGER


class StateRequest(BaseModel):
    session_id: str
    joints: dict[str, float] = Field(default_factory=dict)
    camera: dict | None = None


async def update(body: StateRequest):
    try:
        MANAGER.queue_state(body.session_id, body.joints, body.camera)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown session.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True}
