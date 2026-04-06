from __future__ import annotations

from fastapi import HTTPException
from pydantic import BaseModel

from nanami.manager import MANAGER


class ExportRequest(BaseModel):
    session_id: str
    profile: str = "final"


async def start(body: ExportRequest):
    try:
        return {"job_id": MANAGER.start_export(body.session_id, body.profile)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown session.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
