from __future__ import annotations

from fastapi import HTTPException
from pydantic import BaseModel

from nanami.manager import MANAGER


class LoadRequest(BaseModel):
    session_id: str


async def load(body: LoadRequest):
    try:
        return MANAGER.load_robot(body.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown session.") from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
