from __future__ import annotations

import traceback

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from utils.genie_sim_runtime import (
    execute_scene_code,
    load_assets_module,
    resolve_assets_root,
    search_assets as search_assets_index,
)

router = APIRouter()


class AssetSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=8, ge=1, le=20)
    assets_root: str | None = None


class SceneRequest(BaseModel):
    code: str
    assets_root: str | None = None


def _detail(exc: Exception, fallback: str) -> str:
    return str(exc).strip() or fallback


def _detail_with_traceback(exc: Exception, fallback: str) -> str:
    detail = _detail(exc, fallback)
    stack = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
    if not stack:
        return detail
    if detail in stack:
        return stack
    return f"{detail}\n\n{stack}"


@router.post("/search_assets")
async def search_assets(body: AssetSearchRequest) -> dict:
    try:
        assets_root = resolve_assets_root(body.assets_root)
        assets_module = load_assets_module(assets_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_detail(exc, "GenieSim assets not found.")) from exc
    return {
        "assetsRoot": str(assets_root),
        "items": search_assets_index(assets_module.ASSETS_INDEX, body.query, top_k=body.top_k),
    }


@router.post("/execute")
async def execute(body: SceneRequest) -> dict:
    try:
        return execute_scene_code(body.code, assets_root=body.assets_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_detail(exc, "GenieSim assets not found.")) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_detail_with_traceback(exc, "Scene execution failed.")) from exc
