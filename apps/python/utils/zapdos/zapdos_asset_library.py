from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from pxr import Usd, UsdGeom

from utils.genie_sim import load_assets_module, resolve_assets_root


class AssetRecord(TypedDict):
    asset_id: str
    url: str
    description: dict[str, object]


class AssetBounds(TypedDict):
    min: list[float]
    max: list[float]


def resolve_asset_record(asset_id: str, assets_root: str | Path | None = None) -> AssetRecord:
    root = resolve_assets_root(assets_root)
    module = load_assets_module(root)
    info = module.ASSETS_INDEX.get(asset_id)
    if info is None:
        raise KeyError(asset_id)
    return {"asset_id": asset_id, "url": info["url"], "description": info.get("description", {})}


@lru_cache(maxsize=256)
def asset_local_bounds(asset_path: Path) -> AssetBounds:
    stage = Usd.Stage.Open(str(asset_path))
    if stage is None:
        raise RuntimeError(f"Failed to open asset stage: {asset_path}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise RuntimeError(f"Asset stage has no default prim: {asset_path}")
    cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
    box = cache.ComputeWorldBound(default_prim).ComputeAlignedBox()
    return {
        "min": [float(box.GetMin()[0]), float(box.GetMin()[1]), float(box.GetMin()[2])],
        "max": [float(box.GetMax()[0]), float(box.GetMax()[1]), float(box.GetMax()[2])],
    }
