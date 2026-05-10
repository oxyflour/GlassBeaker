from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from pxr import Gf, Sdf, Usd, UsdGeom

SCENE_ROOT = Path("apps/python/tmp/genie_sim")


def create_scene_dir(repo_root: Path) -> Path:
    scene_dir = repo_root / SCENE_ROOT / uuid.uuid4().hex[:12]
    scene_dir.mkdir(parents=True, exist_ok=True)
    return scene_dir


def write_scene_usda(
    scene_dir: Path,
    assets_root: Path,
    assets_index: dict[str, Any],
    layout_info: dict[str, Any],
) -> Path:
    scene_path = scene_dir / "scene.usda"
    stage = Usd.Stage.CreateNew(scene_path.as_posix())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata(UsdGeom.Tokens.metersPerUnit, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Objects")
    for object_id, placement in layout_info["layout"].items():
        asset_id = placement["usd"]
        relative_payload = Path(assets_index[asset_id]["url"])
        object_path = Sdf.Path(f"/World/Objects/{object_id}")
        payload_path = object_path.AppendChild("Payload")

        object_prim = UsdGeom.Xform.Define(stage, object_path).GetPrim()
        payload_prim = UsdGeom.Xform.Define(stage, payload_path).GetPrim()
        payload_prim.GetPayloads().AddPayload(
            os.path.relpath((assets_root / relative_payload).resolve(), scene_path.parent.resolve())
        )
        ops = UsdGeom.Xformable(object_prim)
        ops.AddTranslateOp().Set(Gf.Vec3d(*placement["xyz"]))
        quat = placement["xyzw"]
        ops.AddOrientOp().Set(Gf.Quatf(quat[3], quat[0], quat[1], quat[2]))
        ops.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().Save()
    return scene_path


def persist_scene_output(
    repo_root: Path,
    assets_root: Path,
    assets_index: dict[str, Any],
    layout_info: dict[str, Any],
) -> dict[str, Any]:
    scene_dir = create_scene_dir(repo_root)
    scene_usda_path = write_scene_usda(scene_dir, assets_root, assets_index, layout_info)
    return {"sceneUsdaPath": scene_usda_path.as_posix()}
