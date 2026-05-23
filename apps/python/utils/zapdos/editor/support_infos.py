from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco  # type: ignore
from fastapi import HTTPException

from utils.genie_sim import resolve_assets_root
from utils.zapdos.editor.state import overlay_body_name
from utils.zapdos.physics.mujoco_tools import body_world_pose
from utils.zapdos.zapdos_asset_library import asset_local_bounds


@dataclass(frozen=True)
class SupportInfoBody:
    body: str
    base_top_z: float
    asset_path: Path | None = None
    instance_id: str | None = None
    add_asset_max_z: bool = False


@dataclass(frozen=True)
class SupportInfoSnapshot:
    bodies: tuple[SupportInfoBody, ...]


def capture_support_info_inputs(editor: Any) -> SupportInfoSnapshot:
    assets_root = resolve_assets_root(editor.overlay_state.get("assets_root"))
    instance_by_body = {
        overlay_body_name(item["id"]): item
        for item in editor.overlay_state["instances"]
    }
    bodies: list[SupportInfoBody] = []
    for body in editor.session.physics.editable_body_names:
        body_id = mujoco.mj_name2id(editor.session.physics.model, mujoco.mjtObj.mjOBJ_BODY, body)  # type: ignore
        instance = instance_by_body.get(body)
        asset_path = assets_root / str(instance["url"]) if instance is not None else None
        world_aabb = editor.session.physics.body_world_aabb(body)
        if world_aabb is not None:
            bodies.append(SupportInfoBody(
                body=body,
                base_top_z=float(world_aabb["max"][2]),
                asset_path=asset_path,
                instance_id=str(instance["id"]) if instance is not None else None,
            ))
            continue
        bodies.append(SupportInfoBody(
            body=body,
            base_top_z=float(body_world_pose(editor.session.physics.data, body_id)[2, 3]),
            asset_path=asset_path,
            instance_id=str(instance["id"]) if instance is not None else None,
            add_asset_max_z=asset_path is not None,
        ))
    return SupportInfoSnapshot(bodies=tuple(bodies))


def resolve_support_infos(snapshot: SupportInfoSnapshot) -> dict[str, dict[str, float]]:
    infos: dict[str, dict[str, float]] = {}
    for item in snapshot.bodies:
        top_z = item.base_top_z
        if item.asset_path is not None:
            try:
                bounds = asset_local_bounds(item.asset_path)
            except (FileNotFoundError, OSError, RuntimeError) as exc:
                raise HTTPException(
                    status_code=409,
                    detail=f"Existing overlay asset unavailable: {item.instance_id}: {exc}",
                ) from exc
            if item.add_asset_max_z:
                top_z += float(bounds["max"][2])
        infos[item.body] = {"top_z": top_z}
    return infos
