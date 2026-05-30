from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco  # type: ignore


@dataclass(frozen=True)
class BodyCapabilities:
    editable_body_names: set[str]
    robot_body_names: set[str]
    robot_root_body_names: set[str]
    movable_body_names: set[str]
    selection_body_by_name: dict[str, str]


def parent_body_name(model: Any, body_name: str) -> str | None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)  # type: ignore
    if body_id < 0:
        return None
    parent_id = int(model.body_parentid[body_id])
    if parent_id <= 0:
        return None
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)  # type: ignore


def build_body_capabilities(model: Any, body_map: dict[str, str]) -> BodyCapabilities:
    editable_body_names = {
        name for name, render_path in body_map.items()
        if not render_path.startswith("MyRobot/")
    }
    robot_body_names = {
        name for name, render_path in body_map.items()
        if render_path.startswith("MyRobot/")
    }
    robot_root_body_names = {
        name for name in robot_body_names
        if parent_body_name(model, name) not in robot_body_names
    }
    selection_body_by_name = {name: name for name in editable_body_names}
    for body_name in robot_body_names:
        current = body_name
        while current not in robot_root_body_names:
            parent = parent_body_name(model, current)
            if parent is None:
                break
            current = parent
        selection_body_by_name[body_name] = current
    return BodyCapabilities(
        editable_body_names=editable_body_names,
        robot_body_names=robot_body_names,
        robot_root_body_names=robot_root_body_names,
        movable_body_names=editable_body_names | robot_root_body_names,
        selection_body_by_name=selection_body_by_name,
    )


__all__ = ["BodyCapabilities", "build_body_capabilities", "parent_body_name"]
