from __future__ import annotations

from pathlib import PurePosixPath
from typing import TypedDict


class BodyVisual(TypedDict):
    name: str
    label: str
    editable: bool
    selectable: bool
    movable: bool
    selectionBody: str | None
    matrix: list[float]


class MeshVisual(TypedDict, total=False):
    name: str
    body: str | None
    kind: str
    color: list[float] | None
    matrix: list[float]
    localMatrix: list[float]
    size: list[float]
    mesh: str
    texture: str


class SceneVisuals(TypedDict):
    bodies: list[BodyVisual]
    meshes: list[MeshVisual]


def body_label(render_path: str) -> str:
    return PurePosixPath(render_path).name or render_path


def serialize_body(
    name: str,
    render_path: str,
    editable: bool,
    matrix: list[float],
    *,
    selectable: bool,
    movable: bool,
    selection_body: str | None,
) -> BodyVisual:
    return {
        "name": name,
        "label": body_label(render_path),
        "editable": editable,
        "selectable": selectable,
        "movable": movable,
        "selectionBody": selection_body,
        "matrix": matrix,
    }


def serialize_mesh(
    name: str,
    body: str | None,
    kind: str,
    color: list[float] | None,
    *,
    matrix: list[float] | None = None,
    local_matrix: list[float] | None = None,
    size: list[float] | None = None,
    mesh: str = "",
    texture: str = "",
) -> MeshVisual:
    payload: MeshVisual = {
        "name": name,
        "body": body,
        "kind": kind,
        "color": color,
    }
    if matrix is not None:
        payload["matrix"] = matrix
    if local_matrix is not None:
        payload["localMatrix"] = local_matrix
    if size is not None:
        payload["size"] = size
    if mesh:
        payload["mesh"] = mesh
    if texture:
        payload["texture"] = texture
    return payload
