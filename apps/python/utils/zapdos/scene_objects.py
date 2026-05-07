from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pxr import Usd, UsdGeom

from utils.zapdos.usd_to_mjcf import sanitize_name

SKIP_OBJECT_NAMES = {"Ground"}
TRANSPARENT_CONTAINER_NAMES = {"Objects"}
SKIP_OBJECT_TYPES = {
    "Camera",
    "DiskLight",
    "DistantLight",
    "DomeLight",
    "Material",
    "NodeGraph",
    "RectLight",
    "Scope",
    "Shader",
    "SphereLight",
}


@dataclass(frozen=True)
class SceneObjectSpec:
    source_path: str
    render_path: str
    sim_path: str
    body_name: str


def collect_scene_objects(scene_usd: Path) -> list[SceneObjectSpec]:
    stage = Usd.Stage.Open(str(scene_usd))
    if stage is None:
        raise RuntimeError(f"Failed to open scene stage: {scene_usd}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise RuntimeError(f"Scene stage has no default prim: {scene_usd}")
    default_path = PurePosixPath(str(default_prim.GetPath()))
    scene_objects: list[SceneObjectSpec] = []
    for child in object_root_prims(default_prim):
        spec = scene_object_spec(default_path, child)
        if spec is not None:
            scene_objects.append(spec)
    return scene_objects


def object_root_prims(default_prim):
    for child in default_prim.GetChildren():
        if child.GetName() in TRANSPARENT_CONTAINER_NAMES and child.IsA(UsdGeom.Xformable):  # type: ignore
            yield from child.GetChildren()
            continue
        yield child


def scene_object_spec(default_path: PurePosixPath, prim) -> SceneObjectSpec | None:
    if not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):  # type: ignore
        return None
    if prim.GetName() in SKIP_OBJECT_NAMES or prim.GetTypeName() in SKIP_OBJECT_TYPES:
        return None
    prim_path = PurePosixPath(str(prim.GetPath()))
    render_path = prim_path.relative_to(default_path).as_posix()
    sim_path = f"/Scene/{render_path}"
    return SceneObjectSpec(
        source_path=str(prim.GetPath()),
        render_path=render_path,
        sim_path=sim_path,
        body_name=sanitize_name(sim_path),
    )

