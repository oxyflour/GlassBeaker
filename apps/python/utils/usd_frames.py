import re
from pathlib import Path

from pxr import Usd, UsdGeom


_SKIP_TYPES = {"Scope", "Material", "Shader"}


def sanitize_name(path: str) -> str:
    name = path.strip("/").replace("/", "_")
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    if not name:
        return "root"
    return f"_{name}" if name[0].isdigit() else name


def build_frame_map(usd: Path) -> dict[str, str]:
    stage = Usd.Stage.Open(str(usd))
    if stage is None:
        return {}

    frame_map: dict[str, str] = {}
    for prim in stage.Traverse():
        if not prim or not prim.IsValid():
            continue
        if prim.GetTypeName() in _SKIP_TYPES:
            continue
        if not prim.IsA(UsdGeom.Xformable):
            continue
        path = str(prim.GetPath())
        frame_map.setdefault(sanitize_name(path), path.lstrip("/"))
    return frame_map
