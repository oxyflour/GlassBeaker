from __future__ import annotations

import os
import warnings
import traceback

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel
from scipy.spatial.transform import Rotation as R

warnings.filterwarnings("ignore")
os.environ.setdefault("MI_DEFAULT_VARIANT", "scalar_rgb")

import geniesim.generator.scene_language.mi_helper  # implements primitive_call
from geniesim.generator.scene_language.engine_utils import primitive_call as _engine_primitive_call
from geniesim.generator.scene_language.shape_utils import transform_shape, concat_shapes
from geniesim.generator.scene_language.math_utils import (
    translation_matrix,
    rotation_matrix,
    scale_matrix,
    identity_matrix,
)
from geniesim.generator.scene_language.dsl_utils import register, library_call
import math

# Shape kwargs that belong inside shape_kwargs, not passed directly
_SHAPE_KWARGS_KEYS = {"scale", "radius", "p0", "p1"}


def primitive_call(name, *args, **kwargs):
    """Flexible wrapper: accepts both shape_kwargs={...} and flat kwargs."""
    # Map aliases to supported types
    _ALIASES = {"plane": "cube", "box": "cube", "cone": "cube"}
    name = _ALIASES.get(name, name)
    if "shape_kwargs" not in kwargs:
        # LLM passed shape args directly — repackage into shape_kwargs
        shape_kwargs = {}
        for key in _SHAPE_KWARGS_KEYS:
            if key in kwargs:
                shape_kwargs[key] = kwargs.pop(key)
        if shape_kwargs:
            kwargs["shape_kwargs"] = shape_kwargs
    if "info" not in kwargs:
        kwargs["info"] = {"id": name, "name": "origin"}
    return _engine_primitive_call(name, *args, **kwargs)


SAFETY_BUILTINS = {
    "abs": abs, "min": min, "max": max, "round": round,
    "len": len, "range": range, "enumerate": enumerate,
    "float": float, "int": int, "str": str, "list": list,
    "dict": dict, "tuple": tuple, "bool": bool,
}

SAFETY_MODULES = {
    "primitive_call": primitive_call,
    "transform_shape": transform_shape,
    "concat_shapes": concat_shapes,
    "translation_matrix": translation_matrix,
    "rotation_matrix": rotation_matrix,
    "scale_matrix": scale_matrix,
    "identity_matrix": identity_matrix,
    "register": register,
    "library_call": library_call,
    "np": np,
    "math": math,
}


class SceneRequest(BaseModel):
    code: str


def shapes_to_json(shapes: list) -> list:
    objects = []
    for s in shapes:
        tw = np.array(s["to_world"])
        pos = tw[:3, 3].tolist()
        sx = float(np.linalg.norm(tw[:3, 0]))
        sy = float(np.linalg.norm(tw[:3, 1]))
        sz = float(np.linalg.norm(tw[:3, 2]))
        rot_mat = tw[:3, :3] / np.array([sx, sy, sz])
        euler = R.from_matrix(rot_mat).as_euler("xyz", degrees=True).tolist()
        color_rgb = [float(c) for c in s["bsdf"]["reflectance"]["value"][:3]]
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(color_rgb[0] * 255), int(color_rgb[1] * 255), int(color_rgb[2] * 255)
        )
        obj_type = s["type"]
        if obj_type == "cube":
            obj_type = "box"
        info = s.get("info", {}).get("info", {})
        label = info.get("id", "object")
        objects.append(
            {
                "id": label,
                "type": obj_type,
                "position": [round(x, 3) for x in pos],
                "rotation": [round(x, 1) for x in euler],
                "scale": [round(sx, 3), round(sy, 3), round(sz, 3)],
                "color": hex_color,
                "label": label,
            }
        )
    return objects


def _strip_imports(code: str) -> str:
    """Remove import/from lines — all DSL functions are pre-injected."""
    import re
    return re.sub(r"^(?:from\s+\S+\s+)?import\s+.+$", "", code, flags=re.MULTILINE)


def execute_dsl(code: str) -> list:
    from geniesim.generator.scene_language.dsl_utils import library
    library.clear()
    code = _strip_imports(code)
    ns = {**SAFETY_BUILTINS, **SAFETY_MODULES}
    exec(code, ns)
    for name, entry in library.items():
        fn = entry.get("__target__")
        if fn and name not in ns:
            ns[name] = fn
    root_fn = ns.get("root_scene")
    if root_fn is None:
        for name in reversed(list(library.keys())):
            fn = library[name].get("__target__")
            if fn:
                root_fn = fn
                break
    if root_fn is None:
        raise ValueError("No @register() function found in code")
    return root_fn()


async def execute(body: SceneRequest) -> dict:
    try:
        shapes = execute_dsl(body.code)
        objects = shapes_to_json(shapes)
        return {"objects": objects, "description": "genie_sim scene"}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Execution error: {str(e)}")
