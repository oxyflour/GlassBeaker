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


class RenderRequest(BaseModel):
    objects: list[dict]


def objects_to_scene_dict(objects: list[dict]) -> dict:
    """Convert scene objects back to Mitsuba scene dict format."""
    import mitsuba as mi
    scene_dict: dict = {"type": "scene"}
    T = mi.scalar_rgb.Transform4f
    for i, obj in enumerate(objects):
        obj_type = obj.get("type", "box")
        if obj_type == "box":
            obj_type = "cube"
        pos = obj.get("position", [0, 0, 0])
        rot = obj.get("rotation", [0, 0, 0])
        scale = obj.get("scale", [1, 1, 1])
        color_hex = obj.get("color", "#888888")
        # Parse hex color
        color = [int(color_hex.lstrip("#")[j:j+2], 16) / 255.0 for j in (0, 2, 4)] if color_hex.startswith("#") else [0.5, 0.5, 0.5]

        # Build transformation using Mitsuba Transform4f
        transform = T.translate(pos)
        # Apply rotations (XYZ order)
        if rot[0] != 0:
            transform = transform.rotate([1, 0, 0], rot[0])
        if rot[1] != 0:
            transform = transform.rotate([0, 1, 0], rot[1])
        if rot[2] != 0:
            transform = transform.rotate([0, 0, 1], rot[2])
        # Apply scale
        transform = transform.scale(scale)

        shape = {
            "type": obj_type,
            "to_world": transform,
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": color}},
        }
        if obj_type == "sphere":
            shape["radius"] = scale[0] / 2
        scene_dict[f"obj_{i:03d}"] = shape
    return scene_dict


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


async def render(body: RenderRequest) -> dict:
    """Render scene objects using Mitsuba and return the image as base64."""
    try:
        from geniesim.generator.scene_language.engine.utils.mitsuba_utils import render_scene_dict
        import io
        import base64

        scene_dict = objects_to_scene_dict(body.objects)
        image = render_scene_dict(scene_dict, verbose=False)

        # Convert PIL image to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"image": f"data:image/png;base64,{img_base64}"}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Render error: {str(e)}")
