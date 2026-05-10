from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys
import uuid
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .genie_sim_bundle import persist_scene_output

DEFAULT_ASSETS_ROOT = (
    Path.home()
    / ".cache"
    / "modelscope"
    / "hub"
    / "datasets"
    / "agibot_world"
    / "GenieSimAssets"
)

HELPER_MODULE_ALIASES = (
    "helper",
    "genie_sim_helper",
    "genie_sim_open",
    "geniesim_helper",
    "geniesim_open",
)


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_assets_root(assets_root: str | Path | None = None) -> Path:
    if assets_root:
        return Path(assets_root).resolve()
    for name in ("GENIE_SIM_ASSETS_ROOT", "SIM_ASSETS"):
        value = os.getenv(name)
        if value:
            return Path(value).resolve()
    return DEFAULT_ASSETS_ROOT.resolve()


@lru_cache(maxsize=8)
def _load_assets_module_cached(assets_root: str):
    root = Path(assets_root)
    init_file = root / "__init__.py"
    if not init_file.exists():
        raise FileNotFoundError(f"GenieSim assets entry not found: {init_file}")
    spec = importlib.util.spec_from_file_location("genie_sim_external_assets", init_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create module spec for {init_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_assets_module(assets_root: str | Path | None = None):
    module = _load_assets_module_cached(str(resolve_assets_root(assets_root)))
    sys.modules["assets"] = module
    sys.modules["geniesim.assets"] = module
    return module


def _description_text(asset_info: dict[str, Any]) -> str:
    description = asset_info.get("description", {})
    parts: list[str] = []
    for key in ("semantic_name", "full_description"):
        value = description.get(key, "")
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    return " ".join(parts)


def search_assets(assets_index: dict[str, Any], query: str, top_k: int = 8) -> list[dict[str, Any]]:
    query = query.strip().lower()
    if not query:
        return []
    tokens = [token for token in re.split(r"[^a-z0-9]+", query) if token]
    hits: list[dict[str, Any]] = []
    for asset_id, asset_info in assets_index.items():
        haystack = f"{asset_id} {_description_text(asset_info)}".lower()
        score = 0.0
        if asset_id.lower() == query:
            score += 1000
        if query in haystack:
            score += 100
        score += sum(3 if token in asset_id.lower() else 1 for token in tokens if token in haystack)
        if score <= 0:
            continue
        hits.append(
            {
                "asset_id": asset_id,
                "description": asset_info.get("description", {}),
                "score": score,
                "url": asset_info.get("url", ""),
            }
        )
    hits.sort(key=lambda item: (-item["score"], item["asset_id"]))
    return [{key: value for key, value in item.items() if key != "score"} for item in hits[:top_k]]


def _prepend_sys_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def _patch_rotation_matrix_compat(helper: Any) -> None:
    try:
        math_utils = importlib.import_module("geniesim.generator.scene_language.math_utils")
    except ModuleNotFoundError as exc:
        if exc.name in {
            "geniesim",
            "geniesim.generator",
            "geniesim.generator.scene_language",
            "geniesim.generator.scene_language.math_utils",
        }:
            return
        raise
    rotation_matrix = math_utils.rotation_matrix
    if getattr(rotation_matrix, "_glassbeaker_numpy2_compat", False):
        helper.rotation_matrix = rotation_matrix
        return

    def rotation_matrix_compat(angle: float, direction: Any, point: Any):
        return rotation_matrix(
            angle,
            np.asarray(direction, dtype=np.float64),
            np.asarray(point, dtype=np.float64),
        )

    rotation_matrix_compat._glassbeaker_numpy2_compat = True
    math_utils.rotation_matrix = rotation_matrix_compat
    helper.rotation_matrix = rotation_matrix_compat


def _install_exposed_primitive_compat() -> None:
    try:
        from geniesim.generator.scene_language._shape_utils import primitive_call
        from geniesim.generator.scene_language.math_utils import _scale_matrix
    except ModuleNotFoundError as exc:
        if exc.name in {
            "geniesim",
            "geniesim.generator",
            "geniesim.generator.scene_language",
            "geniesim.generator.scene_language._shape_utils",
            "geniesim.generator.scene_language.math_utils",
        }:
            return
        raise

    if getattr(primitive_call, "is_implemented", False):
        return

    def _bsdf(color: Any) -> dict[str, Any]:
        return {
            "type": "diffuse",
            "reflectance": {"type": "rgb", "value": np.asarray(color[:3]).clip(0, 1)},
        }

    def cube_fn(*, info: Any, scale: Any, color: Any = (1, 1, 1)) -> list[dict[str, Any]]:
        return [
            {
                "type": "cube",
                "to_world": _scale_matrix(scale, enforce_uniform=False) @ _scale_matrix(0.5),
                "bsdf": _bsdf(color),
                "info": {"stack": [], "info": info},
            }
        ]

    def sphere_fn(*, info: Any, radius: float = 1.0, color: Any = (1, 1, 1)) -> list[dict[str, Any]]:
        return [
            {
                "type": "sphere",
                "to_world": _scale_matrix(radius),
                "bsdf": _bsdf(color),
                "info": {"stack": [], "info": info},
            }
        ]

    def cylinder_fn(
        *,
        info: Any,
        radius: float,
        p0: Any,
        p1: Any,
        color: Any = (1, 1, 1),
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "cylinder",
                "p0": tuple(float(value) for value in p0),
                "p1": tuple(float(value) for value in p1),
                "radius": float(radius),
                "to_world": np.eye(4),
                "bsdf": _bsdf(color),
                "info": {"stack": [], "info": info},
            }
        ]

    def impl_primitive_call():
        def fn(name: str, **kwargs):
            return {
                "cube": cube_fn,
                "sphere": sphere_fn,
                "cylinder": cylinder_fn,
            }.get(name, cube_fn)(**kwargs)

        return fn

    primitive_call.implement(impl_primitive_call)


@lru_cache(maxsize=1)
def _prepare_runtime_cached(assets_root: str):
    repo_root = resolve_repo_root()
    geniesim_root = repo_root / "deps" / "genie_sim"
    source_root = geniesim_root / "source"
    generator_root = source_root / "geniesim" / "generator"
    os.environ["SIM_REPO_ROOT"] = str(geniesim_root)
    os.environ["SIM_ASSETS"] = assets_root
    os.environ.setdefault("ENGINE_MODE", "exposed")
    _prepend_sys_path(source_root)
    _prepend_sys_path(generator_root)
    load_assets_module(assets_root)
    import helper

    _patch_rotation_matrix_compat(helper)
    _install_exposed_primitive_compat()
    return helper


def prepare_runtime(assets_root: str | Path | None = None):
    return _prepare_runtime_cached(str(resolve_assets_root(assets_root)))


def _strip_code_fences(code: str) -> str:
    text = code.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _normalize_helper_imports(code: str) -> str:
    normalized: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("from ") and stripped.endswith(" import *"):
            module_name = stripped[len("from ") : -len(" import *")].strip()
            if module_name in HELPER_MODULE_ALIASES:
                normalized.append("from helper import *")
                continue
        normalized.append(line)
    return "\n".join(normalized)


def _scene_object_id(shape: dict[str, Any]) -> str:
    asset_id = shape["info"]["info"]["id"]
    prefix = "".join(char for char in asset_id if not char.isdigit()).rstrip("_")
    stack = "".join(str(entry[1]) for entry in shape["info"].get("stack", []))
    instance = uuid.uuid5(uuid.NAMESPACE_DNS, stack)
    return f"{prefix}_{str(instance).split('-')[0]}"


def _build_scene_objects(helper, scene_data: list[dict[str, Any]], layout: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for shape in scene_data:
        groups[_scene_object_id(shape)].append(shape)
    objects: list[dict[str, Any]] = []
    for object_id, placement in layout.items():
        size = helper.get_object_info(groups.get(object_id, []))["size"]
        objects.append(
            {
                "assetId": placement["usd"],
                "description": placement["description"],
                "id": object_id,
                "keywords": placement.get("keywords", []),
                "position": placement["xyz"],
                "quaternion": placement["xyzw"],
                "size": [round(float(axis), 3) for axis in size],
            }
        )
    return objects


def _install_helper_aliases(helper: Any) -> None:
    for module_name in HELPER_MODULE_ALIASES:
        sys.modules[module_name] = helper


def _copy_library_entries(library: Any) -> dict[str, Any]:
    if not hasattr(library, "items"):
        return {}
    return {name: value.copy() if isinstance(value, dict) else value for name, value in library.items()}


def _reset_dsl_library(helper: Any, library: Any) -> None:
    if not hasattr(library, "clear"):
        return
    if not hasattr(library, "items") or not hasattr(library, "update"):
        library.clear()
        return
    base_library = getattr(helper, "_GENIE_SIM_BASE_LIBRARY", None)
    if not isinstance(base_library, dict):
        base_library = _copy_library_entries(library)
        setattr(helper, "_GENIE_SIM_BASE_LIBRARY", base_library)
    library.clear()
    library.update(_copy_library_entries(base_library))


def execute_scene_code(code: str, assets_root: str | Path | None = None) -> dict[str, Any]:
    helper = prepare_runtime(assets_root)
    from geniesim.generator.scene_language.dsl_utils import library

    code = _normalize_helper_imports(_strip_code_fences(code))
    _install_helper_aliases(helper)
    _reset_dsl_library(helper, library)
    namespace = dict(helper.__dict__)
    namespace["__name__"] = "__main__"
    namespace.setdefault("Shape", list[dict[str, Any]])
    namespace.setdefault("Scene", namespace["Shape"])
    exec(code, namespace)
    root_scene = namespace.get("root_scene")
    if root_scene is None:
        raise ValueError("Generated scene code must define root_scene().")
    scene_data = root_scene()
    layout_info, _ = helper.gen_scene_layout_info(scene_data)
    resolved_assets_root = resolve_assets_root(assets_root)
    payload = {
        "assetsRoot": str(resolved_assets_root),
        "code": code,
        "description": layout_info["scene_id"],
        "objects": _build_scene_objects(helper, scene_data, layout_info["layout"]),
        "relations": layout_info["relations"]["graph"],
        "seed": layout_info["seed"],
    }
    output = persist_scene_output(
        resolve_repo_root(),
        resolved_assets_root,
        load_assets_module(resolved_assets_root).ASSETS_INDEX,
        layout_info,
    )
    return payload | output
