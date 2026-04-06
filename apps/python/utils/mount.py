from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path

from fastapi import APIRouter, FastAPI

API_ROOT = Path(__file__).resolve().parents[1]


def module_name(abs_path: Path) -> str:
    digest = hashlib.sha1(str(abs_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"glassbeaker_dynamic_{abs_path.stem}_{digest}"


def load_module(abs_path: Path):
    name = module_name(abs_path)
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(name, abs_path)
    module = spec and importlib.util.module_from_spec(spec)
    if module and spec and spec.loader:
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
        return module
    print(f"WARN: load from {abs_path} failed")
    return None


def route_prefix(root: Path, sub_path: str, abs_path: Path) -> str:
    rel_path = abs_path.relative_to(root).with_suffix("")
    parts = [part for part in rel_path.parts if part != "index"]
    return "/" + "/".join((sub_path, *parts))


def mount_module(app: FastAPI, root: Path, sub_path: str, abs_path: Path) -> None:
    module = load_module(abs_path)
    if module is None:
        return

    prefix = route_prefix(root, sub_path, abs_path)
    router = getattr(module, "router", None)
    if isinstance(router, APIRouter):
        print(f"INFO: adding router {prefix}")
        app.include_router(router, prefix=prefix)
        return

    for name, member in inspect.getmembers(module):
        if inspect.iscoroutinefunction(member) and not name.startswith('_'):
            route = f"{prefix}/{name}"
            print(f"INFO: adding api {route}")
            app.add_api_route(route, endpoint=member, methods=["GET", "POST"])


def mount_routes(app: FastAPI, sub_path: str) -> None:
    root = API_ROOT / sub_path
    for abs_path in sorted(root.rglob("*.py")):
        rel_parts = abs_path.relative_to(root).parts
        if any(part.startswith("_") for part in rel_parts):
            continue
        mount_module(app, root, sub_path, abs_path)
