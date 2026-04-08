from __future__ import annotations

import inspect
from pathlib import Path

from fastapi import APIRouter, FastAPI

from utils.module import load_module

API_ROOT = Path(__file__).resolve().parents[1]


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
