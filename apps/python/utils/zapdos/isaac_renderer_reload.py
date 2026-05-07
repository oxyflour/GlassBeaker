from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping


@dataclass(frozen=True)
class CameraBinding:
    annotator: object | None
    render_product: object | None


def validate_camera_topology(
    current_cameras: list[Mapping[str, object]],
    next_cameras: list[Mapping[str, object]],
) -> None:
    if len(current_cameras) != len(next_cameras):
        raise RuntimeError(
            "renderer camera topology changed: expected "
            f"{len(current_cameras)} camera(s), got {len(next_cameras)}"
        )
    for index, (current, nxt) in enumerate(zip(current_cameras, next_cameras)):
        current_name = str(current.get("name") or "")
        next_name = str(nxt.get("name") or "")
        if current_name != next_name:
            raise RuntimeError(
                "renderer camera topology changed at index "
                f"{index}: expected '{current_name}', got '{next_name}'"
            )


def reset_subscriber_caches(
    subscribers: Iterable[object],
    body_name_map: dict[str, str],
) -> None:
    for subscriber in subscribers:
        subscriber.body_name_map = body_name_map
        subscriber._attr_cache = {}
        subscriber._ordered_attrs = None


def rebuild_camera_bindings(
    env_paths: list[str],
    camera_list: list[Mapping[str, object]],
    old_bindings: list[list[CameraBinding]],
    create_binding: Callable[[str], CameraBinding],
    release_binding: Callable[[CameraBinding], None],
) -> list[list[CameraBinding]]:
    for bindings in old_bindings:
        for binding in bindings:
            release_binding(binding)

    rebuilt: list[list[CameraBinding]] = [[] for _ in range(len(camera_list))]
    for env_path in env_paths:
        for camera_index, camera in enumerate(camera_list):
            rebuilt[camera_index].append(
                create_binding(env_path + str(camera["prim"]))
            )
    return rebuilt
