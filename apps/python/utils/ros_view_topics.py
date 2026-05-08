from __future__ import annotations

from urllib.parse import quote

from utils.zapdos.sim_env import IMAGE_TYPE, JOINT_STATE_TYPE

PLOT_COLORS = ["#5eead4", "#f59e0b", "#f472b6"]


def render_path(topic_id: str) -> str:
    return f"/python/ros_view/render/{quote(topic_id, safe='')}"


def normalize_topics(items: object) -> list[tuple[str, list[str]]]:
    topics: list[tuple[str, list[str]]] = []
    if not isinstance(items, list):
        return topics
    for item in items:
        if not isinstance(item, (list, tuple)) or not item:
            continue
        topic = item[0]
        if not isinstance(topic, str):
            continue
        raw_types = item[1] if len(item) > 1 else []
        if isinstance(raw_types, (list, tuple)):
            type_names = [str(type_name) for type_name in raw_types]
        elif raw_types:
            type_names = [str(raw_types)]
        else:
            type_names = []
        topics.append((topic, type_names))
    return topics


def subscription_type(type_names: list[str]) -> str | None:
    if IMAGE_TYPE in type_names:
        return IMAGE_TYPE
    if JOINT_STATE_TYPE in type_names:
        return JOINT_STATE_TYPE
    return None


def build_topic(topic: str, type_names: list[str]) -> dict[str, object]:
    type_name = subscription_type(type_names)
    description = ", ".join(type_names) if type_names else "unknown"
    if type_name == IMAGE_TYPE:
        return {
            "id": topic,
            "kind": "image",
            "label": topic,
            "description": description,
            "src": render_path(topic),
        }
    if type_name == JOINT_STATE_TYPE:
        return {
            "id": topic,
            "kind": "plot",
            "label": topic,
            "description": description,
            "unit": "rad",
            "timestamps": ["now"],
            "series": [],
        }
    return {
        "id": topic,
        "kind": "state",
        "label": topic,
        "description": description,
        "fields": [
            {"label": "Type", "value": description, "tone": "good"},
            {"label": "Status", "value": "Unsupported", "tone": "warn"},
        ],
    }
