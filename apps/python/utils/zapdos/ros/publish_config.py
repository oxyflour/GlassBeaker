from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from utils.user_config import read_user_config
from utils.zapdos.ros.topics import IMAGE_TYPE, JOINT_STATE_TYPE


@dataclass(frozen=True)
class ImagePublishSpec:
    topic: str
    type_name: str
    camera_name: str
    format: str
    quality: int


@dataclass(frozen=True)
class JointStatePublishSpec:
    topic: str
    type_name: str
    joints: tuple[JointCommandTarget, ...]


@dataclass(frozen=True)
class JointCommandTarget:
    name: str
    scale: float


@dataclass(frozen=True)
class JointStateSubscribeSpec:
    topic: str
    type_name: str
    joints: tuple[tuple[JointCommandTarget, ...], ...]


def configured_image_publish_specs(camera_names: Iterable[str] | None = None) -> list[ImagePublishSpec]:
    return image_publish_specs(read_user_config(), camera_names)


def configured_joint_state_publish_specs() -> list[JointStatePublishSpec]:
    return joint_state_publish_specs(read_user_config())


def configured_joint_state_subscribe_specs() -> list[JointStateSubscribeSpec]:
    return joint_state_subscribe_specs(read_user_config())


def image_publish_specs(config: dict[str, Any], camera_names: Iterable[str] | None = None) -> list[ImagePublishSpec]:
    allowed = set(camera_names) if camera_names is not None else None
    specs: list[ImagePublishSpec] = []
    for topic, entry in _ros_entries(config, "publish").items():
        if not isinstance(entry, dict) or entry.get("type") != IMAGE_TYPE:
            continue
        camera_name = _camera_name(topic, entry)
        if not camera_name or (allowed is not None and camera_name not in allowed):
            continue
        specs.append(ImagePublishSpec(
            topic=topic,
            type_name=IMAGE_TYPE,
            camera_name=camera_name,
            format=str(entry.get("format") or "raw").strip().lower(),
            quality=_quality(entry.get("quality")),
        ))
    return specs


def joint_state_publish_specs(config: dict[str, Any]) -> list[JointStatePublishSpec]:
    specs: list[JointStatePublishSpec] = []
    for topic, entry in _ros_entries(config, "publish").items():
        if not isinstance(entry, dict) or entry.get("type") != JOINT_STATE_TYPE:
            continue
        joints = entry.get("joints")
        specs.append(JointStatePublishSpec(
            topic=topic,
            type_name=JOINT_STATE_TYPE,
            joints=tuple(target for item in joints if (target := _joint_command_target(item)) is not None)
            if isinstance(joints, list)
            else (),
        ))
    return specs


def joint_state_subscribe_specs(config: dict[str, Any]) -> list[JointStateSubscribeSpec]:
    specs: list[JointStateSubscribeSpec] = []
    for topic, entry in _ros_entries(config, "subscribe").items():
        if not isinstance(entry, dict) or entry.get("type") != JOINT_STATE_TYPE:
            continue
        specs.append(JointStateSubscribeSpec(
            topic=topic,
            type_name=JOINT_STATE_TYPE,
            joints=_joint_command_groups(entry.get("joints")),
        ))
    return specs


def joint_state_publish_messages(
    joint_state: dict[str, Any],
    specs: Iterable[JointStatePublishSpec],
) -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (spec.topic, spec.type_name, filter_joint_state(joint_state, spec.joints))
        for spec in specs
    ]


def filter_joint_state(joint_state: dict[str, Any], joints: tuple[object, ...]) -> dict[str, Any]:
    if not joints:
        return {
            "name": list(joint_state.get("name") or []),
            "position": list(joint_state.get("position") or []),
            "velocity": list(joint_state.get("velocity") or []),
            "effort": list(joint_state.get("effort") or []),
        }
    names = [str(name) for name in (joint_state.get("name") or [])]
    index_by_name = {name: index for index, name in enumerate(names)}
    targets = tuple(_joint_command_target(joint) for joint in joints)
    selected = [target for target in targets if target is not None and target.name in index_by_name]
    indexes = [index_by_name[target.name] for target in selected]
    scales = [target.scale for target in selected]
    return {
        "name": [target.name for target in selected],
        "position": _select(joint_state.get("position"), indexes, scales),
        "velocity": _select(joint_state.get("velocity"), indexes, scales),
        "effort": _select(joint_state.get("effort"), indexes, scales),
    }


def joint_command_from_subscribe_msg(msg: dict[str, Any], spec: JointStateSubscribeSpec) -> dict[str, Any]:
    source_positions = list(msg.get("position") or [])
    names: list[str] = []
    positions: list[float] = []
    for index, targets in enumerate(spec.joints):
        if index >= len(source_positions):
            continue
        source_position = float(source_positions[index])
        for target in targets:
            names.append(target.name)
            positions.append(source_position * target.scale)
    return {"name": names, "position": positions}


def _ros_entries(config: dict[str, Any], section: str) -> dict[str, Any]:
    ros = config.get("ros")
    if not isinstance(ros, dict):
        return {}
    entries = ros.get(section)
    if not isinstance(entries, dict):
        return {}
    return {str(topic): value for topic, value in entries.items()}


def _joint_command_groups(value: object) -> tuple[tuple[JointCommandTarget, ...], ...]:
    if not isinstance(value, list):
        return ()
    groups: list[tuple[JointCommandTarget, ...]] = []
    for group in value:
        items = group if isinstance(group, list) else [group]
        targets = tuple(target for item in items if (target := _joint_command_target(item)) is not None)
        groups.append(targets)
    return tuple(groups)


def _joint_command_target(value: object) -> JointCommandTarget | None:
    if isinstance(value, JointCommandTarget):
        return value
    if isinstance(value, str):
        return JointCommandTarget(value, 1.0)
    if not isinstance(value, dict):
        return None
    name = str(value.get("name") or "").strip()
    if not name:
        return None
    scale = value.get("scale", 1.0)
    scale_value = 1.0 if isinstance(scale, bool) or not isinstance(scale, (int, float)) else float(scale)
    return JointCommandTarget(name, scale_value)


def _camera_name(topic: str, entry: dict[str, Any]) -> str:
    configured = entry.get("camera") or entry.get("camera_name")
    if configured:
        return str(configured)
    parts = topic.strip("/").split("/")
    if len(parts) >= 2 and parts[-1] in {"image", "image_raw"}:
        return parts[-2]
    return ""


def _quality(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 80
    return max(1, min(100, int(value)))


def _select(values: object, indexes: list[int], scales: list[float] | None = None) -> list[float]:
    source = list(values or [])
    factors = scales if scales is not None else [1.0] * len(indexes)
    return [
        (float(source[index]) if index < len(source) else 0.0) * factors[offset]
        for offset, index in enumerate(indexes)
    ]
