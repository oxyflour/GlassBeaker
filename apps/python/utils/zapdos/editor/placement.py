from __future__ import annotations


def normalize_placement(placement: object):
    if not isinstance(placement, dict):
        raise ValueError("placement must be an object")
    normalized = dict(placement)
    if "position" in normalized and "pos" not in normalized:
        normalized["pos"] = normalized["position"]
    if "orientation" in normalized and "quat" not in normalized:
        normalized["quat"] = normalized["orientation"]
    kind = normalized.get("kind")
    if kind is None:
        kind = _infer_placement_kind(normalized)
        if kind is None:
            raise ValueError("placement.kind is required")
        normalized["kind"] = kind
    if kind == "world_pose":
        normalized["pos"] = _require_float_list(normalized, "pos", 3)
        normalized["quat"] = _require_float_list(normalized, "quat", 4)
        if "payload_quat" in normalized:
            normalized["payload_quat"] = _require_float_list(normalized, "payload_quat", 4)
        return normalized
    if kind == "floor_at_xy":
        normalized["xy"] = _require_float_list(normalized, "xy", 2)
        normalized["z_offset"] = float(normalized.get("z_offset", 0.0))
        if "quat" in normalized:
            normalized["quat"] = _require_float_list(normalized, "quat", 4)
        else:
            normalized["yaw"] = float(normalized.get("yaw", 0.0))
        if "payload_quat" in normalized:
            normalized["payload_quat"] = _require_float_list(normalized, "payload_quat", 4)
        return normalized
    if kind == "on_top_of_body":
        normalized["body"] = _require_nonempty_string(normalized, "body")
        normalized["xy"] = _require_float_list(normalized, "xy", 2)
        normalized["gap"] = float(normalized.get("gap", 0.0))
        if "quat" in normalized:
            normalized["quat"] = _require_float_list(normalized, "quat", 4)
        else:
            normalized["yaw"] = float(normalized.get("yaw", 0.0))
        if "payload_quat" in normalized:
            normalized["payload_quat"] = _require_float_list(normalized, "payload_quat", 4)
        return normalized
    raise ValueError(f"Unsupported placement.kind: {kind}")


def _infer_placement_kind(placement: dict[str, object]) -> str | None:
    if "pos" in placement or "quat" in placement:
        if "pos" in placement and "quat" not in placement:
            placement["quat"] = [1.0, 0.0, 0.0, 0.0]
        return "world_pose" if "pos" in placement and "quat" in placement else None
    if "body" in placement:
        return "on_top_of_body" if "xy" in placement else None
    if "xy" in placement:
        return "floor_at_xy"
    return None


def _require_float_list(placement: dict[str, object], key: str, length: int) -> list[float]:
    value = placement.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"placement.{key} must be a {length}-item list")
    return [float(item) for item in value]


def _require_nonempty_string(placement: dict[str, object], key: str) -> str:
    value = placement.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"placement.{key} must be a non-empty string")
    return value


__all__ = ["normalize_placement"]
