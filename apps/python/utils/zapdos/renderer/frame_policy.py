from __future__ import annotations

from collections.abc import Iterable


def should_render_frame(subscribers: Iterable[object], frame_counter: int) -> bool:
    if frame_counter <= 0:
        return True
    return any(bool(getattr(subscriber, "_dirty", False)) for subscriber in subscribers)


__all__ = ["should_render_frame"]
