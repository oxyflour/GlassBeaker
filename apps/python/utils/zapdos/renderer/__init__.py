from __future__ import annotations

__all__ = ["RendererBackend", "ZapdosRenderer"]


def __getattr__(name: str):
    if name == "RendererBackend":
        from .base import RendererBackend

        return RendererBackend
    if name == "ZapdosRenderer":
        from .zapdos_renderer import ZapdosRenderer

        return ZapdosRenderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
