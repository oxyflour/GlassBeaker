from __future__ import annotations

__all__ = ["DEFAULT_SCENE_USD", "RenderBundle", "ensure_render_bundle"]


def __getattr__(name: str):
    if name == "ensure_render_bundle":
        from .bundle_builder import ensure_render_bundle

        return ensure_render_bundle
    if name in {"DEFAULT_SCENE_USD", "RenderBundle"}:
        from .render_bundle import DEFAULT_SCENE_USD, RenderBundle

        exports = {
            "DEFAULT_SCENE_USD": DEFAULT_SCENE_USD,
            "RenderBundle": RenderBundle,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
