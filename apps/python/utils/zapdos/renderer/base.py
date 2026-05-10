from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from utils.zapdos.bundle import RenderBundle


class RendererBackend(Protocol):
    async def wait_ready(self, timeout: float = 300.0) -> dict[str, Any]: ...

    def read(self, camera_name: str) -> tuple[int, np.ndarray] | None: ...

    def reload_scene(
        self,
        bundle: "RenderBundle",
        timeout: float = 30.0,
    ) -> None: ...

    def snapshot_cameras(self, timeout: float = 5.0) -> list[dict[str, Any]]: ...

    def status(self) -> dict[str, Any]: ...

    def close(self, stop_remote: bool = True) -> None: ...
