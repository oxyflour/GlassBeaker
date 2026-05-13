from __future__ import annotations

from typing import Protocol

from utils.zapdos.physics.visuals import SceneVisuals


class PhysicsBackend(Protocol):
    def get_visual(self) -> SceneVisuals: ...

    def get_pose(self) -> dict[str, list[float]]: ...

    def robot_bounds(self) -> dict[str, list[float]] | None: ...

    def get_camera(self) -> dict[str, list[float]]: ...

    def set_body_pose(
        self,
        body: str,
        pos: list[float],
        quat: list[float],
    ) -> dict[str, object]: ...

    def step(self) -> None: ...

    def close(self) -> None: ...
