from __future__ import annotations

from typing import Protocol, TypedDict

from utils.zapdos.physics.visuals import SceneVisuals


class AttachmentPayload(TypedDict):
    parent_body: str
    child_body: str
    relative_position: list[float]
    relative_quat: list[float]


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

    def reset_pose(self) -> dict[str, object]: ...

    def joint_state_msg(self) -> dict[str, list[float] | list[str]]: ...

    def apply_joint_command(self, message: dict[str, object] | None) -> None: ...

    def attach_body(self, parent_body: str, child_body: str) -> AttachmentPayload: ...

    def detach_body(self, child_body: str) -> dict[str, object]: ...

    def get_attachment(self, child_body: str) -> AttachmentPayload | None: ...

    def step(self) -> None: ...

    def close(self) -> None: ...
