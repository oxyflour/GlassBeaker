from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .camera_specs import RenderCamera

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_SCENE_USD = REPO_ROOT / "apps" / "python" / "assets" / "default_scene.usda"


@dataclass(frozen=True)
class RenderBundle:
    robot_usd: Path
    scene_usd: Path
    bundle_dir: Path
    mjcf: Path
    sim_scene_usda: Path
    scene_render_usda: Path
    robot_wrapper_usda: Path
    render_scene_usda: Path
    body_map_json: Path
    body_map_jsona: Path
    cameras: list[RenderCamera]

    def to_json(self) -> dict[str, object]:
        data = asdict(self)
        data["cameras"] = [camera.to_json() for camera in self.cameras]
        return {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in data.items()
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "RenderBundle":
        path_fields = {
            "robot_usd",
            "scene_usd",
            "bundle_dir",
            "mjcf",
            "sim_scene_usda",
            "scene_render_usda",
            "robot_wrapper_usda",
            "render_scene_usda",
            "body_map_json",
            "body_map_jsona",
        }
        kwargs = {
            key: (Path(value) if key in path_fields else value)
            for key, value in data.items()
            if key != "cameras"
        }
        kwargs["cameras"] = [
            RenderCamera.from_json(item)
            for item in data.get("cameras", [])
        ]
        return cls(**kwargs)  # type: ignore[arg-type]

    def camera_names(self) -> list[str]:
        return [camera.name for camera in self.cameras]

    def outputs(self) -> tuple[Path, ...]:
        return (
            self.mjcf,
            self.sim_scene_usda,
            self.scene_render_usda,
            self.robot_wrapper_usda,
            self.render_scene_usda,
            self.body_map_json,
            self.body_map_jsona,
        )


__all__ = ["DEFAULT_SCENE_USD", "RenderBundle"]
