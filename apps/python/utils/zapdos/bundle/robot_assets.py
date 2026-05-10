from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
import xml.etree.ElementTree as ET

import mujoco  # type: ignore
from pxr import Usd

REPO_ROOT = Path(__file__).resolve().parents[5]
MOZ1_VISUAL_ROOT = "/World/MOZ1"
MOZ1_VISUAL_CANDIDATES = (
    REPO_ROOT
    / "deps"
    / "moz01"
    / "isaac_moz1"
    / "Issacsim_Assets"
    / "spirit01_model"
    / "spirit01_model"
    / "USD"
    / "Moz1_omni_gripper_full.usd",
    REPO_ROOT
    / "deps"
    / "moz01"
    / "isaac_moz1"
    / "Issacsim_Assets"
    / "spirit01_model"
    / "spirit01_model"
    / "USD"
    / "Moz1_omni_gripper.usd",
)
PACKAGE_PREFIX = "package://spirit01_model/"


@dataclass(frozen=True)
class RobotAssetDescriptor:
    robot_input: Path
    physics_input: Path
    visual_usd: Path
    visual_root: str
    attachments_by_body: dict[str, list[str]]
    static_visual_paths: list[str]
    dependency_paths: list[Path]


def resolve_robot_assets(robot_input: Path, bundle_dir: Path) -> RobotAssetDescriptor:
    robot_input = robot_input.resolve()
    if robot_input.suffix.lower() in {".usd", ".usda", ".usdc"}:
        return RobotAssetDescriptor(
            robot_input=robot_input,
            physics_input=robot_input,
            visual_usd=robot_input,
            visual_root="",
            attachments_by_body={},
            static_visual_paths=[],
            dependency_paths=[robot_input],
        )
    if robot_input.suffix.lower() != ".urdf":
        raise RuntimeError(f"Unsupported robot input: {robot_input}")
    return _resolve_moz1_assets(robot_input, bundle_dir)


def rewrite_urdf_for_bundle(urdf_path: Path, bundle_dir: Path) -> Path:
    rewritten, _ = _rewrite_urdf_assets(urdf_path.resolve(), bundle_dir)
    return rewritten


def _resolve_moz1_assets(robot_input: Path, bundle_dir: Path) -> RobotAssetDescriptor:
    physics_input, mesh_dependencies = _rewrite_urdf_assets(robot_input, bundle_dir)
    visual_usd = _pick_visual_usd()
    body_names = _body_names_from_urdf(physics_input)
    dependencies = [robot_input, visual_usd, *mesh_dependencies]
    return RobotAssetDescriptor(
        robot_input=robot_input,
        physics_input=physics_input,
        visual_usd=visual_usd,
        visual_root=MOZ1_VISUAL_ROOT,
        attachments_by_body=_moz1_attachments(visual_usd, body_names),
        static_visual_paths=[f"{MOZ1_VISUAL_ROOT}/base_link"],
        dependency_paths=list(dict.fromkeys(path.resolve() for path in dependencies)),
    )


def _rewrite_urdf_assets(urdf_path: Path, bundle_dir: Path) -> tuple[Path, list[Path]]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    meshes_dir = bundle_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(urdf_path)
    dependencies: list[Path] = []
    for mesh in tree.findall(".//mesh"):
        filename = mesh.get("filename", "")
        if not filename.startswith(PACKAGE_PREFIX):
            continue
        relative = Path(filename.removeprefix(PACKAGE_PREFIX))
        source = (urdf_path.parents[1] / relative).resolve()
        target = meshes_dir / relative.name
        target.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, target)
        mesh.set("filename", Path("meshes", target.name).as_posix())
        dependencies.append(source)
    rewritten = bundle_dir / f"{urdf_path.stem}.bundle.urdf"
    tree.write(rewritten, encoding="utf-8", xml_declaration=True)
    return rewritten, dependencies


def _pick_visual_usd() -> Path:
    for candidate in MOZ1_VISUAL_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()
    raise RuntimeError("Moz1 visual USD is missing.")


def _body_names_from_urdf(urdf_path: Path) -> list[str]:
    model = mujoco.MjModel.from_xml_path(str(urdf_path))  # type: ignore
    names: list[str] = []
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)  # type: ignore
        if name:
            names.append(name)
    return names


def _moz1_attachments(visual_usd: Path, body_names: list[str]) -> dict[str, list[str]]:
    stage = Usd.Stage.Open(str(visual_usd))
    if stage is None:
        raise RuntimeError(f"Failed to open visual USD: {visual_usd}")
    attachments = {body_name: [f"{MOZ1_VISUAL_ROOT}/{body_name}"] for body_name in body_names}
    attachments["waist03"].extend(f"{MOZ1_VISUAL_ROOT}/{name}" for name in ("head21", "head22", "head23"))
    root = stage.GetPrimAtPath(MOZ1_VISUAL_ROOT)
    for child in root.GetChildren():
        name = child.GetName()
        path = str(child.GetPath())
        if name.startswith("left_") and path not in attachments["left07"]:
            attachments["left07"].append(path)
        if name.startswith("right_") and path not in attachments["right07"]:
            attachments["right07"].append(path)
    return attachments


__all__ = [
    "RobotAssetDescriptor",
    "MOZ1_VISUAL_ROOT",
    "resolve_robot_assets",
    "rewrite_urdf_for_bundle",
]
