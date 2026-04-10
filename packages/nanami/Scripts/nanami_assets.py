from __future__ import annotations

from pathlib import PurePosixPath, Path

import unreal  # type: ignore


def log(msg: str) -> None:
    unreal.log(f"[Nanami] {msg}")


def ensure_content_dir(path: str) -> None:
    if not unreal.EditorAssetLibrary.does_directory_exist(path):
        unreal.EditorAssetLibrary.make_directory(path)


def _sanitize(part: str) -> str:
    chars = [char if char.isalnum() or char == "_" else "_" for char in part]
    safe = "".join(chars).strip("_")
    return safe or "asset"


def _mesh_asset_path(obj_path: str, asset_root: str) -> str:
    path = Path(obj_path)
    parts = path.parts
    try:
        index = parts.index("obj")
        rel = PurePosixPath(*parts[index + 1 :]).with_suffix("")
    except ValueError:
        rel = PurePosixPath(path.stem)
    rel_parts = [_sanitize(part) for part in rel.parts]
    package = "/".join(rel_parts)
    return f"{asset_root}/Imported/{package}"


def _texture_asset_path(texture_path: str, asset_root: str) -> str:
    stem = _sanitize(Path(texture_path).stem)
    return f"{asset_root}/HDRI/{stem}"


def _load_existing(asset_path: str, asset_type):
    if not unreal.EditorAssetLibrary.does_asset_exist(asset_path):
        return None
    asset = unreal.EditorAssetLibrary.load_asset(asset_path)
    return asset if isinstance(asset, asset_type) else None


def _import_asset(src_path: str, asset_path: str, asset_type):
    existing = _load_existing(asset_path, asset_type)
    if existing is not None:
        return existing, asset_path
    package_path = asset_path.rsplit("/", 1)[0]
    ensure_content_dir(package_path)
    task = unreal.AssetImportTask()
    task.filename = src_path
    task.destination_path = package_path
    task.destination_name = asset_path.rsplit("/", 1)[-1]
    task.automated = True
    task.save = True
    task.replace_existing = False
    task.replace_existing_settings = False
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    asset = _load_existing(asset_path, asset_type)
    if asset is None:
        raise RuntimeError(f"Failed to import asset: {src_path}")
    return asset, asset_path


def import_static_mesh(obj_path: str, asset_root: str):
    return _import_asset(obj_path, _mesh_asset_path(obj_path, asset_root), unreal.StaticMesh)


def import_texture_cube(texture_path: str, asset_root: str):
    asset, _path = _import_asset(texture_path, _texture_asset_path(texture_path, asset_root), unreal.TextureCube)
    return asset


def ensure_color_material(material_root: str, color_rgba: list[float]):
    ensure_content_dir(material_root)
    rgba = [max(0, min(255, int(round(value * 255)))) for value in color_rgba[:4]]
    name = "M_" + "".join(f"{value:02X}" for value in rgba)
    material_path = f"{material_root}/{name}"
    if unreal.EditorAssetLibrary.does_asset_exist(material_path):
        return unreal.EditorAssetLibrary.load_asset(material_path)

    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    material = asset_tools.create_asset(name, material_root, unreal.Material, unreal.MaterialFactoryNew())
    base_color = unreal.MaterialEditingLibrary.create_material_expression(material, unreal.MaterialExpressionVectorParameter, -400, 0)
    roughness = unreal.MaterialEditingLibrary.create_material_expression(material, unreal.MaterialExpressionScalarParameter, -400, 180)
    metallic = unreal.MaterialEditingLibrary.create_material_expression(material, unreal.MaterialExpressionScalarParameter, -400, 320)
    base_color.set_editor_property("default_value", unreal.LinearColor(*color_rgba))
    roughness.set_editor_property("default_value", 0.42)
    metallic.set_editor_property("default_value", 0.0)
    unreal.MaterialEditingLibrary.connect_material_property(base_color, "", unreal.MaterialProperty.MP_BASE_COLOR)
    unreal.MaterialEditingLibrary.connect_material_property(roughness, "", unreal.MaterialProperty.MP_ROUGHNESS)
    unreal.MaterialEditingLibrary.connect_material_property(metallic, "", unreal.MaterialProperty.MP_METALLIC)
    unreal.MaterialEditingLibrary.recompile_material(material)
    unreal.EditorAssetLibrary.save_loaded_asset(material)
    return material


def apply_material(actor, material) -> None:
    comp = actor.get_component_by_class(unreal.StaticMeshComponent)
    if comp and material is not None:
        comp.set_material(0, material)
