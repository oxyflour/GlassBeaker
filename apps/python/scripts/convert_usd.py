import bpy # type: ignore
import os
import sys
from pathlib import Path


# =========================
# 配置区
# =========================
EXPORT_FORMAT = "stl"   # 可选: "obj" 或 "stl"
RECURSIVE = True

# 是否只导出 mesh 对象
EXPORT_MESH_ONLY = True

# obj 导出时是否写材质
EXPORT_MATERIALS = True


# =========================
# 工具函数
# =========================
USD_EXTS = {".usd", ".usda", ".usdc", ".usdz"}


def log(msg: str):
    print(f"[usd-convert] {msg}")


def reset_scene():
    # 删除所有对象
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # 删除孤立数据块，避免累计占内存
    for _ in range(3):
        try:
            bpy.ops.outliner.orphans_purge(do_recursive=True)
        except Exception:
            pass


def collect_input_files(root: Path, recursive: bool = True):
    if recursive:
        files = [p for p in root.rglob("*") if p.suffix.lower() in USD_EXTS]
    else:
        files = [p for p in root.glob("*") if p.suffix.lower() in USD_EXTS]
    return sorted(files)


def import_usd(filepath: str):
    log(f"Importing: {filepath}")
    bpy.ops.wm.usd_import(filepath=filepath)


def select_mesh_objects_only():
    bpy.ops.object.select_all(action='DESELECT')
    count = 0
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
            count += 1
    return count


def ensure_parent_dir(filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)


def export_obj(filepath: str):
    ensure_parent_dir(filepath)

    # 新版 Blender
    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(
            filepath=filepath,
            export_selected_objects=EXPORT_MESH_ONLY,
            export_materials=EXPORT_MATERIALS,
        )
    else:
        # 兼容旧版 API
        bpy.ops.export_scene.obj(
            filepath=filepath,
            use_selection=EXPORT_MESH_ONLY,
            use_materials=EXPORT_MATERIALS,
        )


def export_stl(filepath: str):
    ensure_parent_dir(filepath)

    # 新版 Blender
    if hasattr(bpy.ops.wm, "stl_export"):
        bpy.ops.wm.stl_export(
            filepath=filepath,
            export_selected_objects=EXPORT_MESH_ONLY,
        )
    else:
        # 兼容旧版 API
        bpy.ops.export_mesh.stl(
            filepath=filepath,
            use_selection=EXPORT_MESH_ONLY,
        )


def convert_one_file(src_path: Path, input_root: Path, output_root: Path, export_format: str):
    reset_scene()

    try:
        import_usd(str(src_path))
    except Exception as e:
        log(f"FAILED to import {src_path}: {e}")
        return False

    mesh_count = select_mesh_objects_only() if EXPORT_MESH_ONLY else len(bpy.data.objects)

    if mesh_count == 0:
        log(f"No mesh objects found in {src_path}")
        return False

    rel = src_path.relative_to(input_root)
    out_rel = rel.with_suffix(f".{export_format.lower()}")
    out_path = output_root / out_rel

    try:
        if export_format.lower() == "obj":
            export_obj(str(out_path))
        elif export_format.lower() == "stl":
            export_stl(str(out_path))
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

        log(f"OK: {src_path} -> {out_path}")
        return True

    except Exception as e:
        log(f"FAILED to export {src_path}: {e}")
        return False


def main():
    root = Path(sys.argv[-1])
    files = collect_input_files(root, recursive=RECURSIVE)
    log(f"Found {len(files)} USD files")

    ok = 0
    fail = 0

    for f in files:
        if convert_one_file(f, root, root, EXPORT_FORMAT):
            ok += 1
        else:
            fail += 1

    log(f"Done. success={ok}, failed={fail}")


if __name__ == "__main__":
    main()