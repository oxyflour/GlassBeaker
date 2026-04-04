import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


UPROJECT_TEMPLATE = {
    "FileVersion": 3,
    "EngineAssociation": "5.6",
    "Category": "",
    "Description": "Headless OBJ render project",
    "Plugins": [
        {"Name": "PythonScriptPlugin", "Enabled": True},
        {"Name": "EditorScriptingUtilities", "Enabled": True},
        {"Name": "SequencerScripting", "Enabled": True},
        {"Name": "MovieRenderPipeline", "Enabled": True},
    ],
}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def ensure_project(project_root: Path, project_name: str, engine_association: str) -> Path:
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / "Content").mkdir(exist_ok=True)
    (project_root / "Config").mkdir(exist_ok=True)
    (project_root / "Scripts").mkdir(exist_ok=True)
    (project_root / "Temp").mkdir(exist_ok=True)

    uproject_path = project_root / f"{project_name}.uproject"
    if not uproject_path.exists():
        data = dict(UPROJECT_TEMPLATE)
        data["EngineAssociation"] = engine_association
        write_text(uproject_path, json.dumps(data, indent=2))
    return uproject_path


def find_unreal_editor_cmd(engine_root: Path) -> Path:
    exe = engine_root / "Engine" / "Binaries" / "Win64" / "UnrealEditor-Cmd.exe"
    if not exe.exists():
        raise FileNotFoundError(f"UnrealEditor-Cmd.exe not found: {exe}")
    return exe


def stream_process(cmd, env):
    print("Running:")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    print()

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    return proc.wait()


def find_first_png(output_dir: Path) -> Path | None:
    pngs = sorted(output_dir.glob("*.png"))
    return pngs[0] if pngs else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-root", required=True, help=r"C:\Program Files\Epic Games\UE_5.6")
    parser.add_argument("--project-root", required=True, help=r"D:\ue_projects\HeadlessObjRender")
    parser.add_argument("--project-name", default="HeadlessObjRender")
    parser.add_argument("--engine-association", default="5.6")
    parser.add_argument("--obj", required=True, help=r"D:\assets\model.obj")
    parser.add_argument("--output-dir", required=True, help=r"D:\renders\shot01")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--rotation-x", type=float, default=0.0)
    parser.add_argument("--rotation-y", type=float, default=0.0)
    parser.add_argument("--rotation-z", type=float, default=0.0)
    parser.add_argument("--sequence-frames", type=int, default=2, help="Use 2 for a still")
    parser.add_argument("--render-script", default=None, help="Path to render_obj_pipeline.py")
    args = parser.parse_args()

    engine_root = Path(args.engine_root).resolve()
    project_root = Path(args.project_root).resolve()
    obj_path = Path(args.obj).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ not found: {obj_path}")

    uproject_path = ensure_project(project_root, args.project_name, args.engine_association)
    unreal_cmd = find_unreal_editor_cmd(engine_root)

    render_script = Path(args.render_script).resolve() if args.render_script else (project_root / "Scripts" / "render_obj_pipeline.py")
    if not render_script.exists():
        raise FileNotFoundError(f"Unreal Python script not found: {render_script}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "obj_path": str(obj_path),
        "output_dir": str(output_dir),
        "width": args.width,
        "height": args.height,
        "rotation_deg": [args.rotation_x, args.rotation_y, args.rotation_z],
        "sequence_frames": args.sequence_frames,
        "content_root": "/Game/Auto",
        "import_dest": "/Game/Auto/Imported",
        "map_path": "/Game/Auto/Maps/AutoMap",
        "sequence_path": "/Game/Auto/Sequences/LS_Auto",
        "job_name": "obj_render",
    }

    cfg_path = project_root / "Temp" / "render_config.json"
    write_text(cfg_path, json.dumps(cfg, indent=2))

    env = os.environ.copy()
    env["UE_HEADLESS_RENDER_CONFIG"] = str(cfg_path)

    cmd = [
        str(unreal_cmd),
        str(uproject_path),
        f"-ExecutePythonScript={render_script}",
        "-unattended",
        "-stdout",
        "-FullStdOutLogOutput",
        "-NoSplash",
        "-NoSound",
    ]

    rc = stream_process(cmd, env)
    if rc != 0:
        raise SystemExit(rc)

    first_png = find_first_png(output_dir)
    if first_png:
        final_png = output_dir / "result.png"
        shutil.copy2(first_png, final_png)
        print(f"\nDone. First frame copied to: {final_png}")
    else:
        print("\nUnreal exited successfully, but no PNG was found.")
        print(f"Check output dir: {output_dir}")


if __name__ == "__main__":
    main()
