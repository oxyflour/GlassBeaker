from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.genie_sim_runtime import resolve_assets_root
from utils.zapdos.rl_bundle import ensure_render_bundle
from utils.zapdos.zapdos_asset_library import asset_local_bounds
from utils.zapdos.zapdos_overlay import scene_revision
from utils.zapdos.zapdos_overlay_scene import write_overlay_scene


def prepare_overlay_rebuild(request: dict[str, object]) -> dict[str, object]:
    next_overlay = request["next_overlay"]
    if not isinstance(next_overlay, dict):
        raise TypeError("next_overlay must be an object")
    support_infos = request["support_infos"]
    if not isinstance(support_infos, dict):
        raise TypeError("support_infos must be an object")
    robot_usd = Path(str(request["robot_usd"]))
    base_scene_usd = Path(str(request["base_scene_usd"]))
    composed_scene_usd = Path(str(request["composed_scene_usd"]))
    assets_root = resolve_assets_root(next_overlay.get("assets_root"))
    instances = next_overlay.get("instances")
    if not isinstance(instances, list):
        raise TypeError("next_overlay.instances must be a list")
    bounds_by_instance = {
        str(item["id"]): asset_local_bounds(assets_root / str(item["url"]))
        for item in instances
    }
    write_overlay_scene(
        composed_scene_usd,
        base_scene_usd,
        assets_root,
        next_overlay,
        support_infos=support_infos,
        asset_bounds_by_instance=bounds_by_instance,
    )
    bundle = ensure_render_bundle(robot_usd, composed_scene_usd)
    return {
        "bundle": bundle.to_json(),
        "next_revision": scene_revision(base_scene_usd, next_overlay),
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: prepare_zapdos_overlay_rebuild.py <request.json> <response.json>", file=sys.stderr)
        return 2
    request_path = Path(args[0])
    response_path = Path(args[1])
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        response = prepare_overlay_rebuild(request)
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(json.dumps(response, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as err:
        print(str(err), file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
