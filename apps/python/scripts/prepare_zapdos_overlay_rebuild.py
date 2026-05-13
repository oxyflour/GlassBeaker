from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.zapdos.editor.rebuild_runner import prepare_overlay_rebuild_request


def _log_stage(stage: str) -> None:
    print(f"stage: {stage}", file=sys.stderr, flush=True)


def prepare_overlay_rebuild(request: dict[str, object]) -> dict[str, object]:
    return prepare_overlay_rebuild_request(request, _log_stage)


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
