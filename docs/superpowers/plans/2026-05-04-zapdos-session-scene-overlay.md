# Zapdos Session Scene Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add agent-callable session-local asset insertion, removal, and pose-preserving overlay rebuilds to `/demo/zapdos` without changing the meaning of `scene_usd` or `robot_usd`.

**Architecture:** Represent mutable scene edits as a persisted overlay JSON plus a composed session USDA that references the immutable base `scene_usd`. The Python session rebuilds MuJoCo and Isaac bundle outputs from `base scene + overlay topology`, keeps editable body pose overrides outside `scene_revision`, and emits a small SSE topology event that the existing React Three Fiber scene uses to reload visuals.

**Tech Stack:** Python 3.12, FastAPI, MuJoCo, USD (`pxr`), Next.js client components, CopilotKit frontend tools, React Three Fiber, Drei, `node:test`, `unittest`

---

## File Structure

- `apps/python/utils/zapdos_overlay.py`
  Keeps overlay types, JSON persistence, stable body naming, and revision hashing.
- `apps/python/utils/zapdos_asset_library.py`
  Reuses the Genie Sim asset index, resolves asset records, and computes asset bounds needed for placement.
- `apps/python/utils/zapdos_overlay_scene.py`
  Resolves placement modes into world poses and writes the composed session USDA.
- `apps/python/api/zapdos/{session}/{action}.py`
  Stores overlay state on `ZapdosSession`, exposes new call routes, rebuilds the runtime safely, and emits `scene_revision`.
- `apps/python/tests/test_zapdos_overlay.py`
  Covers overlay hashing, naming, and JSON round-tripping.
- `apps/python/tests/test_zapdos_overlay_scene.py`
  Covers asset lookup, bounds, placement resolution, and composed USDA output.
- `apps/python/tests/test_zapdos_import.py`
  Covers session rebuild RPCs, rollback behavior, pose-override persistence, and SSE revision events.
- `apps/web/components/zapdos/zapdos-tool-api.ts`
  Typed fetch helpers for scene-editing tool calls.
- `apps/web/components/zapdos/zapdos-tool-api.test.ts`
  Covers frontend request builders and fetch wrappers.
- `apps/web/components/zapdos/zapdos-agent-instructions.ts`
  Provides focused Copilot instructions for search, inspect, add, and remove flows.
- `apps/web/components/zapdos/useZapdosAgentTools.ts`
  Registers the Zapdos frontend tools with CopilotKit.
- `apps/web/components/zapdos/zapdos-runtime.ts`
  Adds helpers for parsing `scene_revision` payloads from SSE.
- `apps/web/components/zapdos/zapdos-runtime.test.ts`
  Covers revision parsing and disconnected-session behavior.
- `apps/web/components/zapdos/zapdos-scene-state.ts`
  Adds pure helpers for revision changes and stale selection cleanup.
- `apps/web/components/zapdos/zapdos-scene-state.test.ts`
  Covers revision change detection and selection invalidation.
- `apps/web/components/zapdos/ZapdosScene.tsx`
  Swaps in Zapdos-specific agent tools, listens for scene revision events, and reloads visuals without disturbing normal pose sync.

### Task 1: Add Overlay State, Naming, and Revision Helpers

**Files:**
- Create: `apps/python/utils/zapdos_overlay.py`
- Create: `apps/python/tests/test_zapdos_overlay.py`

- [ ] **Step 1: Write the failing overlay-state tests**

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from utils import zapdos_overlay


class ZapdosOverlayTest(unittest.TestCase):
    def test_scene_revision_ignores_pose_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene.usda"
            scene.write_text("#usda 1.0\n", encoding="utf-8")
            base = zapdos_overlay.default_overlay_state("C:/assets")
            base["instances"] = [{
                "id": "table_000_01",
                "asset_id": "table_000",
                "url": "objects/table_000/Aligned.usda",
                "motion": "static",
                "placement": {"kind": "floor_at_xy", "xy": [0.0, 0.0], "z_offset": 0.0, "yaw": 0.0},
            }]
            edited = json.loads(json.dumps(base))
            edited["pose_overrides"]["Scene_table_000_01"] = {"pos": [1.0, 2.0, 3.0], "quat": [1.0, 0.0, 0.0, 0.0]}

            self.assertEqual(
                zapdos_overlay.scene_revision(scene, base),
                zapdos_overlay.scene_revision(scene, edited),
            )

    def test_bundle_revision_changes_when_robot_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            robot_a = Path(tmp) / "robot-a.usda"
            robot_b = Path(tmp) / "robot-b.usda"
            scene = Path(tmp) / "scene.usda"
            for path in (robot_a, robot_b, scene):
                path.write_text("#usda 1.0\n", encoding="utf-8")

            overlay = zapdos_overlay.default_overlay_state("C:/assets")
            self.assertNotEqual(
                zapdos_overlay.bundle_revision(robot_a, scene, overlay),
                zapdos_overlay.bundle_revision(robot_b, scene, overlay),
            )

    def test_save_and_load_overlay_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "overlay.json"
            overlay = zapdos_overlay.default_overlay_state("C:/assets")
            overlay["instances"].append({
                "id": "crate_001_01",
                "asset_id": "crate_001",
                "url": "objects/crate_001/Aligned.usda",
                "motion": "dynamic",
                "placement": {"kind": "world_pose", "pos": [1.0, 2.0, 0.5], "quat": [1.0, 0.0, 0.0, 0.0]},
            })

            zapdos_overlay.save_overlay_state(path, overlay)

            self.assertEqual(zapdos_overlay.load_overlay_state(path), overlay)
            self.assertEqual(zapdos_overlay.overlay_body_name("crate_001_01"), "Scene_crate_001_01")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the overlay-state test to verify it fails**

Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_zapdos_overlay -v`

Expected: FAIL with `ModuleNotFoundError` for `utils.zapdos_overlay` or missing exported helpers.

- [ ] **Step 3: Implement the overlay helper module**

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypedDict

from utils.usd_to_mjcf import sanitize_name


class OverlayPoseOverride(TypedDict):
    pos: list[float]
    quat: list[float]


class OverlayPlacement(TypedDict, total=False):
    kind: str
    xy: list[float]
    z_offset: float
    yaw: float
    body: str
    gap: float
    pos: list[float]
    quat: list[float]


class OverlayInstance(TypedDict):
    id: str
    asset_id: str
    url: str
    motion: str
    placement: OverlayPlacement


class OverlayState(TypedDict):
    version: int
    assets_root: str | None
    instances: list[OverlayInstance]
    pose_overrides: dict[str, OverlayPoseOverride]


def default_overlay_state(assets_root: str | None = None) -> OverlayState:
    return {"version": 1, "assets_root": assets_root, "instances": [], "pose_overrides": {}}


def overlay_body_name(instance_id: str) -> str:
    return f"Scene_{sanitize_name(instance_id)}"


def load_overlay_state(path: Path) -> OverlayState:
    if not path.exists():
        return default_overlay_state()
    data = json.loads(path.read_text(encoding="utf-8"))
    state = default_overlay_state(data.get("assets_root"))
    state["instances"] = list(data.get("instances") or [])
    state["pose_overrides"] = dict(data.get("pose_overrides") or {})
    return state


def save_overlay_state(path: Path, state: OverlayState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def scene_revision(scene_usd: Path, state: OverlayState) -> str:
    return _digest(["scene", _fingerprint(scene_usd), _normalized_instances(state)])


def bundle_revision(robot_usd: Path, scene_usd: Path, state: OverlayState) -> str:
    return _digest(["bundle", _fingerprint(robot_usd), _fingerprint(scene_usd), _normalized_instances(state)])


def _normalized_instances(state: OverlayState) -> list[dict[str, object]]:
    return sorted(
        [
            {
                "id": item["id"],
                "asset_id": item["asset_id"],
                "url": item["url"],
                "motion": item["motion"],
                "placement": item["placement"],
            }
            for item in state["instances"]
        ],
        key=lambda item: item["id"],
    )


def _fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path.resolve()), "mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


def _digest(parts: list[object]) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
```

- [ ] **Step 4: Run the overlay-state test to verify it passes**

Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_zapdos_overlay -v`

Expected: PASS with 3 tests.

- [ ] **Step 5: Commit the overlay-state helpers**

```bash
git add apps/python/utils/zapdos_overlay.py apps/python/tests/test_zapdos_overlay.py
git commit -m "feat: add zapdos overlay state helpers"
```

### Task 2: Resolve Asset Bounds and Write the Composed Overlay USDA

**Files:**
- Create: `apps/python/utils/zapdos_asset_library.py`
- Create: `apps/python/utils/zapdos_overlay_scene.py`
- Create: `apps/python/tests/test_zapdos_overlay_scene.py`

- [ ] **Step 1: Write the failing asset and overlay-scene tests**

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pxr import Gf, Usd, UsdGeom

from utils.zapdos_asset_library import asset_local_bounds, resolve_asset_record
from utils.zapdos_overlay import default_overlay_state
from utils.zapdos_overlay_scene import resolve_instance_pose, write_overlay_scene


class ZapdosOverlaySceneTest(unittest.TestCase):
    def make_assets_root(self, tmp: str) -> Path:
        assets_root = Path(tmp) / "GenieSimAssets"
        asset_dir = assets_root / "objects" / "table_000"
        asset_dir.mkdir(parents=True)
        asset_path = asset_dir / "Aligned.usda"
        stage = Usd.Stage.CreateNew(asset_path.as_posix())
        asset = UsdGeom.Xform.Define(stage, "/Asset")
        geom = UsdGeom.Cube.Define(stage, "/Asset/Cube")
        geom.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.375))
        geom.CreateSizeAttr(0.75)
        stage.SetDefaultPrim(asset.GetPrim())
        stage.GetRootLayer().Save()
        (assets_root / "__init__.py").write_text(
            "\n".join(
                [
                    "from pathlib import Path",
                    "ASSETS_PATH = Path(__file__).parent",
                    "ASSETS_INDEX = {'table_000': {'url': 'objects/table_000/Aligned.usda', 'description': {'semantic_name': ['table']}}}",
                    "ASSETS_INDEX_HASH = 'overlay-test-hash'",
                ]
            ),
            encoding="utf-8",
        )
        return assets_root

    def test_resolve_asset_record_and_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_root = self.make_assets_root(tmp)
            record = resolve_asset_record("table_000", assets_root)
            bounds = asset_local_bounds(assets_root / record["url"])

            self.assertEqual(record["asset_id"], "table_000")
            self.assertEqual(bounds["min"][2], 0.0)
            self.assertGreater(bounds["max"][2], 0.7)

    def test_resolve_instance_pose_supports_floor_and_support_body(self):
        pose = resolve_instance_pose(
            {
                "id": "mug_001_01",
                "asset_id": "mug_001",
                "url": "objects/mug_001/Aligned.usda",
                "motion": "dynamic",
                "placement": {"kind": "on_top_of_body", "body": "Scene_table_000_01", "xy": [0.1, 0.2], "gap": 0.0, "yaw": 0.0},
            },
            asset_bounds={"min": [-0.05, -0.05, 0.0], "max": [0.05, 0.05, 0.1]},
            support_infos={"Scene_table_000_01": {"top_z": 0.75}},
            pose_overrides={},
        )

        self.assertEqual(pose["pos"], [0.1, 0.2, 0.75])

    def test_write_overlay_scene_references_base_scene_and_marks_static_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_root = self.make_assets_root(tmp)
            base_scene = Path(tmp) / "scene.usda"
            stage = Usd.Stage.CreateNew(base_scene.as_posix())
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            stage.GetRootLayer().Save()

            overlay = default_overlay_state(str(assets_root))
            overlay["instances"].append({
                "id": "table_000_01",
                "asset_id": "table_000",
                "url": "objects/table_000/Aligned.usda",
                "motion": "static",
                "placement": {"kind": "floor_at_xy", "xy": [1.0, -0.5], "z_offset": 0.0, "yaw": 0.0},
            })

            scene_path = write_overlay_scene(
                Path(tmp) / "overlay_scene.usda",
                base_scene,
                assets_root,
                overlay,
                support_infos={},
                asset_bounds_by_instance={"table_000_01": {"min": [-0.375, -0.375, 0.0], "max": [0.375, 0.375, 0.75]}},
            )

            stage = Usd.Stage.Open(scene_path.as_posix())
            table = stage.GetPrimAtPath("/World/Objects/table_000_01")
            self.assertTrue(table.IsValid())
            self.assertTrue(table.HasAuthoredAttribute("physics:kinematicEnabled"))
```

- [ ] **Step 2: Run the overlay-scene test to verify it fails**

Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_zapdos_overlay_scene -v`

Expected: FAIL with `ModuleNotFoundError` for `utils.zapdos_asset_library` or `utils.zapdos_overlay_scene`.

- [ ] **Step 3: Implement asset lookup, bounds, and scene writing**

```python
# apps/python/utils/zapdos_asset_library.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from pxr import Usd, UsdGeom

from utils.genie_sim_runtime import load_assets_module, resolve_assets_root


class AssetRecord(TypedDict):
    asset_id: str
    url: str
    description: dict[str, object]


class AssetBounds(TypedDict):
    min: list[float]
    max: list[float]


def resolve_asset_record(asset_id: str, assets_root: str | Path | None = None) -> AssetRecord:
    root = resolve_assets_root(assets_root)
    module = load_assets_module(root)
    info = module.ASSETS_INDEX.get(asset_id)
    if info is None:
        raise KeyError(asset_id)
    return {"asset_id": asset_id, "url": info["url"], "description": info.get("description", {})}


@lru_cache(maxsize=256)
def asset_local_bounds(asset_path: Path) -> AssetBounds:
    stage = Usd.Stage.Open(str(asset_path))
    if stage is None:
        raise RuntimeError(f"Failed to open asset stage: {asset_path}")
    default_prim = stage.GetDefaultPrim()
    cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
    box = cache.ComputeWorldBound(default_prim).ComputeAlignedBox()
    return {
        "min": [float(box.GetMin()[0]), float(box.GetMin()[1]), float(box.GetMin()[2])],
        "max": [float(box.GetMax()[0]), float(box.GetMax()[1]), float(box.GetMax()[2])],
    }
```

```python
# apps/python/utils/zapdos_overlay_scene.py
from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

from pxr import Gf, Sdf, Usd, UsdGeom

import math

from utils.zapdos_overlay import OverlayInstance, OverlayPoseOverride, OverlayState, overlay_body_name


class SupportInfo(TypedDict):
    top_z: float


def resolve_instance_pose(
    instance: OverlayInstance,
    *,
    asset_bounds: dict[str, list[float]],
    support_infos: dict[str, SupportInfo],
    pose_overrides: dict[str, OverlayPoseOverride],
) -> dict[str, list[float]]:
    body_name = overlay_body_name(instance["id"])
    if body_name in pose_overrides:
        return pose_overrides[body_name]
    placement = instance["placement"]
    if placement["kind"] == "world_pose":
        return {"pos": list(placement["pos"]), "quat": list(placement["quat"])}
    yaw = float(placement.get("yaw", 0.0))
    quat = [float(math.cos(yaw / 2.0)), 0.0, 0.0, float(math.sin(yaw / 2.0))]
    if placement["kind"] == "floor_at_xy":
        return {
            "pos": [float(placement["xy"][0]), float(placement["xy"][1]), float(placement.get("z_offset", 0.0)) - float(asset_bounds["min"][2])],
            "quat": quat,
        }
    support = support_infos[placement["body"]]
    return {
        "pos": [float(placement["xy"][0]), float(placement["xy"][1]), float(support["top_z"]) + float(placement.get("gap", 0.0)) - float(asset_bounds["min"][2])],
        "quat": quat,
    }


def write_overlay_scene(
    output_path: Path,
    base_scene_usd: Path,
    assets_root: Path,
    overlay_state: OverlayState,
    *,
    support_infos: dict[str, SupportInfo],
    asset_bounds_by_instance: dict[str, dict[str, list[float]]],
) -> Path:
    stage = Usd.Stage.CreateNew(output_path.as_posix())
    world = UsdGeom.Xform.Define(stage, "/World")
    world.GetPrim().GetReferences().AddReference(str(base_scene_usd.resolve()))
    UsdGeom.Xform.Define(stage, "/World/Objects")
    for instance in overlay_state["instances"]:
        pose = resolve_instance_pose(
            instance,
            asset_bounds=asset_bounds_by_instance[instance["id"]],
            support_infos=support_infos,
            pose_overrides=overlay_state["pose_overrides"],
        )
        object_path = Sdf.Path(f"/World/Objects/{instance['id']}")
        payload_path = object_path.AppendChild("Payload")
        object_prim = UsdGeom.Xform.Define(stage, object_path).GetPrim()
        payload_prim = UsdGeom.Xform.Define(stage, payload_path).GetPrim()
        payload_prim.GetPayloads().AddPayload(
            os.path.relpath((assets_root / instance["url"]).resolve(), output_path.parent.resolve())
        )
        xform = UsdGeom.Xformable(object_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(*pose["pos"]))
        quat = pose["quat"]
        xform.AddOrientOp().Set(Gf.Quatf(quat[0], quat[1], quat[2], quat[3]))
        xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        object_prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(instance["motion"] == "static")
        if instance["motion"] == "dynamic":
            object_prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Double).Set(1.0)
    stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().Save()
    return output_path
```

- [ ] **Step 4: Run the overlay-scene test to verify it passes**

Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_zapdos_overlay_scene -v`

Expected: PASS with 3 tests.

- [ ] **Step 5: Commit the overlay-scene helpers**

```bash
git add apps/python/utils/zapdos_asset_library.py apps/python/utils/zapdos_overlay_scene.py apps/python/tests/test_zapdos_overlay_scene.py
git commit -m "feat: add zapdos overlay scene builder"
```

### Task 3: Add Session Rebuild RPCs, Pose Overrides, and Revision Events

**Files:**
- Modify: `apps/python/api/zapdos/{session}/{action}.py`
- Modify: `apps/python/tests/test_zapdos_import.py`

- [ ] **Step 1: Extend the Zapdos session tests with failing rebuild scenarios**

```python
    def test_add_asset_to_scene_returns_body_and_revision(self):
        session = self.build_pose_edit_session()
        session.base_scene_usd = Path("scene.usda")
        session.robot_usd = Path("robot.usda")
        session.overlay_state = MODULE.default_overlay_state("C:/assets")
        session.scene_revision = "rev-1"
        session.overlay_path = Path("overlay.json")
        session.composed_scene_usd = Path("overlay_scene.usda")
        session.rebuilding_scene = False

        with mock.patch.object(MODULE, "resolve_asset_record", return_value={"asset_id": "table_000", "url": "objects/table_000/Aligned.usda", "description": {}}):
            with mock.patch.object(MODULE, "asset_local_bounds", return_value={"min": [-0.5, -0.5, 0.0], "max": [0.5, 0.5, 0.75]}):
                with mock.patch.object(session, "_rebuild_overlay_runtime", return_value="rev-2"):
                    result = MODULE.ZapdosSession.add_asset_to_scene(
                        session,
                        "table_000",
                        "static",
                        {"kind": "floor_at_xy", "xy": [0.0, 0.0], "z_offset": 0.0, "yaw": 0.0},
                    )

        self.assertEqual(result["scene_revision"], "rev-2")
        self.assertEqual(result["body"], "Scene_table_000_01")

    def test_set_body_pose_persists_pose_override_without_changing_scene_revision(self):
        session = self.build_freejoint_pose_edit_session()
        session.overlay_state = MODULE.default_overlay_state("C:/assets")
        session.overlay_path = Path("overlay.json")
        session.scene_revision = "rev-1"

        with mock.patch.object(MODULE, "save_overlay_state") as save_overlay:
            session.call_once("set_body_pose", ("Scene_Crate", [4.0, 5.0, 6.0], [1.0, 0.0, 0.0, 0.0]))

        self.assertEqual(session.scene_revision, "rev-1")
        self.assertEqual(session.overlay_state["pose_overrides"]["Scene_Crate"]["pos"], [4.0, 5.0, 6.0])
        save_overlay.assert_called_once()

    def test_failed_rebuild_restores_previous_overlay_state(self):
        session = self.build_pose_edit_session()
        session.base_scene_usd = Path("scene.usda")
        session.robot_usd = Path("robot.usda")
        session.overlay_state = MODULE.default_overlay_state("C:/assets")
        session.overlay_path = Path("overlay.json")
        session.scene_revision = "rev-1"
        session.rebuilding_scene = False

        with mock.patch.object(MODULE, "save_overlay_state"):
            with mock.patch.object(session, "_build_support_infos", return_value={}):
                with mock.patch.object(MODULE, "write_overlay_scene", side_effect=RuntimeError("bundle exploded")):
                    with self.assertRaises(RuntimeError):
                        session._rebuild_overlay_runtime(
                            lambda state: state["instances"].append({
                                "id": "table_000_01",
                                "asset_id": "table_000",
                                "url": "objects/table_000/Aligned.usda",
                                "motion": "static",
                                "placement": {"kind": "floor_at_xy", "xy": [0.0, 0.0], "z_offset": 0.0, "yaw": 0.0},
                            })
                        )

        self.assertEqual(session.overlay_state["instances"], [])
        self.assertEqual(session.scene_revision, "rev-1")
```

- [ ] **Step 2: Run the Zapdos import test to verify it fails**

Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_zapdos_import -v`

Expected: FAIL with missing `default_overlay_state`, `add_asset_to_scene`, or `_rebuild_overlay_runtime`.

- [ ] **Step 3: Implement the session overlay rebuild flow and call routes**

```python
# imports near the top of apps/python/api/zapdos/{session}/{action}.py
from copy import deepcopy

from utils.zapdos_asset_library import asset_local_bounds, resolve_asset_record
from utils.genie_sim_runtime import resolve_assets_root
from utils.zapdos_overlay import (
    bundle_revision,
    default_overlay_state,
    load_overlay_state,
    overlay_body_name,
    save_overlay_state,
    scene_revision,
)
from utils.zapdos_overlay_scene import write_overlay_scene
```

```python
# inside ZapdosSession.__init__
        self.robot_usd = bundle.robot_usd
        self.base_scene_usd = bundle.scene_usd
        self.session_dir = REPO_ROOT / "apps" / "python" / "tmp" / "zapdos" / sess
        self.overlay_path = self.session_dir / "overlay.json"
        self.composed_scene_usd = self.session_dir / "scene-overlay.usda"
        self.overlay_state = load_overlay_state(self.overlay_path)
        self.scene_revision = scene_revision(self.base_scene_usd, self.overlay_state)
        self.rebuilding_scene = False
```

```python
# new methods on ZapdosSession
    def _build_support_infos(self) -> dict[str, dict[str, float]]:
        infos: dict[str, dict[str, float]] = {}
        assets_root = resolve_assets_root(self.overlay_state.get("assets_root"))
        instance_by_body = {
            overlay_body_name(item["id"]): item
            for item in self.overlay_state["instances"]
        }
        for body in self.editable_body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)  # type: ignore
            top_z = float(self.data.xpos[body_id][2])
            instance = instance_by_body.get(body)
            if instance is not None:
                bounds = asset_local_bounds(assets_root / instance["url"])
                top_z += float(bounds["max"][2])
            infos[body] = {"top_z": top_z}
        return infos

    def _swap_runtime_bundle(self, bundle) -> None:
        snapshot_qpos = np.copy(self.data.qpos)
        snapshot_ctrl = np.copy(self.data.ctrl)
        old_renderer = self.renderer
        self.bundle = bundle
        self.model = mujoco.MjModel.from_xml_path(str(bundle.mjcf))  # type: ignore
        self.data = mujoco.MjData(self.model)  # type: ignore
        self.assets = {}
        self.geoms = self._build_geometry(bundle.mjcf.parent)
        self.body_map = json.loads(bundle.body_map_json.read_text(encoding="utf-8"))
        self.body_labels = {name: path.rsplit("/", 1)[-1] for name, path in self.body_map.items()}
        self.editable_body_names = {
            name for name, path in self.body_map.items() if not str(path).startswith("MyRobot/")
        }
        self.camera_index = camera_name_to_index(bundle.cameras)
        self.last_frame_index = {camera.name: -1 for camera in bundle.cameras}
        old_renderer.close()
        self.renderer = IsaacRenderer(self.sess, bundle, RENDER_SIZE[0], RENDER_SIZE[1], 30, True, 0)
        self.data.qpos[: min(len(snapshot_qpos), len(self.data.qpos))] = snapshot_qpos[: min(len(snapshot_qpos), len(self.data.qpos))]
        self.data.ctrl[: min(len(snapshot_ctrl), len(self.data.ctrl))] = snapshot_ctrl[: min(len(snapshot_ctrl), len(self.data.ctrl))]
        mujoco.mj_forward(self.model, self.data)  # type: ignore
        for body, pose in self.overlay_state["pose_overrides"].items():
            if body in self.editable_body_names:
                self.set_body_pose(body, pose["pos"], pose["quat"])

    def list_placement_bodies(self) -> dict[str, object]:
        items = []
        for body in sorted(self.editable_body_names):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)  # type: ignore
            pose = body_world_pose(self.data, body_id)
            items.append({
                "body": body,
                "label": self.body_labels.get(body, body),
                "matrix": flatten_matrix(pose),
                "support": self._build_support_infos().get(body),
            })
        return {"items": items, "scene_revision": self.scene_revision}

    def add_asset_to_scene(self, asset_id: str, motion: str, placement: dict[str, object]) -> dict[str, object]:
        if self.rebuilding_scene:
            raise HTTPException(status_code=409, detail="Scene rebuild already in progress")
        asset = resolve_asset_record(asset_id, self.overlay_state.get("assets_root"))
        instance_id = f"{asset_id}_{len(self.overlay_state['instances']) + 1:02d}"
        body = overlay_body_name(instance_id)

        def mutate(state):
            state["assets_root"] = state.get("assets_root") or str(resolve_assets_root(self.overlay_state.get("assets_root")))
            state["instances"].append({
                "id": instance_id,
                "asset_id": asset["asset_id"],
                "url": asset["url"],
                "motion": motion,
                "placement": placement,
            })

        revision = self._rebuild_overlay_runtime(mutate)
        return {"ok": True, "instance_id": instance_id, "body": body, "scene_revision": revision}

    def remove_asset_from_scene(self, instance_id: str) -> dict[str, object]:
        body = overlay_body_name(instance_id)

        def mutate(state):
            state["instances"] = [item for item in state["instances"] if item["id"] != instance_id]
            state["pose_overrides"].pop(body, None)

        revision = self._rebuild_overlay_runtime(mutate)
        return {"ok": True, "instance_id": instance_id, "scene_revision": revision}

    def _rebuild_overlay_runtime(self, mutate_overlay) -> str:
        previous_overlay = deepcopy(self.overlay_state)
        previous_revision = self.scene_revision
        self.rebuilding_scene = True
        try:
            next_overlay = deepcopy(previous_overlay)
            mutate_overlay(next_overlay)
            save_overlay_state(self.overlay_path, next_overlay)
            support_infos = self._build_support_infos()
            bounds_by_instance = {
                item["id"]: asset_local_bounds(resolve_assets_root(next_overlay.get("assets_root")) / item["url"])
                for item in next_overlay["instances"]
            }
            write_overlay_scene(
                self.composed_scene_usd,
                self.base_scene_usd,
                resolve_assets_root(next_overlay.get("assets_root")),
                next_overlay,
                support_infos=support_infos,
                asset_bounds_by_instance=bounds_by_instance,
            )
            bundle = ensure_render_bundle(self.robot_usd, self.composed_scene_usd)
            self._swap_runtime_bundle(bundle)
            self.overlay_state = next_overlay
            self.scene_revision = scene_revision(self.base_scene_usd, next_overlay)
            if not self.msgs.full():
                self.msgs.put_nowait({"scene_revision": self.scene_revision})
            return self.scene_revision
        except Exception:
            self.overlay_state = previous_overlay
            self.scene_revision = previous_revision
            save_overlay_state(self.overlay_path, previous_overlay)
            raise
        finally:
            self.rebuilding_scene = False
```

```python
# extend set_body_pose and call_once dispatch
        self.overlay_state["pose_overrides"][body] = {"pos": list(pos), "quat": normalized_quat.tolist()}
        save_overlay_state(self.overlay_path, self.overlay_state)
```

```python
        if method == "list_placement_bodies":
            return self.list_placement_bodies()
        if method == "add_asset_to_scene":
            return self.add_asset_to_scene(*args)
        if method == "remove_asset_from_scene":
            return self.remove_asset_from_scene(*args)
```

- [ ] **Step 4: Run the Zapdos import test to verify it passes**

Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_zapdos_import -v`

Expected: PASS with the new overlay RPC tests plus the existing bootstrap and pose tests.

- [ ] **Step 5: Commit the backend runtime changes**

```bash
git add apps/python/api/zapdos/{session}/{action}.py apps/python/tests/test_zapdos_import.py
git commit -m "feat: add zapdos scene overlay runtime"
```

### Task 4: Register Zapdos Agent Tools and Typed Tool API Calls

**Files:**
- Create: `apps/web/components/zapdos/zapdos-tool-api.ts`
- Create: `apps/web/components/zapdos/zapdos-tool-api.test.ts`
- Create: `apps/web/components/zapdos/zapdos-agent-instructions.ts`
- Create: `apps/web/components/zapdos/useZapdosAgentTools.ts`
- Modify: `apps/web/components/zapdos/ZapdosScene.tsx`

- [ ] **Step 1: Write the failing web API helper tests**

```ts
import assert from "node:assert/strict";
import test from "node:test";

type ZapdosToolApiModule = typeof import("./zapdos-tool-api");

test("createAddAssetToSceneRequest posts asset id motion and placement", async () => {
  const { createAddAssetToSceneRequest } = await loadModule<ZapdosToolApiModule>("./zapdos-tool-api.ts");

  assert.deepEqual(
    createAddAssetToSceneRequest({
      asset_id: "table_000",
      motion: "static",
      placement: { kind: "floor_at_xy", xy: [0, 0], z_offset: 0, yaw: 0 },
    }),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(["table_000", "static", { kind: "floor_at_xy", xy: [0, 0], z_offset: 0, yaw: 0 }]),
    }
  );
});

test("listPlacementBodies posts to the zapdos route", async () => {
  const { createSceneToolRequest, listPlacementBodies } = await loadModule<ZapdosToolApiModule>("./zapdos-tool-api.ts");
  const calls: Array<{ input: unknown; init: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: unknown, init?: unknown) => {
    calls.push({ input, init });
    return new Response(JSON.stringify({ items: [], scene_revision: "rev-1" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    const payload = await listPlacementBodies("sess-1");
    assert.equal(payload.scene_revision, "rev-1");
    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/call/list_placement_bodies");
    assert.deepEqual(calls[0]?.init, createSceneToolRequest([]));
  } finally {
    globalThis.fetch = originalFetch;
  }
});

async function loadModule<TModule>(specifier: string): Promise<TModule> {
  const loaded = await import(specifier);
  return (loaded.default ?? loaded["module.exports"] ?? loaded) as TModule;
}
```

- [ ] **Step 2: Run the web API helper test to verify it fails**

Run: `pnpm --filter glassbeaker-web test -- zapdos-tool-api`

Expected: FAIL with `Cannot find module './zapdos-tool-api.ts'`.

- [ ] **Step 3: Implement the tool fetch wrappers and Copilot registrations**

```ts
// apps/web/components/zapdos/zapdos-tool-api.ts
export type AddAssetToSceneInput = {
  asset_id: string;
  motion: "dynamic" | "static";
  placement: Record<string, unknown>;
};

export function createSceneToolRequest(args: unknown[]): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args),
  };
}

export function createAddAssetToSceneRequest(input: AddAssetToSceneInput): RequestInit {
  return createSceneToolRequest([input.asset_id, input.motion, input.placement]);
}

export async function listPlacementBodies(sess: string) {
  const response = await fetch(`/python/zapdos/${sess}/call/list_placement_bodies`, createSceneToolRequest([]));
  if (!response.ok) throw new Error(await response.text());
  return await response.json() as { items: unknown[]; scene_revision: string };
}

export async function addAssetToScene(sess: string, input: AddAssetToSceneInput) {
  const response = await fetch(`/python/zapdos/${sess}/call/add_asset_to_scene`, createAddAssetToSceneRequest(input));
  if (!response.ok) throw new Error(await response.text());
  return await response.json() as { body: string; instance_id: string; scene_revision: string };
}

export async function removeAssetFromScene(sess: string, instanceId: string) {
  const response = await fetch(`/python/zapdos/${sess}/call/remove_asset_from_scene`, createSceneToolRequest([instanceId]));
  if (!response.ok) throw new Error(await response.text());
  return await response.json() as { instance_id: string; scene_revision: string };
}
```

```ts
// apps/web/components/zapdos/zapdos-agent-instructions.ts
export const ZAPDOS_ADDITIONAL_INSTRUCTIONS = [
  "You are editing the current Zapdos simulation scene.",
  "Use search_assets to find candidate asset ids before inserting new assets.",
  "Use list_placement_bodies before on_top_of_body placement so you know the support body name.",
  "Use add_asset_to_scene for session-local insertion only.",
  "Use remove_asset_from_scene when the user wants an inserted overlay asset gone.",
  "Prefer motion=static for furniture and supports; prefer motion=dynamic for manipulable objects.",
].join("\\n");
```

```ts
// apps/web/components/zapdos/useZapdosAgentTools.ts
import { useCopilotAdditionalInstructions, useFrontendTool } from "@copilotkit/react-core";

import { SEARCH_ASSETS_DESCRIPTION, SEARCH_ASSETS_PARAMETERS_CPK } from "../genie-sim";
import { postToolJson } from "../genie-sim/tool-client";
import { addAssetToScene, listPlacementBodies, removeAssetFromScene } from "./zapdos-tool-api";
import { ZAPDOS_ADDITIONAL_INSTRUCTIONS } from "./zapdos-agent-instructions";

export function useZapdosAgentTools(sess: string) {
  useCopilotAdditionalInstructions({ instructions: ZAPDOS_ADDITIONAL_INSTRUCTIONS }, [sess]);

  useFrontendTool({
    name: "search_assets",
    description: SEARCH_ASSETS_DESCRIPTION,
    followUp: true,
    parameters: SEARCH_ASSETS_PARAMETERS_CPK,
    handler: async (args) => postToolJson(
      fetch,
      "/python/genie_sim/search_assets",
      { query: args.query, top_k: typeof args.top_k === "number" ? args.top_k : 8 },
      "Asset search failed"
    ),
  }, []);

  useFrontendTool({
    name: "list_placement_bodies",
    description: "List editable scene bodies with support metadata for placement.",
    followUp: true,
    parameters: [] as never[],
    handler: async () => await listPlacementBodies(sess),
  }, [sess]);

  useFrontendTool({
    name: "add_asset_to_scene",
    description: "Insert a session-local asset into the active Zapdos scene and rebuild the runtime.",
    followUp: true,
    parameters: [
      { name: "asset_id", type: "string", required: true, description: "Exact asset id from search_assets." },
      { name: "motion", type: "string", required: true, description: "Use static or dynamic." },
      { name: "placement", type: "object", required: true, description: "Placement payload using floor_at_xy, on_top_of_body, or world_pose." },
    ] as never[],
    handler: async (args) => await addAssetToScene(sess, args as never),
  }, [sess]);

  useFrontendTool({
    name: "remove_asset_from_scene",
    description: "Remove a session-local overlay asset by instance id.",
    followUp: true,
    parameters: [
      { name: "instance_id", type: "string", required: true, description: "Overlay instance id to remove." },
    ] as never[],
    handler: async (args) => await removeAssetFromScene(sess, String(args.instance_id)),
  }, [sess]);
}
```

```tsx
// in ZapdosScene.tsx
import { useZapdosAgentTools } from "./useZapdosAgentTools";

export function ZapdosScene({ sess, onRuntimeError }: { sess: string; onRuntimeError: (message: string) => void }) {
  useZapdosAgentTools(sess);
  // remove useGeineSimAssets()
}
```

- [ ] **Step 4: Run the web API helper test to verify it passes**

Run: `pnpm --filter glassbeaker-web test -- zapdos-tool-api`

Expected: PASS with 2 tests.

- [ ] **Step 5: Commit the frontend tool integration**

```bash
git add apps/web/components/zapdos/zapdos-tool-api.ts apps/web/components/zapdos/zapdos-tool-api.test.ts apps/web/components/zapdos/zapdos-agent-instructions.ts apps/web/components/zapdos/useZapdosAgentTools.ts apps/web/components/zapdos/ZapdosScene.tsx
git commit -m "feat: add zapdos agent scene tools"
```

### Task 5: Reload Scene Visuals on `scene_revision` and Clear Stale Selection

**Files:**
- Modify: `apps/web/components/zapdos/zapdos-runtime.ts`
- Modify: `apps/web/components/zapdos/zapdos-runtime.test.ts`
- Modify: `apps/web/components/zapdos/zapdos-scene-state.ts`
- Modify: `apps/web/components/zapdos/zapdos-scene-state.test.ts`
- Modify: `apps/web/components/zapdos/ZapdosScene.tsx`

- [ ] **Step 1: Add failing revision and selection tests**

```ts
import assert from "node:assert/strict";
import test from "node:test";

import { getZapdosSceneRevision } from "./zapdos-runtime";
import { clearMissingSelection, shouldReloadSceneRevision } from "./zapdos-scene-state";

test("getZapdosSceneRevision returns the revision string when present", () => {
  assert.equal(getZapdosSceneRevision({ scene_revision: "rev-2" }), "rev-2");
  assert.equal(getZapdosSceneRevision({ pose: {} }), null);
});

test("shouldReloadSceneRevision ignores duplicate revisions", () => {
  assert.equal(shouldReloadSceneRevision("rev-1", "rev-1"), false);
  assert.equal(shouldReloadSceneRevision("rev-1", "rev-2"), true);
});

test("clearMissingSelection drops a body that is no longer present", () => {
  assert.equal(clearMissingSelection("Scene_table_000_01", new Set(["Scene_crate_000_01"])), null);
  assert.equal(clearMissingSelection("Scene_table_000_01", new Set(["Scene_table_000_01"])), "Scene_table_000_01");
});
```

- [ ] **Step 2: Run the revision helper tests to verify they fail**

Run: `pnpm --filter glassbeaker-web test -- zapdos-runtime zapdos-scene-state`

Expected: FAIL with missing `getZapdosSceneRevision`, `shouldReloadSceneRevision`, or `clearMissingSelection`.

- [ ] **Step 3: Implement revision helpers and scene reload flow**

```ts
// apps/web/components/zapdos/zapdos-runtime.ts
export function getZapdosSceneRevision(payload: unknown) {
  if (!payload || typeof payload !== "object") {
    return null;
  }
  const revision = (payload as { scene_revision?: unknown }).scene_revision;
  return typeof revision === "string" && revision.trim() ? revision : null;
}
```

```ts
// apps/web/components/zapdos/zapdos-scene-state.ts
export function shouldReloadSceneRevision(current: string | null, next: string | null) {
  return !!next && next !== current;
}

export function clearMissingSelection(selectedBody: string | null, nextBodies: Set<string>) {
  if (!selectedBody) return null;
  return nextBodies.has(selectedBody) ? selectedBody : null;
}
```

```tsx
// inside SceneRuntime in ZapdosScene.tsx
import { clearMissingSelection, shouldReloadSceneRevision } from "./zapdos-scene-state";
import { getZapdosSceneRevision, isZapdosInactivePayload } from "./zapdos-runtime";

const sceneRevisionRef = useRef<string | null>(null);

const clearLoadedVisuals = () => {
  bodyObjectsRef.current = {};
  for (const object of topLevel) root.remove(object);
  topLevel.length = 0;
};

const loadVisuals = async () => {
  clearLoadedVisuals();
  const payload = await getSceneVisual(sess);
  for (const body of payload.bodies) {
    const group = new Object3D();
    group.name = body.name;
    group.userData.zapdosBody = body.name;
    group.userData.zapdosEditable = body.editable;
    applyObjectMatrix(group, body.matrix);
    bodyObjectsRef.current[body.name] = group;
    topLevel.push(group);
    root.add(group);
  }
  for (const item of payload.meshes) {
    const geometry = await loadSceneGeometry(item);
    const image = await loadSceneTexture(item.texture);
    const mesh = new Mesh(geometry, getSceneMaterial(item, image));
    mesh.name = item.name;
    mesh.castShadow = !item.name.endsWith(".plane");
    mesh.receiveShadow = true;
    if (item.body) {
      mesh.userData.zapdosBody = item.body;
      mesh.userData.zapdosEditable = bodyObjectsRef.current[item.body]?.userData.zapdosEditable === true;
      applyObjectMatrix(mesh, item.localMatrix as number[]);
      bodyObjectsRef.current[item.body]?.add(mesh);
    } else if (item.matrix) {
      applyObjectMatrix(mesh, item.matrix);
      topLevel.push(mesh);
      root.add(mesh);
    }
  }
  const nextBodies = new Set(Object.keys(bodyObjectsRef.current));
  setSelectedBody(current => clearMissingSelection(current, nextBodies));
};

sse.onmessage = event => {
  const payload = JSON.parse(event.data) as { inactive?: boolean; pose?: Record<string, number[]>; scene_revision?: string };
  if (isZapdosInactivePayload(payload)) return fail(ZAPDOS_RUNTIME_DISCONNECTED_MESSAGE);
  const nextRevision = getZapdosSceneRevision(payload);
  if (shouldReloadSceneRevision(sceneRevisionRef.current, nextRevision)) {
    sceneRevisionRef.current = nextRevision;
    void loadVisuals().catch(fail);
    return;
  }
  if (!payload.pose) return;
  for (const [name, matrix] of Object.entries(payload.pose)) {
    if (!shouldApplyBodyPose(name, draggingBodyRef.current)) continue;
    const object = bodyObjectsRef.current[name];
    if (object) applyObjectMatrix(object, matrix);
  }
};
```

- [ ] **Step 4: Run the web revision helper tests to verify they pass**

Run:

```bash
pnpm --filter glassbeaker-web test -- zapdos-runtime zapdos-scene-state zapdos-tool-api zapdos-scene-api
pnpm --filter glassbeaker-web exec tsc --noEmit
```

Expected: PASS for the focused tests and no TypeScript errors.

- [ ] **Step 5: Commit the scene revision reload changes**

```bash
git add apps/web/components/zapdos/zapdos-runtime.ts apps/web/components/zapdos/zapdos-runtime.test.ts apps/web/components/zapdos/zapdos-scene-state.ts apps/web/components/zapdos/zapdos-scene-state.test.ts apps/web/components/zapdos/ZapdosScene.tsx
git commit -m "feat: reload zapdos visuals on scene revision"
```

### Task 6: Run Focused Verification and Final Integration Checks

**Files:**
- Modify: none

- [ ] **Step 1: Run the focused Python suite**

Run:

```powershell
uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_zapdos_overlay apps.python.tests.test_zapdos_overlay_scene apps.python.tests.test_zapdos_import -v
```

Expected: PASS for all new overlay, scene writer, and runtime tests.

- [ ] **Step 2: Run the focused web suite**

Run:

```bash
pnpm --filter glassbeaker-web test -- zapdos-tool-api zapdos-runtime zapdos-scene-state zapdos-scene-api
pnpm --filter glassbeaker-web exec tsc --noEmit
```

Expected: PASS for the focused tests and no TypeScript diagnostics.

- [ ] **Step 3: Perform manual verification in the browser**

Run:

```powershell
uv run --project apps/python --python 3.12 python apps/python/app.py
```

Manual checks:

```text
1. Open /demo/zapdos with a valid scene_usd.
2. Ask the agent to search for "table" and call add_asset_to_scene with motion=static and placement.kind=floor_at_xy.
3. Confirm the 3D view reloads once, the new table appears, and the MJPEG camera view matches.
4. Drag the inserted table with the existing transform controls and confirm pose sync still works.
5. Ask the agent to insert a dynamic asset on top of the new table with placement.kind=on_top_of_body.
6. Confirm the object appears above the table, collides, and remains selectable.
7. Remove the inserted dynamic asset and confirm the selection clears if that asset had been selected.
8. Insert another asset after moving an existing editable body and confirm the moved body's pose survives the rebuild.
```

- [ ] **Step 4: Commit the integrated overlay feature**

```bash
git add apps/python/utils/zapdos_overlay.py apps/python/utils/zapdos_asset_library.py apps/python/utils/zapdos_overlay_scene.py apps/python/tests/test_zapdos_overlay.py apps/python/tests/test_zapdos_overlay_scene.py apps/python/tests/test_zapdos_import.py apps/python/api/zapdos/{session}/{action}.py apps/web/components/zapdos/zapdos-tool-api.ts apps/web/components/zapdos/zapdos-tool-api.test.ts apps/web/components/zapdos/zapdos-agent-instructions.ts apps/web/components/zapdos/useZapdosAgentTools.ts apps/web/components/zapdos/zapdos-runtime.ts apps/web/components/zapdos/zapdos-runtime.test.ts apps/web/components/zapdos/zapdos-scene-state.ts apps/web/components/zapdos/zapdos-scene-state.test.ts apps/web/components/zapdos/ZapdosScene.tsx
git commit -m "feat: add zapdos session scene overlay tools"
```

## Self-Review

- Spec coverage:
  - Overlay JSON plus revision hashing: Task 1
  - Asset lookup, bounds, placement modes, composed USDA: Task 2
  - Session-local add/remove tools, rebuild rollback, pose override persistence, SSE revision events: Task 3
  - Agent-callable frontend tools and instructions: Task 4
  - Frontend visual reload and stale selection cleanup: Task 5
  - Focused tests and manual verification across MuJoCo, WebGL, and MJPEG: Task 6
- Placeholder scan:
  - No `TBD`, `TODO`, or "similar to above" instructions remain.
  - Every code step includes concrete function names or code blocks.
- Type consistency:
  - Python uses `scene_revision`, `bundle_revision`, `overlay_body_name`, `resolve_asset_record`, `asset_local_bounds`, and `write_overlay_scene` consistently.
  - Web uses `listPlacementBodies`, `addAssetToScene`, `removeAssetFromScene`, `getZapdosSceneRevision`, `shouldReloadSceneRevision`, and `clearMissingSelection` consistently.
