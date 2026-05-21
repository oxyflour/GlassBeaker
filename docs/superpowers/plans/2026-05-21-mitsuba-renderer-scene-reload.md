# Mitsuba Renderer Scene Reload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Zapdos scene changes reach an immediately usable Mitsuba render state without doing Mitsuba scene build, mesh export, and `mi.load_dict()` inside the runtime swap critical section.

**Architecture:** Keep the existing `set_scene_assets` / `remove_asset_from_scene` API and overlay rebuild job model. Add an optional renderer prewarm phase after `ensure_render_bundle()` and before `reserve_world()`, so Mitsuba can build/load the next scene and first frames while the old runtime remains active. At swap time, `MitsubaRenderer.reload_scene()` adopts the prepared state in a short locked section; non-Mitsuba renderers continue using the existing synchronous reload path.

**Tech Stack:** Python 3.12, `uv` environment in `apps/python`, `unittest`, Pixar USD (`pxr`), Mitsuba 3, NumPy.

---

## Current Startup Chain

Scene asset changes currently flow through these files:

- `apps/python/utils/zapdos/editor/zapdos_editor.py`: `set_scene_assets()` / `remove_asset_from_scene()` start an overlay rebuild job; `_swap_runtime_bundle()` creates new physics and calls `old_renderer.reload_scene(bundle)`.
- `apps/python/utils/zapdos/editor/rebuild_manager.py`: `run_overlay_rebuild()` prepares the bundle in a background thread, then enters `reserve_world()` and applies the prepared rebuild on the owner loop.
- `apps/python/utils/zapdos/renderer/zapdos_renderer.py`: `reload_scene()` delegates to the backend and resets camera dedupe state.
- `apps/python/utils/zapdos/renderer/mitsuba_renderer.py`: `reload_scene()` clears frames, marks the renderer not ready, then synchronously calls `_load_scene()`.
- `apps/python/utils/zapdos/renderer/mitsuba_scene.py`: `build_mitsuba_scene_dict()` opens the render USD, traverses every mesh, triangulates geometry, writes PLY files, builds sensors, and returns the Mitsuba scene dictionary.

The slow part is currently inside `MitsubaRenderer.reload_scene()`:

1. `reload_scene()` clears old frames and readiness.
2. `_load_scene()` deletes `apps/python/tmp/mitsuba_<session>/meshes`.
3. `build_mitsuba_scene_dict()` traverses USD meshes and rewrites every PLY.
4. `load_mitsuba()` imports Mitsuba on first use and selects `cuda_ad_rgb`.
5. `_load_pose_scene()` applies body transforms and calls `mi.load_dict()`.
6. The render thread later renders frames camera-by-camera.

For scene changes, this runs while the editor is applying the runtime swap. That makes the owner-loop/world reservation window much longer than needed and also leaves the new renderer in `Waiting` until the first post-swap render finishes.

## File Structure

- Modify `apps/python/utils/zapdos/editor/rebuild_types.py`
  - Add a `renderer_reload` field to `PreparedOverlayRebuild` so the background preparation phase can carry optional backend-specific prepared state into the swap phase.
- Modify `apps/python/utils/zapdos/editor/rebuild_manager.py`
  - After `_prepare_overlay_rebuild()` returns and before `reserve_world()`, call an optional `prepare_scene_reload(bundle)` on the session renderer.
  - Store the returned prepared state on `PreparedOverlayRebuild`.
  - Emit progress stages for timing and diagnostics.
- Modify `apps/python/utils/zapdos/editor/zapdos_editor.py`
  - Pass `prepared.renderer_reload` into `_swap_runtime_bundle()`.
  - Pass it through to `old_renderer.reload_scene()`.
- Modify `apps/python/utils/zapdos/renderer/base.py`
  - Extend the renderer backend protocol with optional prepared reload support while preserving existing call compatibility.
- Modify `apps/python/utils/zapdos/renderer/zapdos_renderer.py`
  - Add `prepare_scene_reload()`.
  - Allow `reload_scene()` to pass a non-`None` prepared state to backends that support it.
- Create `apps/python/utils/zapdos/renderer/mitsuba_pose.py`
  - Move the pure pose-to-scene transformation out of `mitsuba_renderer.py`.
  - This is justified as pure codec/validation logic shared by normal rendering and preload.
- Create `apps/python/utils/zapdos/renderer/mitsuba_scene_state.py`
  - Define `MitsubaPreparedScene`.
  - Build a loaded Mitsuba scene and optional first frames for a target bundle using the current pose.
  - This isolates heavy side effects: USD traversal, PLY writing, `mi.load_dict()`, and first-frame render.
- Modify `apps/python/utils/zapdos/renderer/mitsuba_renderer.py`
  - Add `prepare_scene_reload(bundle)`.
  - Add prepared-state adoption to `reload_scene(bundle, prepared_scene=...)`.
  - Add a scene lock around `_scene` mutation and `mi.render()` to avoid preparing a new scene while the render loop is reading the old one.
- Modify `apps/python/utils/zapdos/renderer/mitsuba_scene.py`
  - Stop deleting all Mitsuba mesh files for every reload.
  - Write deterministic mesh files only when content changes.
- Modify tests:
  - `apps/python/tests/test_mitsuba_renderer.py`
  - `apps/python/tests/test_mitsuba_scene_builder.py`
  - `apps/python/tests/test_zapdos_renderer_entity.py`
  - `apps/python/tests/test_zapdos_import.py`

---

### Task 1: Add Prepared Reload Plumbing Types

**Files:**
- Modify: `apps/python/utils/zapdos/editor/rebuild_types.py`
- Test: `apps/python/tests/test_zapdos_import.py`

- [ ] **Step 1: Write the failing dataclass compatibility test**

Add this test near the existing rebuild tests in `apps/python/tests/test_zapdos_import.py`:

```python
    def test_prepared_overlay_rebuild_defaults_renderer_reload_to_none(self):
        prepared = PreparedOverlayRebuild(
            bundle=SimpleNamespace(),
            next_overlay=default_overlay_state("C:/assets"),
            previous_overlay=default_overlay_state("C:/assets"),
            previous_revision="rev-1",
            next_revision="rev-2",
        )

        self.assertIsNone(prepared.renderer_reload)
```

- [ ] **Step 2: Run test to verify it fails**

Run from `apps/python`:

```powershell
uv run python -m unittest tests.test_zapdos_import.ZapdosImportTest.test_prepared_overlay_rebuild_defaults_renderer_reload_to_none -v
```

Expected: FAIL with `AttributeError: 'PreparedOverlayRebuild' object has no attribute 'renderer_reload'`.

- [ ] **Step 3: Add the dataclass field**

Change `apps/python/utils/zapdos/editor/rebuild_types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from utils.zapdos.bundle import RenderBundle


@dataclass(frozen=True)
class PreparedOverlayRebuild:
    bundle: RenderBundle
    next_overlay: dict[str, object]
    previous_overlay: dict[str, object]
    previous_revision: str
    next_revision: str
    renderer_reload: Any | None = None


@dataclass(frozen=True)
class OverlayRebuildCompletion:
    op_id: str
    prepared: PreparedOverlayRebuild | None = None
    error: Exception | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run from `apps/python`:

```powershell
uv run python -m unittest tests.test_zapdos_import.ZapdosImportTest.test_prepared_overlay_rebuild_defaults_renderer_reload_to_none -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add apps/python/utils/zapdos/editor/rebuild_types.py apps/python/tests/test_zapdos_import.py
git commit -m "test: cover prepared renderer reload payload"
```

---

### Task 2: Add Optional Renderer Prewarm to Rebuild Manager

**Files:**
- Modify: `apps/python/utils/zapdos/editor/rebuild_manager.py`
- Test: `apps/python/tests/test_zapdos_import.py`

- [ ] **Step 1: Write the failing prewarm ordering test**

Add this test near `test_run_overlay_rebuild_resolves_scene_rebuild_job_future_without_manual_drain`:

```python
    async def test_run_overlay_rebuild_prepares_renderer_before_world_reservation(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.sess = "sess-1"
        session.bundle = SimpleNamespace(cameras=[])
        editor = self.attach_editor(session)
        session.editor = editor
        editor.rebuilding_scene = True
        events: list[str] = []
        world_held = False

        @asynccontextmanager
        async def reserve_world():
            nonlocal world_held
            events.append("reserve.enter")
            world_held = True
            try:
                yield object()
            finally:
                world_held = False
                events.append("reserve.exit")

        async def run_sync(fn, world_token=None):
            return fn(session)

        class Renderer:
            def prepare_scene_reload(self, bundle):
                events.append(f"prepare_renderer.world_held={world_held}")
                self.bundle = bundle
                return {"prepared": bundle}

        session.renderer = Renderer()
        session.reserve_world = reserve_world
        session.run_sync = run_sync
        prepared = PreparedOverlayRebuild(
            bundle=SimpleNamespace(cameras=[]),
            next_overlay=default_overlay_state("C:/assets"),
            previous_overlay=default_overlay_state("C:/assets"),
            previous_revision="rev-1",
            next_revision="rev-2",
        )
        EDITOR_REBUILD_EVENTS.create_scene_rebuild_job(editor, "op-1", {"ok": True})

        with mock.patch.object(editor, "_build_support_infos", return_value={}):
            with mock.patch.object(editor, "_prepare_overlay_rebuild", return_value=prepared):
                with mock.patch.object(
                    editor,
                    "_apply_prepared_overlay_rebuild",
                    side_effect=lambda current_prepared, current_op_id: events.append(
                        f"apply.renderer_reload={current_prepared.renderer_reload!r}"
                    ) or setattr(editor, "rebuilding_scene", False) or "rev-2",
                ):
                    await EDITOR_REBUILD_MANAGER.run_overlay_rebuild(
                        editor,
                        "op-1",
                        default_overlay_state("C:/assets"),
                        default_overlay_state("C:/assets"),
                        "rev-1",
                    )

        self.assertEqual(events, [
            "prepare_renderer.world_held=False",
            "reserve.enter",
            "apply.renderer_reload={'prepared': namespace(cameras=[])}",
            "reserve.exit",
        ])
```

- [ ] **Step 2: Run test to verify it fails**

Run from `apps/python`:

```powershell
uv run python -m unittest tests.test_zapdos_import.ZapdosImportTest.test_run_overlay_rebuild_prepares_renderer_before_world_reservation -v
```

Expected: FAIL because `prepare_scene_reload()` is never called and `renderer_reload` remains `None`.

- [ ] **Step 3: Add prewarm inside `run_overlay_rebuild()`**

In `apps/python/utils/zapdos/editor/rebuild_manager.py`, update imports:

```python
from dataclasses import replace
```

Then add this helper near `_run_overlay_rebuild_inline()`:

```python
async def _prepare_renderer_reload(session: Any, op_id: str, bundle: RenderBundle) -> Any | None:
    owner_session = getattr(session, "session", session)
    renderer = getattr(owner_session, "renderer", None)
    prepare = getattr(renderer, "prepare_scene_reload", None)
    if not callable(prepare):
        return None
    emit_scene_rebuild_progress(session, op_id, "prepare_renderer_reload.started")
    prepared = await asyncio.to_thread(prepare, bundle)
    emit_scene_rebuild_progress(session, op_id, "prepare_renderer_reload.done")
    return prepared
```

In `run_overlay_rebuild()`, after `prepared = await asyncio.to_thread(...)` and after emitting `prepare_overlay_rebuild.done`, add:

```python
        renderer_reload = await _prepare_renderer_reload(session, op_id, prepared.bundle)
        if renderer_reload is not None:
            prepared = replace(prepared, renderer_reload=renderer_reload)
```

- [ ] **Step 4: Run targeted tests**

Run from `apps/python`:

```powershell
uv run python -m unittest tests.test_zapdos_import.ZapdosImportTest.test_run_overlay_rebuild_prepares_renderer_before_world_reservation tests.test_zapdos_import.ZapdosImportTest.test_run_overlay_rebuild_resolves_scene_rebuild_job_future_without_manual_drain -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add apps/python/utils/zapdos/editor/rebuild_manager.py apps/python/tests/test_zapdos_import.py
git commit -m "feat: prewarm renderer reload before runtime swap"
```

---

### Task 3: Pass Prepared Renderer State Through Swap

**Files:**
- Modify: `apps/python/utils/zapdos/editor/zapdos_editor.py`
- Test: `apps/python/tests/test_zapdos_import.py`

- [ ] **Step 1: Write the failing swap test**

Add this test near the existing `_swap_runtime_bundle` tests:

```python
    def test_apply_prepared_overlay_rebuild_passes_renderer_reload_to_swap(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.sess = "sess-1"
        session.bundle = SimpleNamespace(cameras=[])
        session.msgs = queue.Queue(maxsize=4)
        editor = self.attach_editor(session)
        session.editor = editor
        editor.rebuilding_scene = True
        prepared = PreparedOverlayRebuild(
            bundle=SimpleNamespace(cameras=[]),
            next_overlay=default_overlay_state("C:/assets"),
            previous_overlay=default_overlay_state("C:/assets"),
            previous_revision="rev-1",
            next_revision="rev-2",
            renderer_reload={"state": "prepared"},
        )

        with mock.patch.object(editor, "_swap_runtime_bundle") as swap:
            revision = EDITOR_REBUILD_MANAGER.apply_prepared_overlay_rebuild(editor, prepared, "op-1")

        self.assertEqual(revision, "rev-2")
        swap.assert_called_once_with(prepared.bundle, prepared.next_overlay, "op-1", {"state": "prepared"})
```

- [ ] **Step 2: Run test to verify it fails**

Run from `apps/python`:

```powershell
uv run python -m unittest tests.test_zapdos_import.ZapdosImportTest.test_apply_prepared_overlay_rebuild_passes_renderer_reload_to_swap -v
```

Expected: FAIL because `_swap_runtime_bundle()` is called without the prepared renderer state.

- [ ] **Step 3: Update apply/swap signatures**

In `apps/python/utils/zapdos/editor/rebuild_manager.py`, change:

```python
        session._swap_runtime_bundle(prepared.bundle, prepared.next_overlay, op_id)
```

to:

```python
        session._swap_runtime_bundle(prepared.bundle, prepared.next_overlay, op_id, prepared.renderer_reload)
```

In `apps/python/utils/zapdos/editor/zapdos_editor.py`, change the method signature:

```python
    def _swap_runtime_bundle(self, bundle, overlay_state, op_id: str | None = None, renderer_reload=None) -> None:
```

Then change the reload call:

```python
                old_renderer.reload_scene(bundle, prepared_scene=renderer_reload)
```

- [ ] **Step 4: Preserve compatibility for current renderer wrappers**

The next task updates `ZapdosRenderer.reload_scene()` to accept `prepared_scene`. Until then, this test will fail if `_swap_runtime_bundle()` uses a bare mock without the keyword. Keep all existing `_swap_runtime_bundle` tests passing by updating their expected mock calls to:

```python
old_renderer.reload_scene.assert_called_once_with(bundle, prepared_scene=None)
```

Tests that use `ZapdosRenderer.reload_scene()` directly should not be changed in this task.

- [ ] **Step 5: Run targeted tests**

Run from `apps/python`:

```powershell
uv run python -m unittest tests.test_zapdos_import -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add apps/python/utils/zapdos/editor/rebuild_manager.py apps/python/utils/zapdos/editor/zapdos_editor.py apps/python/tests/test_zapdos_import.py
git commit -m "feat: pass prepared renderer state during swap"
```

---

### Task 4: Teach ZapdosRenderer About Prepared Reloads

**Files:**
- Modify: `apps/python/utils/zapdos/renderer/base.py`
- Modify: `apps/python/utils/zapdos/renderer/zapdos_renderer.py`
- Test: `apps/python/tests/test_zapdos_renderer_entity.py`

- [ ] **Step 1: Write failing wrapper tests**

Add these tests to `apps/python/tests/test_zapdos_renderer_entity.py`:

```python
    def test_prepare_scene_reload_delegates_when_backend_supports_it(self):
        bundle = SimpleNamespace(cameras=[_camera("head_camera")])
        backend = SimpleNamespace(
            ready=True,
            wait_ready=mock.AsyncMock(return_value={"ready": True}),
            read=mock.Mock(return_value=None),
            reload_scene=mock.Mock(),
            prepare_scene_reload=mock.Mock(return_value={"prepared": True}),
            snapshot_cameras=mock.Mock(return_value=[]),
            close=mock.Mock(),
        )
        renderer = self.make_renderer(backend=backend, bundle=bundle)

        result = renderer.prepare_scene_reload(bundle)

        self.assertEqual(result, {"prepared": True})
        backend.prepare_scene_reload.assert_called_once_with(bundle)

    def test_prepare_scene_reload_returns_none_when_backend_does_not_support_it(self):
        bundle = SimpleNamespace(cameras=[_camera("head_camera")])
        backend = SimpleNamespace(
            ready=True,
            wait_ready=mock.AsyncMock(return_value={"ready": True}),
            read=mock.Mock(return_value=None),
            reload_scene=mock.Mock(),
            snapshot_cameras=mock.Mock(return_value=[]),
            close=mock.Mock(),
        )
        renderer = self.make_renderer(backend=backend, bundle=bundle)

        self.assertIsNone(renderer.prepare_scene_reload(bundle))

    def test_reload_scene_passes_prepared_state_when_present(self):
        first_bundle = SimpleNamespace(cameras=[_camera("head_camera")])
        second_bundle = SimpleNamespace(cameras=[_camera("wrist_camera")])
        backend = SimpleNamespace(
            ready=True,
            wait_ready=mock.AsyncMock(return_value={"ready": True}),
            read=mock.Mock(return_value=None),
            reload_scene=mock.Mock(),
            snapshot_cameras=mock.Mock(return_value=[]),
            close=mock.Mock(),
        )
        renderer = self.make_renderer(backend=backend, bundle=first_bundle)

        renderer.reload_scene(second_bundle, timeout=12.0, prepared_scene={"prepared": True})

        backend.reload_scene.assert_called_once_with(
            second_bundle,
            timeout=12.0,
            prepared_scene={"prepared": True},
        )
        self.assertIs(renderer.bundle, second_bundle)
        self.assertEqual(renderer.last_frame_index, {"wrist_camera": -1})
```

- [ ] **Step 2: Run tests to verify they fail**

Run from `apps/python`:

```powershell
uv run python -m unittest tests.test_zapdos_renderer_entity -v
```

Expected: FAIL because `ZapdosRenderer.prepare_scene_reload()` does not exist and `reload_scene()` does not accept `prepared_scene`.

- [ ] **Step 3: Update renderer protocol**

In `apps/python/utils/zapdos/renderer/base.py`, import `Any` is already present. Change `reload_scene()` to:

```python
    def reload_scene(
        self,
        bundle: "RenderBundle",
        timeout: float = 30.0,
        prepared_scene: Any | None = None,
    ) -> None: ...
```

Add:

```python
    def prepare_scene_reload(
        self,
        bundle: "RenderBundle",
        timeout: float = 30.0,
    ) -> Any | None: ...
```

- [ ] **Step 4: Update ZapdosRenderer**

In `apps/python/utils/zapdos/renderer/zapdos_renderer.py`, add:

```python
    def prepare_scene_reload(self, bundle: "RenderBundle", timeout: float = 30.0):
        prepare = getattr(self.backend, "prepare_scene_reload", None)
        if not callable(prepare):
            return None
        if timeout == 30.0:
            return prepare(bundle)
        return prepare(bundle, timeout=timeout)
```

Change `reload_scene()` to:

```python
    def reload_scene(
        self,
        bundle: "RenderBundle",
        timeout: float = 30.0,
        prepared_scene=None,
    ) -> None:
        if prepared_scene is None:
            if timeout == 30.0:
                self.backend.reload_scene(bundle)
            else:
                self.backend.reload_scene(bundle, timeout=timeout)
        else:
            self.backend.reload_scene(bundle, timeout=timeout, prepared_scene=prepared_scene)
        self.set_bundle(bundle)
```

- [ ] **Step 5: Run wrapper tests**

Run from `apps/python`:

```powershell
uv run python -m unittest tests.test_zapdos_renderer_entity -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add apps/python/utils/zapdos/renderer/base.py apps/python/utils/zapdos/renderer/zapdos_renderer.py apps/python/tests/test_zapdos_renderer_entity.py
git commit -m "feat: add renderer prepared reload hook"
```

---

### Task 5: Extract Mitsuba Pose Application Into a Pure Module

**Files:**
- Create: `apps/python/utils/zapdos/renderer/mitsuba_pose.py`
- Modify: `apps/python/utils/zapdos/renderer/mitsuba_renderer.py`
- Modify: `apps/python/tests/test_mitsuba_renderer.py`

- [ ] **Step 1: Write failing pure pose tests**

Move the two existing pose tests from `MitsubaRendererTest` into direct function tests in `apps/python/tests/test_mitsuba_renderer.py` by adding this import:

```python
from utils.zapdos.renderer.mitsuba_pose import scene_for_pose
```

Replace `renderer._scene_for_pose(scene, pose)` with:

```python
posed = scene_for_pose(scene, pose)
```

in:

```python
    def test_scene_for_pose_updates_body_local_mesh_transform(self):
```

and:

```python
    def test_scene_for_pose_updates_body_attached_camera(self):
```

- [ ] **Step 2: Run tests to verify they fail**

Run from `apps/python`:

```powershell
uv run python -m unittest tests.test_mitsuba_renderer.MitsubaRendererTest.test_scene_for_pose_updates_body_local_mesh_transform tests.test_mitsuba_renderer.MitsubaRendererTest.test_scene_for_pose_updates_body_attached_camera -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'utils.zapdos.renderer.mitsuba_pose'`.

- [ ] **Step 3: Create `mitsuba_pose.py`**

Add `apps/python/utils/zapdos/renderer/mitsuba_pose.py`:

```python
from __future__ import annotations

from typing import Any

import numpy as np


def scene_for_pose(scene_dict: dict[str, Any], pose: dict[str, list[float]]) -> dict[str, Any]:
    scene = dict(scene_dict)
    for key, value in scene_dict.items():
        if not isinstance(value, dict):
            continue
        body = value.get("_zapdos_body")
        local_matrix = value.get("_zapdos_body_local_matrix")
        body_matrix = pose.get(body) if isinstance(body, str) else None
        if body_matrix is None:
            continue
        entry = dict(value)
        body_transform = np.asarray(body_matrix, dtype=float).reshape(4, 4)
        if local_matrix is not None:
            entry["to_world_matrix"] = (
                body_transform
                @ np.asarray(local_matrix, dtype=float).reshape(4, 4)
            ).tolist()
        update_camera_pose(entry, body_transform)
        scene[key] = entry
    return scene


def update_camera_pose(entry: dict[str, Any], body_transform: np.ndarray) -> None:
    origin = entry.get("_zapdos_camera_local_origin")
    target = entry.get("_zapdos_camera_local_target")
    up = entry.get("_zapdos_camera_local_up")
    if origin is None or target is None or up is None:
        return
    world_origin = body_transform @ np.asarray(origin, dtype=float).reshape(4)
    world_target = body_transform @ np.asarray(target, dtype=float).reshape(4)
    world_up = body_transform[:3, :3] @ np.asarray(up, dtype=float).reshape(3)
    entry["to_world_look_at"] = {
        "origin": world_origin[:3].tolist(),
        "target": world_target[:3].tolist(),
        "up": world_up.tolist(),
    }
```

- [ ] **Step 4: Update MitsubaRenderer to use the pure function**

In `apps/python/utils/zapdos/renderer/mitsuba_renderer.py`, add:

```python
from utils.zapdos.renderer.mitsuba_pose import scene_for_pose
```

Change:

```python
        posed_scene = self._scene_for_pose(scene_dict, pose)
```

to:

```python
        posed_scene = scene_for_pose(scene_dict, pose)
```

Delete `_scene_for_pose()` and `_update_camera_pose()` from `MitsubaRenderer`.

- [ ] **Step 5: Run tests**

Run from `apps/python`:

```powershell
uv run python -m unittest tests.test_mitsuba_renderer -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add apps/python/utils/zapdos/renderer/mitsuba_pose.py apps/python/utils/zapdos/renderer/mitsuba_renderer.py apps/python/tests/test_mitsuba_renderer.py
git commit -m "refactor: isolate Mitsuba pose transform logic"
```

---

### Task 6: Add Mitsuba Prepared Scene State Builder

**Files:**
- Create: `apps/python/utils/zapdos/renderer/mitsuba_scene_state.py`
- Modify: `apps/python/tests/test_mitsuba_renderer.py`

- [ ] **Step 1: Write failing state builder test**

Add this import to `apps/python/tests/test_mitsuba_renderer.py`:

```python
from utils.zapdos.renderer.mitsuba_scene_state import build_mitsuba_prepared_scene
```

Add this test:

```python
    def test_build_mitsuba_prepared_scene_loads_scene_and_first_frames(self):
        fake = _FakeMitsuba()
        bundle = _bundle()
        pose = {"body": [1.0] * 16}

        with mock.patch(
            "utils.zapdos.renderer.mitsuba_scene_state.build_mitsuba_scene_dict",
            return_value=({"type": "scene"}, [{"name": "main"}]),
        ) as build_scene:
            prepared = build_mitsuba_prepared_scene(
                bundle=bundle,
                mesh_dir=Path("meshes"),
                width=4,
                height=3,
                spp=2,
                mi=fake,
                pose=pose,
                pose_version=5,
            )

        build_scene.assert_called_once_with(bundle, Path("meshes"), 4, 3, spp=2)
        self.assertIs(prepared.bundle, bundle)
        self.assertEqual(prepared.camera_index, {"main": 0})
        self.assertEqual(prepared.snapshots, [{"name": "main"}])
        self.assertEqual(prepared.pose_version, 5)
        self.assertIn("main", prepared.frames)
        self.assertEqual(prepared.frames["main"].shape, (3, 4, 3))
```

- [ ] **Step 2: Run test to verify it fails**

Run from `apps/python`:

```powershell
uv run python -m unittest tests.test_mitsuba_renderer.MitsubaRendererTest.test_build_mitsuba_prepared_scene_loads_scene_and_first_frames -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'utils.zapdos.renderer.mitsuba_scene_state'`.

- [ ] **Step 3: Create `mitsuba_scene_state.py`**

Add `apps/python/utils/zapdos/renderer/mitsuba_scene_state.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from utils.zapdos.bundle.camera_specs import camera_name_to_index
from utils.zapdos.renderer.mitsuba_pose import scene_for_pose
from utils.zapdos.renderer.mitsuba_scene import apply_mitsuba_transforms, build_mitsuba_scene_dict

if TYPE_CHECKING:
    from utils.zapdos.bundle import RenderBundle


@dataclass(frozen=True)
class MitsubaPreparedScene:
    bundle: "RenderBundle"
    camera_index: dict[str, int]
    scene_dict: dict[str, Any]
    snapshots: list[dict[str, Any]]
    scene: Any
    frames: dict[str, np.ndarray]
    pose_version: int


def build_mitsuba_prepared_scene(
    *,
    bundle: "RenderBundle",
    mesh_dir: Path,
    width: int,
    height: int,
    spp: int,
    mi,
    pose: dict[str, list[float]],
    pose_version: int,
) -> MitsubaPreparedScene:
    scene_dict, snapshots = build_mitsuba_scene_dict(bundle, mesh_dir, width, height, spp=spp)
    scene = mi.load_dict(apply_mitsuba_transforms(scene_for_pose(scene_dict, pose), mi))
    frames = {
        camera.name: _render_frame(mi, scene, sensor)
        for sensor, camera in enumerate(bundle.cameras)
    }
    return MitsubaPreparedScene(
        bundle=bundle,
        camera_index=camera_name_to_index(bundle.cameras),
        scene_dict=scene_dict,
        snapshots=snapshots,
        scene=scene,
        frames=frames,
        pose_version=pose_version,
    )


def _render_frame(mi, scene, sensor: int) -> np.ndarray:
    rendered = mi.render(scene, sensor=sensor)
    frame = np.asarray(rendered)
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    frame = frame[:, :, :3]
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.maximum(frame, 0.0)
        frame = frame / (1.0 + frame)
        frame = np.power(frame, 1.0 / 2.2) * 255.0
    return np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
```

- [ ] **Step 4: Run targeted test**

Run from `apps/python`:

```powershell
uv run python -m unittest tests.test_mitsuba_renderer.MitsubaRendererTest.test_build_mitsuba_prepared_scene_loads_scene_and_first_frames -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add apps/python/utils/zapdos/renderer/mitsuba_scene_state.py apps/python/tests/test_mitsuba_renderer.py
git commit -m "feat: build prepared Mitsuba scene state"
```

---

### Task 7: Adopt Prepared Mitsuba State in Reload

**Files:**
- Modify: `apps/python/utils/zapdos/renderer/mitsuba_renderer.py`
- Modify: `apps/python/utils/zapdos/renderer/mitsuba_scene_state.py`
- Test: `apps/python/tests/test_mitsuba_renderer.py`

- [ ] **Step 1: Write failing prepared reload test**

Add this test to `apps/python/tests/test_mitsuba_renderer.py`:

```python
    async def test_reload_scene_with_prepared_state_does_not_rebuild_scene(self):
        first = _bundle()
        second = _bundle()
        second.cameras = [RenderCamera(**{**second.cameras[0].to_json(), "name": "wrist", "frame_id": "wrist"})]
        fake = _FakeMitsuba()

        with mock.patch("utils.zapdos.renderer.mitsuba_renderer.load_mitsuba", return_value=fake):
            with mock.patch(
                "utils.zapdos.renderer.mitsuba_renderer.build_mitsuba_scene_dict",
                return_value=({"type": "scene"}, []),
            ):
                renderer = MitsubaRenderer("sess-1", first, 4, 3, 30, True, 0)
                try:
                    await renderer.wait_ready(timeout=1.0)
                    prepared = renderer.prepare_scene_reload(second)
                    with mock.patch(
                        "utils.zapdos.renderer.mitsuba_renderer.build_mitsuba_scene_dict",
                        side_effect=AssertionError("reload should adopt prepared scene"),
                    ):
                        renderer.reload_scene(second, prepared_scene=prepared)
                    self.assertIsNotNone(renderer.read("wrist"))
                    self.assertIsNone(renderer.read("main"))
                finally:
                    renderer.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run from `apps/python`:

```powershell
uv run python -m unittest tests.test_mitsuba_renderer.MitsubaRendererTest.test_reload_scene_with_prepared_state_does_not_rebuild_scene -v
```

Expected: FAIL because `prepare_scene_reload()` and `prepared_scene` adoption do not exist.

- [ ] **Step 3: Make first-frame rendering use the renderer's SPP**

In `apps/python/utils/zapdos/renderer/mitsuba_scene_state.py`, change `_render_frame()`:

```python
def _render_frame(mi, scene, sensor: int, spp: int) -> np.ndarray:
    rendered = mi.render(scene, sensor=sensor, spp=spp)
```

Change the caller:

```python
        camera.name: _render_frame(mi, scene, sensor, spp)
```

- [ ] **Step 4: Update MitsubaRenderer imports and initialization**

In `apps/python/utils/zapdos/renderer/mitsuba_renderer.py`, add:

```python
from utils.zapdos.renderer.mitsuba_pose import scene_for_pose
from utils.zapdos.renderer.mitsuba_scene_state import MitsubaPreparedScene, build_mitsuba_prepared_scene
```

In `__init__()`, add:

```python
        self._scene_lock = threading.Lock()
```

Add a small private SPP method:

```python
    def _spp(self) -> int:
        return max(1, int(self.render_hz // 10) or 1)
```

Replace each repeated `max(1, int(self.render_hz // 10) or 1)` with `self._spp()`.

- [ ] **Step 5: Add `prepare_scene_reload()`**

Add this method to `MitsubaRenderer`:

```python
    def prepare_scene_reload(self, bundle: "RenderBundle", timeout: float = 30.0) -> MitsubaPreparedScene:
        del timeout
        self.start()
        with self._lock:
            pose = {name: list(matrix) for name, matrix in self._pose.items()}
            pose_version = self._pose_version
        mesh_dir = self._mesh_dir(bundle)
        self._mi = self._mi or load_mitsuba()
        with self._scene_lock:
            return build_mitsuba_prepared_scene(
                bundle=bundle,
                mesh_dir=mesh_dir,
                width=self.width,
                height=self.height,
                spp=self._spp(),
                mi=self._mi,
                pose=pose,
                pose_version=pose_version,
            )
```

Add:

```python
    def _mesh_dir(self, bundle: "RenderBundle") -> Path:
        bundle_dir = getattr(bundle, "bundle_dir", None)
        if isinstance(bundle_dir, Path):
            return bundle_dir / "mitsuba_meshes"
        return self.work_dir / "meshes"
```

- [ ] **Step 6: Add prepared-state adoption**

Change `reload_scene()` signature:

```python
    def reload_scene(
        self,
        bundle: "RenderBundle",
        timeout: float = 30.0,
        prepared_scene: MitsubaPreparedScene | None = None,
    ) -> None:
```

Implement the prepared branch before the synchronous fallback:

```python
        del timeout
        if prepared_scene is not None:
            self._adopt_prepared_scene(bundle, prepared_scene)
            return
        self.bundle = bundle
        self.camera_index = camera_name_to_index(bundle.cameras)
        with self._lock:
            self._frames.clear()
            self._ready = False
            self._frame_index = 0
            self._scene_dict = None
            self._loaded_pose_version = -1
        self._load_scene(bundle)
```

Add:

```python
    def _adopt_prepared_scene(self, bundle: "RenderBundle", prepared: MitsubaPreparedScene) -> None:
        if prepared.bundle is not bundle:
            raise RuntimeError("Prepared Mitsuba scene does not match reload bundle")
        with self._scene_lock:
            with self._lock:
                self.bundle = bundle
                self.camera_index = dict(prepared.camera_index)
                self._scene_dict = prepared.scene_dict
                self._snapshots = list(prepared.snapshots)
                self._scene = prepared.scene
                self._loaded_pose_version = prepared.pose_version
                self._frames.clear()
                self._frame_index = 0
                for camera in bundle.cameras:
                    frame = prepared.frames.get(camera.name)
                    if frame is None:
                        continue
                    self._frame_index += 1
                    self._frames[camera.name] = (self._frame_index, frame.copy())
                self._ready = bool(self._frames) or not bundle.cameras
                self._error = None
```

- [ ] **Step 7: Make load/render scene-lock aware**

Change `_load_scene()` to accept a bundle:

```python
    def _load_scene(self, bundle: "RenderBundle" | None = None) -> None:
        bundle = self.bundle if bundle is None else bundle
```

Use:

```python
        mesh_dir = self._mesh_dir(bundle)
```

Remove the `shutil.rmtree(mesh_dir)` call from `_load_scene()`.

In `_load_pose_scene()`, wrap `mi.load_dict()`:

```python
        with self._scene_lock:
            self._scene = self._mi.load_dict(apply_mitsuba_transforms(posed_scene, self._mi))
```

In `_render_camera()`, wrap render:

```python
        with self._scene_lock:
            rendered = self._mi.render(self._scene, sensor=sensor, spp=self._spp())
```

- [ ] **Step 8: Run Mitsuba renderer tests**

Run from `apps/python`:

```powershell
uv run python -m unittest tests.test_mitsuba_renderer -v
```

Expected: PASS.

- [ ] **Step 9: Commit**

```powershell
git add apps/python/utils/zapdos/renderer/mitsuba_renderer.py apps/python/utils/zapdos/renderer/mitsuba_scene_state.py apps/python/tests/test_mitsuba_renderer.py
git commit -m "feat: adopt prewarmed Mitsuba scene reloads"
```

---

### Task 8: Cache Mitsuba Mesh Writes by Content

**Files:**
- Modify: `apps/python/utils/zapdos/renderer/mitsuba_scene.py`
- Test: `apps/python/tests/test_mitsuba_scene_builder.py`

- [ ] **Step 1: Write failing mesh cache test**

Add this test to `apps/python/tests/test_mitsuba_scene_builder.py`:

```python
    def test_build_mitsuba_scene_dict_reuses_unchanged_mesh_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            usd_path = root / "scene.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            mesh = UsdGeom.Mesh.Define(stage, "/World/Triangle")
            mesh.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)])
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
            stage.GetRootLayer().Save()
            bundle = SimpleNamespace(render_scene_usda=usd_path, cameras=[_camera()])
            mesh_dir = root / "meshes"

            first, _ = build_mitsuba_scene_dict(bundle, mesh_dir, 64, 48, spp=2)
            mesh_file = Path(next(value["filename"] for value in first.values() if isinstance(value, dict) and value.get("type") == "ply"))
            original = mesh_file.read_text(encoding="utf-8")
            marker = mesh_file.stat().st_mtime_ns
            second, _ = build_mitsuba_scene_dict(bundle, mesh_dir, 64, 48, spp=2)

            self.assertEqual(mesh_file.read_text(encoding="utf-8"), original)
            self.assertEqual(mesh_file.stat().st_mtime_ns, marker)
            self.assertEqual(
                next(value["filename"] for value in first.values() if isinstance(value, dict) and value.get("type") == "ply"),
                next(value["filename"] for value in second.values() if isinstance(value, dict) and value.get("type") == "ply"),
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run from `apps/python`:

```powershell
uv run python -m unittest tests.test_mitsuba_scene_builder.MitsubaSceneBuilderTest.test_build_mitsuba_scene_dict_reuses_unchanged_mesh_file -v
```

Expected: FAIL because `_write_ply()` rewrites the file and changes mtime.

- [ ] **Step 3: Add deterministic mesh file names and skip unchanged writes**

In `apps/python/utils/zapdos/renderer/mitsuba_scene.py`, import `hashlib`:

```python
import hashlib
```

Change mesh path selection in `_add_meshes()`:

```python
        mesh_payload = _ply_text(vertices, triangles)
        mesh_hash = hashlib.sha1(mesh_payload.encode("utf-8")).hexdigest()[:16]
        mesh_path = mesh_dir / f"mesh_{mesh_hash}.ply"
        _write_text_if_changed(mesh_path, mesh_payload)
```

Replace `_write_ply()` with:

```python
def _ply_text(vertices: list[list[float]], faces: list[tuple[int, int, int]]) -> str:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines.extend(f"{x:.9g} {y:.9g} {z:.9g}" for x, y, z in vertices)
    lines.extend(f"3 {a} {b} {c}" for a, b, c in faces)
    return "\n".join(lines) + "\n"


def _write_text_if_changed(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")
```

- [ ] **Step 4: Run scene builder tests**

Run from `apps/python`:

```powershell
uv run python -m unittest tests.test_mitsuba_scene_builder -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add apps/python/utils/zapdos/renderer/mitsuba_scene.py apps/python/tests/test_mitsuba_scene_builder.py
git commit -m "perf: cache Mitsuba mesh files by content"
```

---

### Task 9: Add Stage Timing Evidence to Rebuild Progress

**Files:**
- Modify: `apps/python/utils/zapdos/editor/rebuild_manager.py`
- Test: `apps/python/tests/test_zapdos_import.py`

- [ ] **Step 1: Write the failing timing test**

Add this test near `test_run_overlay_rebuild_prepares_renderer_before_world_reservation`:

```python
    async def test_renderer_reload_done_progress_includes_elapsed_ms(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.sess = "sess-1"
        session.bundle = SimpleNamespace(cameras=[])
        editor = self.attach_editor(session)
        session.editor = editor
        editor.rebuilding_scene = True
        drained: list[tuple[str, dict[str, object]]] = []

        @asynccontextmanager
        async def reserve_world():
            yield object()

        async def run_sync(fn, world_token=None):
            return fn(session)

        class Renderer:
            def prepare_scene_reload(self, bundle):
                del bundle
                time.sleep(0.001)
                return {"prepared": True}

        session.renderer = Renderer()
        session.reserve_world = reserve_world
        session.run_sync = run_sync
        prepared = PreparedOverlayRebuild(
            bundle=SimpleNamespace(cameras=[]),
            next_overlay=default_overlay_state("C:/assets"),
            previous_overlay=default_overlay_state("C:/assets"),
            previous_revision="rev-1",
            next_revision="rev-2",
        )
        EDITOR_REBUILD_EVENTS.create_scene_rebuild_job(editor, "op-1", {"ok": True})

        with mock.patch.object(editor, "_build_support_infos", return_value={}):
            with mock.patch.object(editor, "_prepare_overlay_rebuild", return_value=prepared):
                with mock.patch.object(
                    editor,
                    "_apply_prepared_overlay_rebuild",
                    side_effect=lambda current_prepared, current_op_id: setattr(editor, "rebuilding_scene", False) or "rev-2",
                ):
                    await EDITOR_REBUILD_MANAGER.run_overlay_rebuild(
                        editor,
                        "op-1",
                        default_overlay_state("C:/assets"),
                        default_overlay_state("C:/assets"),
                        "rev-1",
                    )
        drained = EDITOR_REBUILD_EVENTS.drain_scene_rebuild_events(editor, "op-1")
        done_events = [
            payload
            for name, payload in drained
            if name == "progress" and payload.get("stage") == "prepare_renderer_reload.done"
        ]

        self.assertEqual(len(done_events), 1)
        self.assertIsInstance(done_events[0].get("elapsed_ms"), float)
```

- [ ] **Step 2: Run test to verify it fails**

Run from `apps/python`:

```powershell
uv run python -m unittest tests.test_zapdos_import.ZapdosImportTest.test_renderer_reload_done_progress_includes_elapsed_ms -v
```

Expected: FAIL because `prepare_renderer_reload.done` exists but does not include `elapsed_ms`.

- [ ] **Step 3: Add elapsed timing around renderer prewarm**

In `apps/python/utils/zapdos/editor/rebuild_manager.py`, import:

```python
import time
```

Change `_prepare_renderer_reload()`:

```python
    started = time.perf_counter()
    emit_scene_rebuild_progress(session, op_id, "prepare_renderer_reload.started")
    prepared = await asyncio.to_thread(prepare, bundle)
    emit_scene_rebuild_progress(
        session,
        op_id,
        "prepare_renderer_reload.done",
        elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
    )
```

- [ ] **Step 4: Run targeted tests**

Run from `apps/python`:

```powershell
uv run python -m unittest tests.test_zapdos_import.ZapdosImportTest.test_renderer_reload_done_progress_includes_elapsed_ms tests.test_zapdos_import.ZapdosImportTest.test_run_overlay_rebuild_prepares_renderer_before_world_reservation -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add apps/python/utils/zapdos/editor/rebuild_manager.py apps/python/tests/test_zapdos_import.py
git commit -m "chore: report renderer prewarm timing"
```

---

### Task 10: Full Verification

**Files:**
- Verify all touched files.

- [ ] **Step 1: Run focused Python tests**

Run from `apps/python`:

```powershell
uv run python -m unittest tests.test_mitsuba_renderer tests.test_mitsuba_scene_builder tests.test_zapdos_renderer_entity tests.test_zapdos_import -v
```

Expected: PASS.

- [ ] **Step 2: Run broader Zapdos renderer and bundle tests**

Run from `apps/python`:

```powershell
uv run python -m unittest tests.test_sim_env_renderer tests.test_rl_bundle tests.test_zapdos_runtime_swap -v
```

Expected: PASS.

- [ ] **Step 3: Manual timing check**

Start Zapdos with Mitsuba enabled from the project root:

```powershell
$env:ZAPDOS_RENDERER="mitsuba"
cd apps/python
uv run python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

In the web UI, run one `set_scene_assets` operation. Confirm these progress events appear in the operation stream:

```text
prepare_overlay_rebuild.ensure_render_bundle
prepare_renderer_reload.started
prepare_renderer_reload.done
swap_runtime_bundle.reload_scene.started
swap_runtime_bundle.reload_scene.done
```

Expected result:

- `prepare_renderer_reload.done.elapsed_ms` contains the Mitsuba preload cost.
- The wall-clock gap between `swap_runtime_bundle.reload_scene.started` and `swap_runtime_bundle.reload_scene.done` is short compared with the old full Mitsuba load path.
- The camera stream has a non-placeholder frame immediately after the operation emits `done`.

- [ ] **Step 4: Commit final verification notes if implementation changed docs**

If implementation added a short note under `docs/`, commit it:

```powershell
git add docs
git commit -m "docs: record Mitsuba reload verification"
```

If no docs changed during verification, skip this commit.

---

## Self-Review

- Spec coverage: The plan covers the slow synchronous `reload_scene()` chain, prewarm before world reservation, fast adoption at swap, mesh write caching, tests, and manual timing evidence.
- Placeholder scan: No placeholder markers or vague edge-case instructions remain.
- Type consistency: `PreparedOverlayRebuild.renderer_reload`, `ZapdosRenderer.prepare_scene_reload()`, `reload_scene(..., prepared_scene=...)`, `MitsubaPreparedScene`, and `build_mitsuba_prepared_scene()` names are consistent across tasks.
- Scope: This stays within the Python Zapdos renderer/editor path. It does not change front-end API contracts, Isaac renderer behavior, or bundle generation semantics.
