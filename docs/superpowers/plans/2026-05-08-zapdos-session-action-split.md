# Zapdos Session Action Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Zapdos session runtime and overlay scene-operation logic out of `apps/python/api/zapdos/{session}/{action}.py` while preserving behavior and compatibility imports.

**Architecture:** Keep the route file thin and compatibility-oriented. Move `ZapdosSession` into `utils/zapdos/zapdos_session.py`, move overlay rebuild and SSE operation helpers into `utils/zapdos/zapdos_scene_operations.py`, and re-export moved symbols from the route module so callers and existing tests keep working.

**Tech Stack:** Python 3.12, FastAPI, MuJoCo, asyncio, `unittest`, `uv`

---

## File Structure

- `apps/python/api/zapdos/{session}/{action}.py`
  Thin route and compatibility re-export layer.
- `apps/python/utils/zapdos/zapdos_session.py`
  `ZapdosSession` lifecycle and runtime behavior.
- `apps/python/utils/zapdos/zapdos_scene_operations.py`
  Overlay operation state, rebuild subprocess helpers, and scene-operation SSE stream.
- `apps/python/tests/test_zapdos_import.py`
  Regression coverage for the split plus existing route/session behavior.

### Task 1: Add the split regression test

**Files:**
- Modify: `apps/python/tests/test_zapdos_import.py`

- [ ] **Step 1: Write the failing split test**

```python
    def test_action_module_reexports_split_runtime_symbols(self):
        self.assertEqual(MODULE.ZapdosSession.__module__, "utils.zapdos.zapdos_session")
        self.assertEqual(MODULE._stream_scene_operation.__module__, "utils.zapdos.zapdos_scene_operations")
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_import.ZapdosImportTest.test_action_module_reexports_split_runtime_symbols -v`

Expected: FAIL because both symbols still come from the action module.

- [ ] **Step 3: Keep the rest of the suite unchanged as regression coverage**

No extra test scaffolding is needed because `test_zapdos_import.py`, `test_zapdos_overlay_rebuild_subprocess.py`, and `test_zapdos_render_camera.py` already cover the moved behavior.

### Task 2: Extract scene-operation helpers

**Files:**
- Create: `apps/python/utils/zapdos/zapdos_scene_operations.py`
- Modify: `apps/python/api/zapdos/{session}/{action}.py`

- [ ] **Step 1: Move dataclasses and overlay operation helpers into the new module**

Include:

- `PreparedOverlayRebuild`
- `OverlayRebuildCompletion`
- `SceneOperation`
- `build_set_scene_assets_overlay`
- `build_remove_asset_overlay`
- `prepare_overlay_rebuild`
- `run_overlay_rebuild_background`
- `apply_prepared_overlay_rebuild`
- `drain_overlay_completions`
- `scene_operation_future`
- `discard_scene_operation`
- `_stream_scene_operation`

- [ ] **Step 2: Re-export moved symbols from the action module**

Import the moved names into `apps/python/api/zapdos/{session}/{action}.py` so existing tests keep using `MODULE.<name>`.

- [ ] **Step 3: Run the focused Python tests**

Run:

```powershell
uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_import apps.python.tests.test_zapdos_overlay_rebuild_subprocess apps.python.tests.test_zapdos_render_camera -v
```

Expected: PASS.

### Task 3: Extract `ZapdosSession`

**Files:**
- Create: `apps/python/utils/zapdos/zapdos_session.py`
- Modify: `apps/python/api/zapdos/{session}/{action}.py`

- [ ] **Step 1: Move `ZapdosSession` and its runtime helpers into the new module**

Keep method names intact for compatibility:

- `_build_support_infos`
- `_swap_runtime_bundle`
- `_prepare_overlay_rebuild`
- `_run_overlay_rebuild_background`
- `_apply_prepared_overlay_rebuild`
- `_drain_overlay_completions`
- `scene_operation_future`
- `discard_scene_operation`

- [ ] **Step 2: Make the action module import and re-export `ZapdosSession`**

This preserves current test import style and keeps route dispatch code small.

- [ ] **Step 3: Run the focused Python tests again**

Run:

```powershell
uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_import apps.python.tests.test_zapdos_overlay_rebuild_subprocess apps.python.tests.test_zapdos_render_camera apps.python.tests.test_session apps.python.tests.test_session_registry -v
```

Expected: PASS.

### Task 4: Final verification

**Files:**
- Modify: none

- [ ] **Step 1: Check the action file line count**

Run: `(Get-Content 'apps/python/api/zapdos/{session}/{action}.py').Count`

Expected: materially smaller than the current working copy and limited to route/bootstrap concerns.

- [ ] **Step 2: Run the full focused verification set**

Run:

```powershell
uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_import apps.python.tests.test_zapdos_overlay_rebuild_subprocess apps.python.tests.test_zapdos_render_camera apps.python.tests.test_session apps.python.tests.test_session_registry -v
```

Expected: PASS with no regressions in the moved behavior.
