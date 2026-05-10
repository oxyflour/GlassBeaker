# Zapdos R1Pro Moz1 Switching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep `r1pro` working as-is, add `moz1` import support, and make `/demo/zapdos` switch robots only by changing `robot_usd`.

**Architecture:** Keep the public contract unchanged and branch internally by input suffix. `.usd` and `.usda` continue through the current USD pipeline. `.urdf` resolves into a robot asset descriptor that separates physics input from render visuals, compiles MuJoCo XML from URDF, attaches Spirit USD visuals to MuJoCo body poses, and synthesizes a fallback `main` camera when the MuJoCo model has none.

**Tech Stack:** Python 3.12, MuJoCo, Pixar USD, FastAPI session bootstrap, `unittest`, `uv`

---

## Runtime Contract

- Default `r1pro`: `/demo/zapdos`
- Explicit `r1pro`: `/demo/zapdos?robot_usd=deps/galaxea/object/r1pro/r1pro.usda`
- Switch to `moz1`: `/demo/zapdos?robot_usd=deps/moz01/spirit01_model/urdf/moz1.urdf`
- No new query parameter in this change. `robot_usd` remains the only robot selector even when the value is a URDF path.

## File Structure

- Create: `apps/python/utils/zapdos/bundle/robot_assets.py`
- Modify: `apps/python/utils/zapdos/bundle/bundle_builder.py`
- Modify: `apps/python/utils/zapdos/bundle/stage_builder.py`
- Modify: `apps/python/utils/zapdos/bundle/camera_specs.py`
- Modify: `apps/python/utils/zapdos/physics/mujoco_tools.py`
- Modify: `apps/python/tests/test_rl_bundle.py`
- Create: `apps/python/tests/test_rl_bundle_urdf.py`
- Modify: `apps/python/tests/test_zapdos_import.py`

### Task 1: Add a robot asset resolver with failing tests first

**Files:**
- Create: `apps/python/utils/zapdos/bundle/robot_assets.py`
- Create: `apps/python/tests/test_rl_bundle_urdf.py`

- [ ] **Step 1: Write failing tests for the resolver contract**
  Cover both robot types. Assert that `r1pro.usda` resolves to a USD-backed descriptor and `moz1.urdf` resolves to a URDF-backed descriptor with:
  `visual_usd` under `deps/moz01/isaac_moz1/Issacsim_Assets/spirit01_model/spirit01_model/USD/`,
  `visual_root == "/World/MOZ1"`,
  and an attachment map that includes all movable MuJoCo bodies by leaf name.

- [ ] **Step 2: Add failing tests for Moz1 URDF path rewriting**
  Assert that a helper rewrites `package://spirit01_model/...` references into bundle-local mesh paths without mutating the source URDF under `deps/moz01/spirit01_model/urdf/`.

- [ ] **Step 3: Run the focused tests and confirm failure**
  Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_rl_bundle_urdf`
  Expected: FAIL because the resolver module and tests do not exist yet.

- [ ] **Step 4: Implement a small descriptor layer**
  Add a dataclass-oriented helper in `robot_assets.py` that returns a resolved descriptor with fields equivalent to:
  `robot_input`, `physics_input`, `visual_usd`, `visual_root`, `attachments_by_body`, `static_visual_paths`, and `dependency_paths`.
  Keep the existing `r1pro` path as the trivial USD case.

- [ ] **Step 5: Encode Moz1-specific resolution rules**
  Prefer `Moz1_omni_gripper_full.usd`, then `Moz1_omni_gripper.usd`.
  Resolve attachments by prim leaf name under `/World/MOZ1`.
  Hard-code the fixed-joint merges discovered during investigation:
  `base_link` visuals belong to static root visuals and `head21/head22/head23` visuals attach under `waist03`.

- [ ] **Step 6: Re-run the focused tests**
  Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_rl_bundle_urdf`
  Expected: PASS.

### Task 2: Split bundle building into physics and render inputs

**Files:**
- Modify: `apps/python/utils/zapdos/bundle/bundle_builder.py`
- Modify: `apps/python/utils/zapdos/physics/mujoco_tools.py`
- Test: `apps/python/tests/test_rl_bundle.py`
- Test: `apps/python/tests/test_rl_bundle_urdf.py`

- [ ] **Step 1: Add failing bundle tests for URDF input**
  Extend bundle coverage so `ensure_render_bundle(Path("deps/moz01/spirit01_model/urdf/moz1.urdf"), DEFAULT_SCENE_USD)` is expected to emit:
  `sim_scene.xml`, `robot_wrapper.usda`, `render_scene.usda`, and a manifest with the original `robot_usd` preserved.

- [ ] **Step 2: Run the bundle suites and confirm failure**
  Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_rl_bundle apps.python.tests.test_rl_bundle_urdf`
  Expected: FAIL because `ensure_render_bundle()` still assumes `robot_usd` is directly openable as a USD stage.

- [ ] **Step 3: Resolve robot assets before any stage open**
  In `bundle_builder.py`, replace the early `Usd.Stage.Open(str(robot_usd))` assumption with a descriptor from `robot_assets.resolve_robot_assets(robot_usd, bundle_dir)`.
  Continue storing the original requested path in `RenderBundle.robot_usd`.

- [ ] **Step 4: Split physics generation by descriptor type**
  Keep the existing USD path for `r1pro`.
  For URDF input, compile a bundle-local MuJoCo robot XML from the rewritten URDF, build a scene-only MJCF from `scene_usd`, then merge robot XML plus scene XML into the final `sim_scene.xml`.
  Reuse `apps/python/utils/zapdos/physics/mujoco_tools.py` where practical, but remove assumptions that conversion always writes beside the source asset.

- [ ] **Step 5: Update bundle cache invalidation**
  Bump `BUNDLE_VERSION` and add all descriptor dependency paths to the bundle key so changes to the URDF, chosen Moz1 USD, or resolver logic invalidate stale bundles.

- [ ] **Step 6: Re-run the bundle suites**
  Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_rl_bundle apps.python.tests.test_rl_bundle_urdf`
  Expected: PASS.

### Task 3: Teach the wrapper builder to attach multiple visual prims per body

**Files:**
- Modify: `apps/python/utils/zapdos/bundle/stage_builder.py`
- Test: `apps/python/tests/test_rl_bundle_urdf.py`

- [ ] **Step 1: Add failing wrapper tests**
  Assert that the generated Moz1 wrapper preserves:
  `base_link` meshes as static visuals under `/MyRobot`,
  `head21/head22/head23` visuals under `/MyRobot/waist03`,
  and one render body prim for each movable MuJoCo body.

- [ ] **Step 2: Run the URDF suite and confirm failure**
  Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_rl_bundle_urdf`
  Expected: FAIL because `build_robot_wrapper()` only accepts one source prim per body and assumes `visuals` exists directly beneath the matched prim.

- [ ] **Step 3: Generalize wrapper inputs**
  Change `robot_source_map()` and `build_robot_wrapper()` to accept multi-attachment mappings such as `dict[str, list[str]]` instead of a single source path.
  Keep the old USD behavior by adapting the existing one-to-one map into a single-item list.

- [ ] **Step 4: Add static visual support**
  Allow the wrapper builder to emit root-level visual references for fixed visuals that do not belong to a movable MuJoCo body.
  Keep embedded source cameras deactivated just as the current builder already does for referenced visuals.

- [ ] **Step 5: Re-run the URDF suite**
  Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_rl_bundle_urdf`
  Expected: PASS.

### Task 4: Add a fallback `main` camera for camera-less robot inputs

**Files:**
- Modify: `apps/python/utils/zapdos/bundle/camera_specs.py`
- Modify: `apps/python/utils/zapdos/bundle/bundle_builder.py`
- Test: `apps/python/tests/test_rl_bundle.py`
- Test: `apps/python/tests/test_rl_bundle_urdf.py`

- [ ] **Step 1: Add failing camera tests**
  Assert that a camera-less MuJoCo model yields one scene camera named `main` instead of raising `RuntimeError("MuJoCo model has no cameras.")`.
  Assert that existing camera-bearing `r1pro` bundles remain unchanged.

- [ ] **Step 2: Run the camera-related suites and confirm failure**
  Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_rl_bundle apps.python.tests.test_rl_bundle_urdf`
  Expected: FAIL because `build_render_cameras()` currently hard-fails on zero cameras.

- [ ] **Step 3: Implement fallback camera generation**
  Add a helper in `camera_specs.py` that returns a synthesized scene camera:
  `name="main"`, `prim="/SceneRender/main"`, `body=None`.
  Use the same default viewing pose the frontend already expects for initial rendering so `/render/main` becomes stable for Moz1.

- [ ] **Step 4: Keep override compatibility**
  Make sure `apply_camera_overrides()` still runs after fallback creation so user camera overrides continue to work for both robots.

- [ ] **Step 5: Re-run the camera-related suites**
  Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_rl_bundle apps.python.tests.test_rl_bundle_urdf`
  Expected: PASS.

### Task 5: Lock the public switching behavior with API and session tests

**Files:**
- Modify: `apps/python/tests/test_zapdos_import.py`
- Test: `apps/python/tests/test_rl_bundle.py`
- Test: `apps/python/tests/test_rl_bundle_urdf.py`

- [ ] **Step 1: Add bootstrap tests for `robot_usd` switching**
  Cover `ZapdosSession.create()` and `/python/zapdos/{session}/init/start` with:
  default `r1pro`,
  explicit `r1pro.usda`,
  and `moz1.urdf`.
  Assert that session identity still keys off the raw `robot_usd` string so `r1pro` and `moz1` do not collide.

- [ ] **Step 2: Run the API and bundle suites**
  Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_zapdos_import apps.python.tests.test_rl_bundle apps.python.tests.test_rl_bundle_urdf`
  Expected: FAIL until the session bootstrap path accepts the URDF-backed bundle end-to-end.

- [ ] **Step 3: Fix any bootstrap assumptions that still require USD-only robots**
  Keep the request parameter name `robot_usd`.
  Do not add a new `robot` or `robot_urdf` parameter in this change.
  Only normalize code that incorrectly validates by suffix or wording.

- [ ] **Step 4: Run the full focused verification**
  Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_zapdos_import apps.python.tests.test_rl_bundle apps.python.tests.test_rl_bundle_urdf`
  Expected: PASS.

- [ ] **Step 5: Manual verification**
  Open `/demo/zapdos`
  Confirm `r1pro` still renders and behaves normally.
  Open `/demo/zapdos?robot_usd=deps/moz01/spirit01_model/urdf/moz1.urdf`
  Confirm Moz1 renders with base, torso, head, arms, and wheels visible.
  Confirm `GET /python/zapdos/{sess}/render/main` returns frames for Moz1.

## Notes

- Keep the change minimally invasive to the public interface. The important compatibility rule is "same parameter, more accepted file types".
- Do not revert the existing USD path. The `r1pro` path is the regression baseline for every task above.
- If the implementation starts growing, split helper logic aggressively. The bundle layer is already large and should not absorb all Moz1-specific rules inline.
