# Zapdos Scene Object Transform Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to execute this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Allow selecting top-level scene objects in `/demo/zapdos`, moving or rotating them with R3F `TransformControls`, and syncing the pose back into the current MuJoCo session so WebGL, SSE, and Isaac MJPEG stay consistent.

**Architecture:** Promote editable scene objects to named MuJoCo fixed bodies and include them in the existing `render_scene_body_map.json` sidecar so `tf_render` moves the matching USD prims inside Isaac. Change Zapdos visual payload from geom-only world matrices to `bodies + meshes`, then attach `TransformControls` to body groups and send `set_body_pose` RPC calls on change/end.

**Tech Stack:** Python 3.12, MuJoCo, USD, FastAPI, Next.js client components, React Three Fiber, Drei, `node:test`, `unittest`

---

### Task 1: Promote editable scene objects into MuJoCo and Isaac body maps

**Files:**
- Create: `apps/python/utils/scene_objects.py`
- Modify: `apps/python/utils/rl_bundle.py`
- Modify: `apps/python/utils/rl_bundle_stage.py`
- Modify: `apps/python/utils/usd_to_mjcf.py`
- Modify: `apps/python/tests/test_rl_bundle.py`
- Modify: `apps/python/tests/test_usd_to_mjcf.py`

- [ ] Add failing tests for a synthetic scene with one top-level object root. Assert that the object becomes a MuJoCo body, appears in `render_scene_body_map.json`, and maps to the env-relative render prim path.
- [ ] Implement `scene_objects.py` to scan the scene default prim, collect editable top-level `UsdGeom.Xformable` roots, and exclude non-objects such as `Ground`, cameras, materials, and shaders.
- [ ] Extend `USDToMJCFConverter` with `force_body_paths: set[str]`. `should_emit_body()` must treat forced scene roots as bodies even when they have no inertial or joint data.
- [ ] Keep forced scene bodies at the world/root level so runtime `model.body_pos/body_quat` edits represent world-space object motion directly.
- [ ] In `ensure_render_bundle()`, collect scene object specs before conversion, pass `force_body_paths` into the converter, merge scene object entries into the existing robot `body_map`, and bump `BUNDLE_VERSION`.
- [ ] Keep body-map values env-relative, matching the renderer convention already used by robot entries such as `MyRobot/...`.
- [ ] Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_usd_to_mjcf apps.python.tests.test_rl_bundle`

### Task 2: Expose body-grain visuals and a `set_body_pose` RPC in Zapdos

**Files:**
- Create: `apps/python/utils/zapdos_scene_visuals.py`
- Modify: `apps/python/utils/mujoco_tools.py`
- Modify: `apps/python/api/zapdos/{session}/{action}.py`
- Modify: `apps/python/tests/test_zapdos_import.py`

- [ ] Add failing tests for `get_visual()` returning `{ bodies, meshes }`, for `set_body_pose()` updating an editable scene body, and for rejecting robot bodies or unknown names.
- [ ] Add `body_world_pose()` to `mujoco_tools.py` and use it to build stable body matrices keyed by MuJoCo body name.
- [ ] Create `zapdos_scene_visuals.py` with small serializers:
  `BodyVisual = { name, label, editable, matrix }`
  `MeshVisual = { name, body | null, kind, color, matrix | localMatrix, size?, mesh?, texture? }`
- [ ] Change `get_visual()` to return body groups plus meshes. Meshes attached to a body must carry `localMatrix`; static meshes without a body keep absolute `matrix`.
- [ ] Keep SSE `pose` keyed by body name only. Frontend body groups should use these names directly.
- [ ] Add `set_body_pose(body, pos, quat)` to `call_once()`. Run it on the existing session thread, validate `body` against the editable scene body set, write `model.body_pos/body_quat`, call `mujoco.mj_forward`, and let the existing SSE plus `tf_render` loop publish the result.
- [ ] Leave `save_camera_override()` and camera APIs unchanged.
- [ ] Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_zapdos_import`

### Task 3: Add selection and `TransformControls` in the web scene

**Files:**
- Create: `apps/web/components/zapdos/zapdos-scene-api.ts`
- Create: `apps/web/components/zapdos/zapdos-scene-api.test.ts`
- Create: `apps/web/components/zapdos/zapdos-scene-state.ts`
- Create: `apps/web/components/zapdos/zapdos-scene-state.test.ts`
- Create: `apps/web/components/zapdos/ZapdosScene.tsx`
- Modify: `apps/web/components/zapdos/SurfacePivotControls.tsx`
- Modify: `apps/web/components/zapdos/ZapdosTopOverlay.tsx`
- Modify: `apps/web/app/demo/zapdos/page.tsx`

- [ ] Add failing web tests for the request helpers and the small state helpers only. Keep pointer-heavy canvas interaction covered by manual verification.
- [ ] Move the current inline scene-loading logic out of `page.tsx` into `ZapdosScene.tsx` so `page.tsx` stays focused on session bootstrap and top-level layout.
- [ ] Add `enabled?: boolean` to `SurfacePivotControls`. Disable it while a transform gizmo drag is active.
- [ ] In `zapdos-scene-api.ts`, add typed helpers for `get_visual` and `set_body_pose`, including `createSetBodyPoseRequest()` and a payload builder that converts `THREE.Object3D` world transform to `{ pos, quat }`.
- [ ] In `ZapdosScene.tsx`, render:
  - body groups keyed by body name, each with the current world matrix
  - meshes parented under their body group with `localMatrix`
  - static meshes with absolute `matrix`
- [ ] Selection rules: click selects editable scene bodies only; robot bodies and static world meshes stay read-only.
- [ ] Transform rules: default mode is `translate`; `W` switches to move, `E` switches to rotate, `Escape` clears selection.
- [ ] While a body is being dragged, keep its local transform authoritative in the browser and ignore SSE pose updates for that same body to avoid snap-back from slightly older echoes.
- [ ] On `TransformControls` object change, update the selected body group locally; on pointer-up or drag end, send a final `set_body_pose()` RPC.
- [ ] Surface selected-body name and current mode in `ZapdosTopOverlay` so the operator can see what is being edited.
- [ ] Run:
  - `pnpm --filter glassbeaker-web test -- zapdos-scene-api zapdos-scene-state camera-override-save zapdos-import`
  - `pnpm --filter glassbeaker-web exec tsc --noEmit`

### Task 4: Verify MuJoCo, WebGL, and Isaac stay in sync

**Files:**
- Modify: none

- [ ] Run the focused Python suite:
  `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_usd_to_mjcf apps.python.tests.test_rl_bundle apps.python.tests.test_zapdos_import`
- [ ] Run the focused web suite:
  `pnpm --filter glassbeaker-web test -- zapdos-scene-api zapdos-scene-state camera-override-save zapdos-import`
- [ ] Manual verification:
  - start the Python app with the usual `uv` flow under `apps/python`
  - open `/demo/zapdos`
  - click a scene object and confirm the gizmo appears
  - move and rotate it, confirming the main WebGL object, SSE pose, and Isaac MJPEG all follow
  - confirm robot links cannot be selected for transform
  - refresh or rebuild the session and confirm scene edits are gone, matching the current-session-only scope

## Assumptions

- v1 supports scene objects only, not robot links.
- v1 supports translate and rotate only; scaling stays unsupported.
- Editable scene roots are top-level scene object prims under the scene default prim.
- Persistence is intentionally session-only. Do not write scene transforms to user config or source USD files in this pass.
