# Genie Sim Mitsuba Design

**Goal:** Extend the `agent-genie-sim` demo so each generated scene exports a frozen `scene.usda`, and the preview panel can trigger Mitsuba CUDA rendering for that frozen scene without re-executing scene code.

## Decisions

- Keep the current `generate_scene` flow as the single source of scene generation.
- Export `scene.usda` immediately after successful scene execution, not lazily on render.
- Add a dedicated render action in the right preview panel instead of auto-rendering after every scene update.
- Treat USDA export and Mitsuba render as sibling artifacts of one generated scene bundle. Mitsuba does not read USDA back as its scene source in v1.
- Require Mitsuba `cuda_ad_rgb` for rendering on this machine. Do not silently fall back to CPU/LLVM rendering.
- Store generated scene artifacts under `apps/python/tmp/genie_sim/<bundle_id>/`.

## Why This Shape

- Exporting USDA during scene execution guarantees the saved scene matches the structured preview the user sees.
- A preview button keeps the agent workflow responsive. Scene generation can happen repeatedly without forcing a GPU render each time.
- Re-rendering from a frozen bundle avoids re-running arbitrary scene code on every render request.
- Using the upstream Mitsuba scene-language path is substantially simpler than building a new USDA-to-Mitsuba bridge.

## Bundle Layout

Each successful scene generation creates a bundle directory:

- `apps/python/tmp/genie_sim/<bundle_id>/scene.usda`
- `apps/python/tmp/genie_sim/<bundle_id>/shape.json`
- `apps/python/tmp/genie_sim/<bundle_id>/manifest.json`
- `apps/python/tmp/genie_sim/<bundle_id>/renders/`

The manifest records:

- `bundleId`
- `assetsRoot`
- `code`
- `description`
- `seed`
- `sceneUsdaPath`
- `shapePath`
- `renderPreset`
- `objects`
- render output metadata after a render completes

The bundle id should be deterministic enough for local caching but does not need to be stable across process restarts. A UUID is acceptable for v1.

## Backend

### Runtime responsibilities

`apps/python/utils/genie_sim_runtime.py` should own the scene bundle lifecycle:

- resolve asset root
- prepare Genie Sim runtime
- execute scene code once
- build `layout_info`
- serialize a Mitsuba-renderable scene shape payload
- convert layout objects into USDA export input
- write `scene.usda`
- persist a manifest
- render the frozen bundle on demand with Mitsuba CUDA

Keep the scene-generation-specific logic in focused helpers instead of growing one large function.

### USDA export

Reuse upstream export behavior from `deps/genie_sim/source/geniesim/generator/app.py` and `geniesim/generator/utils/usd.py`:

- derive `object_info_list` from `layout_info["layout"]`
- write a Z-up, `metersPerUnit=1.0` scene file
- save the file into the bundle directory

The runtime should return the exported USDA path with the generated scene payload.

### Mitsuba render

Render from the frozen bundle, not directly from a browser payload.

- Load the bundle manifest by `bundle_id`
- Load the serialized shape payload captured during `/execute`
- Rebuild only the in-memory shape structure needed by upstream `mi_helper.execute_from_preset()`
- Force `mi.set_variant("cuda_ad_rgb")`
- Render into bundle-local PNG files
- Return a browser-consumable primary image URL and the per-view URLs

If CUDA initialization or rendering fails, return an explicit API error. Do not attempt CPU fallback.

The critical invariant is that `/render` may deserialize frozen shape data, but it must not re-run `root_scene()` or re-exec the original scene code.

### API

`apps/python/api/genie_sim.py` should expose three operations:

`POST /python/genie_sim/search_assets`
- unchanged

`POST /python/genie_sim/execute`
- input: `{ code, assets_root? }`
- output: existing scene payload plus:
  - `bundleId`
  - `sceneUsdaPath`

`POST /python/genie_sim/render`
- input: `{ bundle_id }`
- output:
  - `primaryImageUrl`
  - `views: [{ name, url }]`

`GET /python/genie_sim/artifacts/{bundle_id}/{name}`
- serves rendered PNG artifacts from the bundle directory
- only allow known generated artifact names, not arbitrary file paths

### Error handling

Use `HTTPException` in the API layer.

- missing assets root or missing assets package: `404`
- bundle id not found: `404`
- bundle exists but is incomplete or stale: `409`
- scene execution failure: `500`
- USDA export failure: `500`
- CUDA unavailable or Mitsuba CUDA backend unavailable: `503`
- render failure after successful CUDA initialization: `500`

Error messages should be short and direct so the preview panel can display them as-is.

## Frontend

### Data model

Extend `SceneData` in `apps/web/components/genie-sim/scene-types.ts` with:

- `bundleId: string`
- `sceneUsdaPath: string`

Add a separate render result type instead of mixing render state into `SceneData`.

Suggested shape:

- `SceneRenderResult = { primaryImageUrl: string; views: { name: string; url: string }[] }`

### State

`useSceneState()` should track:

- `scene`
- `hasScene`
- `renderStatus: "idle" | "running" | "done" | "error"`
- `renderResult`
- `renderError`

When a new scene is generated successfully:

- replace the current scene
- clear any previous render result
- clear any previous render error
- reset render status to `idle`

### Preview panel

In `apps/web/app/demo/agent-genie-sim/page.tsx`:

- keep the current Three.js box preview
- keep the asset list and generated code block
- add a render section in the right panel with:
  - a `Render` button
  - the exported USDA path
  - a loading state
  - a rendered image area
  - an error message area

The button should be disabled when:

- there is no generated scene
- there is no `bundleId`
- a render is already in progress

The render section should never erase the structured 3D preview. Render failure only affects the Mitsuba result area.

## Testing

### Python

Use the existing `unittest` style.

Add runtime tests covering:

- scene execution returns `bundleId` and `sceneUsdaPath`
- exported USDA exists
- serialized shape payload exists
- exported USDA writes stage metadata expected by this repo
- render bundle lookup rejects missing bundle ids

Add API tests covering:

- `/execute` returns the new bundle fields
- `/render` returns image URLs when the renderer helper succeeds
- `/render` returns `404` for unknown bundle ids
- artifact serving rejects missing or invalid artifact names

Do not require real CUDA rendering in normal unit tests. Stub the render helper in tests and keep GPU validation manual.

### Web

Use the existing `tsx --test` style.

Add lightweight UI tests covering:

- render button disabled state without a bundle id
- loading label during render
- error display on render failure
- rendered image display on success

Do not add browser E2E for v1.

## Manual verification

After implementation:

1. Generate a scene from `/demo/agent-genie-sim`.
2. Confirm the right panel shows a USDA path.
3. Confirm the bundle directory contains `scene.usda`, `shape.json`, and `manifest.json`.
4. Click `Render`.
5. Confirm a Mitsuba PNG appears in the preview panel.
6. Confirm the render request does not re-run scene execution.
7. Confirm render failure surfaces a readable error if CUDA rendering is unavailable.

## Out of scope

- automatic rendering after every scene update
- CPU Mitsuba fallback
- USDA download UX beyond showing the local path
- Mitsuba rendering that consumes USDA directly
- persistent artifact management beyond local temp storage
