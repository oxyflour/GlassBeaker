# Genie Sim Zapdos Import Design

**Goal:** Let a generated `GenieSim` scene open directly in the existing `Zapdos` demo so the user can move from scene generation to robot simulation without manual file-path wiring.

## Decisions

- Keep `scene.usda` as the handoff artifact between `GenieSim` and `Zapdos`.
- Use URL query parameters for v1 handoff. `Zapdos` should accept `scene_usd` and continue to support optional `robot_usd`.
- Add an `Open in Zapdos` entry point in `agent-genie-sim` once a scene has been generated successfully.
- Treat the `Zapdos` browser session id as input-specific. Different `scene_usd` values must not reuse the same backend session.
- Do not build a new upload flow or copy USDA contents through the browser. The handoff is path-based only.

## Why This Shape

- `GenieSim` already exports a usable `scene.usda`; reusing it avoids inventing another intermediate format.
- `Zapdos` already knows how to consume an external `scene_usd` on session startup. The missing piece is exposing that input from the frontend.
- The current fixed `useLocalUUID("zapdos-session")` is a correctness bug for imports. If the user opens a second scene, the existing session can silently keep simulating the first one.
- A path-based URL handoff is the smallest change that produces a real end-to-end workflow.

## Current Constraints

- `GenieSim` now returns `sceneUsdaPath` from `/python/genie_sim/execute`.
- `Zapdos` session creation already reads `scene_usd` and `robot_usd` from the request query string.
- `Zapdos` session state is keyed only by `sess`.
- The `Zapdos` page currently creates `sess` from a fixed local-storage key, so all scenes share one session namespace.

## URL Contract

`/demo/zapdos` should support:

- `scene_usd=<absolute-or-repo-relative-usd-path>`
- `robot_usd=<absolute-or-repo-relative-usd-path>` optional

Example:

`/demo/zapdos?scene_usd=C%3A%2FProjects%2FGlassBeaker%2Fapps%2Fpython%2Ftmp%2Fgenie_sim%2Fabc123%2Fscene.usda`

The page does not need to parse USDA contents. It only needs to preserve and forward the path.

## Frontend

### Genie Sim page

`apps/web/app/demo/agent-genie-sim/page.tsx` should expose a single handoff action after scene generation:

- label: `Open in Zapdos`
- target: `/demo/zapdos?scene_usd=${encodeURIComponent(scene.sceneUsdaPath)}`

The existing USDA floating badge remains the passive path display. The new action is the active handoff.

### Zapdos page

`apps/web/app/demo/zapdos/page.tsx` should:

- read `scene_usd` and `robot_usd` from `useSearchParams()`
- build a stable query suffix for the init request
- pass that suffix to `/python/zapdos/${sess}/init/start`

The important behavior change is session identity:

- current: one local-storage session key for all scenes
- target: session key derived from the import inputs

Recommended shape:

- base key prefix: `zapdos-session`
- suffix: normalized `scene_usd` and `robot_usd`
- example local-storage key: `zapdos-session|<scene_usd>|<robot_usd-or-default>`

This gives:

- same scene path -> same browser session id
- different scene path -> different browser session id

That is enough to prevent stale-scene reuse in normal navigation.

### UX states

`ZapdosInit` should keep its current loading gate, but it needs one extra state:

- `loading`
- `started`
- `error`

If session bootstrap fails because the USDA path is invalid or bundle generation crashes, the page should show a readable error instead of hanging forever.

## Backend

### Session bootstrap

`apps/python/api/zapdos/{session}/{action}.py` already accepts `scene_usd` and `robot_usd` during session creation. For v1, keep that contract.

The backend should be tightened in two small ways:

- If session creation fails, do not leave a failed future permanently cached in `sessions[sess]`.
- The `init` stream should surface bootstrap failure in a user-readable way instead of only terminating the `EventSource`.

The backend does not need a new endpoint if the existing `init/start` flow can emit:

- `loading`
- `started`
- `error: <detail>`

### Path rules

Keep the current path resolution policy:

- absolute paths are accepted directly
- relative paths resolve from repo root
- missing files return `404`

Do not add file-upload handling in this task.

## Data Flow

1. User generates a scene in `agent-genie-sim`.
2. `/python/genie_sim/execute` returns `sceneUsdaPath`.
3. User clicks `Open in Zapdos`.
4. Browser navigates to `/demo/zapdos?scene_usd=<encoded path>`.
5. `Zapdos` derives an input-specific session id.
6. `ZapdosInit` opens `/python/zapdos/${sess}/init/start?scene_usd=...`.
7. Backend creates a render bundle from the selected scene USDA and default robot USDA.
8. After `started`, the existing `Zapdos` rendering, SSE pose updates, camera streams, and teleop continue unchanged.

## Testing

### Web

Add focused tests for:

- handoff URL generation from `scene.sceneUsdaPath`
- session-key derivation changes when `scene_usd` changes
- `ZapdosInit` request includes the encoded `scene_usd`
- bootstrap error state rendering

Keep the tests lightweight and helper-driven. Do not add browser E2E.

### Python

Add API-level tests for:

- `_input_path()` accepts absolute and repo-relative `scene_usd`
- failed session bootstrap can be retried instead of poisoning `sessions[sess]`
- `init/start` returns a readable error signal on bootstrap failure

Do not require MuJoCo or full renderer startup in the new unit tests. Stub session creation where possible.

## Manual Verification

1. Generate a scene in `/demo/agent-genie-sim`.
2. Click `Open in Zapdos`.
3. Confirm the `Zapdos` page starts with the generated scene instead of `default_scene.usda`.
4. Generate a second scene and open it in `Zapdos`.
5. Confirm the second navigation does not reuse the first scene's backend session.
6. Load `/demo/zapdos` with an invalid `scene_usd` and confirm the page shows a readable failure state.

## Out of Scope

- uploading local USDA files through the browser
- browsing previously generated scenes from a history list
- changing the default robot selection UI
- persistent scene/session management beyond current temp files
- automatic cleanup of old `GenieSim` scene directories
