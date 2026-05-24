# Zapdos Session Scene Overlay Design

**Goal:** Let an agent add, remove, and reposition assets inside the current `Zapdos` session without changing the meaning of `scene_usd` or `robot_usd`, while keeping new assets active in MuJoCo collisions and IsaacSim rendering.

## Decisions

- Keep `scene_usd` as the immutable base scene input and `robot_usd` as the immutable robot input.
- Add a session-local overlay layer that describes extra scene assets and per-body pose overrides.
- Compute a frontend-facing `scene_revision` from the base `scene_usd` fingerprint plus the normalized overlay payload.
- Rebuild the render bundle from `scene_usd + overlay` whenever overlay topology changes.
- Preserve robot state and editable body poses across bundle rebuilds.
- Reuse the existing `GenieSimAssets` search index and `/python/genie_sim/search_assets` behavior for asset discovery.
- Expose scene-editing actions as agent-callable Zapdos tools instead of building a drag-and-drop asset UI.
- Support both static and dynamic inserted assets, and make both selectable through the existing mouse transform controls.

## Why This Shape

- The current `Zapdos` pipeline already has one authoritative conversion path: USD scene input to MJCF for MuJoCo and render USDA for IsaacSim. Reusing that path keeps collisions, gravity, and rendering consistent.
- `scene_usd` and `robot_usd` already describe stable session inputs. Overloading either field with mutable edits would blur their purpose and make session identity harder to reason about.
- A session-local overlay gives the agent mutable scene editing without introducing a new persistent scene-management feature.
- Runtime spawn hooks for MuJoCo and IsaacSim would require parallel dynamic import paths and extra reconciliation logic. That is more complex than the current architecture needs.

## Current Constraints

- `apps/web/components/zapdos/ZapdosScene.tsx` loads visuals once with `get_visual()` and then only consumes pose updates from SSE.
- `apps/python/api/zapdos/{session}/{action}.py` treats editable bodies as objects that already exist in the current MuJoCo model.
- `apps/python/utils/rl_bundle.py` already converts a single scene USD into both MuJoCo and IsaacSim bundle outputs.
- `apps/python/utils/genie_sim_bundle.py` already contains the basic USDA payload-writing pattern needed to reference external assets.
- `apps/python/utils/usd_to_mjcf.py` already distinguishes kinematic bodies from free bodies and can emit freejoints for scene objects that should move dynamically.

## Overlay Model

Each Zapdos session should maintain an overlay state file in its temp directory, plus an in-memory copy on `ZapdosSession`.

```json
{
  "version": 1,
  "assets_root": "C:/path/to/GenieSimAssets",
  "instances": [
    {
      "id": "table_000_01",
      "asset_id": "table_000",
      "url": "objects/table_000/Aligned.usda",
      "motion": "static",
      "placement": {
        "kind": "floor_at_xy",
        "xy": [0.8, -0.2],
        "z_offset": 0.0,
        "yaw": 1.57
      }
    },
    {
      "id": "mug_001_01",
      "asset_id": "benchmark_mug_001",
      "url": "objects/benchmark_mug_001/Aligned.usda",
      "motion": "dynamic",
      "placement": {
        "kind": "on_top_of_body",
        "body": "Scene_table_000_01",
        "xy": [0.0, 0.15],
        "gap": 0.0,
        "yaw": 0.0
      }
    }
  ],
  "pose_overrides": {
    "Scene_table_000_01": {
      "pos": [0.8, -0.2, 0.75],
      "quat": [0.707, 0.0, 0.0, 0.707]
    }
  }
}
```

- `instances` describes overlay topology. Changing it changes `scene_revision`.
- `pose_overrides` stores edits made after an asset exists. Changing it does not change `scene_revision`.
- `id` is the stable overlay instance id.
- `motion` is `static` or `dynamic`.
- `url` is captured at insertion time so rebuilds do not depend on a later index lookup.
- Body naming for overlay objects should be deterministic, for example `Scene_<sanitized instance id>`.

## Revision Model

Use two separate derived values:

- `scene_revision`: hash of `scene_usd` fingerprint plus normalized `instances`
- `bundle_key`: hash of `robot_usd` fingerprint plus `scene_usd` fingerprint plus normalized `instances`

`scene_revision` exists for the frontend. If it changes, `ZapdosScene` must reload visuals.

`bundle_key` exists for backend caching and rebuilds. It remains aligned with the real render bundle dependency set.

Do not include `pose_overrides` in `scene_revision`. A pose edit should continue flowing through the existing pose update path and should not force full visual reload.

## Asset Semantics

### Static assets

- Participate in MuJoCo collision.
- Do not fall under gravity.
- Are still marked editable and can be moved with the existing transform controls.
- Should be emitted as fixed or kinematic scene bodies in the rebuilt simulation model.

### Dynamic assets

- Participate in MuJoCo collision.
- Fall under gravity unless explicitly placed on support.
- Are marked editable and can be moved with the existing transform controls.
- Should be emitted as free bodies with mass and a freejoint when the source USD supports rigid-body conversion.

In editor terms, both asset classes are editable bodies. "Static" means fixed in simulation, not locked in the page UI.

## Tool Contract

Add Zapdos session tools that the agent can call through the existing frontend tool plumbing.

### `search_assets`

- Reuse the current Genie Sim asset search behavior.
- Return `asset_id`, `description`, `url`, and `assets_root`.

### `list_placement_bodies`

- Return editable body names, labels, current world pose, and support metadata needed for placement.
- Include a coarse support descriptor such as world-space AABB or top surface height.

### `add_asset_to_scene`

Input:

```json
{
  "asset_id": "table_000",
  "motion": "static",
  "placement": {
    "kind": "floor_at_xy",
    "xy": [0.8, -0.2],
    "z_offset": 0.0,
    "yaw": 1.57
  }
}
```

- Resolve `asset_id` through the reusable asset index.
- Append a deterministic overlay instance.
- Rebuild the bundle and hot-swap the session runtime.
- Return the new overlay instance id, created body name, and current `scene_revision`.

### `remove_asset_from_scene`

- Remove one overlay instance by id.
- Delete related pose overrides for bodies owned by that instance.
- Rebuild the bundle and hot-swap the session runtime.

### Placement modes

Support these placement modes in v1:

- `floor_at_xy`
- `on_top_of_body`
- `world_pose`

This keeps the API agent-friendly without requiring a browser-side placement UI.

## Overlay USDA Generation

Add a small helper in `apps/python/utils` that builds a session-local composed scene USDA:

1. Create a new stage.
2. Reference the original `scene_usd` under the normal world root.
3. Add overlay object prims under `/World/Objects`.
4. For each overlay instance:
   - create an xform prim named from the stable instance id
   - add a payload/reference to the asset USDA
   - author translate and orient ops from the requested placement
   - author metadata needed by conversion to distinguish `static` from `dynamic`
5. Save the composed stage into the session temp directory.

The original `scene_usd` file remains untouched.

## Backend Runtime Flow

Extend `ZapdosSession` in `apps/python/api/zapdos/{session}/{action}.py`:

- Store `robot_usd`, `base_scene_usd`, `overlay_state`, `scene_revision`, and the composed scene path.
- Add a rebuild path that:
  - snapshots robot `qpos`, actuator targets, and editable body poses
  - regenerates the composed scene USDA from `base_scene_usd + overlay`
  - calls the existing render bundle pipeline
  - swaps in the new `model`, `data`, renderer, geoms, body map, editable body names, and assets
  - reapplies saved state to matching body names
  - updates `scene_revision`
- Emit an SSE event after a successful rebuild so the page knows topology changed.

Do not change the meaning of `init/start`. Session bootstrap still starts from `scene_usd` and optional `robot_usd`; the overlay is purely a session-local mutation layer after startup.

## Frontend Runtime Flow

Update `apps/web/components/zapdos/ZapdosScene.tsx` so the scene runtime can react to topology changes:

- Keep the initial `get_visual()` load.
- Continue applying normal pose-only SSE messages without clearing the scene.
- Watch for `scene_revision` in SSE payloads.
- If `scene_revision` changes:
  - clear the currently loaded top-level Three.js objects
  - call `get_visual()` again
  - preserve the current transform mode
  - clear invalid body selection if the selected body no longer exists

The existing mouse picking and `TransformControls` logic can remain body-based as long as overlay assets appear as editable bodies in `get_visual()`.

## Error Handling

- If asset lookup fails, return a 404-style error with the missing `asset_id`.
- If a requested placement body is missing, return a 400-style error instead of inserting at origin silently.
- If overlay rebuild fails, keep the old session runtime alive and return the rebuild error without partially swapping state.
- If a dynamic asset cannot be converted into a collidable rigid body, fail insertion instead of silently downgrading it to visual-only.
- If the renderer is rebuilding, reject overlapping scene-edit operations for that session.

## Testing

### Python

- Overlay state serialization and revision hashing.
- USDA overlay generation for `floor_at_xy`, `on_top_of_body`, and `world_pose`.
- Static versus dynamic metadata mapping into the rebuild pipeline.
- Session rebuild preserves robot state and editable body poses when body names are stable.
- Failed rebuild keeps the previous runtime active.
- `add_asset_to_scene` and `remove_asset_from_scene` update `scene_revision` only when topology changes.

### Web

- Scene runtime reloads visuals when SSE `scene_revision` changes.
- Pose-only SSE updates do not trigger a full visual reload.
- Selection clears when a removed body had been selected.
- New overlay bodies remain selectable through existing picking logic.

## Non-Goals

- Persistent saving back into the original `scene_usd`
- Browser-side drag-and-drop asset placement UI
- General scene history, undo, or multi-user collaboration
- Runtime dynamic spawning paths that bypass bundle rebuild
- Changes under `deps/genie_sim`
