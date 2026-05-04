# Zapdos Camera Override Save Design

**Goal:** Let the Zapdos page persist the current IsaacSim camera calibration into `%USERPROFILE%\.glass-beaker\config.json` under `override.camera`, without adding page-side editing or import/export flows.

## Decisions

- The Zapdos page adds a single save action for camera calibration.
- Calibration is adjusted in the IsaacSim UI, not in Zapdos.
- Save reads the current camera prim state from the running IsaacSim renderer process, not from MuJoCo.
- Saved overrides are written to `%USERPROFILE%\.glass-beaker\config.json`.
- Override keys are grouped by camera parent prim path, then by `camera_name`.
- Saved overrides are applied during later render bundle generation so the next session starts with the calibrated camera values.
- Saving does not restart the current session and does not modify the currently running renderer beyond reading its state.

## Config Shape

```json
{
  "override": {
    "camera": {
      "/MyRobot/Root_r1_pro_with_gripper_zed_link": {
        "head_camera": {
          "pos": [0.0, 0.0, 0.0],
          "quat": [0.0, 0.0, 1.0, 0.0],
          "fovy": 45.0,
          "horizontal_aperture": 32.0,
          "vertical_aperture": 24.0,
          "clipping_range": [0.01, 100.0]
        }
      }
    }
  }
}
```

- The first key is the camera parent prim path in bundle-relative form, for example `/MyRobot/...`.
- The second key is the camera name, matching `RenderCamera.name`.
- `pos` and `quat` are the camera's local transform relative to its parent prim.
- `fovy` is stored explicitly even though it can be derived from focal length and aperture, so the saved config matches the render bundle model.
- Aperture and clipping range are stored too so the config can round-trip Isaac values without silently discarding them.

## Backend

- Add a small user config helper in `apps/python` that:
  - resolves `%USERPROFILE%\.glass-beaker\config.json`
  - reads missing or malformed files defensively
  - merges `override.camera` updates without discarding unrelated config fields
  - writes UTF-8 JSON with indentation
- Extend the local Isaac renderer wrapper so the Python service can request a snapshot of the current camera prim parameters from the running IsaacSim process.
- The snapshot payload should include, per camera:
  - `name`
  - `prim`
  - `parent_prim`
  - `pos`
  - `quat`
  - `focal_length`
  - `horizontal_aperture`
  - `vertical_aperture`
  - `clipping_range`
  - derived `fovy`
- Add a Zapdos backend call that:
  - asks the renderer for the current camera snapshot
  - writes the result into `override.camera`
  - returns a compact status payload with the config path and saved camera count

## Bundle Generation

- Extend the render camera model so bundle generation can consume per-camera overrides for:
  - local pose
  - `fovy`
  - horizontal aperture
  - vertical aperture
  - clipping range
- Resolve overrides by matching:
  - `parent_prim_path = PurePosixPath(camera.prim).parent`
  - `camera_name = camera.name`
- Apply overrides after the MuJoCo-derived camera list is built but before the USD camera prims are written.
- Keep override application local to GlassBeaker code. Do not modify `deps/genie_sim`.

## Frontend

- Add a compact `Save camera override` button to the Zapdos page near the existing runtime controls.
- On click, POST to the new Zapdos backend call for the active session.
- Show a small success or error status message in-page.
- Do not add JSON editors, upload inputs, or camera tuning controls.

## Error Handling

- If the renderer is not ready or no session exists, return a clear 409-style runtime error.
- If the renderer snapshot is missing a requested prim or has incomplete camera attributes, fail the save instead of writing partial data.
- If `%USERPROFILE%` is unavailable or the config file cannot be written, surface the filesystem error to the page.
- If the config file exists but is not a JSON object, treat that as an error instead of overwriting it blindly.

## Testing

- Backend tests cover:
  - user config read/write and merge behavior
  - save endpoint error handling for missing renderer or bad config state
  - override lookup by parent prim path and camera name
  - bundle camera override application
- Frontend tests cover:
  - save request dispatch
  - success message rendering
  - error message rendering

## Non-Goals

- No live camera editing controls in Zapdos
- No import/export UI
- No automatic reloading of the current session after save
- No changes under `deps/genie_sim`
