# Zapdos Surface Pivot Controls Design

## Summary

Replace the current `OrbitControls` in `apps/web/app/demo/zapdos/page.tsx` with a custom control component that supports:

- trackball-style unlimited rotation
- left-button surface picking that updates the pivot without changing the current view
- middle-button drag for view panning
- wheel dolly along the current view direction

The design avoids extra on-screen UI and keeps page layout unchanged.

## Goals

- Keep the current scene compact, with no added panels, overlays, or margins.
- Support a strict pivot workflow: changing the pivot must not immediately rotate or reframe the camera.
- Keep files small by splitting math helpers from React and event glue.
- Limit picking to regular mesh surfaces loaded into the scene.

## Non-Goals

- No pivot marker or debugging HUD in v1.
- No object-selection state beyond updating the internal pivot.
- No support for `SparkSplat` picking in v1.
- No special behavior for orthographic cameras in v1.

## Chosen Approach

Do not build on `ArcballControls` or `OrbitControls`.

Those controls bind the orbit target to `camera.lookAt`, so updating the pivot would immediately change the view. That conflicts with the required strict pivot semantics.

Instead, implement a small custom controller that treats the camera transform as a rigid body:

- rotation moves both camera position and camera orientation around the pivot
- panning moves both camera position and pivot together
- dolly moves only camera position along the current forward direction

This preserves the current image when the pivot changes and still allows unlimited rotation.

## File Layout

- `apps/web/components/zapdos/SurfacePivotControls.tsx`
  - owns pointer and wheel events
  - stores controller refs and transient drag state
  - performs raycasts against the scene
  - applies transforms to the active perspective camera
- `apps/web/utils/surfacePivotMath.ts`
  - pure vector and quaternion helpers
  - click-vs-drag threshold helpers
  - no React or DOM dependencies

`apps/web/app/demo/zapdos/page.tsx` only swaps the control component and keeps scene composition unchanged.

## Interaction Model

### Left button

- On `pointerdown`, run a raycast immediately.
- If the raycast hits a regular mesh surface, store the hit point as the pending pivot.
- Do not move or rotate the camera on `pointerdown`.
- If the pointer is released without exceeding the drag threshold, commit the pending pivot and end the gesture.
- If movement exceeds the drag threshold, enter rotate mode and rotate around the most recently committed pivot. If `pointerdown` hit a surface, commit that pivot before the first rotation step.

This gives immediate surface intent detection while preserving the current view until an actual drag begins.

### Middle button

- On drag, pan in screen space.
- Translate camera position and pivot by the same world-space offset.
- Preserve camera orientation while panning.

### Wheel

- Dolly the camera along its current forward axis.
- Do not auto-focus or reorient toward the pivot.
- Clamp minimum camera-to-pivot distance to avoid passing through the pivot accidentally.

## Math Model

The math helper module in `apps/web/utils` will expose small pure functions:

- `isDragGesture(start, current, thresholdPx)`
- `rotateCameraAroundPivot(cameraPosition, cameraQuaternion, pivot, deltaX, deltaY, speed)`
- `panCameraAndPivot(camera, pivot, deltaX, deltaY, viewport, speed)`
- `dollyCamera(camera, pivot, deltaY, speed, minDistance, maxDistance)`

Rotation will be applied by composing quaternions for camera-local right-axis pitch and world- or camera-relative yaw, then rotating the camera position around the pivot with the same delta. The helper returns the next camera transform without mutating React state directly.

## Raycast Rules

- Only intersect visible `Mesh` instances.
- Ignore helper objects and non-mesh scene content.
- Ignore `SparkSplat` content in v1.
- If `pointerdown` misses all valid meshes, keep the existing pivot.

## Error Handling

- If the active camera is not a `PerspectiveCamera`, the custom controller becomes a no-op.
- If a gesture starts with an unsupported button, ignore it.
- If pointer capture fails or a pointer is canceled, clear transient gesture state.
- If the scene has no valid pick targets, the controller still supports rotate, pan, and dolly around the last pivot.

## Testing

Add unit tests under `apps/web` using the existing `pnpm test` setup.

Primary test coverage:

- changing pivot alone does not change camera position or quaternion
- rotation preserves pivot position and camera-to-pivot distance
- panning applies the same translation to camera and pivot
- drag threshold distinguishes click from rotate start
- dolly respects configured min and max distances

Manual verification on the demo page:

- left click on a mesh surface does not visibly move the camera
- left drag rotates freely without pole locking
- middle drag pans the view
- wheel dollies without snapping target or orientation

## Implementation Notes

- Keep `SurfacePivotControls.tsx` focused on event orchestration and camera mutation only.
- Keep math helpers in `apps/web/utils` so they can be tested without a canvas.
- Do not add visual controls, labels, or layout padding.
- Preserve the current lighting, models, SSE updates, and camera stream UI.
