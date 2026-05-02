# Zapdos Surface Pivot Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a custom rigid-body camera controller for the Zapdos demo that supports strict surface pivot picking, unlimited left-drag rotation, middle-drag pan, and wheel dolly without changing the view when the pivot is picked.

**Architecture:** Add a new `SurfacePivotControls` React component that owns canvas pointer and wheel events and mutates the active `PerspectiveCamera` directly. Keep reusable picking, gesture, and rigid-camera math in `apps/web/utils/surfacePivotMath.ts`, cover that utility with `node:test` tests, and limit the demo page change to swapping out `OrbitControls`.

**Tech Stack:** Next.js App Router, React 19, `@react-three/fiber`, `three`, `node:test` via `tsx --test`

---

## File Map

- Create: `apps/web/utils/surfacePivotMath.ts`
  - Pure helpers for pending-pivot commits, drag threshold checks, rigid rotation, panning, dolly, and mesh picking.
- Create: `apps/web/utils/surfacePivotMath.test.ts`
  - Focused unit tests for gesture semantics, camera transforms, dolly limits, and visible-mesh picking.
- Create: `apps/web/components/zapdos/SurfacePivotControls.tsx`
  - Canvas event glue that calls the utility helpers and applies the returned rig state to the active camera.
- Modify: `apps/web/app/demo/zapdos/page.tsx`
  - Remove `OrbitControls`, import the custom control component, keep the scene layout otherwise unchanged.

### Task 1: Build and Test the Surface Pivot Utility

**Files:**
- Create: `apps/web/utils/surfacePivotMath.test.ts`
- Create: `apps/web/utils/surfacePivotMath.ts`

- [ ] **Step 1: Write the failing test**

```ts
import assert from "node:assert/strict";
import test from "node:test";
import {
  BoxGeometry,
  Mesh,
  MeshBasicMaterial,
  PerspectiveCamera,
  Quaternion,
  Scene,
  Vector2,
  Vector3,
} from "three";

import {
  createRigState,
  dollyRig,
  finishGesture,
  isDragGesture,
  pickSurfacePoint,
  startGesture,
  updateGesture,
} from "./surfacePivotMath";

test("finishGesture commits a pending left-click pivot without moving the camera", () => {
  const rig = createRigState(
    new Vector3(2.5, -2.5, 1.8),
    new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), Math.PI / 6),
    new Vector3(0, 0, 0),
  );
  const gesture = startGesture(0, new Vector2(10, 10), new Vector3(1, 2, 3));

  const next = finishGesture(rig, gesture);

  assertVectorClose(next.position, rig.position);
  assertQuaternionClose(next.quaternion, rig.quaternion);
  assertVectorClose(next.pivot, new Vector3(1, 2, 3));
});

test("isDragGesture stays false below the threshold and flips true above it", () => {
  const start = new Vector2(10, 10);

  assert.equal(isDragGesture(start, new Vector2(12, 12), 4), false);
  assert.equal(isDragGesture(start, new Vector2(15, 10), 4), true);
});

test("updateGesture commits the pending pivot before the first left-drag rotation", () => {
  const pendingPivot = new Vector3(2, 0, 0);
  const rig = createRigState(
    new Vector3(0, 0, 5),
    new Quaternion(),
    new Vector3(0, 0, 0),
  );
  const gesture = startGesture(0, new Vector2(0, 0), pendingPivot);
  const originalDistance = rig.position.distanceTo(pendingPivot);

  const next = updateGesture(
    rig,
    gesture,
    new Vector2(40, 16),
    { width: 200, height: 100 },
    45,
    { dragThresholdPx: 4, rotateSpeed: 0.01, panSpeed: 1 },
  );

  assert.equal(next.changed, true);
  assert.equal(next.gesture.dragging, true);
  assert.equal(next.gesture.pendingPivot, null);
  assertVectorClose(next.rig.pivot, pendingPivot);
  assertNumberClose(next.rig.position.distanceTo(next.rig.pivot), originalDistance);
});

test("middle drag pans the camera and pivot by the same world offset", () => {
  const rig = createRigState(
    new Vector3(0, 0, 5),
    new Quaternion(),
    new Vector3(0, 0, 0),
  );
  const gesture = startGesture(1, new Vector2(10, 10), null);

  const next = updateGesture(
    rig,
    gesture,
    new Vector2(30, 20),
    { width: 200, height: 100 },
    45,
    { dragThresholdPx: 4, rotateSpeed: 0.01, panSpeed: 1 },
  );

  const cameraDelta = next.rig.position.clone().sub(rig.position);
  const pivotDelta = next.rig.pivot.clone().sub(rig.pivot);

  assert.equal(next.changed, true);
  assert.equal(next.gesture.dragging, true);
  assertVectorClose(cameraDelta, pivotDelta);
  assertQuaternionClose(next.rig.quaternion, rig.quaternion);
});

test("dollyRig keeps the camera-to-pivot distance inside the configured bounds", () => {
  const rig = createRigState(
    new Vector3(0, 0, 5),
    new Quaternion(),
    new Vector3(0, 0, 0),
  );

  const zoomIn = dollyRig(rig, -1000, 0.002, 2, 10);
  const zoomOut = dollyRig(rig, 1000, 0.002, 2, 6);

  assert.ok(zoomIn.position.distanceTo(zoomIn.pivot) >= 2 - 1e-6);
  assert.ok(zoomOut.position.distanceTo(zoomOut.pivot) <= 6 + 1e-6);
});

test("pickSurfacePoint ignores hidden meshes and returns the first visible mesh hit", () => {
  const scene = new Scene();
  const hidden = new Mesh(new BoxGeometry(1, 1, 1), new MeshBasicMaterial());
  hidden.position.set(0, 0, 2);
  hidden.visible = false;

  const visible = new Mesh(new BoxGeometry(1, 1, 1), new MeshBasicMaterial());

  scene.add(hidden);
  scene.add(visible);
  scene.updateMatrixWorld(true);

  const camera = new PerspectiveCamera(45, 1, 0.1, 100);
  camera.position.set(0, 0, 5);
  camera.lookAt(0, 0, 0);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);

  const hit = pickSurfacePoint(scene, camera, new Vector2(0, 0));

  assert.ok(hit instanceof Vector3);
  assert.ok(hit.distanceTo(new Vector3(0, 0, 0.5)) < 0.25);
});

function assertVectorClose(actual: Vector3, expected: Vector3) {
  assertNumberClose(actual.x, expected.x);
  assertNumberClose(actual.y, expected.y);
  assertNumberClose(actual.z, expected.z);
}

function assertQuaternionClose(actual: Quaternion, expected: Quaternion) {
  assertNumberClose(actual.x, expected.x);
  assertNumberClose(actual.y, expected.y);
  assertNumberClose(actual.z, expected.z);
  assertNumberClose(actual.w, expected.w);
}

function assertNumberClose(actual: number, expected: number, epsilon = 1e-6) {
  assert.ok(Math.abs(actual - expected) <= epsilon, `${actual} !~= ${expected}`);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm --dir apps/web test -- utils/surfacePivotMath.test.ts`

Expected: FAIL with a module resolution error for `./surfacePivotMath` or missing exported functions.

- [ ] **Step 3: Write minimal implementation**

```ts
import {
  MathUtils,
  Mesh,
  PerspectiveCamera,
  Quaternion,
  Raycaster,
  Scene,
  Vector2,
  Vector3,
} from "three";

export interface SurfacePivotRig {
  position: Vector3;
  quaternion: Quaternion;
  pivot: Vector3;
}

export interface SurfacePivotGesture {
  button: 0 | 1;
  start: Vector2;
  last: Vector2;
  pendingPivot: Vector3 | null;
  dragging: boolean;
}

interface ViewportSize {
  width: number;
  height: number;
}

interface GestureConfig {
  dragThresholdPx: number;
  rotateSpeed: number;
  panSpeed: number;
}

interface GestureUpdate {
  rig: SurfacePivotRig;
  gesture: SurfacePivotGesture;
  changed: boolean;
}

export function createRigState(
  position: Vector3,
  quaternion: Quaternion,
  pivot: Vector3,
): SurfacePivotRig {
  return {
    position: position.clone(),
    quaternion: quaternion.clone(),
    pivot: pivot.clone(),
  };
}

export function startGesture(
  button: 0 | 1,
  start: Vector2,
  pendingPivot: Vector3 | null,
): SurfacePivotGesture {
  return {
    button,
    start: start.clone(),
    last: start.clone(),
    pendingPivot: pendingPivot?.clone() ?? null,
    dragging: false,
  };
}

export function finishGesture(
  rig: SurfacePivotRig,
  gesture: SurfacePivotGesture,
): SurfacePivotRig {
  if (gesture.button === 0 && !gesture.dragging && gesture.pendingPivot) {
    return setPivotWithoutMovingCamera(rig, gesture.pendingPivot);
  }
  return createRigState(rig.position, rig.quaternion, rig.pivot);
}

export function isDragGesture(
  start: Vector2,
  current: Vector2,
  thresholdPx: number,
): boolean {
  return start.distanceToSquared(current) >= thresholdPx * thresholdPx;
}

export function updateGesture(
  rig: SurfacePivotRig,
  gesture: SurfacePivotGesture,
  point: Vector2,
  viewport: ViewportSize,
  fov: number,
  config: GestureConfig,
): GestureUpdate {
  const nextGesture: SurfacePivotGesture = {
    ...gesture,
    last: point.clone(),
  };
  const deltaX = point.x - gesture.last.x;
  const deltaY = point.y - gesture.last.y;
  const startedDragging =
    gesture.dragging || isDragGesture(gesture.start, point, config.dragThresholdPx);

  if (!startedDragging) {
    return {
      rig: createRigState(rig.position, rig.quaternion, rig.pivot),
      gesture: nextGesture,
      changed: false,
    };
  }

  if (gesture.button === 0) {
    const baseRig =
      !gesture.dragging && gesture.pendingPivot
        ? setPivotWithoutMovingCamera(rig, gesture.pendingPivot)
        : createRigState(rig.position, rig.quaternion, rig.pivot);

    return {
      rig: rotateCameraAroundPivot(baseRig, deltaX, deltaY, config.rotateSpeed),
      gesture: {
        ...nextGesture,
        dragging: true,
        pendingPivot: null,
      },
      changed: true,
    };
  }

  return {
    rig: panCameraAndPivot(rig, deltaX, deltaY, viewport, fov, config.panSpeed),
    gesture: {
      ...nextGesture,
      dragging: true,
    },
    changed: true,
  };
}

export function dollyRig(
  rig: SurfacePivotRig,
  deltaY: number,
  speed: number,
  minDistance: number,
  maxDistance: number,
): SurfacePivotRig {
  const distance = rig.position.distanceTo(rig.pivot);
  const forward = new Vector3(0, 0, -1).applyQuaternion(rig.quaternion).normalize();
  const desiredStep = -deltaY * speed * Math.max(distance, minDistance);
  const proposedPosition = rig.position.clone().addScaledVector(forward, desiredStep);
  const proposedDistance = proposedPosition.distanceTo(rig.pivot);

  if (proposedDistance < minDistance || proposedDistance > maxDistance) {
    return createRigState(rig.position, rig.quaternion, rig.pivot);
  }

  return {
    position: proposedPosition,
    quaternion: rig.quaternion.clone(),
    pivot: rig.pivot.clone(),
  };
}

export function pickSurfacePoint(
  scene: Scene,
  camera: PerspectiveCamera,
  pointerNdc: Vector2,
): Vector3 | null {
  const raycaster = new Raycaster();
  raycaster.near = camera.near;
  raycaster.far = camera.far;
  raycaster.setFromCamera(pointerNdc, camera);

  for (const hit of raycaster.intersectObjects(scene.children, true)) {
    if (hit.object instanceof Mesh && hit.object.visible) {
      return hit.point.clone();
    }
  }

  return null;
}

function setPivotWithoutMovingCamera(
  rig: SurfacePivotRig,
  nextPivot: Vector3,
): SurfacePivotRig {
  return {
    position: rig.position.clone(),
    quaternion: rig.quaternion.clone(),
    pivot: nextPivot.clone(),
  };
}

function rotateCameraAroundPivot(
  rig: SurfacePivotRig,
  deltaX: number,
  deltaY: number,
  speed: number,
): SurfacePivotRig {
  const up = new Vector3(0, 1, 0).applyQuaternion(rig.quaternion).normalize();
  const right = new Vector3(1, 0, 0).applyQuaternion(rig.quaternion).normalize();
  const yaw = new Quaternion().setFromAxisAngle(up, -deltaX * speed);
  const pitch = new Quaternion().setFromAxisAngle(right, -deltaY * speed);
  const rotation = yaw.multiply(pitch).normalize();
  const offset = rig.position.clone().sub(rig.pivot).applyQuaternion(rotation);

  return {
    position: rig.pivot.clone().add(offset),
    quaternion: rotation.multiply(rig.quaternion.clone()).normalize(),
    pivot: rig.pivot.clone(),
  };
}

function panCameraAndPivot(
  rig: SurfacePivotRig,
  deltaX: number,
  deltaY: number,
  viewport: ViewportSize,
  fov: number,
  speed: number,
): SurfacePivotRig {
  const safeHeight = Math.max(viewport.height, 1);
  const safeWidth = Math.max(viewport.width, 1);
  const distance = rig.position.distanceTo(rig.pivot);
  const worldHeight = 2 * Math.tan(MathUtils.degToRad(fov) / 2) * distance;
  const worldWidth = worldHeight * (safeWidth / safeHeight);
  const right = new Vector3(1, 0, 0).applyQuaternion(rig.quaternion).normalize();
  const up = new Vector3(0, 1, 0).applyQuaternion(rig.quaternion).normalize();
  const translation = right
    .multiplyScalar((-deltaX / safeWidth) * worldWidth * speed)
    .add(up.multiplyScalar((deltaY / safeHeight) * worldHeight * speed));

  return {
    position: rig.position.clone().add(translation),
    quaternion: rig.quaternion.clone(),
    pivot: rig.pivot.clone().add(translation),
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm --dir apps/web test -- utils/surfacePivotMath.test.ts`

Expected: PASS with all tests green.

- [ ] **Step 5: Commit**

```bash
git add apps/web/utils/surfacePivotMath.ts apps/web/utils/surfacePivotMath.test.ts
git commit -m "add zapdos surface pivot math"
```

### Task 2: Add the React Control Component and Wire It Into the Demo

**Files:**
- Create: `apps/web/components/zapdos/SurfacePivotControls.tsx`
- Modify: `apps/web/app/demo/zapdos/page.tsx`

- [ ] **Step 1: Write the failing integration change**

Update `apps/web/app/demo/zapdos/page.tsx` so it stops importing `OrbitControls` and references the new component instead:

```tsx
import { Environment, Lightformer } from "@react-three/drei";

import { SurfacePivotControls } from "../../../components/zapdos/SurfacePivotControls";
```

Replace the control inside the canvas:

```tsx
<SurfacePivotControls />
```

- [ ] **Step 2: Run integration verification to confirm it fails first**

Run: `pnpm --dir apps/web exec tsc --noEmit`

Expected: FAIL with `Cannot find module '../../../components/zapdos/SurfacePivotControls'`.

- [ ] **Step 3: Write minimal implementation**

Create `apps/web/components/zapdos/SurfacePivotControls.tsx`:

```tsx
'use client'

import { useEffect, useRef } from "react";
import { useThree } from "@react-three/fiber";
import { PerspectiveCamera, Vector2, Vector3 } from "three";

import {
  createRigState,
  dollyRig,
  finishGesture,
  pickSurfacePoint,
  startGesture,
  type SurfacePivotGesture,
  updateGesture,
} from "../../utils/surfacePivotMath";

const DRAG_THRESHOLD_PX = 4;
const ROTATE_SPEED = 0.01;
const PAN_SPEED = 1;
const DOLLY_SPEED = 0.002;
const MIN_DISTANCE = 0.1;
const MAX_DISTANCE = 100;

export function SurfacePivotControls() {
  const camera = useThree((state) => state.camera);
  const gl = useThree((state) => state.gl);
  const scene = useThree((state) => state.scene);
  const invalidate = useThree((state) => state.invalidate);
  const gestureRef = useRef<SurfacePivotGesture | null>(null);
  const pivotRef = useRef(new Vector3(0, 0, 0));

  useEffect(() => {
    if (!(camera instanceof PerspectiveCamera)) {
      return;
    }

    const element = gl.domElement;
    camera.lookAt(pivotRef.current);
    camera.updateMatrixWorld();
    invalidate();

    const applyRig = (rig: ReturnType<typeof createRigState>) => {
      camera.position.copy(rig.position);
      camera.quaternion.copy(rig.quaternion);
      pivotRef.current.copy(rig.pivot);
      camera.updateMatrixWorld();
      invalidate();
    };

    const toScreenPoint = (event: PointerEvent) =>
      new Vector2(event.clientX, event.clientY);

    const toNdc = (event: PointerEvent) => {
      const rect = element.getBoundingClientRect();
      return new Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      );
    };

    const onPointerDown = (event: PointerEvent) => {
      if (event.button !== 0 && event.button !== 1) {
        return;
      }

      element.setPointerCapture(event.pointerId);
      const hitPoint =
        event.button === 0 ? pickSurfacePoint(scene, camera, toNdc(event)) : null;
      gestureRef.current = startGesture(
        event.button as 0 | 1,
        toScreenPoint(event),
        hitPoint,
      );
    };

    const onPointerMove = (event: PointerEvent) => {
      const gesture = gestureRef.current;
      if (!gesture) {
        return;
      }

      const rig = createRigState(camera.position, camera.quaternion, pivotRef.current);
      const next = updateGesture(
        rig,
        gesture,
        toScreenPoint(event),
        { width: element.clientWidth, height: element.clientHeight },
        camera.fov,
        {
          dragThresholdPx: DRAG_THRESHOLD_PX,
          rotateSpeed: ROTATE_SPEED,
          panSpeed: PAN_SPEED,
        },
      );

      gestureRef.current = next.gesture;
      if (next.changed) {
        applyRig(next.rig);
      }
    };

    const onPointerUp = (event: PointerEvent) => {
      const gesture = gestureRef.current;
      if (!gesture) {
        return;
      }

      const rig = createRigState(camera.position, camera.quaternion, pivotRef.current);
      applyRig(finishGesture(rig, gesture));
      gestureRef.current = null;

      if (element.hasPointerCapture(event.pointerId)) {
        element.releasePointerCapture(event.pointerId);
      }
    };

    const onPointerCancel = (event: PointerEvent) => {
      gestureRef.current = null;
      if (element.hasPointerCapture(event.pointerId)) {
        element.releasePointerCapture(event.pointerId);
      }
    };

    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rig = createRigState(camera.position, camera.quaternion, pivotRef.current);
      applyRig(dollyRig(rig, event.deltaY, DOLLY_SPEED, MIN_DISTANCE, MAX_DISTANCE));
    };

    element.addEventListener("pointerdown", onPointerDown);
    element.addEventListener("pointermove", onPointerMove);
    element.addEventListener("pointerup", onPointerUp);
    element.addEventListener("pointercancel", onPointerCancel);
    element.addEventListener("wheel", onWheel, { passive: false });

    return () => {
      element.removeEventListener("pointerdown", onPointerDown);
      element.removeEventListener("pointermove", onPointerMove);
      element.removeEventListener("pointerup", onPointerUp);
      element.removeEventListener("pointercancel", onPointerCancel);
      element.removeEventListener("wheel", onWheel);
    };
  }, [camera, gl, invalidate, scene]);

  return null;
}
```

Keep the final `apps/web/app/demo/zapdos/page.tsx` control-related edits limited to:

```tsx
import { Environment, Lightformer } from "@react-three/drei";

import { SurfacePivotControls } from "../../../components/zapdos/SurfacePivotControls";
```

and:

```tsx
<SurfacePivotControls />
```

- [ ] **Step 4: Run verification to confirm the integration works**

Run:

`pnpm --dir apps/web test -- utils/surfacePivotMath.test.ts`

`pnpm --dir apps/web exec tsc --noEmit`

Expected: both commands PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/web/app/demo/zapdos/page.tsx apps/web/components/zapdos/SurfacePivotControls.tsx
git commit -m "add zapdos surface pivot controls"
```

### Task 3: Verify Behavior in the Demo

**Files:**
- No code changes required if Task 2 passes cleanly.

- [ ] **Step 1: Run the focused automated checks again from a clean tree**

Run:

`pnpm --dir apps/web test -- utils/surfacePivotMath.test.ts`

`pnpm --dir apps/web exec tsc --noEmit`

Expected: both commands PASS with no new failures.

- [ ] **Step 2: Run the web app for manual verification**

Run: `pnpm --dir apps/web dev`

Open: `http://127.0.0.1:3000/demo/zapdos`

Manual checks:

- left-click a visible mesh surface and confirm the image does not jump
- left-drag after picking and confirm the camera rotates freely without pole locking
- middle-drag and confirm camera and pivot move together
- wheel forward/back and confirm dolly works without snapping the camera orientation
- click empty space and confirm the existing pivot is preserved

- [ ] **Step 3: Confirm the tree is clean except for intentional feature files**

Run: `git status --short`

Expected: no unexpected modifications beyond the files in this plan.

## Self-Review

- Spec coverage:
  - strict surface-pivot picking without view jump: Task 1 `finishGesture` test + Task 2 component wiring
  - left-button `pointerdown` raycast: Task 1 `pickSurfacePoint` test + Task 2 `onPointerDown`
  - unlimited rotation: Task 1 rigid rotation test + Task 2 left-drag event flow
  - middle-button pan: Task 1 pan test + Task 2 middle-button pointer flow
  - wheel dolly and bounds: Task 1 dolly test + Task 2 wheel handler
  - compact page with no extra UI: Task 2 limits page edits to swapping the control component
- Placeholder scan: no `TODO`, `TBD`, or unresolved references remain.
- Type consistency:
  - `SurfacePivotRig`, `SurfacePivotGesture`, `createRigState`, `startGesture`, `updateGesture`, `finishGesture`, `dollyRig`, and `pickSurfacePoint` are named consistently across tasks.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-02-zapdos-surface-pivot-controls.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
