# Zapdos Wheel Pivot Anchor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the current pivot fixed at the same screen position while wheel zooming in the Zapdos surface pivot controls.

**Architecture:** Extend the existing pure camera math in `apps/web/utils/surfacePivotMath.ts` with a zoom helper that performs the current dolly step and then applies a screen-space compensation pan for the pivot. Keep `apps/web/components/zapdos/SurfacePivotControls.tsx` limited to event wiring by swapping the wheel handler to that helper, and prove the behavior with a focused regression test in `apps/web/utils/surfacePivotMath.test.ts`.

**Tech Stack:** Next.js App Router, React 19, `@react-three/fiber`, `three`, `node:test`, `tsx`

---

## File Map

- Modify: `apps/web/utils/surfacePivotMath.test.ts`
  - add a regression test for pivot screen-position preservation during wheel zoom
- Modify: `apps/web/utils/surfacePivotMath.ts`
  - add a pure anchored wheel-zoom helper built on top of the existing dolly math
- Modify: `apps/web/components/zapdos/SurfacePivotControls.tsx`
  - switch the wheel handler to the new helper

### Task 1: Add the Regression Test

**Files:**
- Modify: `apps/web/utils/surfacePivotMath.test.ts`

- [ ] **Step 1: Write the failing test**

Add a test that creates an off-center pivot, runs the new anchored zoom helper, and asserts the pivot's NDC coordinates stay unchanged:

```ts
test("zoomRig keeps the pivot at the same screen position during wheel zoom", () => {
  const rig = createRigState(
    new Vector3(2.5, -2.5, 1.8),
    new PerspectiveCamera(45, 1, 0.01, 100).quaternion,
    new Vector3(1.2, 0.6, 0.5),
  );
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm --dir apps/web test -- utils/surfacePivotMath.test.ts`

Expected: FAIL because the anchored zoom helper does not exist yet.

### Task 2: Implement Anchored Wheel Zoom

**Files:**
- Modify: `apps/web/utils/surfacePivotMath.ts`
- Modify: `apps/web/components/zapdos/SurfacePivotControls.tsx`

- [ ] **Step 1: Write the minimal implementation**

Add a pure helper in `apps/web/utils/surfacePivotMath.ts` that:

```ts
const dolly = dollyRig(rig, deltaY, speed, minDistance, maxDistance);
```

then computes the pre/post pivot projection and translates both camera position and pivot along camera-local right and up to cancel the NDC error before returning the next rig.

- [ ] **Step 2: Wire the wheel handler**

Update `apps/web/components/zapdos/SurfacePivotControls.tsx` so `onWheel` calls the new helper instead of calling `dollyRig` directly.

- [ ] **Step 3: Run verification to confirm it passes**

Run:

`pnpm --dir apps/web test -- utils/surfacePivotMath.test.ts`

`pnpm --dir apps/web exec tsc --noEmit`

Expected: both commands PASS.

## Self-Review

- Spec coverage:
  - wheel zoom preserves the pivot screen position: Task 1 regression test + Task 2 helper and wiring
- Placeholder scan:
  - no `TODO`, `TBD`, or unresolved names remain
- Type consistency:
  - the wheel helper is defined in `surfacePivotMath.ts` and consumed from `SurfacePivotControls.tsx`
