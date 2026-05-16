# Zapdos Place The Apple Button Design

**Goal:** Add a `Place the apple` control to the Zapdos debug overlay that releases the currently grabbed apple in place by opening the gripper and detaching the logical attachment, without introducing a new placement planner or torso motion.

## Decisions

- Add a new `PlaceTheAppleButton` beside the existing `GrabTheAppleButton` in the debug popover of `ZapdosTopOverlay`.
- Add a dedicated frontend action module that posts to `/python/zapdos/${sess}/call/place_apple`.
- Add a dedicated backend `place_apple()` runtime action instead of overloading `pick_object` or introducing a generic release API in this change.
- Treat "place" as an in-place release only:
  - keep the current end-effector pose
  - open the active gripper
  - detach the apple from the gripper link
- Keep the action scoped to the canned apple flow:
  - target query remains the apple selected by the existing `grab_apple` path
  - no new user-provided target pose or support query
- Do not move the torso or introduce a new approach, retreat, or support-aware place planner.

## Frontend Structure

- Add `apps/web/components/zapdos/place-the-apple.ts` that mirrors `grab-the-apple.ts`:
  - request builder returns the standard empty manipulation payload
  - action posts to the `place_apple` route
  - response shape matches the existing canned manipulation actions
- Add `apps/web/components/zapdos/PlaceTheAppleButton.tsx` that mirrors `GrabTheAppleButton.tsx`:
  - local `busy`, `message`, and `error` state
  - disabled while busy or when `sess` is empty
  - success message reports the released target body
- Update `ZapdosTopOverlay.tsx` to render the new button in the existing debug panel.

## Backend Structure

- Add `place_apple()` to `ManipulationRuntime`.
- `place_apple()` resolves the same apple target used by `grab_apple()` so the route stays deterministic.
- Before releasing, verify that the apple is currently attached. If not, return a clear `409` error instead of pretending the release succeeded.
- Build a minimal release plan that:
  - uses the same arm as `grab_apple`
  - targets the apple body for result reporting
  - contains a short `open_gripper` stage
  - marks the plan as a release action so the executor detaches instead of attaches
- Add `place_apple` dispatch in `ZapdosSession.call_once`.

## Executor Behavior

- Extend `PickExecutor.execute()` to support a release-style plan in addition to the existing pick-style flow.
- A release plan must:
  - hold the current end-effector pose
  - drive the gripper open for a small number of steps
  - detach the target body after the gripper-open stage completes
- Release execution must not require `descend_to_grasp`, attach proximity checks, or post-release retreat motion.
- If the target body is not attached at release time, the executor should raise a `409` error with a clear message.
- Keep the existing pick path behavior unchanged.

## Error Handling

- `place_apple` returns `409` when the apple is not currently attached.
- Frontend surfaces backend error text in the button status area, matching `GrabTheAppleButton`.
- The action is idempotent only in the failure-reporting sense:
  - first successful release detaches the apple
  - subsequent clicks fail clearly until the apple is grabbed again

## Testing

- Add a frontend route test for `place-the-apple.ts` mirroring `grab-the-apple.test.ts`.
- Add executor tests covering:
  - successful release opens the gripper, detaches the target, and reports success
  - release fails with `409` when the target is not attached
- Add runtime or import tests covering:
  - `ZapdosSession.call_once(..., "place_apple", ...)` dispatch
  - `ManipulationRuntime.place_apple()` builds and executes the release flow

## Non-Goals

- No generic `place_object` API
- No support for releasing arbitrary attached bodies
- No target-pose placement on the table
- No physics-only release path that relies on gripper opening without detaching the logical attachment
