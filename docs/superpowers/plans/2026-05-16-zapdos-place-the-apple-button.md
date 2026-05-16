# Zapdos Place The Apple Button Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `Place the apple` debug control that releases the currently attached apple in place by opening the gripper and detaching the logical attachment.

**Architecture:** Keep the canned apple workflow split across the existing frontend action/button modules and the Python manipulation runtime. Add a dedicated `place_apple` route that builds a tiny release plan, then extend `PickExecutor` with a release branch that opens the gripper while holding pose and detaches the target instead of attaching it.

**Tech Stack:** Next.js App Router, React 19, `node:test` via `pnpm exec tsx --test`, Python 3.12, FastAPI, `unittest`, MuJoCo-backed Zapdos manipulation runtime

**Working Directories:** Run frontend commands from `apps/web`. Run Python commands from `apps/python` using `uv`.

---

### Task 1: Add the frontend place action helper

**Files:**
- Create: `apps/web/components/zapdos/place-the-apple.ts`
- Create: `apps/web/components/zapdos/place-the-apple.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
import assert from "node:assert/strict";
import test from "node:test";

type PlaceTheAppleModule = typeof import("./place-the-apple");

test("createPlaceTheAppleRequest posts the canned apple release payload", async () => {
  const { createPlaceTheAppleRequest } = await loadModule<PlaceTheAppleModule>("./place-the-apple.ts");

  assert.deepEqual(createPlaceTheAppleRequest(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify([]),
  });
});

test("placeTheApple posts to the zapdos place_apple route", async () => {
  const { createPlaceTheAppleRequest, placeTheApple } = await loadModule<PlaceTheAppleModule>("./place-the-apple.ts");
  const calls: Array<{ input: unknown; init: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: unknown, init?: unknown) => {
    calls.push({ input, init });
    return new Response(JSON.stringify({
      ok: true,
      arm: "left",
      target_body: "Scene_apple_01",
      scene_revision: "rev-4",
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    const payload = await placeTheApple("sess-1");
    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/call/place_apple");
    assert.deepEqual(calls[0]?.init, createPlaceTheAppleRequest());
    assert.equal(payload.target_body, "Scene_apple_01");
    assert.equal(payload.scene_revision, "rev-4");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

async function loadModule<TModule>(specifier: string): Promise<TModule> {
  const loaded = await import(specifier);
  return (loaded.default ?? loaded["module.exports"] ?? loaded) as TModule;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm exec tsx --test components/zapdos/place-the-apple.test.ts`

Expected: FAIL with a module resolution error for `./place-the-apple.ts`.

- [ ] **Step 3: Write minimal implementation**

```ts
import { createManipulationToolRequest } from "./zapdos-manipulation-tool-api";

export function createPlaceTheAppleRequest(): RequestInit {
  return createManipulationToolRequest([]);
}

export async function placeTheApple(sess: string) {
  const response = await fetch(
    `/python/zapdos/${sess}/call/place_apple`,
    createPlaceTheAppleRequest(),
  );
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return await response.json() as {
    arm?: string;
    ok?: boolean;
    scene_revision: string;
    status?: string;
    target_body?: string;
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm exec tsx --test components/zapdos/place-the-apple.test.ts`

Expected: PASS with `2` passing tests and no TypeScript/runtime errors.

- [ ] **Step 5: Commit**

```bash
git add apps/web/components/zapdos/place-the-apple.ts apps/web/components/zapdos/place-the-apple.test.ts
git commit -m "feat: add zapdos place-apple action helper"
```

If this session is sharing a dirty workspace, stage only the two paths above.

### Task 2: Add the PlaceTheApple button and debug-menu wiring

**Files:**
- Create: `apps/web/components/zapdos/PlaceTheAppleButton.tsx`
- Modify: `apps/web/components/zapdos/ZapdosTopOverlay.tsx`
- Modify: `apps/web/components/zapdos/zapdos-top-overlay.test.tsx`

- [ ] **Step 1: Write the failing test**

Add this assertion to the existing debug-menu test in `apps/web/components/zapdos/zapdos-top-overlay.test.tsx`:

```tsx
test("ZapdosTopOverlay renders Add benchmark table in the debug menu", () => {
  const html = renderToStaticMarkup(
    <ZapdosTopOverlay
      activeRobotModelKey="moz1"
      defaultDebugOpen
      onRobotModelChange={ () => undefined }
      sess="sess-1"
      sse={1} />
  );

  assert.match(html, /Add benchmark table/);
  assert.match(html, /Grab the apple/);
  assert.match(html, /Place the apple/);
  assert.doesNotMatch(html, /Save camera override/);
  assert.doesNotMatch(html, /Robot model/);
  assert.doesNotMatch(html, /SpaceMouse/);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm exec tsx --test components/zapdos/zapdos-top-overlay.test.tsx`

Expected: FAIL because `Place the apple` is not yet rendered in the debug menu.

- [ ] **Step 3: Write minimal implementation**

Create `apps/web/components/zapdos/PlaceTheAppleButton.tsx`:

```tsx
'use client'

import { useState } from "react";

import { placeTheApple } from "./place-the-apple";

export function PlaceTheAppleButton({ sess }: { sess: string }) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function handleClick() {
    setBusy(true);
    setMessage("");
    setError("");
    try {
      const payload = await placeTheApple(sess);
      setMessage(`Placed ${payload.target_body ?? "apple"}`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Place apple failed");
    } finally {
      setBusy(false);
    }
  }

  return <div className="rounded-md bg-black/60 px-3 py-2 text-white backdrop-blur-sm">
    <button
      className="rounded border border-white/20 bg-black/40 px-3 py-1 text-sm disabled:opacity-60"
      disabled={ busy || !sess }
      onClick={ () => void handleClick() }>
      { busy ? "Placing..." : "Place the apple" }
    </button>
    { message ? <div className="mt-2 max-w-80 text-xs text-emerald-200">{ message }</div> : null }
    { error ? <div className="mt-2 max-w-80 text-xs text-red-200">{ error }</div> : null }
  </div>;
}
```

Update `apps/web/components/zapdos/ZapdosTopOverlay.tsx` imports and debug-panel body:

```tsx
import { GrabTheAppleButton } from "./GrabTheAppleButton";
import { PlaceTheAppleButton } from "./PlaceTheAppleButton";
```

```tsx
{ debugOpen ? <div className="absolute right-0 mt-3 flex w-72 max-w-[calc(100vw-4rem)] flex-col gap-3">
  <AddBenchmarkTableButton sess={ sess } />
  <GrabTheAppleButton sess={ sess } />
  <PlaceTheAppleButton sess={ sess } />
</div> : null }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm exec tsx --test components/zapdos/place-the-apple.test.ts components/zapdos/zapdos-top-overlay.test.tsx`

Expected: PASS with the new helper tests and overlay test green.

- [ ] **Step 5: Commit**

```bash
git add apps/web/components/zapdos/PlaceTheAppleButton.tsx apps/web/components/zapdos/ZapdosTopOverlay.tsx apps/web/components/zapdos/zapdos-top-overlay.test.tsx
git commit -m "feat: add place-the-apple debug button"
```

If this session is sharing a dirty workspace, stage only the three paths above.

### Task 3: Add executor release behavior for in-place detach

**Files:**
- Modify: `apps/python/utils/zapdos/manipulation/executor.py`
- Modify: `apps/python/tests/test_zapdos_pick_executor.py`

- [ ] **Step 1: Write the failing test**

Add these tests to `apps/python/tests/test_zapdos_pick_executor.py`:

```python
    def test_execute_release_opens_gripper_and_detaches_attached_target(self):
        physics = _FakePhysics()
        physics.attached.append(("Root_r1_pro_with_gripper_left_gripper_link", "Scene_Crate"))
        ik = _MutableIK(_pose(0.0, 0.0, 0.08))
        executor = PickExecutor(physics, bundle=_bundle(), ik_controller=ik)
        driven: list[tuple[tuple[float, ...], float, int]] = []

        def drive(_ik_controller, _arm, target, gripper, steps=12, **_kwargs):
            driven.append((target["position"], gripper, steps))
            ik.pose = target

        with mock.patch.object(executor, "_drive_pose", side_effect=drive):
            result = executor.execute({
                "kind": "release",
                "arm": "left",
                "target_body": "Scene_Crate",
                "stages": [
                    {"name": "open_gripper", "kind": "gripper", "width": 0.05, "steps": 18},
                ],
            })

        self.assertTrue(result["ok"])
        self.assertEqual(driven, [((0.0, 0.0, 0.08), 0.05, 18)])
        self.assertEqual(physics.detached, ["Scene_Crate"])
        self.assertIsNone(result["attachment"])

    def test_execute_release_rejects_target_that_is_not_attached(self):
        executor = PickExecutor(_FakePhysics(), bundle=_bundle(), ik_controller=_MutableIK(_pose(0.0, 0.0, 0.08)))

        with self.assertRaises(HTTPException) as err:
            executor.execute({
                "kind": "release",
                "arm": "left",
                "target_body": "Scene_Crate",
                "stages": [
                    {"name": "open_gripper", "kind": "gripper", "width": 0.05, "steps": 18},
                ],
            })

        self.assertEqual(err.exception.status_code, 409)
        self.assertIn("not attached", err.exception.detail)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_zapdos_pick_executor.PickExecutorTest.test_execute_release_opens_gripper_and_detaches_attached_target tests.test_zapdos_pick_executor.PickExecutorTest.test_execute_release_rejects_target_that_is_not_attached`

Expected: FAIL because `PickExecutor.execute()` only knows the pick/attach flow.

- [ ] **Step 3: Write minimal implementation**

In `apps/python/utils/zapdos/manipulation/executor.py`, branch on `plan.get("kind")` before the pick-specific `descend_to_grasp` and attach logic:

```python
    def execute(self, plan: dict[str, object]) -> dict[str, object]:
        if str(plan.get("kind") or "pick") == "release":
            return self._execute_release(plan)
        ...

    def _execute_release(self, plan: dict[str, object]) -> dict[str, object]:
        arm = str(plan.get("arm") or "left")
        target_body = str(plan["target_body"])
        attachment = self.physics.get_attachment(target_body)
        if attachment is None:
            raise HTTPException(status_code=409, detail=f"Release failed: {target_body} is not attached")

        ik = self._ensure_ik()
        ik.sync_joint_state(self.physics.joint_state_msg())
        hold_pose = ik.get_end_effector_pose(arm)
        for raw_stage in plan.get("stages", []):
            stage = raw_stage if isinstance(raw_stage, dict) else {}
            if str(stage.get("kind")) != "gripper":
                raise HTTPException(status_code=409, detail=f"Release failed: unsupported stage kind {stage.get('kind')}")
            self._drive_pose(
                ik,
                arm,
                hold_pose,
                float(stage.get("width", 0.0)),
                steps=max(1, int(stage.get("steps", 6))),
            )
        self.physics.detach_body(target_body)
        return {"ok": True, "arm": arm, "target_body": target_body, "attachment": self.physics.get_attachment(target_body)}
```

Keep the existing pick path unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_zapdos_pick_executor.PickExecutorTest.test_execute_release_opens_gripper_and_detaches_attached_target tests.test_zapdos_pick_executor.PickExecutorTest.test_execute_release_rejects_target_that_is_not_attached`

Expected: `Ran 2 tests` and `OK`.

- [ ] **Step 5: Commit**

```bash
git add apps/python/utils/zapdos/manipulation/executor.py apps/python/tests/test_zapdos_pick_executor.py
git commit -m "feat: add release path to zapdos pick executor"
```

If this session is sharing a dirty workspace, stage only the two paths above.

### Task 4: Add the canned place_apple runtime route and session dispatch

**Files:**
- Modify: `apps/python/utils/zapdos/manipulation/runtime.py`
- Modify: `apps/python/utils/zapdos/zapdos_session.py`
- Modify: `apps/python/tests/test_zapdos_import.py`

- [ ] **Step 1: Write the failing test**

Update `apps/python/tests/test_zapdos_import.py` in two places.

First, extend the session dispatch test:

```python
    def test_call_once_dispatches_manipulation_runtime_methods(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.runtime = mock.Mock(
            list_scene_objects=mock.Mock(return_value={"items": [{"body": "Scene_Crate"}], "scene_revision": "rev-1"}),
            grab_apple=mock.Mock(return_value={"ok": True, "target_body": "Scene_apple_1", "scene_revision": "rev-1"}),
            place_apple=mock.Mock(return_value={"ok": True, "target_body": "Scene_apple_1", "scene_revision": "rev-2"}),
            pick_object=mock.Mock(return_value={"ok": True, "target_body": "Scene_Crate", "scene_revision": "rev-1"}),
        )

        listed = MODULE.ZapdosSession.call_once(session, "list_scene_objects", ())
        grabbed = MODULE.ZapdosSession.call_once(session, "grab_apple", ())
        placed = MODULE.ZapdosSession.call_once(session, "place_apple", ())
        picked = MODULE.ZapdosSession.call_once(session, "pick_object", ({"target_query": "crate"},))

        self.assertEqual(placed["target_body"], "Scene_apple_1")
        session.runtime.place_apple.assert_called_once_with()
```

Then add a new runtime test:

```python
    def test_manipulation_runtime_executes_arm_only_place_apple_plan(self):
        from utils.zapdos.manipulation.runtime import ManipulationRuntime

        target = {
            "body": "Scene_apple_1",
            "label": "apple",
            "motion": "dynamic",
            "position": [0.5, 0.0, 0.83],
            "world_aabb": {"min": [0.454, -0.046, 0.784], "max": [0.546, 0.046, 0.876]},
        }
        support = {
            "body": "table_body",
            "label": "benchmark table",
            "motion": "static",
            "position": [0.5, 0.0, 0.75],
            "world_aabb": {"min": [0.1, -0.3, 0.7], "max": [0.9, 0.3, 0.8]},
        }
        physics = mock.Mock()
        physics.get_attachment.return_value = {
            "parent_body": "Root_r1_pro_with_gripper_left_gripper_link",
            "child_body": "Scene_apple_1",
            "relative_position": [0.0, 0.0, 0.0],
            "relative_quat": [1.0, 0.0, 0.0, 0.0],
        }
        session = SimpleNamespace(
            editor=SimpleNamespace(
                scene_revision="rev-2",
                overlay_state={},
                list_scene_bodies=mock.Mock(return_value={"items": []}),
            ),
            bundle=SimpleNamespace(scene_usd=Path("scene.usda"), robot_usd=Path("robot.usda")),
            physics=physics,
        )
        executor = mock.Mock(
            execute=mock.Mock(return_value={"ok": True, "target_body": "Scene_apple_1", "attachment": None}),
        )
        catalog_loader = mock.Mock(return_value=[target, support])
        grounder = mock.Mock(return_value={"target": target, "support": support})

        runtime = ManipulationRuntime(
            session,
            catalog_loader=catalog_loader,
            grounding_fn=grounder,
            executor=executor,
        )
        result = runtime.place_apple()

        self.assertEqual(result["target_body"], "Scene_apple_1")
        self.assertEqual(result["scene_revision"], "rev-2")
        physics.get_attachment.assert_called_once_with("Scene_apple_1")
        executor.execute.assert_called_once_with({
            "kind": "release",
            "arm": "left",
            "target_body": "Scene_apple_1",
            "stages": [{"name": "open_gripper", "kind": "gripper", "width": 0.05, "steps": 18}],
        })
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_zapdos_import.ZapdosImportTest.test_call_once_dispatches_manipulation_runtime_methods tests.test_zapdos_import.ZapdosImportTest.test_manipulation_runtime_executes_arm_only_place_apple_plan`

Expected: FAIL because `place_apple` does not exist on the runtime/session path.

- [ ] **Step 3: Write minimal implementation**

In `apps/python/utils/zapdos/manipulation/runtime.py`, add:

```python
    def place_apple(self) -> dict[str, object]:
        objects = self._scene_objects()
        grounded = self._ground_target({
            "target_query": GRAB_APPLE_TARGET_QUERY,
            "support_query": GRAB_APPLE_SUPPORT_QUERY,
        }, objects)
        target = grounded["target"]
        if self.session.physics.get_attachment(target["body"]) is None:
            raise HTTPException(status_code=409, detail=f"Place apple requires {target['body']} to be attached")
        self._sync_executor_state()
        result = self.executor.execute({
            "kind": "release",
            "arm": GRAB_APPLE_ARM,
            "target_body": target["body"],
            "stages": [
                {
                    "name": "open_gripper",
                    "kind": "gripper",
                    "width": GRAB_APPLE_OPEN_WIDTH,
                    "steps": 18,
                },
            ],
        })
        return {**result, "scene_revision": self.session.editor.scene_revision}
```

In `apps/python/utils/zapdos/zapdos_session.py`, add:

```python
        if method == "place_apple":
            return self.runtime.place_apple()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_zapdos_import.ZapdosImportTest.test_call_once_dispatches_manipulation_runtime_methods tests.test_zapdos_import.ZapdosImportTest.test_manipulation_runtime_executes_arm_only_place_apple_plan tests.test_zapdos_pick_executor.PickExecutorTest.test_execute_release_opens_gripper_and_detaches_attached_target tests.test_zapdos_pick_executor.PickExecutorTest.test_execute_release_rejects_target_that_is_not_attached`

Expected: `Ran 4 tests` and `OK`.

- [ ] **Step 5: Commit**

```bash
git add apps/python/utils/zapdos/manipulation/runtime.py apps/python/utils/zapdos/zapdos_session.py apps/python/tests/test_zapdos_import.py
git commit -m "feat: add canned zapdos place-apple action"
```

If this session is sharing a dirty workspace, stage only the three paths above.

### Task 5: Run focused regression coverage across web and Python

**Files:**
- Test only: `apps/web/components/zapdos/place-the-apple.test.ts`
- Test only: `apps/web/components/zapdos/zapdos-top-overlay.test.tsx`
- Test only: `apps/python/tests/test_zapdos_pick_executor.py`
- Test only: `apps/python/tests/test_zapdos_import.py`

- [ ] **Step 1: Run the focused web regression suite**

Run: `pnpm exec tsx --test components/zapdos/place-the-apple.test.ts components/zapdos/zapdos-top-overlay.test.tsx components/zapdos/grab-the-apple.test.ts`

Expected: PASS with all Zapdos canned-button web tests green.

- [ ] **Step 2: Run the focused Python regression suite**

Run: `uv run python -m unittest tests.test_zapdos_pick_executor tests.test_zapdos_import.ZapdosImportTest.test_call_once_dispatches_manipulation_runtime_methods tests.test_zapdos_import.ZapdosImportTest.test_manipulation_runtime_executes_arm_only_grab_apple_plan tests.test_zapdos_import.ZapdosImportTest.test_manipulation_runtime_executes_arm_only_place_apple_plan`

Expected: PASS with the existing grab-apple flow still green and the new place-apple flow green.

- [ ] **Step 3: Run TypeScript checking for the touched frontend surface**

Run: `pnpm exec tsc --noEmit --pretty false`

Expected: exit code `0`.

- [ ] **Step 4: Review changed files before final handoff**

Run: `git diff -- apps/web/components/zapdos apps/python/utils/zapdos/manipulation apps/python/utils/zapdos/zapdos_session.py apps/python/tests/test_zapdos_pick_executor.py apps/python/tests/test_zapdos_import.py`

Expected: diff shows only the new place-apple flow, button, and tests.

- [ ] **Step 5: Commit the final integrated change**

```bash
git add apps/web/components/zapdos/place-the-apple.ts apps/web/components/zapdos/place-the-apple.test.ts apps/web/components/zapdos/PlaceTheAppleButton.tsx apps/web/components/zapdos/ZapdosTopOverlay.tsx apps/web/components/zapdos/zapdos-top-overlay.test.tsx apps/python/utils/zapdos/manipulation/executor.py apps/python/utils/zapdos/manipulation/runtime.py apps/python/utils/zapdos/zapdos_session.py apps/python/tests/test_zapdos_pick_executor.py apps/python/tests/test_zapdos_import.py
git commit -m "feat: add zapdos place-the-apple release flow"
```

If this session is sharing a dirty workspace, stage only the paths above and leave unrelated user changes untouched.
