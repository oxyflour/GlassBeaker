# Zapdos SpaceMouse Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Autostart the SpaceMouse teleop backend and let the Zapdos page switch between `off`, `left`, and `right` modes.

**Architecture:** Keep the existing global `SpaceMouseManager`, add explicit mode state to it, expose one high-level mode API, and wire a dedicated Zapdos UI selector to that API. The thread stays alive in `off` mode so device and ROS connectivity stay warm while command publishing is suppressed.

**Tech Stack:** FastAPI, Python `unittest`, Next.js, React, `tsx --test`

---

### Task 1: Backend Mode Semantics

**Files:**
- Modify: `apps/python/tests/test_spacemouse_manager.py`
- Modify: `apps/python/teleop/manager.py`

- [ ] **Step 1: Write the failing test**

```python
def test_off_mode_keeps_thread_alive_but_suppresses_publish(self):
    manager = SpaceMouseManager(
        device=_FakeDevice([self.sample()]),
        ros_client=_FakeRosClient({"name": [], "position": []}),
        ik_controller=_FakeIKController(),
    )
    manager.set_mode("off")
    manager.step_once()
    self.assertEqual(manager.status()["mode"], "off")
    self.assertEqual(manager.ros_client.published, [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_spacemouse_manager.SpaceMouseManagerTest.test_off_mode_keeps_thread_alive_but_suppresses_publish`
Expected: FAIL because `set_mode` / `mode` do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
self._mode = "off"

def set_mode(self, mode: str) -> dict[str, Any]:
    ...
    if mode in {"left", "right"}:
        self._active_arm = mode
    self._mode = mode
    return self._status_unlocked()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_spacemouse_manager.SpaceMouseManagerTest.test_off_mode_keeps_thread_alive_but_suppresses_publish`
Expected: PASS

### Task 2: Backend Autostart And API

**Files:**
- Modify: `apps/python/tests/test_spacemouse_api.py`
- Modify: `apps/python/api/teleop/spacemouse.py`

- [ ] **Step 1: Write the failing test**

```python
def test_startup_event_autostarts_manager(self):
    with self.make_client()[0]:
        pass
    self.assertIn(("start", {}), stub.calls)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_spacemouse_api.SpaceMouseApiTest.test_startup_event_autostarts_manager`
Expected: FAIL because startup does not call `manager.start()`.

- [ ] **Step 3: Write minimal implementation**

```python
@router.on_event("startup")
async def startup() -> None:
    manager.start()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_spacemouse_api.SpaceMouseApiTest.test_startup_event_autostarts_manager`
Expected: PASS

### Task 3: Zapdos Selector Logic

**Files:**
- Create: `apps/web/components/zapdos/spacemouse-mode.ts`
- Create: `apps/web/components/zapdos/spacemouse-mode.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
test("deriveSpaceMouseMode prefers explicit backend mode", () => {
  expect(deriveSpaceMouseMode({ running: true, mode: "left" })).toBe("left")
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm --filter glassbeaker-web test -- components/zapdos/spacemouse-mode.test.ts`
Expected: FAIL because helper file does not exist.

- [ ] **Step 3: Write minimal implementation**

```ts
export function deriveSpaceMouseMode(status: { mode?: string; running?: boolean; active_arm?: string }) {
  ...
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm --filter glassbeaker-web test -- components/zapdos/spacemouse-mode.test.ts`
Expected: PASS

### Task 4: Zapdos Selector UI

**Files:**
- Create: `apps/web/components/zapdos/SpaceMouseModeSelect.tsx`
- Modify: `apps/web/app/demo/zapdos/page.tsx`

- [ ] **Step 1: Implement the selector component**

```tsx
<select value={mode} onChange={...}>
  <option value="off">关闭</option>
  <option value="left">左臂</option>
  <option value="right">右臂</option>
</select>
```

- [ ] **Step 2: Wire mount-time status load and change handling**

```tsx
fetch("/python/teleop/spacemouse/status")
fetch("/python/teleop/spacemouse/set_mode", { method: "POST", ... })
```

- [ ] **Step 3: Run the targeted tests**

Run: `pnpm --filter glassbeaker-web test -- components/zapdos/spacemouse-mode.test.ts`
Expected: PASS

### Task 5: Regression Verification

**Files:**
- Test: `apps/python/tests/test_spacemouse_api.py`
- Test: `apps/python/tests/test_spacemouse_manager.py`
- Test: `apps/web/components/zapdos/spacemouse-mode.test.ts`

- [ ] **Step 1: Run backend teleop tests**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_spacemouse_api apps.python.tests.test_spacemouse_manager`
Expected: PASS

- [ ] **Step 2: Run frontend selector tests**

Run: `pnpm --filter glassbeaker-web test -- components/zapdos/spacemouse-mode.test.ts`
Expected: PASS

- [ ] **Step 3: Manual smoke**

Run the desktop app, open Zapdos, and verify:
- selector shows `关闭` on first load
- switching to `左臂` or `右臂` updates teleop status
- switching back to `关闭` leaves the service running but stops motion commands
