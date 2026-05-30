# Zapdos Robot Whole-Body Move Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let any picked `r1pro` or `moz1` robot link select the owning robot root body and move the whole robot with the existing Zapdos transform controls.

**Architecture:** Add explicit body capability metadata in the Python `get_visual()` payload so selection and movement rules come from runtime body data instead of frontend model-name conditionals. Keep scene-object editing behavior unchanged, allow `set_body_pose()` for movable robot roots, and teach the web scene to select via `selectionBody` while gating transform controls on `movable`.

**Tech Stack:** Python 3.12 with `uv`, MuJoCo, FastAPI session runtime, Next.js/React client components, TypeScript, `node:test`

---

### Task 1: Backend Body Capability Contract

**Files:**
- Create: `apps/python/utils/zapdos/physics/body_capabilities.py`
- Modify: `apps/python/utils/zapdos/physics/visuals.py`
- Modify: `apps/python/utils/zapdos/physics/mujoco_physics.py`
- Test: `apps/python/tests/test_zapdos_import.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_robot_root_detection_uses_parent_links(self):
    session = self.build_robot_root_pose_session()
    self.assertEqual(session.physics.robot_root_body_names, {"Root_base_link"})
    self.assertEqual(session.physics.movable_body_names, {"Root_base_link", "Scene_Crate"})

def test_get_visual_exposes_selection_body_and_movable_flags(self):
    session = self.build_robot_root_pose_session()
    bodies = {body["name"]: body for body in session.call_once("get_visual", ())["bodies"]}
    self.assertEqual(bodies["Root_base_link"]["selectionBody"], "Root_base_link")
    self.assertTrue(bodies["Root_base_link"]["movable"])
    self.assertEqual(bodies["Arm_link"]["selectionBody"], "Root_base_link")
    self.assertFalse(bodies["Arm_link"]["movable"])
    self.assertTrue(bodies["Arm_link"]["selectable"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest apps/python/tests/test_zapdos_import.py -k "robot_root_detection_uses_parent_links or get_visual_exposes_selection_body_and_movable_flags" -q`
Expected: FAIL with missing attributes like `robot_root_body_names`, `movable_body_names`, `selectionBody`, `movable`, or `selectable`

- [ ] **Step 3: Write the minimal backend capability implementation**

```python
@dataclass(frozen=True)
class BodyCapabilities:
    editable_body_names: set[str]
    robot_body_names: set[str]
    robot_root_body_names: set[str]
    movable_body_names: set[str]
    selection_body_by_name: dict[str, str]

def build_body_capabilities(model, body_map: dict[str, str]) -> BodyCapabilities:
    editable = {name for name, path in body_map.items() if not path.startswith("MyRobot/")}
    robot = {name for name, path in body_map.items() if path.startswith("MyRobot/")}
    roots = {name for name in robot if parent_body_name(model, name) not in robot}
    selection_map = {name: name for name in editable}
    for name in robot:
        current = name
        while current not in roots:
            current = parent_body_name(model, current) or name
        selection_map[name] = current
    return BodyCapabilities(
        editable_body_names=editable,
        robot_body_names=robot,
        robot_root_body_names=roots,
        movable_body_names=editable | roots,
        selection_body_by_name=selection_map,
    )
```

```python
serialize_body(
    name,
    self.body_map.get(name, name),
    name in self.editable_body_names,
    flatten_matrix(matrix),
    selectable=name in self.selection_body_by_name,
    movable=name in self.movable_body_names,
    selection_body=self.selection_body_by_name.get(name),
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest apps/python/tests/test_zapdos_import.py -k "robot_root_detection_uses_parent_links or get_visual_exposes_selection_body_and_movable_flags" -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/python/utils/zapdos/physics/body_capabilities.py apps/python/utils/zapdos/physics/visuals.py apps/python/utils/zapdos/physics/mujoco_physics.py apps/python/tests/test_zapdos_import.py
git commit -m "feat: add zapdos body selection metadata"
```

### Task 2: Backend Pose Editing And Runtime Replay

**Files:**
- Modify: `apps/python/utils/zapdos/physics/mujoco_physics.py`
- Modify: `apps/python/utils/zapdos/session/runtime_mixin.py`
- Test: `apps/python/tests/test_zapdos_import.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_set_body_pose_accepts_robot_root_and_rejects_non_root_robot_link(self):
    session = self.build_robot_root_pose_session()
    session.call_once("set_body_pose", ("Root_base_link", [4.0, 5.0, 6.0], [1.0, 0.0, 0.0, 0.0]))
    with self.assertRaises(MODULE.HTTPException):
        session.call_once("set_body_pose", ("Arm_link", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]))

def test_swap_runtime_bundle_replays_pose_overrides_for_movable_robot_roots(self):
    overlay_state = {"pose_overrides": {"Root_base_link": {"pos": [1.0, 2.0, 3.0], "quat": [1.0, 0.0, 0.0, 0.0]}}}
    new_physics = SimpleNamespace(
        movable_body_names={"Root_base_link"},
        editable_body_names=set(),
        model=object(),
        data=SimpleNamespace(qpos=[0.0], ctrl=[0.0]),
        set_body_pose=mock.Mock(),
        close=mock.Mock(),
    )
    session._swap_runtime_bundle(bundle, overlay_state)
    new_physics.set_body_pose.assert_called_once_with("Root_base_link", [1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest apps/python/tests/test_zapdos_import.py -k "set_body_pose_accepts_robot_root_and_rejects_non_root_robot_link or swap_runtime_bundle_replays_pose_overrides_for_movable_robot_roots" -q`
Expected: FAIL because robot roots are still rejected and swap only replays `editable_body_names`

- [ ] **Step 3: Write the minimal pose-edit implementation**

```python
if body not in self.movable_body_names:
    raise HTTPException(status_code=403, detail=f"Body is not movable: {body}")
```

```python
for body, pose in overlay_state["pose_overrides"].items():
    if body in new_physics.movable_body_names:
        new_physics.set_body_pose(body, pose["pos"], pose["quat"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest apps/python/tests/test_zapdos_import.py -k "set_body_pose_accepts_robot_root_and_rejects_non_root_robot_link or swap_runtime_bundle_replays_pose_overrides_for_movable_robot_roots" -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/python/utils/zapdos/physics/mujoco_physics.py apps/python/utils/zapdos/session/runtime_mixin.py apps/python/tests/test_zapdos_import.py
git commit -m "feat: allow whole-body zapdos robot movement"
```

### Task 3: Frontend Selection Mapping And Transform Control Gating

**Files:**
- Modify: `apps/web/components/zapdos/zapdos-scene-api.ts`
- Modify: `apps/web/components/zapdos/zapdos-scene-state.ts`
- Modify: `apps/web/components/zapdos/zapdos-scene-state.test.ts`
- Modify: `apps/web/components/zapdos/ZapdosScene.tsx`

- [ ] **Step 1: Write the failing tests**

```typescript
test("pickSelectableBodyFromHits returns the mapped selection body before the raw hit body", () => {
  assert.equal(
    pickSelectableBodyFromHits([{ body: "Arm_link", editable: false, selectionBody: "Root_base_link" }]),
    "Root_base_link"
  );
});

test("getTransformBodyName only returns movable selections", () => {
  assert.equal(
    getTransformBodyName("Root_base_link", { Root_base_link: { movable: true }, Arm_link: { movable: false } }),
    "Root_base_link"
  );
  assert.equal(getTransformBodyName("Arm_link", { Arm_link: { movable: false } }), null);
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm --dir apps/web test components/zapdos/zapdos-scene-state.test.ts`
Expected: FAIL because `selectionBody` and `getTransformBodyName` do not exist yet

- [ ] **Step 3: Write the minimal frontend implementation**

```typescript
export interface BodyVisual {
  name: string;
  label: string;
  editable: boolean;
  selectable: boolean;
  movable: boolean;
  selectionBody: string | null;
  matrix: number[];
}
```

```typescript
export interface ZapdosPickHit {
  body: string | null;
  editable: boolean;
  selectionBody?: string | null;
}

export function pickSelectableBodyFromHits(hits: ZapdosPickHit[]) {
  for (const hit of hits) {
    if (hit.selectionBody) return hit.selectionBody;
    if (hit.body) return hit.body;
  }
  return null;
}
```

```typescript
const selectedMovableObject = selectedObject?.userData.zapdosMovable === true ? selectedObject : null;
group.userData.zapdosMovable = body.movable;
group.userData.zapdosSelectionBody = body.selectionBody;
mesh.userData.zapdosSelectionBody = bodyObjectsRef.current[item.body]?.userData.zapdosSelectionBody;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm --dir apps/web test components/zapdos/zapdos-scene-state.test.ts components/zapdos/robot-model.test.ts components/zapdos/zapdos-top-overlay.test.tsx components/zapdos/zapdos-import.test.ts`
Expected: PASS with 0 failures

- [ ] **Step 5: Commit**

```bash
git add apps/web/components/zapdos/zapdos-scene-api.ts apps/web/components/zapdos/zapdos-scene-state.ts apps/web/components/zapdos/zapdos-scene-state.test.ts apps/web/components/zapdos/ZapdosScene.tsx
git commit -m "feat: map zapdos robot picks to movable roots"
```
