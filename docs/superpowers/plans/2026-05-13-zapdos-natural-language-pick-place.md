# Zapdos Natural-Language Pick/Place Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a CopilotKit-connected agent execute first-version natural-language `pick/place` commands such as "pick up the apple on the table" in Zapdos using scene metadata, a local IK execution loop, and a kinematic held-object attachment.

**Architecture:** Keep the LLM at the intent layer. The frontend exposes high-level tools like `pick_object` and `pick_and_place_object`; the Python backend grounds `target_query` and `support_query` against existing scene metadata, builds a small manipulation plan using Genie Sim-inspired pose/action sequencing, and executes that plan through the existing `utils.teleop.IKController`. Because true grasp physics is out of scope for v1, closing the gripper transitions the picked object into a kinematic "attached" mode that follows the end effector until released.

**Tech Stack:** Python 3.12, FastAPI session calls, MuJoCo, existing `utils.teleop.IKController`, TypeScript, CopilotKit, Zod, `unittest`, `node:test`, `uv`, `pnpm`

---

## File Structure

- `apps/python/utils/zapdos/manipulation/types.py`
  Shared dataclasses and typed result payloads.
- `apps/python/utils/zapdos/manipulation/catalog.py`
  Build scene object records from `overlay_state`, body poses, and support metadata.
- `apps/python/utils/zapdos/manipulation/grounding.py`
  Resolve `target_query` and optional `support_query` into one grounded object or an ambiguity error.
- `apps/python/utils/zapdos/manipulation/planner.py`
  Build first-version `pick` and `place` step sequences with Genie Sim-style pre-grasp / grasp / lift staging.
- `apps/python/utils/zapdos/manipulation/executor.py`
  Execute plan steps through `IKController` and MuJoCo stepping.
- `apps/python/utils/zapdos/manipulation/runtime.py`
  Session-facing orchestration API: `list_manipulation_objects`, `pick_object`, `place_held_object`, `pick_and_place_object`.
- `apps/python/utils/zapdos/physics/attachment.py`
  Held-object attachment state and pose-follow helper.
- `apps/python/utils/zapdos/physics/base.py`
  Extend the physics protocol with joint-state and attachment methods used by the executor.
- `apps/python/utils/zapdos/physics/mujoco_physics.py`
  Implement attachment and expose physics helpers needed by the runtime.
- `apps/python/utils/zapdos/zapdos_session.py`
  Construct the manipulation runtime and expose it through `call_once`.
- `apps/python/tests/test_zapdos_manipulation.py`
  Focused backend tests for grounding, planning, attachment, and runtime delegation.
- `apps/web/components/zapdos/zapdos-manipulation-tool-schemas.ts`
  Zod schemas for high-level manipulation tools.
- `apps/web/components/zapdos/zapdos-manipulation-tool-api.ts`
  Frontend request builders for Zapdos manipulation calls.
- `apps/web/components/zapdos/useZapdosAgentTools.ts`
  Register manipulation tools with CopilotKit.
- `apps/web/components/zapdos/zapdos-agent-instructions.ts`
  Add tool-use guidance for direct-execution pick/place.
- `apps/web/components/zapdos/zapdos-manipulation-tools.test.ts`
  Frontend request/schema regression coverage.

### Task 1: Build scene-object grounding from existing metadata

**Files:**
- Create: `apps/python/utils/zapdos/manipulation/types.py`
- Create: `apps/python/utils/zapdos/manipulation/catalog.py`
- Create: `apps/python/utils/zapdos/manipulation/grounding.py`
- Create: `apps/python/tests/test_zapdos_manipulation.py`

- [ ] **Step 1: Write the failing grounding test**

```python
class ZapdosManipulationTest(unittest.TestCase):
    def test_ground_pick_target_prefers_dynamic_asset_on_named_support(self):
        objects = [
            SceneObjectRecord(
                body="Scene_apple_red_001_01",
                label="apple",
                asset_id="apple_red_001",
                tags=("apple", "fruit", "red"),
                motion="dynamic",
                support_body="bench_table_top",
                world_pos=(0.45, 0.05, 0.82),
                top_z=0.86,
            ),
            SceneObjectRecord(
                body="bench_table_top",
                label="table",
                asset_id=None,
                tags=("table", "support"),
                motion="static",
                support_body=None,
                world_pos=(0.40, 0.00, 0.75),
                top_z=0.80,
            ),
        ]
        match = ground_pick_target(objects, target_query="apple", support_query="table")
        self.assertEqual(match.body, "Scene_apple_red_001_01")
```

- [ ] **Step 2: Run the focused backend test to verify it fails**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_manipulation.ZapdosManipulationTest.test_ground_pick_target_prefers_dynamic_asset_on_named_support -v`

Expected: FAIL with `ModuleNotFoundError` for `utils.zapdos.manipulation`.

- [ ] **Step 3: Write the minimal catalog and grounding implementation**

```python
@dataclass(frozen=True)
class SceneObjectRecord:
    body: str
    label: str
    asset_id: str | None
    tags: tuple[str, ...]
    motion: Literal["static", "dynamic"]
    support_body: str | None
    world_pos: tuple[float, float, float]
    top_z: float

def ground_pick_target(objects: Sequence[SceneObjectRecord], target_query: str, support_query: str | None):
    tokens = {token.strip().lower() for token in target_query.split() if token.strip()}
    support_tokens = {token.strip().lower() for token in (support_query or "").split() if token.strip()}
    candidates = [obj for obj in objects if obj.motion == "dynamic" and (tokens & ({obj.label.lower(), *(tag.lower() for tag in obj.tags)}))]
    if support_tokens:
        candidates = [obj for obj in candidates if support_tokens & {obj.support_body or "", *(tag.lower() for tag in obj.tags)}]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one grounded target, got {len(candidates)}")
    return candidates[0]
```

- [ ] **Step 4: Run the grounding test to verify it passes**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_manipulation.ZapdosManipulationTest.test_ground_pick_target_prefers_dynamic_asset_on_named_support -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/python/utils/zapdos/manipulation/types.py apps/python/utils/zapdos/manipulation/catalog.py apps/python/utils/zapdos/manipulation/grounding.py apps/python/tests/test_zapdos_manipulation.py
git commit -m "feat: add zapdos manipulation grounding"
```

### Task 2: Add a first-version Genie Sim-style pick/place planner

**Files:**
- Create: `apps/python/utils/zapdos/manipulation/planner.py`
- Modify: `apps/python/tests/test_zapdos_manipulation.py`

- [ ] **Step 1: Write the failing plan-shape test**

```python
    def test_plan_pick_generates_pregrasp_close_attach_and_lift(self):
        target = SceneObjectRecord("Scene_apple_red_001_01", "apple", "apple_red_001", ("apple",), "dynamic", "bench_table_top", (0.45, 0.05, 0.82), 0.86)
        plan = plan_pick(target, arm="right")
        self.assertEqual([step.kind for step in plan.steps], ["move_pose", "move_pose", "gripper", "attach", "move_pose"])
        self.assertGreater(plan.steps[0].pose.position[2], plan.steps[1].pose.position[2])
        self.assertGreater(plan.steps[-1].pose.position[2], plan.steps[1].pose.position[2])
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_manipulation.ZapdosManipulationTest.test_plan_pick_generates_pregrasp_close_attach_and_lift -v`

Expected: FAIL because `plan_pick` does not exist yet.

- [ ] **Step 3: Implement the minimal planner**

```python
@dataclass(frozen=True)
class PoseTarget:
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]

@dataclass(frozen=True)
class ManipulationStep:
    kind: Literal["move_pose", "gripper", "attach", "detach"]
    pose: PoseTarget | None = None
    gripper_opening: float | None = None

def plan_pick(target: SceneObjectRecord, arm: str) -> ManipulationPlan:
    grasp = PoseTarget(position=(target.world_pos[0], target.world_pos[1], target.top_z + 0.02), rotation=(1.0, 0.0, 0.0, 0.0))
    pregrasp = PoseTarget(position=(grasp.position[0], grasp.position[1], grasp.position[2] + 0.08), rotation=grasp.rotation)
    lift = PoseTarget(position=(grasp.position[0], grasp.position[1], grasp.position[2] + 0.12), rotation=grasp.rotation)
    return ManipulationPlan(arm=arm, target_body=target.body, steps=[
        ManipulationStep(kind="move_pose", pose=pregrasp),
        ManipulationStep(kind="move_pose", pose=grasp),
        ManipulationStep(kind="gripper", gripper_opening=0.0),
        ManipulationStep(kind="attach"),
        ManipulationStep(kind="move_pose", pose=lift),
    ])
```

- [ ] **Step 4: Run the planner tests**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_manipulation -v`

Expected: PASS for grounding and planner tests.

- [ ] **Step 5: Commit**

```bash
git add apps/python/utils/zapdos/manipulation/planner.py apps/python/tests/test_zapdos_manipulation.py
git commit -m "feat: add zapdos pick place planner"
```

### Task 3: Execute plans locally with IKController and kinematic attachment

**Files:**
- Create: `apps/python/utils/zapdos/physics/attachment.py`
- Create: `apps/python/utils/zapdos/manipulation/executor.py`
- Modify: `apps/python/utils/zapdos/physics/base.py`
- Modify: `apps/python/utils/zapdos/physics/mujoco_physics.py`
- Modify: `apps/python/tests/test_zapdos_manipulation.py`

- [ ] **Step 1: Write the failing executor test**

```python
    def test_executor_attaches_target_after_close_and_lifts(self):
        physics = FakePhysics()
        ik = FakeIKController()
        plan = plan_pick(self.apple, arm="right")
        result = ZapdosPickExecutor(physics, ik).execute(plan)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(physics.attached_body, "Scene_apple_red_001_01")
        self.assertGreater(physics.last_attached_z, 0.90)
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_manipulation.ZapdosManipulationTest.test_executor_attaches_target_after_close_and_lifts -v`

Expected: FAIL because neither the executor nor attachment helpers exist.

- [ ] **Step 3: Implement attachment plus the executor loop**

```python
@dataclass
class AttachedBodyState:
    body: str
    arm: str
    gripper_to_body: np.ndarray

def attach_body(self, body: str, arm: str) -> None:
    ee = self.body_pose_matrix(get_arm_config(arm).end_effector_body)
    target = self.body_pose_matrix(body)
    self._attached = AttachedBodyState(body=body, arm=arm, gripper_to_body=np.linalg.inv(ee) @ target)

def _sync_attached_body(self) -> None:
    if self._attached is None:
        return
    ee = self.body_pose_matrix(get_arm_config(self._attached.arm).end_effector_body)
    target = ee @ self._attached.gripper_to_body
    self._set_body_pose_matrix(self._attached.body, target)

class ZapdosPickExecutor:
    def execute(self, plan: ManipulationPlan) -> dict[str, object]:
        for step in plan.steps:
            if step.kind == "move_pose":
                self._move_arm(plan.arm, step.pose)
            elif step.kind == "gripper":
                self._set_gripper(plan.arm, step.gripper_opening or 0.0)
            elif step.kind == "attach":
                self.physics.attach_body(plan.target_body, plan.arm)
        return {"status": "ok", "target_body": plan.target_body, "arm": plan.arm}
```

- [ ] **Step 4: Run backend execution coverage**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_manipulation -v`

Expected: PASS, including attachment/executor coverage.

- [ ] **Step 5: Commit**

```bash
git add apps/python/utils/zapdos/physics/attachment.py apps/python/utils/zapdos/physics/base.py apps/python/utils/zapdos/physics/mujoco_physics.py apps/python/utils/zapdos/manipulation/executor.py apps/python/tests/test_zapdos_manipulation.py
git commit -m "feat: execute zapdos manipulation plans locally"
```

### Task 4: Expose the manipulation runtime through Zapdos sessions

**Files:**
- Create: `apps/python/utils/zapdos/manipulation/runtime.py`
- Modify: `apps/python/utils/zapdos/zapdos_session.py`
- Modify: `apps/python/tests/test_zapdos_import.py`

- [ ] **Step 1: Write the failing session-delegation test**

```python
    def test_zapdos_session_call_once_delegates_pick_object(self):
        session = object.__new__(SESSION_MODULE.ZapdosSession)
        session.manipulation = mock.Mock()
        session.manipulation.pick_object.return_value = {"status": "ok", "target_body": "Scene_apple_red_001_01"}
        result = SESSION_MODULE.ZapdosSession.call_once(session, "pick_object", ({"target_query": "apple", "support_query": "table", "arm": "right"},))
        self.assertEqual(result["target_body"], "Scene_apple_red_001_01")
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_import.ZapdosImportTest.test_zapdos_session_call_once_delegates_pick_object -v`

Expected: FAIL because `call_once` has no `pick_object` branch.

- [ ] **Step 3: Add the runtime and session call handlers**

```python
class ZapdosManipulationRuntime:
    def list_manipulation_objects(self) -> dict[str, object]: ...
    def pick_object(self, args: dict[str, object]) -> dict[str, object]:
        grounded = ground_pick_target(self._catalog(), args["target_query"], args.get("support_query"))
        plan = plan_pick(grounded, arm=str(args.get("arm", "right")))
        return self.executor.execute(plan)

def call_once(self, method: str, args: tuple):
    if method == "list_manipulation_objects":
        return self.manipulation.list_manipulation_objects()
    if method == "pick_object":
        return self.manipulation.pick_object(*args)
```

- [ ] **Step 4: Run the session-related backend tests**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_import apps.python.tests.test_zapdos_manipulation -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/python/utils/zapdos/manipulation/runtime.py apps/python/utils/zapdos/zapdos_session.py apps/python/tests/test_zapdos_import.py
git commit -m "feat: expose zapdos manipulation runtime"
```

### Task 5: Register CopilotKit manipulation tools and direct-execution guidance

**Files:**
- Create: `apps/web/components/zapdos/zapdos-manipulation-tool-schemas.ts`
- Create: `apps/web/components/zapdos/zapdos-manipulation-tool-api.ts`
- Create: `apps/web/components/zapdos/zapdos-manipulation-tools.test.ts`
- Modify: `apps/web/components/zapdos/useZapdosAgentTools.ts`
- Modify: `apps/web/components/zapdos/zapdos-agent-instructions.ts`

- [ ] **Step 1: Write the failing frontend request/schema test**

```ts
test("createPickObjectRequest posts the high-level pick payload", async () => {
  const { createPickObjectRequest } = await loadModule<typeof import("./zapdos-manipulation-tool-api")>("./zapdos-manipulation-tool-api.ts");
  assert.deepEqual(createPickObjectRequest({ target_query: "apple", support_query: "table", arm: "right" }), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify([{ target_query: "apple", support_query: "table", arm: "right" }]),
  });
});
```

- [ ] **Step 2: Run the frontend test to verify it fails**

Run: `pnpm --dir apps/web test apps/web/components/zapdos/zapdos-manipulation-tools.test.ts`

Expected: FAIL because the manipulation tool files do not exist yet.

- [ ] **Step 3: Add schemas, API helpers, and CopilotKit tool registration**

```ts
export const pickObjectToolArgsSchema = z.object({
  target_query: z.string().trim().min(1),
  support_query: z.string().trim().optional(),
  arm: z.enum(["left", "right"]).default("right"),
}).strict();

useTypedTool({
  name: "pick_object",
  description: "Pick one dynamic object matched by scene metadata and an optional support query.",
  followUp: true,
  parameters: pickObjectToolArgsSchema,
  handler: async (args) => await pickObject(sess, args),
}, [sess]);
```

- [ ] **Step 4: Run the frontend coverage**

Run: `pnpm --dir apps/web test apps/web/components/zapdos/zapdos-manipulation-tools.test.ts apps/web/utils/agent/tool.test.ts`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/web/components/zapdos/zapdos-manipulation-tool-schemas.ts apps/web/components/zapdos/zapdos-manipulation-tool-api.ts apps/web/components/zapdos/zapdos-manipulation-tools.test.ts apps/web/components/zapdos/useZapdosAgentTools.ts apps/web/components/zapdos/zapdos-agent-instructions.ts
git commit -m "feat: add zapdos manipulation copilot tools"
```

### Task 6: Final verification and manual smoke

**Files:**
- Modify: none

- [ ] **Step 1: Run the focused backend suite**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_manipulation apps.python.tests.test_zapdos_import apps.python.tests.test_zapdos_send_ros -v`

Expected: PASS.

- [ ] **Step 2: Run the focused frontend suite**

Run: `pnpm --dir apps/web test apps/web/components/zapdos/zapdos-manipulation-tools.test.ts apps/web/utils/agent/tool.test.ts`

Expected: PASS.

- [ ] **Step 3: Manual smoke test the natural-language path**

Run:

```powershell
pnpm --dir apps/web dev
uv run --project apps/python python apps/python/app.py
```

Expected: In `/demo/zapdos`, after the user says "pick up the apple on the table", the agent calls `pick_object`, the right arm performs pre-grasp -> grasp -> close -> lift, and the apple follows the gripper after attachment.

Alternative:

```powershell
pnpm dev
```

This starts the integrated desktop development flow, including the Python backend.
