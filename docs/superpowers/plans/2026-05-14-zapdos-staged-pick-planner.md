# Zapdos Staged Pick Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current support-footprint escape special case with a generic staged pick planner that routes the end effector around scene obstacles using body world AABBs before descending to the target.

**Architecture:** Keep the external `pick_object` request unchanged. Extend `list_placement_bodies()` and the manipulation-object catalog so each editable body carries a world-space AABB, then have the planner build a staged route from the current end-effector pose to the target: optional `escape_xy`, `raise_to_transit`, `approach_xy`, `descend_to_pregrasp`, `descend_to_grasp`, `close_gripper`, and `retreat`. The executor consumes that ordered stage list and reports stage-specific failures instead of hard-coding `pre_grasp/grasp/lift`.

**Tech Stack:** Python 3.12, FastAPI session calls, MuJoCo, existing `utils.teleop.IKController`, `unittest`, `uv`

---

## File Structure

- `apps/python/utils/zapdos/physics/mujoco_physics.py`
  Publish body-level world AABBs by aggregating MuJoCo geom bounds.
- `apps/python/utils/zapdos/editor/zapdos_editor.py`
  Include `world_aabb` in `list_placement_bodies()`.
- `apps/python/utils/zapdos/manipulation/types.py`
  Extend typed contracts with `WorldAabb`, `PlanningPose`, `PickStage`, and staged `PickPlan`.
- `apps/python/utils/zapdos/manipulation/catalog.py`
  Preserve `world_aabb` on each `SceneObject`.
- `apps/python/utils/zapdos/manipulation/planner.py`
  Replace the fixed three-pose plan with AABB-aware staged routing.
- `apps/python/utils/zapdos/manipulation/executor.py`
  Add `current_pose()` and execute `stages` in order with stage-specific error reporting.
- `apps/python/utils/zapdos/manipulation/runtime.py`
  Pass `arm`, `scene_objects`, and the current end-effector pose into the planner.
- `apps/python/tests/test_zapdos_import.py`
  Cover `list_placement_bodies()` payload shape and runtime planner call shape.
- `apps/python/tests/test_zapdos_manipulation.py`
  Cover `world_aabb` preservation and staged planner routing.
- `apps/python/tests/test_zapdos_pick_executor.py`
  Cover staged execution order, stage-specific failures, and the real MuJoCo loop.

### Task 1: Publish world AABBs in the scene-body payload

**Files:**
- Modify: `apps/python/utils/zapdos/physics/mujoco_physics.py`
- Modify: `apps/python/utils/zapdos/editor/zapdos_editor.py`
- Modify: `apps/python/tests/test_zapdos_import.py`

- [ ] **Step 1: Write the failing editor payload test**

```python
    def test_list_placement_bodies_includes_world_aabb_for_editable_bodies(self):
        session = self.build_pose_edit_session()

        payload = session.call_once("list_placement_bodies", ())

        self.assertEqual(payload["items"][0]["body"], "Scene_Crate")
        self.assertEqual(payload["items"][0]["world_aabb"], {
            "min": [0.8, 1.8, 2.8],
            "max": [1.2, 2.2, 3.2],
        })
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_import.ZapdosImportTest.test_list_placement_bodies_includes_world_aabb_for_editable_bodies -v`

Expected: FAIL with `KeyError: 'world_aabb'`.

- [ ] **Step 3: Implement body-level AABB aggregation and editor serialization**

```python
def body_world_aabb(self, body_name: str) -> dict[str, list[float]] | None:
    body_id = self._body_id(body_name)
    bounds_min = None
    bounds_max = None
    for geom_id in range(self.model.ngeom):
        if int(self.model.geom_bodyid[geom_id]) != body_id:
            continue
        geom_bounds = self._geom_world_bounds(geom_id)
        if geom_bounds is None:
            continue
        geom_min, geom_max = geom_bounds
        bounds_min = geom_min if bounds_min is None else np.minimum(bounds_min, geom_min)
        bounds_max = geom_max if bounds_max is None else np.maximum(bounds_max, geom_max)
    if bounds_min is None or bounds_max is None:
        return None
    return {
        "min": bounds_min.astype(float).round(6).tolist(),
        "max": bounds_max.astype(float).round(6).tolist(),
    }

items.append(
    {
        "body": body,
        "label": self.session.physics.body_labels.get(body, body),
        "matrix": flatten_matrix(body_world_pose(self.session.physics.data, body_id)),
        "support": support_infos.get(body),
        "world_aabb": self.session.physics.body_world_aabb(body),
    }
)
```

- [ ] **Step 4: Run the focused import tests**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_import.ZapdosImportTest.test_list_placement_bodies_includes_world_aabb_for_editable_bodies apps.python.tests.test_zapdos_import.ZapdosImportTest.test_list_placement_bodies_returns_top_level_robot_bounds -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/python/utils/zapdos/physics/mujoco_physics.py apps/python/utils/zapdos/editor/zapdos_editor.py apps/python/tests/test_zapdos_import.py
git commit -m "feat: expose zapdos scene body aabbs"
```

### Task 2: Extend the manipulation contracts for staged planning

**Files:**
- Modify: `apps/python/utils/zapdos/manipulation/types.py`
- Modify: `apps/python/utils/zapdos/manipulation/catalog.py`
- Modify: `apps/python/utils/zapdos/manipulation/runtime.py`
- Modify: `apps/python/tests/test_zapdos_manipulation.py`
- Modify: `apps/python/tests/test_zapdos_import.py`

- [ ] **Step 1: Write the failing contract tests**

```python
    def test_build_scene_object_catalog_preserves_world_aabb(self, resolve_asset_record, asset_local_bounds):
        catalog = build_scene_object_catalog(self._scene_bodies(), self._overlay_state())
        apple = next(item for item in catalog if item["body"] == "Scene_apple_1")
        self.assertEqual(apple["world_aabb"], {
            "min": [0.4539, -0.0469, 0.7834],
            "max": [0.5461, 0.0469, 0.8766],
        })

    def test_manipulation_runtime_passes_start_pose_and_scene_objects_to_planner(self):
        session = SimpleNamespace(
            editor=SimpleNamespace(
                scene_revision="rev-1",
                overlay_state={},
                list_placement_bodies=mock.Mock(return_value={"items": []}),
            ),
            bundle=SimpleNamespace(scene_usd=Path("scene.usda"), robot_usd=Path("robot.usda")),
            physics=mock.Mock(name="physics"),
        )
        catalog_loader = mock.Mock(return_value=[{
            "body": "Scene_Crate",
            "label": "crate",
            "asset_id": None,
            "motion": "dynamic",
            "tags": ["crate"],
            "support_body": None,
            "position": [0.2, 0.0, 0.2],
            "matrix": None,
            "top_z": 0.25,
            "bounds_min": [-0.05, -0.05, -0.05],
            "bounds_max": [0.05, 0.05, 0.05],
            "world_aabb": {"min": [0.15, -0.05, 0.15], "max": [0.25, 0.05, 0.25]},
        }])
        grounder = mock.Mock(return_value={"target": catalog_loader.return_value[0], "support": None})
        planner = mock.Mock(return_value={"kind": "pick", "target_body": "Scene_Crate", "stages": []})
        executor = mock.Mock(
            current_pose=mock.Mock(return_value={"position": [0.2, 0.1, 0.4], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}),
            execute=mock.Mock(return_value={"ok": True, "target_body": "Scene_Crate"}),
        )

        runtime = ManipulationRuntime(
            session,
            catalog_loader=catalog_loader,
            grounding_fn=grounder,
            planning_fn=planner,
            executor=executor,
        )
        runtime.pick_object({"target_query": "crate"})

        planner.assert_called_once_with(
            grounder.return_value["target"],
            support=grounder.return_value["support"],
            scene_objects=catalog_loader.return_value,
            arm="left",
            start_pose={"position": [0.2, 0.1, 0.4], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
        )
```

- [ ] **Step 2: Run the focused contract tests to verify they fail**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_manipulation.ZapdosManipulationTest.test_build_scene_object_catalog_preserves_world_aabb apps.python.tests.test_zapdos_import.ZapdosImportTest.test_manipulation_runtime_passes_start_pose_and_scene_objects_to_planner -v`

Expected: FAIL because `SceneObject` has no `world_aabb` and `ManipulationRuntime` does not pass `start_pose` or `scene_objects`.

- [ ] **Step 3: Extend the typed contracts and runtime planner call**

```python
class WorldAabb(TypedDict):
    min: list[float]
    max: list[float]

class PlanningPose(TypedDict):
    position: list[float]
    quat_wxyz: list[float]

class PickStage(TypedDict):
    name: str
    kind: str
    pose: NotRequired[PickPose]
    width: NotRequired[float]

class SceneObject(TypedDict):
    body: str
    label: str
    asset_id: str | None
    motion: str | None
    tags: list[str]
    support_body: str | None
    position: list[float] | None
    matrix: list[float] | None
    top_z: float | None
    bounds_min: list[float] | None
    bounds_max: list[float] | None
    world_aabb: WorldAabb | None

class PickPlan(TypedDict):
    kind: str
    target_body: str
    orientation: PickOrientation
    stages: list[PickStage]
    support_surface: NotRequired[SupportSurface]

def _plan_pick(self, grounded: GroundedPick, *, arm: str, objects: list[SceneObject]) -> PickPlan:
    return self.planning_fn(
        grounded["target"],
        support=grounded["support"],
        scene_objects=objects,
        arm=arm,
        start_pose=self.executor.current_pose(arm),
    )
```

- [ ] **Step 4: Run the runtime and catalog tests**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_manipulation apps.python.tests.test_zapdos_import.ZapdosImportTest.test_manipulation_runtime_passes_start_pose_and_scene_objects_to_planner -v`

Expected: PASS for the new catalog/runtime assertions and existing grounding coverage.

- [ ] **Step 5: Commit**

```bash
git add apps/python/utils/zapdos/manipulation/types.py apps/python/utils/zapdos/manipulation/catalog.py apps/python/utils/zapdos/manipulation/runtime.py apps/python/tests/test_zapdos_manipulation.py apps/python/tests/test_zapdos_import.py
git commit -m "refactor: prepare zapdos staged pick planning"
```

### Task 3: Implement the staged planner with scene AABB routing

**Files:**
- Modify: `apps/python/utils/zapdos/manipulation/planner.py`
- Modify: `apps/python/tests/test_zapdos_manipulation.py`

- [ ] **Step 1: Write the failing staged-plan tests**

```python
    def test_plan_pick_adds_escape_and_transit_when_start_is_under_support(self, resolve_asset_record, asset_local_bounds):
        catalog = build_scene_object_catalog(self._scene_bodies(), self._overlay_state())
        grounded = ground_pick_target(catalog, target_query="apple", support_query="table")
        start_pose = {"position": [0.25, 0.0, 0.2], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}

        plan = plan_pick(
            grounded["target"],
            support=grounded["support"],
            scene_objects=catalog,
            arm="left",
            start_pose=start_pose,
        )

        self.assertEqual([stage["name"] for stage in plan["stages"][:3]], [
            "escape_xy",
            "raise_to_transit",
            "approach_xy",
        ])
        self.assertGreater(plan["stages"][1]["pose"]["position"][2], 0.84)

    def test_plan_pick_rejects_missing_support_geometry(self, resolve_asset_record, asset_local_bounds):
        target = {
            "body": "Scene_apple_1",
            "label": "apple",
            "asset_id": "apple_red",
            "motion": "dynamic",
            "tags": ["apple"],
            "support_body": "table_body",
            "position": [0.5, 0.0, 0.83],
            "matrix": _matrix_at(0.5, 0.0, 0.83),
            "top_z": 0.86,
            "bounds_min": [-0.0461, -0.0469, -0.0466],
            "bounds_max": [0.0461, 0.0469, 0.0466],
            "world_aabb": {
                "min": [0.4539, -0.0469, 0.7834],
                "max": [0.5461, 0.0469, 0.8766],
            },
        }
        with self.assertRaises(ValueError) as err:
            plan_pick(
                target,
                support=None,
                scene_objects=[target],
                arm="left",
                start_pose={"position": [0.0, 0.0, 0.5], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
            )
        self.assertEqual(str(err.exception), "planner_insufficient_geometry: support surface bounds are required")
```

- [ ] **Step 2: Run the focused planner tests to verify they fail**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_manipulation.ZapdosManipulationTest.test_plan_pick_adds_escape_and_transit_when_start_is_under_support apps.python.tests.test_zapdos_manipulation.ZapdosManipulationTest.test_plan_pick_rejects_missing_support_geometry -v`

Expected: FAIL because `plan_pick()` still returns only `pre_grasp`, `grasp`, and `lift`.

- [ ] **Step 3: Implement staged routing**

```python
def plan_pick(
    target: SceneObject,
    *,
    support: SceneObject | None,
    scene_objects: list[SceneObject],
    arm: str,
    start_pose: PlanningPose,
    xy_margin: float = 0.06,
    z_margin: float = 0.05,
) -> PickPlan:
    support_surface = _support_surface(support)
    if support_surface is None:
        raise ValueError("planner_insufficient_geometry: support surface bounds are required")
    pre_grasp = _pose(target["position"][0], target["position"][1], _grasp_z(target) + 0.12)
    grasp = _pose(target["position"][0], target["position"][1], _grasp_z(target))
    transit_z = _transit_z(start_pose, pre_grasp, scene_objects, ignore={target["body"]}, z_margin=z_margin)
    stages: list[PickStage] = []
    if _needs_escape(start_pose["position"], support_surface):
        escape_xy = _escape_xy(start_pose["position"], support_surface, xy_margin)
        stages.append(_move_stage("escape_xy", escape_xy[0], escape_xy[1], start_pose["position"][2], start_pose["quat_wxyz"]))
        stages.append(_move_stage("raise_to_transit", escape_xy[0], escape_xy[1], transit_z, pre_grasp["quat_wxyz"]))
    elif start_pose["position"][2] < transit_z:
        stages.append(_move_stage("raise_to_transit", start_pose["position"][0], start_pose["position"][1], transit_z, pre_grasp["quat_wxyz"]))
    stages.extend([
        _move_stage("approach_xy", pre_grasp["position"][0], pre_grasp["position"][1], transit_z, pre_grasp["quat_wxyz"]),
        _move_stage("descend_to_pregrasp", *pre_grasp["position"], pre_grasp["quat_wxyz"]),
        _move_stage("descend_to_grasp", *grasp["position"], grasp["quat_wxyz"]),
        {"name": "close_gripper", "kind": "gripper", "width": 0.0},
        _move_stage("retreat", pre_grasp["position"][0], pre_grasp["position"][1], transit_z, pre_grasp["quat_wxyz"]),
    ])
    return {
        "kind": "pick",
        "target_body": target["body"],
        "orientation": {"mode": "top_down", "quat_wxyz": list(TOP_DOWN_QUAT_WXYZ)},
        "stages": stages,
        "support_surface": support_surface,
    }
```

- [ ] **Step 4: Run the planner suite**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_manipulation -v`

Expected: PASS, including the staged-plan assertions and the existing center-of-object pose checks updated to assert the last movement stage instead of `lift`.

- [ ] **Step 5: Commit**

```bash
git add apps/python/utils/zapdos/manipulation/planner.py apps/python/tests/test_zapdos_manipulation.py
git commit -m "feat: add zapdos staged pick planner"
```

### Task 4: Refactor the executor to consume staged plans

**Files:**
- Modify: `apps/python/utils/zapdos/manipulation/executor.py`
- Modify: `apps/python/tests/test_zapdos_pick_executor.py`

- [ ] **Step 1: Write the failing executor regressions**

```python
    def test_execute_runs_stage_sequence_in_order(self):
        physics = _FakePhysics()
        ik = _MutableIK(_pose(0.25, 0.0, 0.2))
        executor = PickExecutor(physics, bundle=_bundle(), ik_controller=ik)
        visited: list[str] = []
        plan = {
            "arm": "left",
            "target_body": "Scene_Crate",
            "stages": [
                {"name": "escape_xy", "kind": "move_pose", "pose": {"position": [0.46, 0.0, 0.2], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
                {"name": "raise_to_transit", "kind": "move_pose", "pose": {"position": [0.46, 0.0, 0.92], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
                {"name": "close_gripper", "kind": "gripper", "width": 0.0},
            ],
        }

        def drive(_ik_controller, _arm, target, _gripper, steps=12):
            del steps
            visited.append(target["position"])

        with mock.patch.object(executor, "_drive_pose", side_effect=drive):
            with self.assertRaises(HTTPException):
                executor.execute(plan)

        self.assertEqual(visited[:2], [(0.46, 0.0, 0.2), (0.46, 0.0, 0.92)])

    def test_execute_reports_stage_name_when_motion_stage_fails(self):
        executor = PickExecutor(_FakePhysics(), bundle=_bundle(), ik_controller=_StaticIK(_pose(1.0, 0.0, 0.0)))
        with self.assertRaises(HTTPException) as err:
            executor.execute(_staged_plan())
        self.assertIn("descend_to_grasp", err.exception.detail)
```

- [ ] **Step 2: Run the focused executor tests to verify they fail**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_pick_executor.PickExecutorTest.test_execute_runs_stage_sequence_in_order apps.python.tests.test_zapdos_pick_executor.PickExecutorTest.test_execute_reports_stage_name_when_motion_stage_fails -v`

Expected: FAIL because `PickExecutor.execute()` still expects `pre_grasp`, `grasp`, and `lift`.

- [ ] **Step 3: Refactor the executor around `stages`**

```python
def current_pose(self, arm: str) -> dict[str, list[float]]:
    ik = self._ensure_ik()
    ik.sync_joint_state(self.physics.joint_state_msg())
    pose = ik.get_end_effector_pose(arm)
    return {
        "position": [float(v) for v in pose["position"]],
        "quat_wxyz": [float(v) for v in pose["rotation"]],
    }

def execute(self, plan: dict[str, object]) -> dict[str, object]:
    arm = str(plan.get("arm") or "left")
    ik = self._ensure_ik()
    ik.sync_joint_state(self.physics.joint_state_msg())
    closed_width = 0.0
    attached = False
    for stage in plan.get("stages") or []:
        stage_name = str(stage["name"])
        if stage["kind"] == "move_pose":
            target = self._pose(stage["pose"])
            self._drive_pose(ik, arm, target, closed_width if attached else float(plan.get("open_gripper", 0.04)))
            self._require_pose_reached(ik, arm, target, DRIVE_POSE_TOLERANCE, stage_name)
        elif stage["kind"] == "gripper":
            closed_width = float(stage.get("width", 0.0))
            grasp_stage = next(item for item in plan["stages"] if item["name"] == "descend_to_grasp")
            grasp_pose = self._pose(grasp_stage["pose"])
            self._drive_pose(ik, arm, grasp_pose, closed_width, steps=6)
            self._require_pose_reached(ik, arm, grasp_pose, GRASP_POSE_TOLERANCE, stage_name)
            self._require_target_near_gripper(ik, arm, str(plan["target_body"]), TARGET_ATTACH_TOLERANCE)
            self.physics.attach_body(str(plan.get("gripper_body") or get_arm_config(arm).end_effector_body), str(plan["target_body"]))
            attached = True
    return {
        "ok": True,
        "arm": arm,
        "target_body": str(plan["target_body"]),
        "attachment": self.physics.get_attachment(str(plan["target_body"])),
    }
```

- [ ] **Step 4: Run the executor suite, including the real MuJoCo loop**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_pick_executor -v`

Expected: PASS, including the real `test_execute_attaches_and_lifts_collisionless_target_in_mujoco_physics_loop` updated to build a staged plan with `approach_xy`, `descend_to_pregrasp`, `descend_to_grasp`, `close_gripper`, and `retreat`.

- [ ] **Step 5: Commit**

```bash
git add apps/python/utils/zapdos/manipulation/executor.py apps/python/tests/test_zapdos_pick_executor.py
git commit -m "refactor: execute staged zapdos pick plans"
```

### Task 5: Verify the end-to-end runtime and smoke the UI path

**Files:**
- Modify: none

- [ ] **Step 1: Run the focused backend regression suite**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_manipulation apps.python.tests.test_zapdos_pick_executor apps.python.tests.test_zapdos_import apps.python.tests.test_spacemouse_ik -v`

Expected: PASS.

- [ ] **Step 2: Run a direct Python call smoke against the session runtime**

Run:

```powershell
@'
import asyncio
from utils.zapdos.zapdos_session import ZapdosSession, DEFAULT_ROBOT_USD
from utils.zapdos.bundle import DEFAULT_SCENE_USD

async def main():
    session = await ZapdosSession.create("plan-smoke", DEFAULT_ROBOT_USD, DEFAULT_SCENE_USD)
    print(session.call_once("list_manipulation_objects", ())["scene_revision"])
    print(session.call_once("pick_object", ({"target_query": "apple", "support_query": "table"},)))
    session.destroy()

asyncio.run(main())
'@ | uv run --project apps/python python -
```

Expected: The first line prints a non-empty scene revision, and the second line prints a dict containing `"ok": True` or a clear `HTTPException` detail naming the failed stage.

- [ ] **Step 3: Manual UI smoke for the natural-language path**

Run:

```powershell
pnpm --dir apps/web dev
uv run --project apps/python python apps/python/app.py
```

Expected: In `/demo/zapdos`, the prompt `pick up the apple on the table` triggers `pick_object`, and if the motion fails the error mentions the exact stage such as `escape_xy` or `descend_to_grasp` rather than a generic attach failure.

- [ ] **Step 4: Capture any geometry gaps before widening scope**

Run: none

Expected: If the smoke fails because a support body lacks AABB data or the chosen target asset needs a non-central grasp offset, add those as follow-up tasks instead of weakening the staged planner back to direct IK.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-05-14-zapdos-staged-pick-planner.md
git commit -m "docs: add zapdos staged pick planner plan"
```
