# Zapdos Physics/Renderer Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `apps/python/utils/zapdos` into clear `physics`, `renderer`, `bundle`, `overlay`, `rebuild`, `ros`, and `session` layers so `ZapdosSession` becomes a thin coordinator with maintainable extension points.

**Architecture:** Replace the current flat utility pile with backend-oriented packages. `ZapdosSession` will hold only session state plus `physics`, `renderer`, ROS, and rebuild collaborators; all MuJoCo specifics move under `physics`, all Isaac specifics move under `renderer`, bundle assembly lives under `bundle`, and expensive scene rebuild orchestration moves under `rebuild`.

**Tech Stack:** Python 3.12, FastAPI, MuJoCo, Isaac Sim, asyncio, `unittest`, `uv`

## Session Handoff (2026-05-10)

- Continue in the current checkout. The worktree already contains uncommitted rewrite changes across `apps/python/utils/zapdos/*`, `apps/python/tests/*`, and `apps/python/api/zapdos/{session}/{action}.py`.
- Package migration completed so far:
  - `apps/python/api/zapdos/{session}/{action}.py` now imports from `utils.zapdos.bundle`, `utils.zapdos.overlay.overlay_state`, and `utils.zapdos.rebuild.scene_rebuild_manager`.
  - `apps/python/utils/zapdos/bundle/camera_specs.py` now owns `RenderCamera`, `build_render_cameras`, `camera_name_to_index`, `cameras_json`, and `image_topic`; `apps/python/utils/zapdos/rl_cameras.py` is now a legacy compatibility wrapper.
  - Consumer imports were migrated off flat modules in `apps/python/teleop/manager.py`, `apps/python/teleop/ik_controller.py`, `apps/python/utils/camera_override.py`, `apps/python/utils/zapdos/renderer/base.py`, `apps/python/utils/zapdos/renderer/isaac_renderer.py`, and `apps/python/utils/zapdos/zapdos_overlay_rebuild_runner.py`.
  - `apps/python/utils/zapdos/bundle/__init__.py` now uses lazy exports to avoid the `camera_override -> bundle -> rl_bundle` circular import.
  - `apps/python/utils/zapdos/rebuild/scene_rebuild_manager.py` now owns overlay rebuild orchestration and `scene_rebuild_job` state/stream helpers; `apps/python/utils/zapdos/session/zapdos_session.py` now tracks `scene_rebuild_jobs` instead of `scene_operations`.
  - `apps/python/utils/zapdos/session/request_router.py` now owns request bootstrap/dispatch helpers; `apps/python/api/zapdos/{session}/{action}.py` is now a thin adapter that re-exports the stable route surface.
  - `apps/python/utils/zapdos/rebuild/scene_rebuild_service.py` now backs `ZapdosSession` scene-asset submission and overlay-completion draining so rebuild orchestration has a dedicated seam.
  - `apps/python/utils/zapdos/bundle/{render_bundle,bundle_builder,scene_catalog,stage_builder}.py` now own their implementations instead of delegating back into `rl_bundle.py`, `scene_objects.py`, and `rl_bundle_stage.py`.
  - Legacy bundle files `apps/python/utils/zapdos/{rl_bundle,scene_objects,rl_bundle_stage}.py` are now compatibility wrappers over the new bundle package instead of being the primary owners.
  - `apps/python/utils/zapdos/rebuild/overlay_rebuild_runner.py` now owns inline overlay rebuild preparation; `scene_rebuild_manager.py` and `apps/python/scripts/prepare_zapdos_overlay_rebuild.py` now delegate to it.
  - `apps/python/utils/zapdos/overlay/overlay_commands.py` now owns `build_set_scene_assets_overlay` and `build_remove_asset_overlay`.
  - `apps/python/utils/zapdos/overlay/{overlay_state,overlay_repository,overlay_placement,overlay_scene_writer}.py` now own their implementations instead of re-exporting the flat overlay files.
  - Overlay rebuild tests were aligned to the new seams, including patching through `SESSION_MODULE.rebuild_manager` and the new bundle/overlay package imports.
  - `apps/python/utils/zapdos/{zapdos_scene_operations,zapdos_overlay,zapdos_overlay_scene}.py` were deleted after their callers migrated.
- Fresh verification on 2026-05-10:

```powershell
uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_import apps.python.tests.test_session apps.python.tests.test_session_registry apps.python.tests.test_zapdos_overlay apps.python.tests.test_zapdos_overlay_scene apps.python.tests.test_zapdos_overlay_rebuild_diagnostics apps.python.tests.test_zapdos_overlay_rebuild_subprocess -v
```

- Verification result: `Ran 69 tests in 31.355s`, `OK`.
- Focused verification on 2026-05-10 after the router/service extraction:

```powershell
uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_import apps.python.tests.test_session apps.python.tests.test_session_registry apps.python.tests.test_zapdos_package_namespace -v
```

- Verification result: `Ran 49 tests in 15.988s`, `OK`.
- Combined verification on 2026-05-10 after the bundle owner migration and rebuild-runner extraction:

```powershell
uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_import apps.python.tests.test_session apps.python.tests.test_session_registry apps.python.tests.test_zapdos_package_namespace apps.python.tests.test_rl_bundle apps.python.tests.test_rl_bundle_ground apps.python.tests.test_zapdos_overlay_rebuild_diagnostics apps.python.tests.test_zapdos_overlay_rebuild_subprocess -v
```

- Verification result: `Ran 73 tests in 306.894s`, `OK`.
- Remaining production migration surface:
  - `apps/python/utils/zapdos/session/zapdos_session.py` is still 364 lines, so Task 6's file-size target is not met yet; the next cut should keep splitting coordinator logic out of the session module.
  - Flat compatibility modules still present and scheduled for deletion include `rl_bundle.py`, `rl_bundle_stage.py`, `rl_cameras.py`, `scene_objects.py`, `sim_env.py`, `zapdos_physics.py`, `zapdos_scene_visuals.py`, `zapdos_session.py`, and `zapdos_overlay_rebuild_runner.py`.
  - The remaining `rg` hits for `utils.zapdos.rl_*` or deleted overlay module names are now test string literals, compatibility-wrapper files, or test stubs rather than new-package production owners.
  - `rg -n "scene_operation|SceneOperation|stream_scene_operation" apps/python apps/python/tests` is now clean; remaining `rg` hits for legacy modules are test string literals, test stubs, or the bundle/flat-module files listed above.
- When re-running `rg` for legacy imports, ignore hits that are only test string literals or test stubs. The real remaining production owners are the files listed above.

---

## File Structure

- `apps/python/utils/zapdos/session/`
  Thin session coordinator, state, and request dispatch helpers.
- `apps/python/utils/zapdos/physics/`
  Physics backend protocol plus MuJoCo implementation and visual serialization.
- `apps/python/utils/zapdos/renderer/`
  Renderer backend protocol plus Isaac implementation, process control, IPC, and frame transport.
- `apps/python/utils/zapdos/bundle/`
  Render bundle models, builders, stage assembly, scene catalog, and camera specs.
- `apps/python/utils/zapdos/overlay/`
  Overlay state, persistence, placement normalization, pure commands, and USD scene writing.
- `apps/python/utils/zapdos/rebuild/`
  Rebuild request types, job tracking, SSE stream, and rebuild orchestration.
- `apps/python/utils/zapdos/ros/`
  Shared ROS topic/type constants and publish loop support.
- `apps/python/api/zapdos/{session}/{action}.py`
  Keep the route stable while switching imports to the new package layout.

### Task 1: Lay down package boundaries and base protocols

**Files:**
- Create: `apps/python/utils/zapdos/session/__init__.py`
- Create: `apps/python/utils/zapdos/session/session_state.py`
- Create: `apps/python/utils/zapdos/physics/__init__.py`
- Create: `apps/python/utils/zapdos/physics/base.py`
- Create: `apps/python/utils/zapdos/renderer/__init__.py`
- Create: `apps/python/utils/zapdos/renderer/base.py`
- Create: `apps/python/utils/zapdos/rebuild/__init__.py`
- Create: `apps/python/utils/zapdos/rebuild/scene_rebuild_job.py`
- Modify: `apps/python/tests/test_zapdos_package_namespace.py`

- [ ] **Step 1: Write the failing namespace test for the new package layout**

```python
        module_names = [
            "utils.zapdos.session.session_state",
            "utils.zapdos.physics.base",
            "utils.zapdos.renderer.base",
            "utils.zapdos.rebuild.scene_rebuild_job",
        ]
```

- [ ] **Step 2: Run the focused namespace test to verify it fails**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_package_namespace.ZapdosPackageNamespaceTest.test_moved_zapdos_modules_are_importable_from_package_namespace -v`

Expected: FAIL because the new packages do not exist yet.

- [ ] **Step 3: Create the package skeleton and base interfaces**

```python
class PhysicsBackend(Protocol):
    def get_visual(self) -> SceneVisuals: ...
    def get_pose(self) -> dict[str, list[float]]: ...
    def get_camera(self) -> dict[str, list[float]]: ...
    def set_body_pose(self, body: str, pos: list[float], quat: list[float]) -> dict[str, object]: ...
    def step(self) -> None: ...
    def close(self) -> None: ...
```

```python
class RendererBackend(Protocol):
    async def wait_ready(self, timeout: float = 300.0) -> dict[str, Any]: ...
    def read(self, camera_name: str) -> tuple[int, np.ndarray] | None: ...
    def reload_scene(self, bundle: RenderBundle, timeout: float = 30.0) -> None: ...
    def snapshot_cameras(self, timeout: float = 5.0) -> list[dict[str, Any]]: ...
    def status(self) -> dict[str, Any]: ...
    def close(self, stop_remote: bool = True) -> None: ...
```

- [ ] **Step 4: Run the namespace test again**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_package_namespace -v`

Expected: PASS with the new packages importable.

- [ ] **Step 5: Commit the scaffolding**

```powershell
git add apps/python/utils/zapdos/session apps/python/utils/zapdos/physics apps/python/utils/zapdos/renderer apps/python/utils/zapdos/rebuild apps/python/tests/test_zapdos_package_namespace.py
git commit -m "refactor: add zapdos package boundaries"
```

### Task 2: Extract shared ROS and visual data contracts

**Files:**
- Create: `apps/python/utils/zapdos/ros/topics.py`
- Create: `apps/python/utils/zapdos/physics/visuals.py`
- Modify: `apps/python/teleop/ros_client.py`
- Modify: `apps/python/utils/ros_view_store.py`
- Modify: `apps/python/utils/ros_view_topics.py`
- Modify: `apps/python/tests/test_spacemouse_ros_client.py`

- [ ] **Step 1: Write the failing ROS import regression**

```python
from utils.zapdos.ros.topics import JOINT_COMMAND_TOPIC, JOINT_STATE_TYPE, JOINT_STATES_TOPIC
```

- [ ] **Step 2: Run the focused ROS test to verify it fails**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_spacemouse_ros_client -v`

Expected: FAIL with `ModuleNotFoundError` for `utils.zapdos.ros.topics`.

- [ ] **Step 3: Move topic/type constants and scene visual typed dicts out of backend files**

```python
JOINT_COMMAND_TOPIC = "/env_0/joint_command"
JOINT_STATES_TOPIC = "/env_0/joint_states"
TF_RENDER_TOPIC = "/env_0/tf_render"
JOINT_STATE_TYPE = "sensor_msgs/msg/JointState"
IMAGE_TYPE = "sensor_msgs/msg/Image"
TF_RENDER_TYPE = "tf2_msgs/msg/TFMessage"
```

- [ ] **Step 4: Update Python callers to import the new shared modules**

Run: `rg -n "sim_env import JOINT_|zapdos_scene_visuals" apps/python`

Expected: only the new `ros/topics.py` and `physics/visuals.py` remain as owners of those contracts.

- [ ] **Step 5: Run the focused ROS and visual tests**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_spacemouse_ros_client apps.python.tests.test_zapdos_package_namespace -v`

Expected: PASS.

### Task 3: Rebuild the physics layer around `MujocoPhysics`

**Files:**
- Create: `apps/python/utils/zapdos/physics/mujoco_tools.py`
- Create: `apps/python/utils/zapdos/physics/mujoco_physics.py`
- Modify: `apps/python/api/zapdos/{session}/{action}.py`
- Modify: `apps/python/tests/test_zapdos_import.py`

- [ ] **Step 1: Write the failing session import assertions for the renamed physics layer**

```python
        self.assertEqual(MODULE.ZapdosSession.__module__, "utils.zapdos.session.zapdos_session")
```

- [ ] **Step 2: Run the focused Zapdos import test to verify it fails**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_import.ZapdosImportTest.test_action_module_reexports_split_runtime_symbols -v`

Expected: FAIL because the action module still points at the old flat session/physics files.

- [ ] **Step 3: Move MuJoCo-specific behavior under `physics/`**

```python
class MujocoPhysics:
    def __init__(self, sess: str, bundle: Any, body_map: dict[str, str]) -> None: ...
    def get_visual(self) -> SceneVisuals: ...
    def get_pose(self) -> dict[str, list[float]]: ...
    def get_camera(self) -> dict[str, list[float]]: ...
    def set_body_pose(self, body: str, pos: list[float], quat: list[float]) -> dict[str, object]: ...
```

- [ ] **Step 4: Update routes and tests to import `session.zapdos_session` and `physics.mujoco_physics`**

Run: `rg -n "zapdos_physics|zapdos_session" apps/python apps/python/tests`

Expected: old flat physics/session imports only remain in files scheduled for deletion.

- [ ] **Step 5: Run the focused physics/session regression set**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_import apps.python.tests.test_session apps.python.tests.test_session_registry -v`

Expected: PASS.

### Task 4: Rebuild the renderer layer and delete `sim_env.py` responsibilities

**Files:**
- Create: `apps/python/utils/zapdos/renderer/isaac_process.py`
- Create: `apps/python/utils/zapdos/renderer/control_channel.py`
- Create: `apps/python/utils/zapdos/renderer/frame_buffer.py`
- Create: `apps/python/utils/zapdos/renderer/isaac_renderer.py`
- Modify: `apps/python/tests/test_sim_env_renderer.py`
- Modify: `apps/python/tests/test_renderer_reload.py`

- [ ] **Step 1: Rename the renderer tests to target the new package path**

```python
from utils.zapdos.renderer.isaac_renderer import IsaacRenderer
```

- [ ] **Step 2: Run the focused renderer test to verify it fails**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_sim_env_renderer -v`

Expected: FAIL because the renderer package has not been implemented yet.

- [ ] **Step 3: Split process control, control IPC, and shared-memory frame reads into separate modules**

```python
class IsaacRenderer:
    def __init__(self, sess: str, bundle: RenderBundle, width: int, height: int, render_hz: float, headless: bool, ros_domain_id: int) -> None: ...
    async def wait_ready(self, timeout: float = 300.0) -> dict[str, Any]: ...
    def read(self, camera_name: str) -> tuple[int, np.ndarray] | None: ...
    def reload_scene(self, bundle: RenderBundle, timeout: float = 30.0) -> None: ...
```

- [ ] **Step 4: Move helper constants and functions out of the deleted `sim_env.py` surface**

Run: `rg -n "utils\\.zapdos\\.sim_env|from utils\\.zapdos\\.sim_env" apps/python apps/python/tests`

Expected: no remaining production imports from `sim_env.py`.

- [ ] **Step 5: Run the focused renderer regression set**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_sim_env_renderer apps.python.tests.test_renderer_reload -v`

Expected: PASS with the new renderer package.

### Task 5: Rebuild bundle, overlay, and rebuild orchestration layers

**Files:**
- Create: `apps/python/utils/zapdos/bundle/render_bundle.py`
- Create: `apps/python/utils/zapdos/bundle/bundle_builder.py`
- Create: `apps/python/utils/zapdos/bundle/stage_builder.py`
- Create: `apps/python/utils/zapdos/bundle/camera_specs.py`
- Create: `apps/python/utils/zapdos/bundle/scene_catalog.py`
- Create: `apps/python/utils/zapdos/bundle/usd_to_mjcf_adapter.py`
- Create: `apps/python/utils/zapdos/overlay/overlay_state.py`
- Create: `apps/python/utils/zapdos/overlay/overlay_repository.py`
- Create: `apps/python/utils/zapdos/overlay/overlay_commands.py`
- Create: `apps/python/utils/zapdos/overlay/overlay_placement.py`
- Create: `apps/python/utils/zapdos/overlay/overlay_scene_writer.py`
- Create: `apps/python/utils/zapdos/rebuild/scene_rebuild_manager.py`
- Create: `apps/python/utils/zapdos/rebuild/scene_rebuild_service.py`
- Modify: `apps/python/tests/test_rl_bundle.py`
- Modify: `apps/python/tests/test_rl_bundle_ground.py`
- Modify: `apps/python/tests/test_zapdos_overlay.py`
- Modify: `apps/python/tests/test_zapdos_overlay_scene.py`
- Modify: `apps/python/tests/test_zapdos_overlay_rebuild_diagnostics.py`
- Modify: `apps/python/tests/test_zapdos_overlay_rebuild_subprocess.py`

- [ ] **Step 1: Point existing bundle and overlay tests at the new package layout**

```python
from utils.zapdos.bundle.bundle_builder import ensure_render_bundle
from utils.zapdos.overlay.overlay_state import default_overlay_state
from utils.zapdos.rebuild.scene_rebuild_manager import stream_scene_rebuild_job
```

- [ ] **Step 2: Run the focused bundle/overlay suite to verify it fails**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_rl_bundle apps.python.tests.test_rl_bundle_ground apps.python.tests.test_zapdos_overlay apps.python.tests.test_zapdos_overlay_scene apps.python.tests.test_zapdos_overlay_rebuild_diagnostics apps.python.tests.test_zapdos_overlay_rebuild_subprocess -v`

Expected: FAIL because the new package modules do not exist yet.

- [ ] **Step 3: Rebuild the bundle and overlay layers with the new ownership rules**

```python
class SceneRebuildService:
    def submit_replace(self, assets: list[dict[str, object]]) -> dict[str, object]: ...
    def submit_remove(self, instance_id: str) -> dict[str, object]: ...
    def drain_completions(self) -> None: ...
```

```python
@dataclass
class SceneRebuildJob:
    future: ConcurrentFuture
    success_payload: dict[str, object]
    events: queue.Queue[tuple[str, dict[str, object]]]
```

- [ ] **Step 4: Replace the `scene_operation` name everywhere with `scene_rebuild_job` terminology**

Run: `rg -n "scene_operation|SceneOperation|stream_scene_operation" apps/python apps/python/tests`

Expected: no remaining occurrences outside commit history and unrelated docs.

- [ ] **Step 5: Run the focused bundle/overlay/rebuild suite again**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_rl_bundle apps.python.tests.test_rl_bundle_ground apps.python.tests.test_zapdos_overlay apps.python.tests.test_zapdos_overlay_scene apps.python.tests.test_zapdos_overlay_rebuild_diagnostics apps.python.tests.test_zapdos_overlay_rebuild_subprocess -v`

Expected: PASS.

### Task 6: Rewrite `ZapdosSession`, migrate the route, and remove flat modules

**Files:**
- Create: `apps/python/utils/zapdos/session/request_router.py`
- Create: `apps/python/utils/zapdos/session/zapdos_session.py`
- Modify: `apps/python/api/zapdos/{session}/{action}.py`
- Delete: `apps/python/utils/zapdos/sim_env.py`
- Delete: `apps/python/utils/zapdos/rl_bundle.py`
- Delete: `apps/python/utils/zapdos/rl_bundle_stage.py`
- Delete: `apps/python/utils/zapdos/rl_cameras.py`
- Delete: `apps/python/utils/zapdos/scene_objects.py`
- Delete: `apps/python/utils/zapdos/zapdos_overlay.py`
- Delete: `apps/python/utils/zapdos/zapdos_overlay_scene.py`
- Delete: `apps/python/utils/zapdos/zapdos_physics.py`
- Delete: `apps/python/utils/zapdos/zapdos_scene_operations.py`
- Delete: `apps/python/utils/zapdos/zapdos_scene_visuals.py`

- [ ] **Step 1: Rebuild `ZapdosSession` as a thin coordinator**

```python
class ZapdosSession(Session):
    @staticmethod
    async def create(sess: str, robot_usd: Path, scene_usd: Path): ...
    def call_once(self, method: str, args: tuple): ...
    def send_sse(self): ...
    async def render(self, camera_name: str): ...
    def destroy(self): ...
```

- [ ] **Step 2: Route all API imports through the new package structure**

Run: `rg -n "utils\\.zapdos\\.(sim_env|rl_bundle|rl_bundle_stage|rl_cameras|scene_objects|zapdos_overlay|zapdos_overlay_scene|zapdos_physics|zapdos_scene_operations|zapdos_scene_visuals)" apps/python apps/python/tests`

Expected: no remaining production or test imports from the deleted flat modules.

- [ ] **Step 3: Run the end-to-end Zapdos Python regression set**

Run:

```powershell
uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_import apps.python.tests.test_zapdos_render_camera apps.python.tests.test_session apps.python.tests.test_session_registry apps.python.tests.test_zapdos_package_namespace -v
```

Expected: PASS with `apps/python/api/zapdos/{session}/{action}.py` still serving the same route contract.

- [ ] **Step 4: Check the session file size and delete any leftover dead modules**

Run:

```powershell
(Get-Content 'apps/python/utils/zapdos/session/zapdos_session.py').Count
Get-ChildItem 'apps/python/utils/zapdos' -File | Select-Object Name
```

Expected: `zapdos_session.py` is roughly within the 200-line target, and the old flat modules listed above are gone.

- [ ] **Step 5: Commit the rewrite**

```powershell
git add apps/python/utils/zapdos apps/python/api/zapdos/{session}/{action}.py apps/python/tests apps/python/teleop apps/python/utils/ros_view_store.py apps/python/utils/ros_view_topics.py
git commit -m "refactor: rewrite zapdos around physics and renderer layers"
```

## Verification Checklist

- `ZapdosSession` owns only session coordination, not backend implementation details.
- `physics/` owns MuJoCo-only behavior.
- `renderer/` owns Isaac-only behavior.
- `bundle/` is the only layer allowed to call the USD-to-MJCF adapter.
- `scene_operation` terminology is fully removed in favor of `scene_rebuild_job`.
- `/python/zapdos` routes and the current web demo behavior stay intact.

## Assumptions

- This rewrite keeps the current user-visible behavior where scene edits still trigger rebuilds immediately.
- Breaking Python import paths inside `apps/python` is acceptable as long as all callers and tests migrate in the same branch.
- `usd_to_mjcf.py` stays as a legacy implementation behind `bundle/usd_to_mjcf_adapter.py`; it is not rewritten in this plan.
