# Zapdos Async Session Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current `actor`/`idle_step_once`/`overlay_completions` design with coroutine-driven long operations that hand off small world-state mutations to the session thread via awaited calls.

**Architecture:** Keep the session thread as the only owner of `physics`, `renderer`, `bundle`, and attachment state. Run long manipulation and rebuild flows as asyncio tasks on the main event loop. Each task acquires explicit world ownership for the duration of the operation and advances by repeatedly awaiting session-thread work. Remove the actor list and the completion queue; keep the session loop responsible only for timers, queued calls, and the default physics tick when no world operation is active.

**Tech Stack:** Python `asyncio`, background session thread, `unittest`, existing FastAPI/ROS/Zapdos utilities.

---

## File Structure

- Modify: `apps/python/utils/session.py`
  - Replace actor scheduling with explicit session-thread callable execution plus world-ownership state.
- Modify: `apps/python/tests/test_session.py`
  - Cover `run_sync`/`call` handoff, world ownership, and default-tick suppression.
- Modify: `apps/python/utils/zapdos/zapdos_session.py`
  - Remove `idle_step_once` split and make the default tick the ordinary physics tick again.
- Modify: `apps/python/utils/zapdos/manipulation/runtime.py`
  - Replace `ManipulationActor` with async operation tasks that advance the existing iterator through awaited session-thread calls.
- Modify: `apps/python/utils/zapdos/manipulation/executor.py`
  - Keep the iterator contract stable if possible; only expose the minimal session-thread stepping helper the async runtime needs.
- Modify: `apps/python/utils/zapdos/editor/zapdos_editor.py`
  - Give the editor explicit rebuild state and an async rebuild runner instead of a dynamic completion queue.
- Modify: `apps/python/utils/zapdos/editor/rebuild_events.py`
  - Remove dynamic `hasattr` state injection and convert job state initialization to explicit construction.
- Modify: `apps/python/utils/zapdos/editor/rebuild_manager.py`
  - Replace queue-based prepare/apply handoff with an async flow that uses `asyncio.to_thread(...)` and awaited session-thread apply.
- Modify: `apps/python/tests/test_zapdos_import.py`
  - Update session, manipulation, and rebuild tests to the async-operation model.
- Modify: `apps/python/tests/test_zapdos_overlay_rebuild_diagnostics.py`
  - Verify rebuild diagnostics still stream progress/failure without `overlay_completions`.

### Task 1: Replace actor scheduling with explicit session-thread calls and world ownership

**Files:**
- Modify: `apps/python/utils/session.py`
- Test: `apps/python/tests/test_session.py`

- [ ] **Step 1: Write the failing tests**

```python
class SessionWorldOwnershipTest(unittest.IsolatedAsyncioTestCase):
    async def test_run_sync_executes_callable_on_session_thread(self):
        class DummySession(Session):
            def __init__(self) -> None:
                self.default_ticks = 0
                super().__init__(0)

            def step_once(self):
                self.default_ticks += 1
                time.sleep(0.001)

        session = DummySession()
        self.addCleanup(lambda: setattr(session, "active", 0))
        self.addCleanup(lambda: session.proc.join(timeout=1))

        result = await asyncio.wait_for(
            session.run_sync(lambda current: current.default_ticks),
            timeout=0.5,
        )

        self.assertIsInstance(result, int)

    async def test_default_step_pauses_while_world_is_reserved(self):
        class DummySession(Session):
            def __init__(self) -> None:
                self.default_ticks = 0
                super().__init__(0)

            def step_once(self):
                self.default_ticks += 1
                time.sleep(0.001)

        session = DummySession()
        self.addCleanup(lambda: setattr(session, "active", 0))
        self.addCleanup(lambda: session.proc.join(timeout=1))

        async with session.reserve_world():
            before = await session.run_sync(lambda current: current.default_ticks)
            await asyncio.sleep(0.05)
            after = await session.run_sync(lambda current: current.default_ticks)

        self.assertEqual(after, before)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m unittest tests.test_session.SessionWorldOwnershipTest -v`  
Expected: `ERROR`/`FAIL` because `Session.run_sync` and `Session.reserve_world` do not exist yet.

- [ ] **Step 3: Implement the minimal session API**

```python
@dataclass
class SessionCall:
    fn: Callable[["Session"], Any]
    future: asyncio.Future


class Session:
    def __init__(self, timeout=120) -> None:
        self.loop = asyncio.get_event_loop()
        self.calls: queue.Queue[SessionCall] = queue.Queue()
        self._world_owner_token: object | None = None

    async def run_sync(self, fn: Callable[["Session"], T]) -> T:
        future = asyncio.get_running_loop().create_future()
        self.calls.put_nowait(SessionCall(fn=fn, future=future))
        return await future

    async def call(self, method: str, *args):
        return await self.run_sync(lambda session: session.call_once(method, args))

    @asynccontextmanager
    async def reserve_world(self):
        token = object()
        await self.run_sync(lambda session: session._claim_world(token))
        try:
            yield
        finally:
            await self.run_sync(lambda session: session._release_world(token))
```

- [ ] **Step 4: Simplify the run loop around world ownership**

```python
def proc_once(self):
    call = self.calls.get(False)
    loop = call.future.get_loop()
    self.active = time.time()
    try:
        ret = call.fn(self)
        loop.call_soon_threadsafe(call.future.set_result, ret)
    except Exception as err:
        loop.call_soon_threadsafe(call.future.set_exception, err)

def run(self):
    while self.is_active():
        now = time.time()
        for timer in self.timers:
            timer.step(now)
        self.proc_calls()
        try:
            if self._world_owner_token is None:
                self.step_once()
        except Exception:
            traceback.print_exc()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m unittest tests.test_session -v`  
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add apps/python/utils/session.py apps/python/tests/test_session.py
git commit -m "refactor: replace session actors with world ownership"
```

### Task 2: Convert manipulation operations from actors to async tasks

**Files:**
- Modify: `apps/python/utils/zapdos/manipulation/runtime.py`
- Modify: `apps/python/utils/zapdos/manipulation/executor.py`
- Modify: `apps/python/utils/zapdos/zapdos_session.py`
- Test: `apps/python/tests/test_zapdos_import.py`

- [ ] **Step 1: Write the failing manipulation tests**

```python
async def test_grab_apple_schedules_async_operation_without_session_actors(self):
    queued_tasks = []

    def create_task(coro):
        task = asyncio.create_task(coro)
        queued_tasks.append(task)
        return task

    runtime = ManipulationRuntime(
        session,
        catalog_loader=catalog_loader,
        grounding_fn=grounder,
        executor=executor,
        create_task=create_task,
    )

    result = runtime.grab_apple()

    self.assertEqual(result, {"ok": True, "op_id": "op-1"})
    self.assertEqual(len(queued_tasks), 1)
```

```python
async def test_manipulation_operation_reserves_world_until_iterator_finishes(self):
    session = build_test_session()
    runtime = ManipulationRuntime(session, executor=executor)
    runtime.pick_object({"target_query": "crate"})

    await asyncio.wait_for(asyncio.gather(*runtime.operation_tasks), timeout=1)

    self.assertEqual(
        session.editor.scene_rebuild_jobs["op-1"].future.result(timeout=1),
        {"ok": True, "target_body": "Scene_Crate", "scene_revision": "rev-1"},
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m unittest tests.test_zapdos_import.ZapdosImportTest.test_manipulation_runtime_executes_grounded_pick_plan tests.test_zapdos_import.ZapdosImportTest.test_manipulation_runtime_executes_arm_only_grab_apple_plan -v`  
Expected: `FAIL` because the runtime still depends on `ManipulationActor` and session actor state.

- [ ] **Step 3: Replace `ManipulationActor` with an async task wrapper**

```python
class ManipulationRuntime:
    def __init__(..., create_task: Callable[[Coroutine[Any, Any, None]], asyncio.Task] | None = None) -> None:
        self.create_task = create_task or asyncio.create_task
        self.operation_tasks: set[asyncio.Task] = set()

    def _start_operation(self, iterator: Iterator[None]) -> dict[str, object]:
        if self.session.world_owned():
            raise HTTPException(status_code=409, detail="Manipulation already in progress")
        op_id = next_scene_rebuild_job_id(self.session.editor)
        create_scene_rebuild_job(self.session.editor, op_id, {"ok": True})
        task = self.create_task(self._run_operation(op_id, iterator))
        self.operation_tasks.add(task)
        task.add_done_callback(self.operation_tasks.discard)
        return {"ok": True, "op_id": op_id}
```

- [ ] **Step 4: Drive the existing iterator through awaited session-thread steps**

```python
async def _run_operation(self, op_id: str, iterator: Iterator[None]) -> None:
    try:
        async with self.session.reserve_world():
            while True:
                try:
                    payload = await self.session.run_sync(
                        lambda session: self._advance_iterator(iterator)
                    )
                except StopIteration as stop:
                    result = stop.value if isinstance(stop.value, dict) else {}
                    self._resolve_operation(op_id, result)
                    return
    except Exception as err:
        fail_scene_rebuild_job(self.session.editor, op_id, err)

def _advance_iterator(self, iterator: Iterator[None]) -> None:
    next(iterator)
```

- [ ] **Step 5: Fold `ZapdosSession` back to a single default tick**

```python
class ZapdosSession(Session):
    def step_once(self):
        self.physics.apply_joint_command(self._latest_joint_command())
        self.physics.step()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run python -m unittest tests.test_zapdos_import tests.test_zapdos_pick_executor -v`  
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add apps/python/utils/zapdos/manipulation/runtime.py apps/python/utils/zapdos/manipulation/executor.py apps/python/utils/zapdos/zapdos_session.py apps/python/tests/test_zapdos_import.py
git commit -m "refactor: run manipulation as async world operations"
```

### Task 3: Replace editor completion queue with explicit async rebuild flow

**Files:**
- Modify: `apps/python/utils/zapdos/editor/zapdos_editor.py`
- Modify: `apps/python/utils/zapdos/editor/rebuild_events.py`
- Modify: `apps/python/utils/zapdos/editor/rebuild_manager.py`
- Test: `apps/python/tests/test_zapdos_import.py`
- Test: `apps/python/tests/test_zapdos_overlay_rebuild_diagnostics.py`

- [ ] **Step 1: Write the failing rebuild tests**

```python
async def test_overlay_rebuild_runs_without_overlay_completion_queue(self):
    session = build_session()
    editor = session.editor

    result = editor.set_scene_assets([asset_payload])

    self.assertEqual(result, {"ok": True, "op_id": "op-1"})
    self.assertFalse(hasattr(editor, "overlay_completions"))
```

```python
async def test_apply_prepared_overlay_rebuild_runs_through_session_call(self):
    prepared = PreparedOverlayRebuild(...)
    await editor._run_overlay_operation("op-1", next_overlay, support_infos, previous_overlay, "rev-1")
    self.assertEqual(editor.scene_revision, "rev-2")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m unittest tests.test_zapdos_import tests.test_zapdos_overlay_rebuild_diagnostics -v`  
Expected: `FAIL` because `ensure_scene_rebuild_state()` still creates `overlay_completions` and rebuild completion depends on `drain_completions()`.

- [ ] **Step 3: Give the editor explicit rebuild state**

```python
@dataclass
class EditorRebuildState:
    job_counter: int = 0
    jobs: dict[str, SceneRebuildJob] = field(default_factory=dict)
    jobs_lock: Lock = field(default_factory=Lock)


class ZapdosEditor:
    def __init__(...):
        self.rebuild_state = EditorRebuildState()
```

- [ ] **Step 4: Replace `submit(...) + queue + drain` with a single async rebuild coroutine**

```python
def set_scene_assets(self, assets: list[dict[str, object]]) -> dict[str, object]:
    next_overlay, items = build_set_scene_assets_overlay(self, assets)
    return self._start_overlay_operation(next_overlay, {"ok": True, "items": items})

def _start_overlay_operation(self, next_overlay, success_payload):
    op_id = next_scene_rebuild_job_id(self)
    create_scene_rebuild_job(self, op_id, success_payload)
    asyncio.create_task(self._run_overlay_operation(op_id, next_overlay))
    return {"ok": True, "op_id": op_id}

async def _run_overlay_operation(self, op_id: str, next_overlay):
    prepared = await asyncio.to_thread(
        self._prepare_overlay_rebuild,
        next_overlay,
        support_infos,
        previous_overlay,
        previous_revision,
        op_id,
    )
    async with self.session.reserve_world():
        revision = await self.session.run_sync(
            lambda session: session.editor._apply_prepared_overlay_rebuild(prepared, op_id)
        )
```

- [ ] **Step 5: Remove dynamic state injection and completion draining**

```python
def ensure_scene_rebuild_state(session: Any) -> None:
    return None

def lookup_scene_rebuild_job(session: Any, op_id: str) -> SceneRebuildJob | None:
    with session.rebuild_state.jobs_lock:
        return session.rebuild_state.jobs.get(op_id)
```

```python
class ZapdosSession(Session):
    # delete editor.drain_completions() from the loop entirely
    pass
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run python -m unittest tests.test_zapdos_import tests.test_zapdos_overlay_rebuild_diagnostics -v`  
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add apps/python/utils/zapdos/editor/zapdos_editor.py apps/python/utils/zapdos/editor/rebuild_events.py apps/python/utils/zapdos/editor/rebuild_manager.py apps/python/tests/test_zapdos_import.py apps/python/tests/test_zapdos_overlay_rebuild_diagnostics.py
git commit -m "refactor: replace rebuild completion queue with async flow"
```

### Task 4: Remove dead scheduling hooks and re-verify the full Zapdos flow

**Files:**
- Modify: `apps/python/utils/session.py`
- Modify: `apps/python/utils/zapdos/zapdos_session.py`
- Test: `apps/python/tests/test_session.py`
- Test: `apps/python/tests/test_zapdos_import.py`
- Test: `apps/python/tests/test_zapdos_send_ros.py`
- Test: `apps/python/tests/test_zapdos_pick_executor.py`

- [ ] **Step 1: Write the cleanup assertions**

```python
def test_session_has_no_actor_api_after_async_refactor(self):
    self.assertFalse(hasattr(Session, "add_actor"))
    self.assertFalse(hasattr(Session, "proc_actors"))

def test_zapdos_session_has_no_idle_step_once_after_async_refactor(self):
    self.assertFalse(hasattr(ZapdosSession, "idle_step_once"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m unittest tests.test_session tests.test_zapdos_import -v`  
Expected: `FAIL` because the dead APIs still exist.

- [ ] **Step 3: Delete the compatibility scaffolding**

```python
class Session:
    # remove: Actor, add_actor, proc_actors, has_actors, idle_step_once
    # keep: timers, run_sync/call queue, reserve_world, step_once
    ...
```

```python
class ZapdosSession(Session):
    # keep only the physics tick in step_once
    ...
```

- [ ] **Step 4: Run the focused Python regression suite**

Run: `uv run python -m unittest tests.test_session tests.test_zapdos_import tests.test_zapdos_send_ros tests.test_zapdos_pick_executor -v`  
Expected: `OK`

- [ ] **Step 5: Run the web regression suite**

Run: `pnpm --dir apps/web exec tsx --test components/zapdos/overlay/debug/grab-the-apple.test.ts components/zapdos/overlay/debug/place-the-apple.test.ts components/zapdos/agent/zapdos-manipulation-tools.test.ts`  
Expected: all tests `pass`

- [ ] **Step 6: Final whitespace check**

Run: `git diff --check`  
Expected: no whitespace errors; CRLF warnings are acceptable on this repo.

- [ ] **Step 7: Commit**

```bash
git add apps/python/utils/session.py apps/python/utils/zapdos/zapdos_session.py apps/python/tests/test_session.py apps/python/tests/test_zapdos_import.py apps/python/tests/test_zapdos_send_ros.py apps/python/tests/test_zapdos_pick_executor.py
git commit -m "refactor: simplify zapdos session scheduling"
```

## Self-Review

- Spec coverage:
  - Async manipulation via awaited session-thread calls: covered by Task 2.
  - Remove actor list and `idle_step_once`: covered by Tasks 1 and 4.
  - Remove dynamic editor completion queue/state injection: covered by Task 3.
  - Keep session thread as world-state owner: covered by Tasks 1, 2, and 3.
- Placeholder scan:
  - No `TODO`/`TBD` placeholders remain.
  - Every task includes exact files, commands, and expected outcomes.
- Type consistency:
  - `run_sync`, `reserve_world`, `step_once`, and `_run_overlay_operation` names are used consistently across tasks.
