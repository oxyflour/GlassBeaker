from __future__ import annotations

import asyncio
import inspect
from concurrent.futures import Future as ConcurrentFuture
from typing import Any, Callable, Iterator

from fastapi import HTTPException

from utils.zapdos.editor.rebuild_events import (
    create_scene_rebuild_job,
    discard_scene_rebuild_job,
    ensure_scene_rebuild_state,
    fail_scene_rebuild_job,
    lookup_scene_rebuild_job,
    next_scene_rebuild_job_id,
)
from utils.zapdos.manipulation.catalog import build_scene_object_catalog
from utils.zapdos.manipulation.executor import PickExecutor
from utils.zapdos.manipulation.grounding import ground_pick_target
from utils.zapdos.manipulation.planner import plan_pick
from utils.zapdos.manipulation.types import GroundedPick, PickPlan, SceneObject

BENCHMARK_OBJECT_ARM = "left"
BENCHMARK_OBJECT_TARGET_QUERY = "cube"
BENCHMARK_OBJECT_SUPPORT_QUERY = "benchmark table"
BENCHMARK_OBJECT_OPEN_WIDTH = 0.05


def _advance_operation(iterator: Iterator[None]) -> dict[str, object] | None:
    try:
        next(iterator)
    except StopIteration as stop:
        return stop.value if isinstance(stop.value, dict) else {}
    return None


class ManipulationRuntime:
    def __init__(
        self,
        session,
        *,
        catalog_loader: Callable[[dict[str, object], dict[str, object]], list[SceneObject]] | None = None,
        grounding_fn: Callable[..., GroundedPick] | None = None,
        planning_fn: Callable[..., PickPlan] | None = None,
        executor: PickExecutor | None = None,
    ) -> None:
        self.session = session
        self.catalog_loader = catalog_loader or build_scene_object_catalog
        self.grounding_fn = grounding_fn or ground_pick_target
        self.planning_fn = planning_fn or plan_pick
        self.executor = executor or PickExecutor(session.physics, session.bundle)
        self.operation_tasks: dict[str, ConcurrentFuture[Any] | asyncio.Task[Any]] = {}

    def list_scene_objects(self) -> dict[str, object]:
        return {
            "items": self._scene_objects(),
            "scene_revision": self.session.editor.scene_revision,
        }

    def pick_object(self, args: dict[str, object]) -> dict[str, object]:
        if not isinstance(args, dict):
            raise HTTPException(status_code=400, detail="pick_object expects a single object argument")
        arm = str(args.get("arm") or "left")
        objects = self._scene_objects()
        grounded = self._ground_target(args, objects)
        target = grounded["target"]
        if target.get("motion") != "dynamic":
            raise HTTPException(status_code=400, detail="Pick target must be a dynamic scene object")
        self._sync_executor_state()
        return self._start_operation(self.executor.iter_execute({
            **self._plan_pick(grounded, arm=arm, objects=objects),
            "arm": arm,
        }))

    def pick_apple(self) -> dict[str, object]:
        objects = self._scene_objects()
        grounded = self._ground_target({
            "target_query": BENCHMARK_OBJECT_TARGET_QUERY,
            "support_query": BENCHMARK_OBJECT_SUPPORT_QUERY,
        }, objects)
        target = grounded["target"]
        if target.get("motion") != "dynamic":
            raise HTTPException(status_code=400, detail="Pick target must be a dynamic scene object")
        self._sync_executor_state()
        return self._start_operation(self.executor.iter_execute({
            **self._plan_pick(grounded, arm=BENCHMARK_OBJECT_ARM, objects=objects),
            "arm": BENCHMARK_OBJECT_ARM,
            "open_gripper": BENCHMARK_OBJECT_OPEN_WIDTH,
            "pick_tolerance": 0.025,
            "attach_tolerance": 0.015,
        }))

    def place_apple(self) -> dict[str, object]:
        objects = self._scene_objects()
        grounded = self._ground_target({
            "target_query": BENCHMARK_OBJECT_TARGET_QUERY,
            "support_query": BENCHMARK_OBJECT_SUPPORT_QUERY,
        }, objects)
        target = grounded["target"]
        if self.session.physics.get_attachment(target["body"]) is None:
            raise HTTPException(status_code=409, detail=f"Place cube requires {target['body']} to be attached")
        self._sync_executor_state()
        return self._start_operation(self.executor.iter_execute({
            "kind": "release",
            "arm": BENCHMARK_OBJECT_ARM,
            "target_body": target["body"],
            "stages": [
                {
                    "name": "open_gripper",
                    "kind": "gripper",
                    "width": BENCHMARK_OBJECT_OPEN_WIDTH,
                    "steps": 18,
                },
            ],
        }))

    def _scene_objects(self) -> list[SceneObject]:
        return self.catalog_loader(
            self.session.editor.list_scene_bodies(),
            self.session.editor.overlay_state,
        )

    def _ground_target(self, request: dict[str, object], objects: list[SceneObject]) -> GroundedPick:
        try:
            return self.grounding_fn(
                objects,
                target_query=str(request.get("target_query") or ""),
                support_query=self._support_query(request),
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _plan_pick(self, grounded: GroundedPick, *, arm: str, objects: list[SceneObject]) -> PickPlan:
        try:
            kwargs: dict[str, object] = {"support": grounded["support"]}
            params = inspect.signature(self.planning_fn).parameters
            accepts_any_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
            if "scene_objects" in params or accepts_any_kwargs:
                kwargs["scene_objects"] = objects
            if "arm" in params or accepts_any_kwargs:
                kwargs["arm"] = arm
            if "start_pose" in params or accepts_any_kwargs:
                kwargs["start_pose"] = self.executor.current_pose(arm, target_point="finger_center")
            return self.planning_fn(grounded["target"], **kwargs)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _support_query(self, request: dict[str, object]) -> str | None:
        value = request.get("support_query")
        return value if isinstance(value, str) and value.strip() else None

    def _sync_executor_state(self) -> None:
        self.executor.physics = self.session.physics
        self.executor.bundle = self.session.bundle

    def _start_operation(self, iterator: Iterator[None]) -> dict[str, object]:
        if getattr(self.session.editor, "rebuilding_scene", False):
            raise HTTPException(status_code=409, detail="Scene rebuild already in progress")
        for current_op_id, task in list(self.operation_tasks.items()):
            if task.done():
                self.operation_tasks.pop(current_op_id, None)
                continue
            raise HTTPException(status_code=409, detail="Manipulation already in progress")
        ensure_scene_rebuild_state(self.session.editor)
        op_id = next_scene_rebuild_job_id(self.session.editor)
        create_scene_rebuild_job(self.session.editor, op_id, {"ok": True})
        try:
            task = self.session.schedule_on_owner_loop(self._run_operation(op_id, iterator))
        except Exception:
            discard_scene_rebuild_job(self.session.editor, op_id)
            raise
        self.operation_tasks[op_id] = task
        task.add_done_callback(lambda _task, current_op_id=op_id: self.operation_tasks.pop(current_op_id, None))
        return {"ok": True, "op_id": op_id}

    async def _run_operation(self, op_id: str, iterator: Iterator[None]) -> None:
        try:
            async with self.session.reserve_world() as world_token:
                while True:
                    payload = await self.session.run_sync(
                        lambda _current: _advance_operation(iterator),
                        world_token=world_token,
                    )
                    if payload is None:
                        continue
                    self._resolve_operation(op_id, payload)
                    return
        except asyncio.CancelledError:
            fail_scene_rebuild_job(self.session.editor, op_id, RuntimeError("Manipulation cancelled"))
            raise
        except Exception as err:
            fail_scene_rebuild_job(self.session.editor, op_id, err)

    def _resolve_operation(self, op_id: str, payload: dict[str, object]) -> None:
        job = lookup_scene_rebuild_job(self.session.editor, op_id)
        if job is not None and not job.future.done():
            job.future.set_result({**job.success_payload, **payload, "scene_revision": self.session.editor.scene_revision})
