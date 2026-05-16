from __future__ import annotations

import inspect
from typing import Any, Callable

from fastapi import HTTPException

from utils.zapdos.manipulation.catalog import build_scene_object_catalog
from utils.zapdos.manipulation.executor import PickExecutor
from utils.zapdos.manipulation.grounding import ground_pick_target
from utils.zapdos.manipulation.planner import plan_pick
from utils.zapdos.manipulation.types import GroundedPick, PickPlan, SceneObject

GRAB_APPLE_ARM = "left"
GRAB_APPLE_TARGET_QUERY = "apple"
GRAB_APPLE_SUPPORT_QUERY = "benchmark table"
GRAB_APPLE_OPEN_WIDTH = 0.05


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
        result = self.executor.execute({
            **self._plan_pick(grounded, arm=arm, objects=objects),
            "arm": arm,
        })
        return {**result, "scene_revision": self.session.editor.scene_revision}

    def grab_apple(self) -> dict[str, object]:
        objects = self._scene_objects()
        grounded = self._ground_target({
            "target_query": GRAB_APPLE_TARGET_QUERY,
            "support_query": GRAB_APPLE_SUPPORT_QUERY,
        }, objects)
        target = grounded["target"]
        if target.get("motion") != "dynamic":
            raise HTTPException(status_code=400, detail="Pick target must be a dynamic scene object")
        self._sync_executor_state()
        result = self.executor.execute(self._grab_apple_plan(target, arm=GRAB_APPLE_ARM))
        return {**result, "scene_revision": self.session.editor.scene_revision}

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
                kwargs["start_pose"] = self.executor.current_pose(arm)
            return self.planning_fn(grounded["target"], **kwargs)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _support_query(self, request: dict[str, object]) -> str | None:
        value = request.get("support_query")
        return value if isinstance(value, str) and value.strip() else None

    def _sync_executor_state(self) -> None:
        self.executor.physics = self.session.physics
        self.executor.bundle = self.session.bundle

    def _grab_apple_plan(self, target: SceneObject, *, arm: str) -> dict[str, object]:
        center = self._target_center(target)
        side = 1.0 if arm == "left" else -1.0
        current_pose = self.executor.current_pose(arm)
        return {
            "kind": "pick",
            "arm": arm,
            "target_body": target["body"],
            "grasp_tolerance": 0.16,
            "attach_tolerance": 0.11,
            "stages": [
                {
                    "name": "open_gripper",
                    "kind": "gripper",
                    "width": GRAB_APPLE_OPEN_WIDTH,
                    "steps": 18,
                },
                {
                    "name": "descend_to_grasp",
                    "kind": "move_pose",
                    "pose": {
                        "position": [
                            round(center[0], 6),
                            round(center[1], 6),
                            round(center[2], 6),
                        ],
                        "quat_wxyz": list(current_pose["quat_wxyz"]),
                    },
                    "position_only": True,
                    "steps": 24,
                    "tolerance": 0.16,
                },
                {"name": "close_gripper", "kind": "gripper", "width": 0.0},
                {
                    "name": "retreat",
                    "kind": "move_pose",
                    "pose": {
                        "position": [
                            round(center[0] - 0.10, 6),
                            round(center[1] + side * 0.18, 6),
                            round(center[2] + 0.09, 6),
                        ],
                        "quat_wxyz": list(current_pose["quat_wxyz"]),
                    },
                    "position_only": True,
                    "steps": 20,
                    "tolerance": 0.08,
                },
            ],
        }

    def _target_center(self, target: SceneObject) -> tuple[float, float, float]:
        aabb = target.get("world_aabb")
        if aabb is not None:
            return (
                0.5 * (float(aabb["min"][0]) + float(aabb["max"][0])),
                0.5 * (float(aabb["min"][1]) + float(aabb["max"][1])),
                0.5 * (float(aabb["min"][2]) + float(aabb["max"][2])),
            )
        position = target.get("position")
        if position is not None:
            return float(position[0]), float(position[1]), float(position[2])
        raise HTTPException(status_code=400, detail=f"Grab apple requires world position for {target['body']}")
