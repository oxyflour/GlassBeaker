from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.zapdos.manipulation import (
    build_scene_object_catalog,
    ground_pick_target,
    plan_pick,
)


def _matrix_at(x: float, y: float, z: float) -> list[float]:
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        x, y, z, 1.0,
    ]


class ZapdosManipulationTest(unittest.TestCase):
    def _scene_bodies(self) -> dict[str, object]:
        return {
            "items": [
                {
                    "body": "table_body",
                    "label": "table",
                    "matrix": _matrix_at(0.5, 0.0, 0.75),
                    "support": {"top_z": 0.8},
                    "world_aabb": {"min": [0.1, -0.3, 0.7], "max": [0.9, 0.3, 0.8]},
                },
                {
                    "body": "counter_body",
                    "label": "counter",
                    "matrix": _matrix_at(-0.5, 0.0, 0.9),
                    "support": {"top_z": 0.95},
                    "world_aabb": {"min": [-0.9, -0.3, 0.85], "max": [-0.1, 0.3, 0.95]},
                },
                {
                    "body": "Scene_apple_1",
                    "label": "apple",
                    "matrix": _matrix_at(0.5, 0.0, 0.83),
                    "support": {"top_z": 0.86},
                    "world_aabb": {"min": [0.4539, -0.0469, 0.7834], "max": [0.5461, 0.0469, 0.8766]},
                },
                {
                    "body": "Scene_pineapple_1",
                    "label": "pineapple",
                    "matrix": _matrix_at(0.45, 0.05, 0.9),
                    "support": {"top_z": 0.98},
                    "world_aabb": {"min": [0.4039, 0.0031, 0.8534], "max": [0.4961, 0.0969, 0.9466]},
                },
                {
                    "body": "Scene_apple_2",
                    "label": "apple",
                    "matrix": _matrix_at(-0.5, 0.0, 0.98),
                    "support": {"top_z": 1.01},
                    "world_aabb": {"min": [-0.5461, -0.0469, 0.9334], "max": [-0.4539, 0.0469, 1.0266]},
                },
            ],
            "scene_revision": "rev-1",
        }

    def _overlay_state(self) -> dict[str, object]:
        return {
            "version": 1,
            "assets_root": "C:/assets",
            "instances": [
                {
                    "id": "apple_1",
                    "asset_id": "apple_red",
                    "url": "objects/apple_red/Aligned.usda",
                    "motion": "dynamic",
                    "placement": {
                        "kind": "on_top_of_body",
                        "body": "table_body",
                        "xy": [0.5, 0.0],
                        "gap": 0.0,
                    },
                },
                {
                    "id": "pineapple_1",
                    "asset_id": "pineapple_001",
                    "url": "objects/pineapple/Aligned.usda",
                    "motion": "dynamic",
                    "placement": {
                        "kind": "on_top_of_body",
                        "body": "table_body",
                        "xy": [0.45, 0.05],
                        "gap": 0.0,
                    },
                },
                {
                    "id": "apple_2",
                    "asset_id": "apple_green",
                    "url": "objects/apple_green/Aligned.usda",
                    "motion": "dynamic",
                    "placement": {
                        "kind": "on_top_of_body",
                        "body": "counter_body",
                        "xy": [-0.5, 0.0],
                        "gap": 0.0,
                    },
                },
            ],
            "pose_overrides": {},
        }

    @mock.patch("utils.zapdos.manipulation.catalog.asset_local_bounds")
    @mock.patch("utils.zapdos.manipulation.catalog.resolve_asset_record")
    def test_build_scene_object_catalog_merges_scene_and_overlay_metadata(self, resolve_asset_record, asset_local_bounds):
        records = {
            "apple_red": {
                "asset_id": "apple_red",
                "url": "objects/apple_red/Aligned.usda",
                "description": {
                    "semantic_name": ["apple", "fruit"],
                    "full_description": ["red apple"],
                },
            },
            "pineapple_001": {
                "asset_id": "pineapple_001",
                "url": "objects/pineapple/Aligned.usda",
                "description": {"semantic_name": ["pineapple"]},
            },
            "apple_green": {
                "asset_id": "apple_green",
                "url": "objects/apple_green/Aligned.usda",
                "description": {"semantic_name": ["apple", "fruit"]},
            },
        }
        resolve_asset_record.side_effect = lambda asset_id, assets_root=None: records[asset_id]
        asset_local_bounds.side_effect = lambda path: {
            "min": [-0.0461, -0.0469, -0.0466],
            "max": [0.0461, 0.0469, 0.0466],
        }

        catalog = build_scene_object_catalog(self._scene_bodies(), self._overlay_state())

        apple = next(item for item in catalog if item["body"] == "Scene_apple_1")
        table = next(item for item in catalog if item["body"] == "table_body")
        self.assertEqual(apple["asset_id"], "apple_red")
        self.assertEqual(apple["motion"], "dynamic")
        self.assertEqual(apple["support_body"], "table_body")
        self.assertEqual(apple["position"], [0.5, 0.0, 0.83])
        self.assertEqual(apple["top_z"], 0.86)
        self.assertEqual(apple["bounds_min"], [-0.0461, -0.0469, -0.0466])
        self.assertEqual(apple["bounds_max"], [0.0461, 0.0469, 0.0466])
        self.assertIn("fruit", apple["tags"])
        self.assertIn("red apple", apple["tags"])
        self.assertIsNone(table["asset_id"])
        self.assertIn("table", table["tags"])

    @mock.patch("utils.zapdos.manipulation.catalog.asset_local_bounds")
    @mock.patch("utils.zapdos.manipulation.catalog.resolve_asset_record")
    def test_ground_pick_target_prefers_exact_matches_and_support_filter(self, resolve_asset_record, asset_local_bounds):
        records = {
            "apple_red": {
                "asset_id": "apple_red",
                "url": "objects/apple_red/Aligned.usda",
                "description": {"semantic_name": ["apple", "fruit"]},
            },
            "pineapple_001": {
                "asset_id": "pineapple_001",
                "url": "objects/pineapple/Aligned.usda",
                "description": {"semantic_name": ["pineapple"]},
            },
            "apple_green": {
                "asset_id": "apple_green",
                "url": "objects/apple_green/Aligned.usda",
                "description": {"semantic_name": ["apple"]},
            },
        }
        resolve_asset_record.side_effect = lambda asset_id, assets_root=None: records[asset_id]
        asset_local_bounds.side_effect = lambda path: {
            "min": [-0.0461, -0.0469, -0.0466],
            "max": [0.0461, 0.0469, 0.0466],
        }
        catalog = build_scene_object_catalog(self._scene_bodies(), self._overlay_state())

        grounded = ground_pick_target(catalog, target_query="apple", support_query="table")

        self.assertEqual(grounded["target"]["body"], "Scene_apple_1")
        self.assertEqual(grounded["support"]["body"], "table_body")

    @mock.patch("utils.zapdos.manipulation.catalog.asset_local_bounds")
    @mock.patch("utils.zapdos.manipulation.catalog.resolve_asset_record")
    def test_build_scene_object_catalog_preserves_world_aabb(self, resolve_asset_record, asset_local_bounds):
        resolve_asset_record.return_value = {
            "asset_id": "apple_red",
            "url": "objects/apple_red/Aligned.usda",
            "description": {"semantic_name": ["apple"]},
        }
        asset_local_bounds.return_value = {
            "min": [-0.0461, -0.0469, -0.0466],
            "max": [0.0461, 0.0469, 0.0466],
        }

        scene_bodies = self._scene_bodies()
        scene_bodies["items"][2]["world_aabb"] = {
            "min": [0.4539, -0.0469, 0.7834],
            "max": [0.5461, 0.0469, 0.8766],
        }
        catalog = build_scene_object_catalog(scene_bodies, self._overlay_state())

        apple = next(item for item in catalog if item["body"] == "Scene_apple_1")
        self.assertEqual(apple["world_aabb"], {
            "min": [0.4539, -0.0469, 0.7834],
            "max": [0.5461, 0.0469, 0.8766],
        })

    @mock.patch("utils.zapdos.manipulation.catalog.asset_local_bounds")
    @mock.patch("utils.zapdos.manipulation.catalog.resolve_asset_record")
    def test_plan_pick_targets_object_center_and_keeps_fingers_facing_down(self, resolve_asset_record, asset_local_bounds):
        resolve_asset_record.return_value = {
            "asset_id": "apple_red",
            "url": "objects/apple_red/Aligned.usda",
            "description": {"semantic_name": ["apple"]},
        }
        asset_local_bounds.return_value = {
            "min": [-0.0461, -0.0469, -0.0466],
            "max": [0.0461, 0.0469, 0.0466],
        }
        catalog = build_scene_object_catalog(self._scene_bodies(), self._overlay_state())
        grounded = ground_pick_target(catalog, target_query="apple", support_query="table")
        target = grounded["target"]

        plan = plan_pick(
            target,
            support=grounded["support"],
            scene_objects=catalog,
            arm="left",
            start_pose={"position": [0.0, 0.0, 0.5], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
        )

        self.assertEqual(plan["kind"], "pick")
        self.assertEqual(plan["target_body"], "Scene_apple_1")
        self.assertEqual(plan["orientation"]["mode"], "top_down")
        self.assertEqual(plan["orientation"]["quat_wxyz"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual([stage["name"] for stage in plan["stages"]], [
            "raise_to_transit",
            "approach_xy",
            "descend_to_pregrasp",
            "descend_to_grasp",
            "close_gripper",
            "retreat",
        ])
        self.assertTrue(all(stage["kind"] == "move_pose" for stage in plan["stages"] if stage["name"] != "close_gripper"))
        self.assertEqual(plan["stages"][0]["pose"]["position"], [0.0, 0.0, 1.03])
        self.assertEqual(plan["stages"][1]["pose"]["position"], [0.5, 0.0, 1.03])
        self.assertTrue(all(stage.get("target_point") == "finger_center" for stage in plan["stages"] if stage["name"] != "close_gripper"))
        self.assertEqual(plan["stages"][2]["pose"]["position"], [0.5, 0.0, 0.95])
        self.assertEqual(plan["stages"][3]["pose"]["position"], [0.5, 0.0, 0.83])
        self.assertEqual(plan["stages"][4]["width"], 0.0)
        self.assertEqual(plan["stages"][5]["pose"]["position"], [0.5, 0.0, 1.03])

    @mock.patch("utils.zapdos.manipulation.catalog.asset_local_bounds")
    @mock.patch("utils.zapdos.manipulation.catalog.resolve_asset_record")
    def test_plan_pick_includes_support_surface_metadata_for_escape_routing(self, resolve_asset_record, asset_local_bounds):
        resolve_asset_record.return_value = {
            "asset_id": "apple_red",
            "url": "objects/apple_red/Aligned.usda",
            "description": {"semantic_name": ["apple"]},
        }
        asset_local_bounds.return_value = {
            "min": [-0.0461, -0.0469, -0.0466],
            "max": [0.0461, 0.0469, 0.0466],
        }
        catalog = build_scene_object_catalog(self._scene_bodies(), self._overlay_state())
        grounded = ground_pick_target(catalog, target_query="apple", support_query="table")
        target = grounded["target"]
        support = {
            "body": "table_body",
            "label": "table",
            "asset_id": "table_000",
            "motion": "static",
            "tags": ["table"],
            "support_body": None,
            "position": [0.5, 0.0, 0.75],
            "matrix": _matrix_at(0.5, 0.0, 0.75),
            "top_z": 0.8,
            "bounds_min": [-0.4, -0.3, -0.05],
            "bounds_max": [0.4, 0.3, 0.05],
        }

        plan = plan_pick(
            target,
            support=support,
            scene_objects=catalog,
            arm="left",
            start_pose={"position": [0.0, 0.0, 0.5], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
        )

        self.assertEqual(plan["support_surface"]["top_z"], 0.8)
        self.assertEqual(plan["support_surface"]["xy_min"], [0.1, -0.3])
        self.assertEqual(plan["support_surface"]["xy_max"], [0.9, 0.3])
        self.assertEqual(plan["stages"][0]["kind"], "move_pose")

    def test_plan_pick_accepts_support_world_aabb_without_position(self):
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
        support = {
            "body": "table_body",
            "label": "table",
            "asset_id": None,
            "motion": None,
            "tags": ["table"],
            "support_body": None,
            "position": None,
            "matrix": None,
            "top_z": 0.8,
            "bounds_min": [-0.4, -0.3, -0.05],
            "bounds_max": [0.4, 0.3, 0.05],
            "world_aabb": {
                "min": [0.1, -0.3, 0.7],
                "max": [0.9, 0.3, 0.8],
            },
        }

        plan = plan_pick(
            target,
            support=support,
            scene_objects=[target, support],
            arm="left",
            start_pose={"position": [0.0, 0.0, 0.5], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
        )

        self.assertEqual(plan["support_surface"]["xy_min"], [0.1, -0.3])
        self.assertEqual(plan["support_surface"]["xy_max"], [0.9, 0.3])
        self.assertEqual(plan["stages"][0]["kind"], "move_pose")

    def test_plan_pick_prefers_world_aabb_center_over_body_position_for_grasp(self):
        target = {
            "body": "Scene_apple_1",
            "label": "apple",
            "asset_id": "apple_red",
            "motion": "dynamic",
            "tags": ["apple"],
            "support_body": "table_body",
            "position": [0.2, 0.3, 0.4],
            "matrix": _matrix_at(0.2, 0.3, 0.4),
            "top_z": 0.9,
            "bounds_min": [-0.0461, -0.0469, -0.0466],
            "bounds_max": [0.0461, 0.0469, 0.0466],
            "world_aabb": {
                "min": [0.55, -0.15, 0.75],
                "max": [0.65, -0.05, 0.85],
            },
        }
        support = {
            "body": "table_body",
            "label": "table",
            "asset_id": None,
            "motion": None,
            "tags": ["table"],
            "support_body": None,
            "position": None,
            "matrix": None,
            "top_z": 0.8,
            "bounds_min": [-0.4, -0.3, -0.05],
            "bounds_max": [0.4, 0.3, 0.05],
            "world_aabb": {
                "min": [0.1, -0.3, 0.7],
                "max": [0.9, 0.3, 0.8],
            },
        }

        plan = plan_pick(
            target,
            support=support,
            scene_objects=[target, support],
            arm="left",
            start_pose={"position": [0.0, 0.0, 1.1], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
        )

        self.assertEqual(plan["pre_grasp"]["position"], [0.6, -0.1, 0.92])
        self.assertEqual(plan["grasp"]["position"], [0.6, -0.1, 0.8])
        self.assertEqual(plan["stages"][0]["pose"]["position"], [0.6, -0.1, 1.1])
        self.assertEqual(plan["stages"][1]["pose"]["position"], [0.6, -0.1, 0.92])
        self.assertEqual(plan["stages"][2]["pose"]["position"], [0.6, -0.1, 0.8])

    @mock.patch("utils.zapdos.manipulation.catalog.asset_local_bounds")
    @mock.patch("utils.zapdos.manipulation.catalog.resolve_asset_record")
    def test_plan_pick_adds_escape_and_transit_when_start_is_under_support(self, resolve_asset_record, asset_local_bounds):
        resolve_asset_record.side_effect = lambda asset_id, assets_root=None: {
            "apple_red": {
                "asset_id": "apple_red",
                "url": "objects/apple_red/Aligned.usda",
                "description": {"semantic_name": ["apple"]},
            },
            "pineapple_001": {
                "asset_id": "pineapple_001",
                "url": "objects/pineapple/Aligned.usda",
                "description": {"semantic_name": ["pineapple"]},
            },
            "apple_green": {
                "asset_id": "apple_green",
                "url": "objects/apple_green/Aligned.usda",
                "description": {"semantic_name": ["apple"]},
            },
        }[asset_id]
        asset_local_bounds.side_effect = lambda path: {
            "min": [-0.0461, -0.0469, -0.0466],
            "max": [0.0461, 0.0469, 0.0466],
        }
        catalog = build_scene_object_catalog(self._scene_bodies(), self._overlay_state())
        grounded = ground_pick_target(catalog, target_query="apple", support_query="table")
        catalog.append({
            "body": "post_escape_blocker",
            "label": "blocker",
            "asset_id": None,
            "motion": "static",
            "tags": ["blocker"],
            "support_body": None,
            "position": None,
            "matrix": None,
            "top_z": 1.1,
            "bounds_min": None,
            "bounds_max": None,
            "world_aabb": {
                "min": [0.47, -0.34, 1.1],
                "max": [0.53, -0.32, 1.2],
            },
        })
        start_pose = {"position": [0.5, -0.25, 0.2], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}

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
        self.assertEqual(plan["stages"][0]["kind"], "move_pose")
        self.assertEqual(plan["stages"][1]["kind"], "move_pose")
        self.assertEqual(plan["stages"][2]["kind"], "move_pose")
        self.assertEqual(plan["stages"][0]["pose"]["position"], [0.5, -0.36, 0.2])
        self.assertEqual(plan["stages"][1]["pose"]["position"], [0.5, -0.36, 1.25])
        self.assertEqual(plan["stages"][2]["pose"]["position"], [0.5, 0.0, 1.25])

    def test_plan_pick_rejects_missing_support_geometry(self):
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


if __name__ == "__main__":
    unittest.main()
