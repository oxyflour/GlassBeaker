from __future__ import annotations

import unittest
from types import SimpleNamespace

from utils.isaac_renderer_reload import (
    CameraBinding,
    rebuild_camera_bindings,
    reset_subscriber_caches,
    validate_camera_topology,
)


class RendererReloadHelperTest(unittest.TestCase):
    def test_validate_camera_topology_rejects_name_changes(self):
        with self.assertRaises(RuntimeError) as err:
            validate_camera_topology(
                [{"name": "head_camera", "prim": "/robot/head"}],
                [{"name": "wrist_camera", "prim": "/robot/wrist"}],
            )

        self.assertIn("camera", str(err.exception))

    def test_reset_subscriber_caches_replaces_body_map_and_clears_lookup_cache(self):
        subscribers = [
            SimpleNamespace(
                body_name_map={"OldBody": "Old/Body"},
                _attr_cache={"OldBody": object()},
                _ordered_attrs=[object()],
            )
        ]
        next_map = {"NewBody": "New/Body"}

        reset_subscriber_caches(subscribers, next_map)

        self.assertIs(subscribers[0].body_name_map, next_map)
        self.assertEqual(subscribers[0]._attr_cache, {})
        self.assertIsNone(subscribers[0]._ordered_attrs)

    def test_rebuild_camera_bindings_releases_old_before_creating_new(self):
        events: list[str] = []
        old_bindings = [[
            CameraBinding(annotator="old-head", render_product="rp-old-head"),
            CameraBinding(annotator="old-wrist", render_product="rp-old-wrist"),
        ]]

        def release(binding: CameraBinding) -> None:
            events.append(f"release:{binding.annotator}:{binding.render_product}")

        def create(camera_path: str) -> CameraBinding:
            events.append(f"create:{camera_path}")
            return CameraBinding(
                annotator=f"ann:{camera_path}",
                render_product=f"rp:{camera_path}",
            )

        rebuilt = rebuild_camera_bindings(
            ["/World/envs/env_0"],
            [
                {"name": "head_camera", "prim": "/robot/head_camera"},
                {"name": "wrist_camera", "prim": "/robot/wrist_camera"},
            ],
            old_bindings,
            create,
            release,
        )

        self.assertEqual(events[:2], [
            "release:old-head:rp-old-head",
            "release:old-wrist:rp-old-wrist",
        ])
        self.assertEqual(rebuilt[0][0].annotator, "ann:/World/envs/env_0/robot/head_camera")
        self.assertEqual(rebuilt[1][0].annotator, "ann:/World/envs/env_0/robot/wrist_camera")


if __name__ == "__main__":
    unittest.main()
