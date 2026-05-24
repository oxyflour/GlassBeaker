from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.zapdos.renderer import ZapdosRenderer


def _camera(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, frame_id=f"{name}_frame")


class ZapdosRendererEntityTest(unittest.IsolatedAsyncioTestCase):
    def make_renderer(self, backend=None, bundle=None):
        return ZapdosRenderer(
            backend=backend or SimpleNamespace(
                ready=True,
                wait_ready=mock.AsyncMock(return_value={"ready": True}),
                read=mock.Mock(return_value=None),
                reload_scene=mock.Mock(),
                snapshot_cameras=mock.Mock(return_value=[]),
                close=mock.Mock(),
            ),
            bundle=bundle or SimpleNamespace(cameras=[_camera("head_camera")]),
            render_size=(4, 3),
            is_active=lambda: True,
            image_topic=lambda name: f"/{name}",
        )

    async def test_lifecycle_methods_delegate_to_backend(self):
        backend = SimpleNamespace(
            ready=True,
            wait_ready=mock.AsyncMock(return_value={"ready": True}),
            read=mock.Mock(return_value=None),
            reload_scene=mock.Mock(),
            snapshot_cameras=mock.Mock(return_value=[{"name": "head_camera"}]),
            close=mock.Mock(),
        )
        renderer = self.make_renderer(backend=backend)

        self.assertEqual(await renderer.wait_ready(), {"ready": True})
        renderer.close()

        backend.wait_ready.assert_awaited_once_with()
        backend.close.assert_called_once_with()

    def test_reload_scene_updates_bundle_and_resets_dedupe(self):
        first_bundle = SimpleNamespace(cameras=[_camera("head_camera")])
        second_bundle = SimpleNamespace(cameras=[_camera("wrist_camera")])
        backend = SimpleNamespace(
            ready=True,
            wait_ready=mock.AsyncMock(return_value={"ready": True}),
            read=mock.Mock(return_value=None),
            reload_scene=mock.Mock(),
            snapshot_cameras=mock.Mock(return_value=[]),
            close=mock.Mock(),
        )
        renderer = self.make_renderer(backend=backend, bundle=first_bundle)
        renderer.last_frame_index["head_camera"] = 8

        renderer.reload_scene(second_bundle, timeout=12.0)

        backend.reload_scene.assert_called_once_with(second_bundle, timeout=12.0)
        self.assertIs(renderer.bundle, second_bundle)
        self.assertEqual(renderer.last_frame_index, {"wrist_camera": -1})

    def test_update_pose_delegates_to_backend_when_supported(self):
        backend = SimpleNamespace(
            ready=True,
            wait_ready=mock.AsyncMock(return_value={"ready": True}),
            read=mock.Mock(return_value=None),
            reload_scene=mock.Mock(),
            update_pose=mock.Mock(),
            snapshot_cameras=mock.Mock(return_value=[]),
            close=mock.Mock(),
        )
        renderer = self.make_renderer(backend=backend)
        pose = {"box": [1.0] * 16}
        update_pose = getattr(renderer, "update_pose", None)

        self.assertTrue(callable(update_pose))
        update_pose(pose)

        backend.update_pose.assert_called_once_with(pose)

    def test_default_frame_delay_matches_previous_ros_cadence(self):
        renderer = self.make_renderer()

        self.assertAlmostEqual(renderer.frame_delay, 0.03333, places=5)


if __name__ == "__main__":
    unittest.main()
