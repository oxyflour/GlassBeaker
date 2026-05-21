from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

import utils.zapdos.renderer as renderer_package
from utils.zapdos.bundle.camera_specs import image_topic
from utils.zapdos.renderer.isaac_renderer import mjpeg_chunk


def _camera(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, frame_id=f"{name}_frame")


def _renderer_module():
    return importlib.import_module("utils.zapdos.renderer.zapdos_renderer")


class ZapdosRendererTest(unittest.IsolatedAsyncioTestCase):
    def make_renderer(self, **overrides):
        renderer_cls = getattr(renderer_package, "ZapdosRenderer", None)
        self.assertIsNotNone(renderer_cls, "ZapdosRenderer should be exported from utils.zapdos.renderer")
        kwargs = {
            "backend": SimpleNamespace(ready=True, read=mock.Mock(return_value=None), snapshot_cameras=mock.Mock(return_value=[])),
            "bundle": SimpleNamespace(cameras=[]),
            "render_size": (4, 3),
            "is_active": lambda: True,
            "image_topic": image_topic,
            "image_subscriptions": {},
        }
        kwargs.update(overrides)
        return renderer_cls(**kwargs)

    def test_snapshot_returns_latest_frame_as_jpeg_bytes(self):
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        renderer = self.make_renderer(
            backend=SimpleNamespace(ready=True, read=mock.Mock(return_value=(7, frame)), snapshot_cameras=mock.Mock(return_value=[])),
        )
        module = _renderer_module()

        def fake_fromarray(image: np.ndarray):
            self.assertIs(image, frame)
            return SimpleNamespace(save=lambda data, format, quality: data.write(b"jpeg-frame"))

        with mock.patch.object(module.Image, "fromarray", side_effect=fake_fromarray):
            payload = renderer.snapshot("head_camera")

        self.assertEqual(payload, b"jpeg-frame")

    def test_snapshot_returns_waiting_placeholder_before_backend_ready(self):
        renderer = self.make_renderer(
            backend=SimpleNamespace(ready=False, read=mock.Mock(return_value=None), snapshot_cameras=mock.Mock(return_value=[])),
        )
        module = _renderer_module()

        with mock.patch.object(module, "placeholder_jpeg", return_value=b"jpeg-waiting") as placeholder:
            payload = renderer.snapshot("head_camera")

        self.assertEqual(payload, b"jpeg-waiting")
        placeholder.assert_called_once_with(4, 3, "Waiting")

    async def test_render_starts_backend_before_yielding_waiting_placeholder(self):
        active = {"value": True}
        backend = SimpleNamespace(
            ready=False,
            read=mock.Mock(return_value=None),
            start=mock.Mock(side_effect=lambda: active.update(value=False)),
            snapshot_cameras=mock.Mock(return_value=[]),
        )
        renderer = self.make_renderer(backend=backend, is_active=lambda: active["value"])
        module = _renderer_module()

        with mock.patch.object(module, "placeholder_jpeg", return_value=b"jpeg-waiting"):
            stream = renderer.render("head_camera")
            try:
                first = await anext(stream)
            finally:
                await stream.aclose()

        self.assertEqual(first, mjpeg_chunk(b"jpeg-waiting"))
        backend.start.assert_called_once_with()

    async def test_render_skips_jpeg_reencode_when_frame_index_does_not_advance(self):
        active = {"value": True}
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        renderer = self.make_renderer(
            backend=SimpleNamespace(
                ready=True,
                read=mock.Mock(side_effect=[(11, frame), (11, frame)]),
                snapshot_cameras=mock.Mock(return_value=[]),
            ),
            is_active=lambda: active["value"],
        )
        module = _renderer_module()
        sleep_calls = {"count": 0}

        async def fake_sleep(_: float):
            sleep_calls["count"] += 1
            if sleep_calls["count"] >= 2:
                active["value"] = False

        def fake_fromarray(_: np.ndarray):
            return SimpleNamespace(save=lambda data, format, quality: data.write(b"jpeg-live"))

        with mock.patch.object(module.asyncio, "sleep", side_effect=fake_sleep):
            with mock.patch.object(module.Image, "fromarray", side_effect=fake_fromarray) as fromarray:
                with mock.patch.object(module, "placeholder_jpeg", return_value=b"jpeg-closed"):
                    stream = renderer.render("head_camera")
                    first = await anext(stream)
                    second = await anext(stream)
                    with self.assertRaises(StopAsyncIteration):
                        await anext(stream)

        self.assertEqual(first, mjpeg_chunk(b"jpeg-live"))
        self.assertEqual(second, mjpeg_chunk(b"jpeg-closed"))
        self.assertEqual(fromarray.call_count, 1)

    async def test_render_multi_camera_stitches_three_frames_and_skips_duplicate_indices(self):
        active = {"value": True}
        frames = [
            np.full((2, 2, 3), 10, dtype=np.uint8),
            np.full((2, 2, 3), 20, dtype=np.uint8),
            np.full((2, 2, 3), 30, dtype=np.uint8),
        ]
        renderer = self.make_renderer(
            bundle=SimpleNamespace(cameras=[
                _camera("head_camera"),
                _camera("left_wrist_camera"),
                _camera("right_wrist_camera"),
            ]),
            backend=SimpleNamespace(
                ready=True,
                read=mock.Mock(side_effect=[
                    (11, frames[0]),
                    (11, frames[1]),
                    (11, frames[2]),
                    (11, frames[0]),
                    (11, frames[1]),
                    (11, frames[2]),
                ]),
                snapshot_cameras=mock.Mock(return_value=[]),
            ),
            is_active=lambda: active["value"],
        )
        module = _renderer_module()
        sleep_calls = {"count": 0}
        encoded: list[np.ndarray] = []

        async def fake_sleep(_: float):
            sleep_calls["count"] += 1
            if sleep_calls["count"] >= 2:
                active["value"] = False

        with mock.patch.object(module.asyncio, "sleep", side_effect=fake_sleep):
            with mock.patch.object(renderer, "_encode_jpeg", side_effect=lambda frame: encoded.append(frame.copy()) or b"jpeg-strip"):
                with mock.patch.object(module, "placeholder_jpeg", return_value=b"jpeg-closed"):
                    stream = renderer.render_multi_camera()
                    first = await anext(stream)
                    second = await anext(stream)
                    with self.assertRaises(StopAsyncIteration):
                        await anext(stream)

        self.assertEqual(first, mjpeg_chunk(b"jpeg-strip"))
        self.assertEqual(second, mjpeg_chunk(b"jpeg-closed"))
        self.assertEqual(len(encoded), 1)
        self.assertEqual(encoded[0].shape, (2, 6, 3))
        np.testing.assert_array_equal(encoded[0][:, :2], frames[0])
        np.testing.assert_array_equal(encoded[0][:, 2:4], frames[1])
        np.testing.assert_array_equal(encoded[0][:, 4:6], frames[2])

    def test_image_messages_skip_duplicate_frame_indices_per_camera(self):
        first = np.full((2, 2, 3), 1, dtype=np.uint8)
        second = np.full((2, 2, 3), 2, dtype=np.uint8)
        third = np.full((2, 2, 3), 3, dtype=np.uint8)
        renderer = self.make_renderer(
            bundle=SimpleNamespace(cameras=[_camera("head_camera"), _camera("wrist_camera")]),
            backend=SimpleNamespace(
                ready=True,
                read=mock.Mock(side_effect=[
                    (5, first),
                    (7, second),
                    (5, first),
                    (8, third),
                ]),
                snapshot_cameras=mock.Mock(return_value=[]),
            ),
        )

        first_batch = renderer.image_messages()
        second_batch = renderer.image_messages()

        self.assertEqual([topic for topic, _ in first_batch], [image_topic("head_camera"), image_topic("wrist_camera")])
        self.assertEqual([topic for topic, _ in second_batch], [image_topic("wrist_camera")])
        self.assertEqual(second_batch[0][1]["header"]["frame_id"], "wrist_camera_frame")
        self.assertEqual(second_batch[0][1]["data"], third.tobytes())

    def test_should_publish_camera_images_uses_env_override_and_subscriptions(self):
        subscriptions = {image_topic("head_camera"): object()}
        renderer = self.make_renderer(
            bundle=SimpleNamespace(cameras=[_camera("head_camera")]),
            image_subscriptions=subscriptions,
        )

        with mock.patch.dict(os.environ, {}, clear=False):
            self.assertTrue(renderer.should_publish_camera_images())
        with mock.patch.dict(os.environ, {"ZAPDOS_PUBLISH_CAMERA_IMAGES": "no"}, clear=False):
            self.assertFalse(renderer.should_publish_camera_images())
        with mock.patch.dict(os.environ, {"ZAPDOS_PUBLISH_CAMERA_IMAGES": "yes"}, clear=False):
            self.assertTrue(renderer.should_publish_camera_images())

    def test_save_camera_override_persists_backend_snapshot_payload(self):
        snapshot = [{"name": "head_camera", "parent_prim": "/Robot", "pos": [1, 2, 3], "quat": [1, 0, 0, 0], "fovy": 45.0, "horizontal_aperture": 32.0, "vertical_aperture": 24.0, "clipping_range": [0.01, 100.0]}]
        backend = SimpleNamespace(ready=True, read=mock.Mock(return_value=None), snapshot_cameras=mock.Mock(return_value=snapshot))
        renderer = self.make_renderer(backend=backend)
        module = _renderer_module()

        with mock.patch.object(module, "save_camera_overrides", return_value=(Path("camera.json"), 1)) as save_mock:
            result = renderer.save_camera_override()

        self.assertEqual(result, {"ok": True, "saved": 1, "path": "camera.json"})
        backend.snapshot_cameras.assert_called_once_with()
        save_mock.assert_called_once_with(snapshot)


if __name__ == "__main__":
    unittest.main()
