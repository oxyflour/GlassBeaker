from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from fastapi import Request

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.zapdos.session import request_router
import utils.zapdos.session.zapdos_session as session_module
from utils.zapdos.session.streaming_mixin import SessionStreamingMixin


class _FakeSession(SessionStreamingMixin):
    pass


class ZapdosCameraStreamingTest(unittest.IsolatedAsyncioTestCase):
    def make_request(self, action: str, name: str) -> Request:
        return Request(
            {
                "type": "http",
                "method": "GET",
                "path": f"/python/zapdos/sess-1/{action}/{name}",
                "path_params": {"session": "sess-1", "action": action, "name": name},
                "query_string": b"",
                "headers": [],
            }
        )

    async def test_snapshot_routes_to_session_snapshot_with_no_store_headers(self):
        future: asyncio.Future[object] = asyncio.Future()
        session = SimpleNamespace(
            camera_index={"head_camera": 0},
            snapshot=mock.Mock(return_value=b"jpeg-bytes"),
        )
        future.set_result(session)

        async def await_session_future(sess: str, resolved: asyncio.Future[object]) -> object:
            self.assertEqual(sess, "sess-1")
            self.assertIs(resolved, future)
            return await resolved

        handler = request_router.build_name_handler(
            bridge=SimpleNamespace(),
            heartbeat_sec_getter=lambda: 1.0,
            get_or_create_session_future=lambda req, sess: future,
            require_session_future=lambda sess: future,
            await_session_future=await_session_future,
            require_active_session=lambda sess, resolved, active: active,
            require_camera_name=request_router.require_camera_name,
            stream_scene_rebuild_job=lambda session, name: (),
        )

        response = await handler(self.make_request("snapshot", "head_camera"))

        self.assertEqual(response.media_type, "image/jpeg")
        self.assertEqual(response.body, b"jpeg-bytes")
        self.assertEqual(response.headers["cache-control"], "no-store, no-cache, must-revalidate")
        session.snapshot.assert_called_once_with("head_camera")

    async def test_multicam_route_streams_a_single_mjpeg_strip(self):
        future: asyncio.Future[object] = asyncio.Future()
        session = SimpleNamespace(render_multi_camera=mock.Mock(return_value=iter(())))
        future.set_result(session)

        async def await_session_future(sess: str, resolved: asyncio.Future[object]) -> object:
            self.assertEqual(sess, "sess-1")
            self.assertIs(resolved, future)
            return await resolved

        handler = request_router.build_name_handler(
            bridge=SimpleNamespace(),
            heartbeat_sec_getter=lambda: 1.0,
            get_or_create_session_future=lambda req, sess: future,
            require_session_future=lambda sess: future,
            await_session_future=await_session_future,
            require_active_session=lambda sess, resolved, active: active,
            require_camera_name=request_router.require_camera_name,
            stream_scene_rebuild_job=lambda session, name: (),
        )

        response = await handler(self.make_request("multicam", "stream"))

        self.assertEqual(response.media_type, "multipart/x-mixed-replace; boundary=frame")
        session.render_multi_camera.assert_called_once_with()

    def test_snapshot_returns_latest_frame_as_jpeg_bytes(self):
        session = _FakeSession()
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        session.renderer = SimpleNamespace(ready=True, read=mock.Mock(return_value=(7, frame)))

        def fake_fromarray(image: np.ndarray):
            self.assertIs(image, frame)
            return SimpleNamespace(save=lambda data, format, quality: data.write(b"jpeg-frame"))

        with mock.patch.object(session_module.Image, "fromarray", side_effect=fake_fromarray):
            payload = session.snapshot("head_camera")

        self.assertEqual(payload, b"jpeg-frame")

    async def test_render_multi_camera_stitches_three_frames_and_skips_duplicate_indices(self):
        session = _FakeSession()
        session.bundle = SimpleNamespace(cameras=[
            SimpleNamespace(name="head_camera"),
            SimpleNamespace(name="left_wrist_camera"),
            SimpleNamespace(name="right_wrist_camera"),
        ])
        active = {"value": True}
        frames = [
            np.full((2, 2, 3), 10, dtype=np.uint8),
            np.full((2, 2, 3), 20, dtype=np.uint8),
            np.full((2, 2, 3), 30, dtype=np.uint8),
        ]
        session.renderer = SimpleNamespace(
            ready=True,
            read=mock.Mock(side_effect=[
                (11, frames[0]),
                (11, frames[1]),
                (11, frames[2]),
                (11, frames[0]),
                (11, frames[1]),
                (11, frames[2]),
            ]),
        )
        session.is_active = lambda: active["value"]

        sleep_calls = {"count": 0}
        encoded: list[np.ndarray] = []

        async def fake_sleep(_: float):
            sleep_calls["count"] += 1
            if sleep_calls["count"] >= 2:
                active["value"] = False

        with mock.patch.object(session_module.asyncio, "sleep", side_effect=fake_sleep):
            with mock.patch.object(session, "_encode_jpeg", side_effect=lambda frame: encoded.append(frame.copy()) or b"jpeg-strip"):
                with mock.patch.object(session_module, "placeholder_jpeg", return_value=b"jpeg-closed"):
                    stream = session.render_multi_camera()
                    first = await anext(stream)
                    second = await anext(stream)
                    with self.assertRaises(StopAsyncIteration):
                        await anext(stream)

        self.assertEqual(first, session_module.mjpeg_chunk(b"jpeg-strip"))
        self.assertEqual(second, session_module.mjpeg_chunk(b"jpeg-closed"))
        self.assertEqual(len(encoded), 1)
        self.assertEqual(encoded[0].shape, (2, 6, 3))
        np.testing.assert_array_equal(encoded[0][:, :2], frames[0])
        np.testing.assert_array_equal(encoded[0][:, 2:4], frames[1])
        np.testing.assert_array_equal(encoded[0][:, 4:6], frames[2])

    async def test_render_skips_jpeg_reencode_when_frame_index_does_not_advance(self):
        session = _FakeSession()
        active = {"value": True}
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        session.renderer = SimpleNamespace(
            ready=True,
            read=mock.Mock(side_effect=[(11, frame), (11, frame)]),
        )
        session.is_active = lambda: active["value"]

        sleep_calls = {"count": 0}

        async def fake_sleep(_: float):
            sleep_calls["count"] += 1
            if sleep_calls["count"] >= 2:
                active["value"] = False

        def fake_fromarray(_: np.ndarray):
            return SimpleNamespace(save=lambda data, format, quality: data.write(b"jpeg-live"))

        with mock.patch.object(session_module.asyncio, "sleep", side_effect=fake_sleep):
            with mock.patch.object(session_module.Image, "fromarray", side_effect=fake_fromarray) as fromarray:
                with mock.patch.object(session_module, "placeholder_jpeg", return_value=b"jpeg-closed"):
                    stream = session.render("head_camera")
                    first = await anext(stream)
                    second = await anext(stream)
                    with self.assertRaises(StopAsyncIteration):
                        await anext(stream)

        self.assertEqual(first, session_module.mjpeg_chunk(b"jpeg-live"))
        self.assertEqual(second, session_module.mjpeg_chunk(b"jpeg-closed"))
        self.assertEqual(fromarray.call_count, 1)


if __name__ == "__main__":
    unittest.main()
