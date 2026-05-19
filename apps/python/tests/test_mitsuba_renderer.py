from __future__ import annotations

import sys
import asyncio
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.zapdos.bundle.camera_specs import RenderCamera
import utils.zapdos.renderer.mitsuba_renderer as mitsuba_renderer_module
from utils.zapdos.renderer.mitsuba_renderer import MitsubaRenderer


def _bundle() -> SimpleNamespace:
    camera = RenderCamera(
        name="main",
        prim="/SceneRender/main",
        topic="/env_0/main/image_raw",
        frame_id="main",
        body=None,
        pos=[0.0, -3.0, 1.0],
        quat=[1.0, 0.0, 0.0, 0.0],
        fovy=45.0,
    )
    return SimpleNamespace(render_scene_usda=Path("scene.usda"), cameras=[camera])


class _FakeMitsuba:
    def __init__(self) -> None:
        self.render_calls = 0
        self.variant: str | None = None

    def set_variant(self, variant: str) -> None:
        self.variant = variant

    def load_dict(self, scene_dict):
        return scene_dict

    def render(self, scene, sensor=0, spp=0):
        del scene, sensor, spp
        self.render_calls += 1
        return np.full((3, 4, 3), self.render_calls, dtype=np.float32)


class MitsubaRendererTest(unittest.IsolatedAsyncioTestCase):
    def test_load_mitsuba_selects_cuda_backend(self):
        fake = _FakeMitsuba()

        with mock.patch.object(mitsuba_renderer_module.importlib, "import_module", return_value=fake):
            loaded = mitsuba_renderer_module.load_mitsuba()

        self.assertIs(loaded, fake)
        self.assertEqual(fake.variant, "cuda_ad_rgb")

    async def test_wait_ready_starts_background_loop_and_read_returns_rgb_frame(self):
        fake = _FakeMitsuba()
        with mock.patch("utils.zapdos.renderer.mitsuba_renderer.load_mitsuba", return_value=fake):
            with mock.patch("utils.zapdos.renderer.mitsuba_renderer.build_mitsuba_scene_dict", return_value=({"type": "scene"}, [])):
                renderer = MitsubaRenderer("sess-1", _bundle(), 4, 3, 30, True, 7)
                try:
                    status = await renderer.wait_ready(timeout=1.0)
                    deadline = time.time() + 1.0
                    frame_state = None
                    while time.time() < deadline and frame_state is None:
                        frame_state = renderer.read("main")
                        await asyncio.sleep(0.01)
                finally:
                    renderer.close()

        self.assertEqual(status["ros_domain_id"], 7)
        self.assertIsNotNone(frame_state)
        index, frame = frame_state
        self.assertGreaterEqual(index, 1)
        self.assertEqual(frame.shape, (3, 4, 3))
        self.assertEqual(frame.dtype, np.uint8)

    async def test_reload_scene_rebuilds_camera_mapping_and_frame_indices(self):
        first = _bundle()
        second = _bundle()
        second.cameras = [RenderCamera(**{**second.cameras[0].to_json(), "name": "wrist", "frame_id": "wrist"})]
        fake = _FakeMitsuba()
        with mock.patch("utils.zapdos.renderer.mitsuba_renderer.load_mitsuba", return_value=fake):
            with mock.patch("utils.zapdos.renderer.mitsuba_renderer.build_mitsuba_scene_dict", return_value=({"type": "scene"}, [])):
                renderer = MitsubaRenderer("sess-1", first, 4, 3, 30, True, 0)
                try:
                    await renderer.wait_ready(timeout=1.0)
                    renderer.reload_scene(second)
                    self.assertIsNone(renderer.read("main"))
                    deadline = time.time() + 1.0
                    frame_state = None
                    while time.time() < deadline and frame_state is None:
                        frame_state = renderer.read("wrist")
                        await asyncio.sleep(0.01)
                finally:
                    renderer.close()

        self.assertIsNotNone(frame_state)
        self.assertEqual(renderer.camera_index, {"wrist": 0})

    async def test_close_stops_renderer(self):
        fake = _FakeMitsuba()
        with mock.patch("utils.zapdos.renderer.mitsuba_renderer.load_mitsuba", return_value=fake):
            with mock.patch("utils.zapdos.renderer.mitsuba_renderer.build_mitsuba_scene_dict", return_value=({"type": "scene"}, [])):
                renderer = MitsubaRenderer("sess-1", _bundle(), 4, 3, 30, True, 0)
                await renderer.wait_ready(timeout=1.0)
                renderer.close()

        self.assertFalse(renderer.running)
        self.assertFalse(renderer.ready)

    def test_render_camera_tonemaps_hdr_values_before_uint8_conversion(self):
        fake = _FakeMitsuba()
        fake.render = mock.Mock(return_value=np.array([[[4.0, 1.0, 0.25]]], dtype=np.float32))
        renderer = MitsubaRenderer("sess-1", _bundle(), 1, 1, 30, True, 0)
        renderer._mi = fake
        renderer._scene = {"type": "scene"}

        frame = renderer._render_camera(0)

        self.assertEqual(frame.dtype, np.uint8)
        self.assertLess(frame[0, 0, 0], 255)
        self.assertGreater(frame[0, 0, 0], frame[0, 0, 1])

    async def test_wait_ready_reports_mitsuba_startup_failure_with_cuda_hint(self):
        with mock.patch("utils.zapdos.renderer.mitsuba_renderer.load_mitsuba", side_effect=RuntimeError("CUDA unavailable")):
            with mock.patch("utils.zapdos.renderer.mitsuba_renderer.build_mitsuba_scene_dict", return_value=({"type": "scene"}, [])):
                renderer = MitsubaRenderer("sess-1", _bundle(), 4, 3, 30, True, 0)

                with self.assertRaisesRegex(RuntimeError, "cuda_ad_rgb"):
                    await renderer.wait_ready(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
