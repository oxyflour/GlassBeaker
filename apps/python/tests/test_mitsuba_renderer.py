from __future__ import annotations

import sys
import asyncio
import tempfile
import threading
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


_UNCHANGED = object()


class _FakeParams(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.update_calls = 0

    def update(self) -> None:
        self.update_calls += 1


class _FakeTransform:
    def __init__(self, value=None) -> None:
        self.value = value

    @classmethod
    def look_at(cls, *, origin, target, up):
        return cls({"origin": origin, "target": target, "up": up})


class _FakeMutableMitsuba(_FakeMitsuba):
    ScalarTransform4f = _FakeTransform

    def __init__(self) -> None:
        super().__init__()
        self.load_calls: list[dict] = []
        self.params = _FakeParams({
            "mesh_0.vertex_positions": None,
            "mesh_static.vertex_positions": _UNCHANGED,
            "sensor_main.to_world": None,
        })

    def load_dict(self, scene_dict):
        self.load_calls.append(scene_dict)
        return {"scene": len(self.load_calls)}

    def traverse(self, scene):
        del scene
        return self.params


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

    def test_start_returns_before_scene_load_finishes(self):
        gate = threading.Event()
        fake = _FakeMitsuba()

        def build_scene(*args, **kwargs):
            del args, kwargs
            gate.wait(timeout=1.0)
            return {"type": "scene"}, []

        with mock.patch("utils.zapdos.renderer.mitsuba_renderer.load_mitsuba", return_value=fake):
            with mock.patch("utils.zapdos.renderer.mitsuba_renderer.build_mitsuba_scene_dict", side_effect=build_scene):
                renderer = MitsubaRenderer("sess-1", _bundle(), 4, 3, 30, True, 0)
                worker = threading.Thread(target=renderer.start)
                worker.start()
                worker.join(timeout=0.1)
                try:
                    self.assertFalse(worker.is_alive())
                finally:
                    gate.set()
                    worker.join(timeout=1.0)
                    renderer.close()

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

    def test_scene_for_pose_updates_body_local_mesh_transform(self):
        renderer = MitsubaRenderer("sess-1", _bundle(), 1, 1, 30, True, 0)
        scene = {
            "type": "scene",
            "mesh_0": {
                "type": "ply",
                "filename": "mesh.ply",
                "_zapdos_body": "box",
                "_zapdos_body_local_matrix": np.eye(4).tolist(),
                "to_world_matrix": np.eye(4).tolist(),
            },
        }
        pose = {"box": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 3.0, 4.0, 1.0]}

        posed = renderer._scene_for_pose(scene, pose)

        self.assertEqual(posed["mesh_0"]["to_world_matrix"][0][3], 2.0)
        self.assertEqual(posed["mesh_0"]["to_world_matrix"][1][3], 3.0)
        self.assertEqual(posed["mesh_0"]["to_world_matrix"][2][3], 4.0)

    def test_scene_for_pose_updates_body_attached_camera(self):
        renderer = MitsubaRenderer("sess-1", _bundle(), 1, 1, 30, True, 0)
        scene = {
            "type": "scene",
            "sensor_main": {
                "type": "perspective",
                "_zapdos_body": "head",
                "_zapdos_camera_local_origin": [0.0, 0.0, 1.0, 1.0],
                "_zapdos_camera_local_target": [0.0, 0.0, 0.0, 1.0],
                "_zapdos_camera_local_up": [0.0, 1.0, 0.0],
            },
        }
        pose = {"head": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 3.0, 4.0, 1.0]}

        posed = renderer._scene_for_pose(scene, pose)

        self.assertEqual(posed["sensor_main"]["to_world_look_at"]["origin"], [2.0, 3.0, 5.0])
        self.assertEqual(posed["sensor_main"]["to_world_look_at"]["target"], [2.0, 3.0, 4.0])

    def test_pose_update_updates_scene_params_without_reloading_ply(self):
        fake = _FakeMutableMitsuba()
        pose = {
            "box": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 3.0, 4.0, 1.0],
            "head": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 3.0, 4.0, 1.0],
        }
        with tempfile.TemporaryDirectory() as tmp:
            mesh_path = Path(tmp) / "mesh.ply"
            mesh_path.write_text(
                "\n".join([
                    "ply",
                    "format ascii 1.0",
                    "element vertex 3",
                    "property float x",
                    "property float y",
                    "property float z",
                    "element face 1",
                    "property list uchar int vertex_indices",
                    "end_header",
                    "0 0 0",
                    "1 0 0",
                    "0 1 0",
                    "3 0 1 2",
                ]) + "\n",
                encoding="utf-8",
            )
            scene = {
                "type": "scene",
                "sensor_main": {
                    "type": "perspective",
                    "_zapdos_body": "head",
                    "_zapdos_camera_local_origin": [0.0, 0.0, 1.0, 1.0],
                    "_zapdos_camera_local_target": [0.0, 0.0, 0.0, 1.0],
                    "_zapdos_camera_local_up": [0.0, 1.0, 0.0],
                    "to_world_look_at": {"origin": [0.0, 0.0, 1.0], "target": [0.0, 0.0, 0.0], "up": [0.0, 1.0, 0.0]},
                },
                "mesh_0": {
                    "type": "ply",
                    "filename": str(mesh_path),
                    "_zapdos_body": "box",
                    "_zapdos_body_local_matrix": np.eye(4).tolist(),
                    "to_world_matrix": np.eye(4).tolist(),
                },
                "mesh_static": {
                    "type": "ply",
                    "filename": str(mesh_path),
                    "to_world_matrix": np.eye(4).tolist(),
                },
            }
            with mock.patch("utils.zapdos.renderer.mitsuba_renderer.load_mitsuba", return_value=fake):
                with mock.patch("utils.zapdos.renderer.mitsuba_renderer.build_mitsuba_scene_dict", return_value=(scene, [])):
                    renderer = MitsubaRenderer("sess-1", _bundle(), 4, 3, 30, True, 0)
                    renderer._load_scene()
                    renderer.update_pose(pose)
                    renderer._load_pose_scene()

        self.assertEqual(len(fake.load_calls), 1)
        self.assertEqual(fake.params.update_calls, 1)
        self.assertEqual(list(fake.params["mesh_0.vertex_positions"]), [2.0, 3.0, 4.0, 3.0, 3.0, 4.0, 2.0, 4.0, 4.0])
        self.assertIs(fake.params["mesh_static.vertex_positions"], _UNCHANGED)
        self.assertEqual(fake.params["sensor_main.to_world"].value["origin"], [2.0, 3.0, 5.0])

    async def test_wait_ready_reports_mitsuba_startup_failure_with_cuda_hint(self):
        with mock.patch("utils.zapdos.renderer.mitsuba_renderer.load_mitsuba", side_effect=RuntimeError("CUDA unavailable")):
            with mock.patch("utils.zapdos.renderer.mitsuba_renderer.build_mitsuba_scene_dict", return_value=({"type": "scene"}, [])):
                renderer = MitsubaRenderer("sess-1", _bundle(), 4, 3, 30, True, 0)

                with self.assertRaisesRegex(RuntimeError, "cuda_ad_rgb"):
                    await renderer.wait_ready(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
