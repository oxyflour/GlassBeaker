from __future__ import annotations

import builtins
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
RENDERER_ENTRY = REPO_ROOT / "apps" / "isaac" / "rl_renderer_entry.py"


class ChangeBlock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def load_renderer_entry():
    spec = importlib.util.spec_from_file_location("_test_rl_renderer_entry", RENDERER_ENTRY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "geniesim.rl.renderer.rl_renderer":
            raise AssertionError("renderer entry must not import upstream renderer")
        return original_import(name, globals, locals, fromlist, level)

    with mock.patch.object(builtins, "__import__", side_effect=guarded_import):
        spec.loader.exec_module(module)
    return module


class RLRendererEntryTest(unittest.TestCase):
    def test_module_import_does_not_import_upstream_or_start_isaac(self):
        entry = load_renderer_entry()

        self.assertTrue(hasattr(entry, "RLRenderer"))
        self.assertIsNone(entry.simulation_app)

    def test_setup_camera_attaches_annotator_to_render_product_list(self):
        entry = load_renderer_entry()
        events: list[object] = []

        class Annotator:
            def attach(self, render_products):
                events.append(render_products)

        entry.is_prim_path_valid = lambda path: True
        entry.rep = SimpleNamespace(
            create=SimpleNamespace(render_product=lambda path, resolution: "render-product"),
            AnnotatorRegistry=SimpleNamespace(get_annotator=lambda name: Annotator()),
        )
        renderer = entry.RLRenderer.__new__(entry.RLRenderer)
        renderer.args = SimpleNamespace(cam_width=160, cam_height=120)

        binding = renderer._setup_camera("/World/camera")

        self.assertEqual(events, [["render-product"]])
        self.assertEqual(binding.render_product, "render-product")

    def test_run_steps_replicator_when_frame_policy_allows_render(self):
        entry = load_renderer_entry()
        events: list[str] = []
        running = {"calls": 0}

        def is_running():
            running["calls"] += 1
            return running["calls"] == 1

        entry.simulation_app = SimpleNamespace(is_running=is_running, close=lambda: events.append("close"))
        entry.rep = SimpleNamespace(
            orchestrator=SimpleNamespace(
                step=lambda **kwargs: events.append(f"rep:{kwargs['wait_for_render']}")
            )
        )
        entry.rclpy = SimpleNamespace(shutdown=lambda: events.append("rclpy"))
        renderer = entry.RLRenderer.__new__(entry.RLRenderer)
        renderer.env_subscribers = []
        renderer.frame_counter = np.array([0], dtype=np.uint32)
        renderer._service_control_request = lambda: events.append("service")
        renderer._ros_executor = SimpleNamespace(
            spin_once=lambda timeout_sec: events.append("ros"),
            shutdown=lambda timeout_sec: events.append("shutdown"),
        )
        renderer.world = SimpleNamespace(
            step=lambda render: events.append(f"world:{render}"),
            stop=lambda: events.append("stop"),
        )
        renderer.shm = None

        renderer.run()

        self.assertEqual(events[:4], ["service", "ros", "rep:True", "world:True"])

    def test_force_render_env_steps_replicator_after_warmup(self):
        entry = load_renderer_entry()
        events: list[str] = []
        running = {"calls": 0}

        def is_running():
            running["calls"] += 1
            return running["calls"] == 1

        entry.simulation_app = SimpleNamespace(is_running=is_running, close=lambda: None)
        entry.rep = SimpleNamespace(
            orchestrator=SimpleNamespace(
                step=lambda **kwargs: events.append(f"rep:{kwargs['wait_for_render']}")
            )
        )
        entry.rclpy = SimpleNamespace(shutdown=lambda: None)
        renderer = entry.RLRenderer.__new__(entry.RLRenderer)
        renderer.env_subscribers = [SimpleNamespace(_dirty=False, destroy_node=lambda: None)]
        renderer.frame_counter = np.array([99], dtype=np.uint32)
        renderer._service_control_request = lambda: None
        renderer._ros_executor = SimpleNamespace(
            spin_once=lambda timeout_sec: None,
            shutdown=lambda timeout_sec: None,
        )
        renderer.world = SimpleNamespace(
            step=lambda render: events.append(f"world:{render}"),
            stop=lambda: None,
        )
        renderer.shm = None

        with mock.patch.dict(entry.os.environ, {"GB_RENDERER_FORCE_RENDER": "1"}):
            renderer.run()

        self.assertEqual(events[:2], ["rep:True", "world:True"])

    def test_render_callback_copies_flat_rgba_annotator_data(self):
        entry = load_renderer_entry()
        entry.Sdf = SimpleNamespace(ChangeBlock=ChangeBlock)
        renderer = entry.RLRenderer.__new__(entry.RLRenderer)
        flat_rgba = np.arange(2 * 3 * 4, dtype=np.uint8)
        renderer.args = SimpleNamespace(cam_height=2, cam_width=3)
        renderer.num_envs = 1
        renderer._num_cams = 1
        renderer.env_subscribers = [SimpleNamespace(_dirty=False, apply_tf=lambda: None)]
        renderer.frame_counter = np.array([0], dtype=np.uint32)
        renderer.shm_array = np.zeros((1, 1, 2, 3, 3), dtype=np.uint8)
        renderer.cam_annotators_all = [[
            SimpleNamespace(
                annotator=SimpleNamespace(get_data=lambda: flat_rgba),
                render_product=None,
            )
        ]]

        renderer._render_callback(0.0)

        np.testing.assert_array_equal(
            renderer.shm_array[0, 0],
            flat_rgba.reshape(2, 3, 4)[:, :, :3],
        )
        self.assertEqual(int(renderer.frame_counter[0]), 1)

    def test_force_render_env_callback_copies_after_warmup(self):
        entry = load_renderer_entry()
        entry.Sdf = SimpleNamespace(ChangeBlock=ChangeBlock)
        renderer = entry.RLRenderer.__new__(entry.RLRenderer)
        flat_rgba = np.arange(2 * 3 * 4, dtype=np.uint8)
        renderer.args = SimpleNamespace(cam_height=2, cam_width=3)
        renderer.num_envs = 1
        renderer._num_cams = 1
        renderer.env_subscribers = [SimpleNamespace(_dirty=False, apply_tf=lambda: None)]
        renderer.frame_counter = np.array([99], dtype=np.uint32)
        renderer.shm_array = np.zeros((1, 1, 2, 3, 3), dtype=np.uint8)
        renderer.cam_annotators_all = [[
            SimpleNamespace(
                annotator=SimpleNamespace(get_data=lambda: flat_rgba),
                render_product=None,
            )
        ]]

        with mock.patch.dict(entry.os.environ, {"GB_RENDERER_FORCE_RENDER": "1"}):
            renderer._render_callback(0.0)

        np.testing.assert_array_equal(
            renderer.shm_array[0, 0],
            flat_rgba.reshape(2, 3, 4)[:, :, :3],
        )
        self.assertEqual(int(renderer.frame_counter[0]), 100)

    def test_render_callback_publishes_configured_image_after_copying_frame(self):
        entry = load_renderer_entry()
        entry.Sdf = SimpleNamespace(ChangeBlock=ChangeBlock)
        published: list[object] = []

        class ImageMsg:
            def __init__(self):
                self.header = SimpleNamespace(stamp=None, frame_id="")
                self.height = 0
                self.width = 0
                self.encoding = ""
                self.is_bigendian = 0
                self.step = 0
                self.data = b""

        entry.ImageMsg = ImageMsg
        renderer = entry.RLRenderer.__new__(entry.RLRenderer)
        frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        renderer.args = SimpleNamespace(cam_height=2, cam_width=3)
        renderer.num_envs = 1
        renderer._num_cams = 1
        renderer._camera_list = [{"name": "head_camera", "prim": "/camera"}]
        renderer.env_subscribers = [SimpleNamespace(_dirty=False, apply_tf=lambda: None)]
        renderer.frame_counter = np.array([0], dtype=np.uint32)
        renderer.shm_array = np.zeros((1, 1, 2, 3, 3), dtype=np.uint8)
        renderer.cam_annotators_all = [[
            SimpleNamespace(
                annotator=SimpleNamespace(get_data=lambda: frame),
                render_product=None,
            )
        ]]
        renderer._ros_publish_node = SimpleNamespace(
            get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: "stamp"))
        )
        renderer._ros_image_specs_by_camera = {
            0: [SimpleNamespace(topic="/cameras/head_camera/image", camera_name="head_camera")]
        }
        renderer._ros_image_publishers = {
            "/cameras/head_camera/image": SimpleNamespace(publish=published.append)
        }

        renderer._render_callback(0.0)

        self.assertEqual(len(published), 1)
        msg = published[0]
        self.assertEqual(msg.header.stamp, "stamp")
        self.assertEqual(msg.header.frame_id, "head_camera")
        self.assertEqual(msg.height, 2)
        self.assertEqual(msg.width, 3)
        self.assertEqual(msg.encoding, "rgb8")
        self.assertEqual(msg.step, 9)
        self.assertEqual(msg.data, frame.tobytes())


if __name__ == "__main__":
    unittest.main()
