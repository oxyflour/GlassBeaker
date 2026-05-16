from __future__ import annotations

import asyncio
import importlib.util
import json
import queue
import sys
import tempfile
import threading
import unittest
from concurrent.futures import Future as ConcurrentFuture
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.parse import urlencode

import mujoco  # type: ignore
from fastapi import HTTPException, Request
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

import utils.zapdos.editor.commands as EDITOR_COMMANDS_MODULE
from utils.zapdos.editor.rebuild_job import SceneRebuildJob
import utils.zapdos.editor.rebuild_manager as EDITOR_REBUILD_MANAGER
from utils.zapdos.editor.rebuild_types import OverlayRebuildCompletion, PreparedOverlayRebuild
from utils.zapdos.editor.state import default_overlay_state
import utils.zapdos.editor.zapdos_editor as EDITOR_SESSION_MODULE
from utils.zapdos.physics.mujoco_physics import MujocoPhysics as ZapdosPhysics
from utils.session import Session
import utils.zapdos.zapdos_session as SESSION_MODULE
from utils.zapdos.bundle import ensure_render_bundle

MODULE_PATH = REPO_ROOT / "apps" / "python" / "api" / "zapdos" / "{session}" / "{action}.py"
SPEC = importlib.util.spec_from_file_location("zapdos_session_action_import", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

R1PRO_USD = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"
MOZ1_USDA = REPO_ROOT / "deps" / "spirit01_model" / "USD" / "Moz1_robot_only.usda"


class ZapdosImportTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        MODULE.sessions.clear()

    def test_action_module_reexports_split_runtime_symbols(self):
        self.assertEqual(MODULE.ZapdosSession.__module__, "utils.zapdos.zapdos_session")
        self.assertEqual(MODULE._name_.__module__, "utils.zapdos.request_router")
        self.assertEqual(MODULE._stream_scene_rebuild_job.__module__, MODULE.__name__)

    def test_consumer_modules_import_new_bundle_and_camera_packages(self):
        cases = {
            REPO_ROOT / "apps" / "python" / "utils" / "teleop" / "manager.py": [
                "from utils.zapdos.bundle import DEFAULT_SCENE_USD",
                "from utils.zapdos.rl_bundle import DEFAULT_SCENE_USD",
            ],
            REPO_ROOT / "apps" / "python" / "utils" / "teleop" / "ik_controller.py": [
                "from utils.zapdos.bundle import ensure_render_bundle",
                "from utils.zapdos.rl_bundle import ensure_render_bundle",
            ],
            REPO_ROOT / "apps" / "python" / "api" / "teleop" / "spacemouse.py": [
                "from utils.teleop.manager import SpaceMouseManager",
                "from teleop.manager import SpaceMouseManager",
            ],
            REPO_ROOT / "apps" / "python" / "utils" / "camera_override.py": [
                "from utils.zapdos.bundle.camera_specs import RenderCamera",
                "from utils.zapdos.rl_cameras import RenderCamera",
            ],
            REPO_ROOT / "apps" / "python" / "utils" / "zapdos" / "renderer" / "base.py": [
                "from utils.zapdos.bundle import RenderBundle",
                "from utils.zapdos.rl_bundle import RenderBundle",
            ],
            REPO_ROOT / "apps" / "python" / "utils" / "zapdos" / "renderer" / "isaac_renderer.py": [
                "from utils.zapdos.bundle.camera_specs import camera_name_to_index, cameras_json",
                "from utils.zapdos.rl_cameras import camera_name_to_index, cameras_json",
            ],
            REPO_ROOT / "apps" / "python" / "utils" / "zapdos" / "editor" / "rebuild_runner.py": [
                "from utils.zapdos.bundle import ensure_render_bundle",
                "from utils.zapdos.rl_bundle import ensure_render_bundle",
            ],
        }

        for path, (expected_import, legacy_import) in cases.items():
            source = path.read_text(encoding="utf-8")
            self.assertIn(expected_import, source, path.as_posix())
            self.assertNotIn(legacy_import, source, path.as_posix())

    def test_camera_specs_owns_render_camera_contracts(self):
        source = (
            REPO_ROOT
            / "apps"
            / "python"
            / "utils"
            / "zapdos"
            / "bundle"
            / "camera_specs.py"
        ).read_text(encoding="utf-8")

        self.assertIn("class RenderCamera", source)
        self.assertIn("def build_render_cameras", source)
        self.assertNotIn("from utils.zapdos.rl_cameras import", source)

    def test_bundle_package_owns_render_bundle_contracts(self):
        source = (
            REPO_ROOT
            / "apps"
            / "python"
            / "utils"
            / "zapdos"
            / "bundle"
            / "render_bundle.py"
        ).read_text(encoding="utf-8")

        self.assertIn("class RenderBundle", source)
        self.assertIn("DEFAULT_SCENE_USD", source)
        self.assertNotIn("from utils.zapdos.rl_bundle import", source)

    def test_bundle_package_owns_bundle_builder_without_legacy_bridge(self):
        source = (
            REPO_ROOT
            / "apps"
            / "python"
            / "utils"
            / "zapdos"
            / "bundle"
            / "bundle_builder.py"
        ).read_text(encoding="utf-8")

        self.assertIn("def ensure_render_bundle", source)
        self.assertNotIn("import utils.zapdos.rl_bundle as legacy_bundle", source)

    def test_bundle_package_owns_scene_catalog_and_stage_builder(self):
        scene_catalog = (
            REPO_ROOT
            / "apps"
            / "python"
            / "utils"
            / "zapdos"
            / "bundle"
            / "scene_catalog.py"
        ).read_text(encoding="utf-8")
        stage_builder = (
            REPO_ROOT
            / "apps"
            / "python"
            / "utils"
            / "zapdos"
            / "bundle"
            / "stage_builder.py"
        ).read_text(encoding="utf-8")

        self.assertIn("class SceneObjectSpec", scene_catalog)
        self.assertNotIn("from utils.zapdos.scene_objects import", scene_catalog)
        self.assertIn("def build_robot_wrapper", stage_builder)
        self.assertNotIn("from utils.zapdos.rl_bundle_stage import", stage_builder)

    def test_physics_package_owns_mujoco_tools_without_root_bridge(self):
        source = (
            REPO_ROOT
            / "apps"
            / "python"
            / "utils"
            / "zapdos"
            / "physics"
            / "mujoco_tools.py"
        ).read_text(encoding="utf-8")

        self.assertIn("def decode_mesh_path", source)
        self.assertIn("def body_world_pose", source)
        self.assertNotIn("from utils.zapdos.mujoco_tools import", source)

    def test_renderer_package_owns_reload_helpers_without_root_bridge(self):
        source = (
            REPO_ROOT
            / "apps"
            / "python"
            / "utils"
            / "zapdos"
            / "renderer"
            / "isaac_renderer_reload.py"
        ).read_text(encoding="utf-8")

        self.assertIn("class CameraBinding", source)
        self.assertIn("def rebuild_camera_bindings", source)
        self.assertNotIn("from utils.zapdos.isaac_renderer_reload import", source)

    def test_legacy_flat_zapdos_modules_are_deleted(self):
        legacy_paths = [
            "apps/python/utils/zapdos/rl_bundle.py",
            "apps/python/utils/zapdos/rl_bundle_stage.py",
            "apps/python/utils/zapdos/rl_cameras.py",
            "apps/python/utils/zapdos/scene_objects.py",
            "apps/python/utils/zapdos/sim_env.py",
            "apps/python/utils/zapdos/zapdos_overlay.py",
            "apps/python/utils/zapdos/zapdos_overlay_rebuild_runner.py",
            "apps/python/utils/zapdos/zapdos_overlay_scene.py",
            "apps/python/utils/zapdos/zapdos_physics.py",
            "apps/python/utils/zapdos/zapdos_scene_operations.py",
            "apps/python/utils/zapdos/zapdos_scene_visuals.py",
        ]

        for raw_path in legacy_paths:
            with self.subTest(path=raw_path):
                self.assertFalse((REPO_ROOT / raw_path).exists(), raw_path)

    def test_root_helper_shells_are_deleted(self):
        helper_paths = [
            "apps/python/utils/zapdos/mujoco_tools.py",
            "apps/python/utils/zapdos/isaac_renderer_reload.py",
            "apps/python/utils/zapdos/renderer_ipc.py",
        ]

        for raw_path in helper_paths:
            with self.subTest(path=raw_path):
                self.assertFalse((REPO_ROOT / raw_path).exists(), raw_path)

    def test_session_coordinator_module_stays_within_line_budget(self):
        path = (
            REPO_ROOT
            / "apps"
            / "python"
            / "utils"
            / "zapdos"
            / "zapdos_session.py"
        )

        self.assertLessEqual(len(path.read_text(encoding="utf-8").splitlines()), 200)

    def test_rebuild_manager_owns_overlay_rebuild_runner_import(self):
        source = (
            REPO_ROOT
            / "apps"
            / "python"
            / "utils"
            / "zapdos"
            / "editor"
            / "rebuild_manager.py"
        ).read_text(encoding="utf-8")

        self.assertIn("from utils.zapdos.editor.rebuild_runner import prepare_overlay_rebuild_request", source)
        self.assertNotIn("from utils.zapdos.zapdos_overlay_rebuild_runner import prepare_overlay_rebuild_request", source)

    def make_request(self, query: str = ""):
        return Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/python/zapdos/sess-1/init/start",
                "path_params": {"session": "sess-1", "action": "init", "name": "start"},
                "query_string": query.encode("utf-8"),
                "headers": [],
            }
        )

    def attach_editor(self, session):
        session.editor = EDITOR_SESSION_MODULE.ZapdosEditor(
            session,
            repo_root=SESSION_MODULE.REPO_ROOT,
            default_robot_usd=SESSION_MODULE.DEFAULT_ROBOT_USD,
            default_scene_usd=MODULE.DEFAULT_SCENE_USD,
        )
        return session.editor

    def build_physics_session(self, xml: str, body_map: dict[str, str]):
        with tempfile.TemporaryDirectory() as tmp:
            xml_path = Path(tmp) / "scene.xml"
            xml_path.write_text(xml.strip(), encoding="utf-8")
            physics = ZapdosPhysics("sess-1", SimpleNamespace(mjcf=xml_path), body_map)
            session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
            session.sess = "sess-1"
            session.physics = physics
            session.bundle = SimpleNamespace(cameras=[])
            session.msgs = queue.Queue(maxsize=64)
            self.attach_editor(session)
            return session

    def build_pose_edit_session(self):
        return self.build_physics_session(
            """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="RobotLink" pos="0 0 0.2">
      <geom name="robot-box" type="box" size="0.1 0.1 0.1" rgba="0 0 1 1"/>
    </body>
    <body name="Scene_Crate" pos="1 2 3">
      <geom name="crate-box" type="box" size="0.2 0.2 0.2" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
""",
            {
                "RobotLink": "MyRobot/RobotLink",
                "Scene_Crate": "Crate",
            },
        )

    def build_robot_root_pose_edit_session(self):
        return self.build_physics_session(
            """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="Root_base_link" pos="0 0 0.2">
      <freejoint name="Root_base_link_freejoint"/>
      <geom name="root-box" type="box" size="0.1 0.1 0.1" rgba="0 0 1 1"/>
      <body name="Arm_link" pos="0.3 0 0">
        <geom name="arm-box" type="box" size="0.05 0.05 0.2" rgba="0 1 0 1"/>
      </body>
    </body>
    <body name="Scene_Crate" pos="1 2 3">
      <geom name="crate-box" type="box" size="0.2 0.2 0.2" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
""",
            {
                "Root_base_link": "MyRobot/Root_base_link",
                "Arm_link": "MyRobot/Arm_link",
                "Scene_Crate": "Crate",
            },
        )

    def build_nested_pose_edit_session(self):
        return self.build_physics_session(
            """
<mujoco>
  <worldbody>
    <body name="Scene_Crate" pos="1 2 3">
      <body name="Scene_Crate_payload" pos="0.5 0 0">
        <geom name="crate-box" type="box" size="0.2 0.2 0.2" rgba="1 0 0 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""",
            {"Scene_Crate": "Objects/Crate"},
        )

    def build_freejoint_pose_edit_session(self):
        return self.build_physics_session(
            """
<mujoco>
  <worldbody>
    <body name="Scene_Crate" pos="1 2 3">
      <freejoint name="Scene_Crate_freejoint"/>
      <geom name="crate-box" type="box" size="0.2 0.2 0.2" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
""",
            {"Scene_Crate": "Crate"},
        )

    def build_attachment_session(self):
        return self.build_physics_session(
            """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="Root_base_link" pos="0 0 0.5">
      <freejoint name="Root_base_link_freejoint"/>
      <geom name="root-box" type="box" size="0.1 0.1 0.1" rgba="0 0 1 1"/>
    </body>
    <body name="Scene_Crate" pos="1 0 0.5">
      <freejoint name="Scene_Crate_freejoint"/>
      <geom name="crate-box" type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
""",
            {
                "Root_base_link": "MyRobot/Root_base_link",
                "Scene_Crate": "Scene/Crate",
            },
        )

    def test_input_path_accepts_absolute_scene_usd(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene.usda"
            scene.write_text("#usda 1.0\n", encoding="utf-8")
            req = self.make_request(urlencode({"scene_usd": str(scene)}))

            resolved = MODULE._input_path(req, "scene_usd", Path("unused.usda"))

            self.assertEqual(resolved, scene.resolve())

    def test_input_path_accepts_repo_relative_scene_usd(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            scene = repo_root / "tmp" / "scene.usda"
            scene.parent.mkdir(parents=True)
            scene.write_text("#usda 1.0\n", encoding="utf-8")
            req = self.make_request(urlencode({"scene_usd": "tmp/scene.usda"}))

            with mock.patch.object(MODULE, "REPO_ROOT", repo_root):
                resolved = MODULE._input_path(req, "scene_usd", Path("unused.usda"))

            self.assertEqual(resolved, scene.resolve())

    async def test_zapdos_session_create_accepts_moz1_usda_bundle(self):
        create = SESSION_MODULE.ZapdosSession.create
        bundle = SimpleNamespace()
        with mock.patch.object(SESSION_MODULE.asyncio, "to_thread", new=mock.AsyncMock(return_value=bundle)) as to_thread:
            with mock.patch.object(SESSION_MODULE, "ZapdosSession", side_effect=lambda sess, built_bundle: (sess, built_bundle)):
                result = await create("sess-1", MOZ1_USDA, MODULE.DEFAULT_SCENE_USD)

        self.assertEqual(result, ("sess-1", bundle))
        to_thread.assert_awaited_once_with(ensure_render_bundle, MOZ1_USDA, MODULE.DEFAULT_SCENE_USD)

    async def test_get_or_create_session_future_uses_default_r1pro_when_query_is_empty(self):
        req = self.make_request()
        create = mock.AsyncMock(return_value=SimpleNamespace(camera_index={}))

        with mock.patch.object(MODULE.ZapdosSession, "create", new=create):
            future = MODULE._get_or_create_session_future(req, "sess-default")
            await MODULE._await_session_future("sess-default", future)

        create.assert_awaited_once_with("sess-default", MODULE.DEFAULT_ROBOT_USD, MODULE.DEFAULT_SCENE_USD)

    async def test_get_or_create_session_future_accepts_explicit_r1pro_robot_usd(self):
        req = self.make_request(urlencode({"robot_usd": "deps/galaxea/object/r1pro/r1pro.usda"}))
        create = mock.AsyncMock(return_value=SimpleNamespace(camera_index={}))

        with mock.patch.object(MODULE.ZapdosSession, "create", new=create):
            future = MODULE._get_or_create_session_future(req, "sess-r1pro")
            await MODULE._await_session_future("sess-r1pro", future)

        create.assert_awaited_once_with("sess-r1pro", R1PRO_USD.resolve(), MODULE.DEFAULT_SCENE_USD)

    async def test_failed_bootstrap_is_evicted_and_can_retry(self):
        req = self.make_request()
        create = mock.AsyncMock(side_effect=[RuntimeError("boom"), SimpleNamespace(camera_index={})])
        with mock.patch.object(MODULE, "_input_path", side_effect=[Path("robot.usda"), Path("scene.usda"), Path("robot.usda"), Path("scene.usda")]):
            with mock.patch.object(MODULE.ZapdosSession, "create", new=create):
                future = MODULE._get_or_create_session_future(req, "sess-1")
                with self.assertRaises(RuntimeError):
                    await MODULE._await_session_future("sess-1", future)
                self.assertNotIn("sess-1", MODULE.sessions)

                retry = MODULE._get_or_create_session_future(req, "sess-1")
                self.assertIs(MODULE.sessions["sess-1"], retry)
                await MODULE._await_session_future("sess-1", retry)

    async def test_init_recreates_inactive_cached_session(self):
        req = self.make_request()
        stale: asyncio.Future[object] = asyncio.Future()
        stale.set_result(SimpleNamespace(is_active=lambda: False))
        MODULE.sessions["sess-1"] = stale
        create = mock.AsyncMock(return_value=SimpleNamespace(camera_index={}))

        with mock.patch.object(MODULE, "_input_path", side_effect=[Path("robot.usda"), Path("scene.usda")]):
            with mock.patch.object(MODULE.ZapdosSession, "create", new=create):
                fresh = MODULE._get_or_create_session_future(req, "sess-1")

        self.assertIs(MODULE.sessions["sess-1"], fresh)
        self.assertIsNot(fresh, stale)
        await MODULE._await_session_future("sess-1", fresh)

    async def test_init_stream_emits_readable_error(self):
        future: asyncio.Future[object] = asyncio.Future()
        future.set_exception(RuntimeError("scene_usd not found"))
        MODULE.sessions["sess-1"] = future

        events = [chunk async for chunk in MODULE._init_stream("sess-1", future)]

        self.assertEqual(events, ["data: loading\n\n", "data: error: scene_usd not found\n\n"])

    async def test_init_stream_emits_heartbeat_while_bootstrap_is_pending(self):
        future = asyncio.create_task(asyncio.sleep(0.03, result=SimpleNamespace(camera_index={})))
        MODULE.sessions["sess-1"] = future

        with mock.patch.object(MODULE, "INIT_STREAM_HEARTBEAT_SEC", 0.01):
            events = [chunk async for chunk in MODULE._init_stream("sess-1", future)]

        self.assertEqual(events[0], "data: loading\n\n")
        self.assertIn("data: loading: preparing render bundle\n\n", events[:-1])
        self.assertEqual(events[-1], "data: started\n\n")

    def test_build_geometry_only_assigns_texture_when_material_has_rgb_texture(self):
        bundle = ensure_render_bundle(MODULE.DEFAULT_ROBOT_USD, MODULE.DEFAULT_SCENE_USD)
        physics = ZapdosPhysics("sess-1", bundle, {})

        textured = [geom for geom in physics.geoms.values() if geom.kind == "mesh" and geom.texture]
        untextured = [geom for geom in physics.geoms.values() if geom.kind == "mesh" and not geom.texture]

        self.assertTrue(textured)
        self.assertTrue(untextured)
        for geom in untextured:
            mat_id = int(physics.model.geom_matid[geom.geom_id])
            if mat_id < 0:
                continue
            tex_id = int(physics.model.mat_texid[mat_id, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
            self.assertLess(tex_id, 0)

    def test_session_init_reads_body_map_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xml_path = root / "scene.xml"
            xml_path.write_text(
                """
<mujoco>
  <worldbody>
    <body name="Scene_Crate" pos="1 2 3">
      <geom name="crate-box" type="box" size="0.2 0.2 0.2" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
""".strip(),
                encoding="utf-8",
            )
            body_map_path = root / "render_scene_body_map.json"
            body_map_path.write_text(json.dumps({"Scene_Crate": "Crate"}), encoding="utf-8")
            bundle = SimpleNamespace(
                mjcf=xml_path,
                body_map_json=body_map_path,
                cameras=[],
            )

            def fake_session_init(instance, timeout=120):
                instance.loop = asyncio.get_event_loop()
                instance.timers = []
                instance.msgs = None
                instance.calls = None
                instance.active = 0
                instance.timeout = timeout

            with mock.patch.object(Session, "__init__", new=fake_session_init):
                with mock.patch.object(SESSION_MODULE, "IsaacRenderer", return_value=SimpleNamespace(wait_ready=mock.AsyncMock(), read=mock.Mock(return_value=None), close=mock.Mock())):
                    with mock.patch.object(
                        MODULE.asyncio,
                        "run_coroutine_threadsafe",
                        side_effect=lambda coro, loop: coro.close(),
                    ):
                        session = MODULE.ZapdosSession("sess-1", bundle)

        self.assertIsInstance(session.physics, ZapdosPhysics)
        self.assertEqual(session.physics.body_map, {"Scene_Crate": "Crate"})
        self.assertEqual(session.physics.editable_body_names, {"Scene_Crate"})
        self.assertIs(session.editor.session, session)

    def test_session_init_discards_stale_overlay_instances(self):
        rewritten_overlay = None
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xml_path = root / "scene.xml"
            xml_path.write_text(
                """
<mujoco>
  <worldbody>
    <body name="Scene_Crate" pos="1 2 3">
      <geom name="crate-box" type="box" size="0.2 0.2 0.2" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
""".strip(),
                encoding="utf-8",
            )
            body_map_path = root / "render_scene_body_map.json"
            body_map_path.write_text(json.dumps({"Scene_Crate": "Crate"}), encoding="utf-8")
            bundle = SimpleNamespace(
                mjcf=xml_path,
                body_map_json=body_map_path,
                cameras=[],
            )
            overlay_path = root / "apps" / "python" / "tmp" / "zapdos" / "sess-1" / "overlay.json"
            overlay_path.parent.mkdir(parents=True)
            overlay_path.write_text(json.dumps({
                "version": 1,
                "assets_root": "C:/assets",
                "instances": [{
                    "id": "benchmark_table_000_01",
                    "asset_id": "benchmark_table_000",
                    "url": "objects/benchmark/table/benchmark_table_000/Aligned.usda",
                    "motion": "static",
                    "placement": {"kind": "floor_at_xy", "xy": [0, 0], "z_offset": 0, "yaw": 0},
                }],
                "pose_overrides": {"Scene_benchmark_table_000_01": {"pos": [0, 0, 0], "quat": [1, 0, 0, 0]}},
            }), encoding="utf-8")

            def fake_session_init(instance, timeout=120):
                instance.loop = asyncio.get_event_loop()
                instance.timers = []
                instance.msgs = None
                instance.calls = None
                instance.active = 0
                instance.timeout = timeout

            with mock.patch.object(SESSION_MODULE, "REPO_ROOT", root):
                with mock.patch.object(Session, "__init__", new=fake_session_init):
                    with mock.patch.object(SESSION_MODULE, "IsaacRenderer", return_value=SimpleNamespace(wait_ready=mock.AsyncMock(), read=mock.Mock(return_value=None), close=mock.Mock())):
                        with mock.patch.object(
                            MODULE.asyncio,
                            "run_coroutine_threadsafe",
                            side_effect=lambda coro, loop: coro.close(),
                        ):
                            session = MODULE.ZapdosSession("sess-1", bundle)
            rewritten_overlay = json.loads(overlay_path.read_text(encoding="utf-8"))

        self.assertEqual(session.editor.overlay_state["instances"], [])
        self.assertEqual(session.editor.overlay_state["pose_overrides"], {})
        self.assertEqual(rewritten_overlay["instances"], [])

    def test_session_does_not_keep_read_only_physics_passthrough_helpers(self):
        self.assertFalse(hasattr(MODULE.ZapdosSession, "_bind_physics"))
        self.assertNotIn("get_visual", MODULE.ZapdosSession.__dict__)
        self.assertNotIn("get_pose", MODULE.ZapdosSession.__dict__)
        self.assertNotIn("get_camera", MODULE.ZapdosSession.__dict__)

    def test_session_does_not_keep_legacy_sync_overlay_rebuild_helper(self):
        self.assertFalse(hasattr(MODULE.ZapdosSession, "_rebuild_overlay_runtime"))

    def test_get_visual_returns_body_groups_and_meshes(self):
        session = self.build_pose_edit_session()

        payload = session.call_once("get_visual", ())

        self.assertEqual(sorted(payload.keys()), ["bodies", "meshes"])
        bodies = {body["name"]: body for body in payload["bodies"]}
        self.assertTrue(bodies["Scene_Crate"]["editable"])
        self.assertTrue(bodies["Scene_Crate"]["movable"])
        self.assertTrue(bodies["Scene_Crate"]["selectable"])
        self.assertEqual(bodies["Scene_Crate"]["selectionBody"], "Scene_Crate")
        self.assertFalse(bodies["RobotLink"]["editable"])
        self.assertTrue(bodies["RobotLink"]["movable"])
        self.assertTrue(bodies["RobotLink"]["selectable"])
        self.assertEqual(bodies["RobotLink"]["selectionBody"], "RobotLink")
        self.assertEqual(bodies["Scene_Crate"]["label"], "Crate")
        attached = [mesh for mesh in payload["meshes"] if mesh.get("body") == "Scene_Crate"]
        self.assertTrue(attached)
        self.assertIn("localMatrix", attached[0])
        self.assertNotIn("matrix", attached[0])
        static_mesh = next(mesh for mesh in payload["meshes"] if mesh.get("body") is None)
        self.assertIn("matrix", static_mesh)
        self.assertNotIn("localMatrix", static_mesh)

    def test_robot_root_detection_uses_parent_links_for_robot_bodies(self):
        session = self.build_robot_root_pose_edit_session()

        self.assertEqual(session.physics.robot_root_body_names, {"Root_base_link"})
        self.assertEqual(session.physics.movable_body_names, {"Root_base_link", "Scene_Crate"})

    def test_get_visual_maps_non_root_robot_body_to_root_selection_body(self):
        session = self.build_robot_root_pose_edit_session()

        payload = session.call_once("get_visual", ())

        bodies = {body["name"]: body for body in payload["bodies"]}
        self.assertTrue(bodies["Root_base_link"]["movable"])
        self.assertEqual(bodies["Root_base_link"]["selectionBody"], "Root_base_link")
        self.assertFalse(bodies["Arm_link"]["movable"])
        self.assertTrue(bodies["Arm_link"]["selectable"])
        self.assertEqual(bodies["Arm_link"]["selectionBody"], "Root_base_link")

    def test_list_scene_bodies_returns_top_level_robot_bounds(self):
        session = self.build_robot_root_pose_edit_session()

        payload = session.call_once("list_scene_bodies", ())

        self.assertEqual([item["body"] for item in payload["items"]], ["Scene_Crate"])
        self.assertEqual(payload["robot_bounds"], {
            "min": [-0.1, -0.1, 0.0],
            "max": [0.35, 0.1, 0.4],
        })

    def test_list_scene_bodies_includes_world_aabb_for_editable_bodies(self):
        session = self.build_pose_edit_session()

        payload = session.call_once("list_scene_bodies", ())

        self.assertEqual(payload["items"][0]["body"], "Scene_Crate")
        self.assertEqual(payload["items"][0]["world_aabb"], {
            "min": [0.8, 1.8, 2.8],
            "max": [1.2, 2.2, 3.2],
        })
        self.assertEqual(payload["items"][0]["support"], {"top_z": 3.2})

    def test_list_scene_bodies_includes_descendant_geom_world_aabb_for_editable_body(self):
        session = self.build_nested_pose_edit_session()

        payload = session.call_once("list_scene_bodies", ())

        self.assertEqual(payload["items"][0]["body"], "Scene_Crate")
        self.assertEqual(payload["items"][0]["world_aabb"], {
            "min": [1.3, 1.8, 2.8],
            "max": [1.7, 2.2, 3.2],
        })

    def test_mesh_world_aabb_uses_vertices_not_bounding_sphere(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh_path = root / "thin_table.obj"
            mesh_path.write_text(
                "\n".join((
                    "v -0.3 -0.6 0.0",
                    "v 0.3 -0.6 0.0",
                    "v 0.3 0.6 0.0",
                    "v -0.3 0.6 0.0",
                    "v -0.3 -0.6 0.05",
                    "v 0.3 -0.6 0.05",
                    "v 0.3 0.6 0.05",
                    "v -0.3 0.6 0.05",
                    "f 1 2 3",
                    "f 1 3 4",
                    "f 5 8 7",
                    "f 5 7 6",
                    "f 1 5 6",
                    "f 1 6 2",
                    "f 2 6 7",
                    "f 2 7 3",
                    "f 3 7 8",
                    "f 3 8 4",
                    "f 4 8 5",
                    "f 4 5 1",
                )),
                encoding="utf-8",
            )
            xml_path = root / "scene.xml"
            xml_path.write_text(
                f"""
<mujoco>
  <asset>
    <mesh name="thin_table" file="{mesh_path.as_posix()}"/>
  </asset>
  <worldbody>
    <body name="Scene_Table" pos="0.5 0 0.4">
      <geom name="table-mesh" type="mesh" mesh="thin_table" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
""".strip(),
                encoding="utf-8",
            )
            physics = ZapdosPhysics("sess-1", SimpleNamespace(mjcf=xml_path), {"Scene_Table": "Table"})
            try:
                self.assertEqual(physics.body_world_aabb("Scene_Table"), {
                    "min": [0.2, -0.6, 0.4],
                    "max": [0.8, 0.6, 0.45],
                })
            finally:
                physics.close()

    def test_get_visual_attaches_descendant_scene_mesh_to_editable_ancestor(self):
        session = self.build_nested_pose_edit_session()

        payload = session.call_once("get_visual", ())

        mesh = next(item for item in payload["meshes"] if item["name"] == "geom-0")
        self.assertEqual(mesh["body"], "Scene_Crate")
        self.assertIn("localMatrix", mesh)

    def test_get_visual_ignores_collision_mesh_that_reuses_visual_asset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xml_path = root / "scene.xml"
            mesh_path = root / "shared.obj"
            mesh_path.write_text(
                "\n".join((
                    "v -1 0 0",
                    "v 1 0 0",
                    "v 0 1 0",
                    "v 0 0 1",
                    "f 1 2 3",
                    "f 1 2 4",
                    "f 2 3 4",
                    "f 1 3 4",
                    "",
                )),
                encoding="utf-8",
            )
            xml_path.write_text(
                """
<mujoco>
  <asset>
    <mesh name="shared" file="shared.obj"/>
  </asset>
  <worldbody>
    <body name="RobotLink" pos="0 0 0.2">
      <geom name="RobotLink_visuals_geom" type="mesh" mesh="shared" rgba="0.2 0.3 0.4 1"/>
      <geom name="RobotLink_collisions_geom" type="mesh" mesh="shared" rgba="1 1 1 1"/>
    </body>
  </worldbody>
</mujoco>
""".strip(),
                encoding="utf-8",
            )
            physics = ZapdosPhysics(
                "sess-1",
                SimpleNamespace(mjcf=xml_path),
                {"RobotLink": "MyRobot/RobotLink"},
            )

            payload = physics.get_visual()

        self.assertEqual(len(payload["meshes"]), 1)
        self.assertEqual(
            [round(value, 3) for value in payload["meshes"][0]["color"]],
            [0.2, 0.3, 0.4, 1.0],
        )

    def test_set_body_pose_updates_editable_scene_body(self):
        session = self.build_pose_edit_session()
        body_id = mujoco.mj_name2id(session.physics.model, mujoco.mjtObj.mjOBJ_BODY, "Scene_Crate")  # type: ignore

        session.call_once("set_body_pose", ("Scene_Crate", [4.0, 5.0, 6.0], [1.0, 0.0, 0.0, 0.0]))

        self.assertEqual(session.physics.model.body_pos[body_id].tolist(), [4.0, 5.0, 6.0])
        self.assertEqual(session.physics.data.xpos[body_id].tolist(), [4.0, 5.0, 6.0])

    def test_set_body_pose_updates_editable_freejoint_scene_body(self):
        session = self.build_freejoint_pose_edit_session()
        body_id = mujoco.mj_name2id(session.physics.model, mujoco.mjtObj.mjOBJ_BODY, "Scene_Crate")  # type: ignore
        joint_id = mujoco.mj_name2id(session.physics.model, mujoco.mjtObj.mjOBJ_JOINT, "Scene_Crate_freejoint")  # type: ignore
        qpos_adr = int(session.physics.model.jnt_qposadr[joint_id])

        session.call_once("set_body_pose", ("Scene_Crate", [4.0, 5.0, 6.0], [1.0, 0.0, 0.0, 0.0]))

        self.assertEqual(session.physics.data.xpos[body_id].tolist(), [4.0, 5.0, 6.0])
        self.assertEqual(session.physics.data.qpos[qpos_adr:qpos_adr + 3].tolist(), [4.0, 5.0, 6.0])

    def test_attachment_updates_attached_body_pose_during_step(self):
        session = self.build_attachment_session()

        attached = session.physics.attach_body("Root_base_link", "Scene_Crate")
        session.call_once("set_body_pose", ("Root_base_link", [2.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0]))
        session.physics.step()

        pose = session.physics.get_pose()["Scene_Crate"]
        self.assertEqual(attached["parent_body"], "Root_base_link")
        self.assertEqual(pose[12:15], [3.0, 0.0, 0.5])
        self.assertEqual(session.physics.get_attachment("Scene_Crate")["child_body"], "Scene_Crate")

    def test_set_body_pose_accepts_robot_root_and_rejects_non_root_robot_body(self):
        session = self.build_robot_root_pose_edit_session()
        root_body_id = mujoco.mj_name2id(session.physics.model, mujoco.mjtObj.mjOBJ_BODY, "Root_base_link")  # type: ignore

        session.call_once("set_body_pose", ("Root_base_link", [4.0, 5.0, 6.0], [1.0, 0.0, 0.0, 0.0]))

        self.assertEqual(session.physics.data.xpos[root_body_id].tolist(), [4.0, 5.0, 6.0])

        with self.assertRaises(HTTPException) as robot_error:
            session.call_once("set_body_pose", ("Arm_link", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]))
        self.assertEqual(robot_error.exception.status_code, 403)

        with self.assertRaises(HTTPException) as missing_error:
            session.call_once("set_body_pose", ("MissingBody", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]))
        self.assertEqual(missing_error.exception.status_code, 404)

    def test_set_body_pose_rejects_while_scene_rebuild_is_running(self):
        session = self.build_pose_edit_session()
        session.editor.rebuilding_scene = True

        with self.assertRaises(HTTPException) as err:
            session.call_once("set_body_pose", ("Scene_Crate", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]))

        self.assertEqual(err.exception.status_code, 409)
        self.assertIn("Scene rebuild already in progress", err.exception.detail)

    def test_set_scene_assets_returns_operation_id(self):
        session = self.build_pose_edit_session()
        editor = session.editor
        editor.base_scene_usd = Path("scene.usda")
        editor.robot_usd = Path("robot.usda")
        editor.overlay_state = default_overlay_state("C:/assets")
        editor.scene_revision = "rev-1"
        editor.overlay_path = Path("overlay.json")
        editor.composed_scene_usd = Path("overlay_scene.usda")
        editor.rebuilding_scene = False

        with mock.patch.object(
            EDITOR_COMMANDS_MODULE,
            "resolve_asset_record",
            return_value={"asset_id": "table_000", "url": "objects/table_000/Aligned.usda", "description": {}},
        ):
            with mock.patch.object(editor, "_start_overlay_operation", return_value={"ok": True, "op_id": "op-1"}) as start_op:
                result = session.editor.set_scene_assets(
                    [{
                        "asset_id": "table_000",
                        "motion": "static",
                        "placement": {"kind": "floor_at_xy", "xy": [0.0, 0.0], "z_offset": 0.0, "yaw": 0.0},
                    }],
                )

        self.assertEqual(result, {"ok": True, "op_id": "op-1"})
        start_args = start_op.call_args.args
        self.assertEqual(start_args[0]["instances"], [{
            "id": "table_000_01",
            "asset_id": "table_000",
            "url": "objects/table_000/Aligned.usda",
            "motion": "static",
            "placement": {"kind": "floor_at_xy", "xy": [0.0, 0.0], "z_offset": 0.0, "yaw": 0.0},
        }])
        self.assertEqual(start_args[1], {
            "ok": True,
            "items": [{
                "asset_id": "table_000",
                "instance_id": "table_000_01",
                "body": "Scene_table_000_01",
            }],
        })

    def test_set_scene_assets_delegates_to_editor(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.editor = mock.Mock(
            set_scene_assets=mock.Mock(return_value={"ok": True, "op_id": "op-9"}),
        )

        result = MODULE.ZapdosSession.call_once(session, "set_scene_assets", ([{"asset_id": "table_000"}],))

        self.assertEqual(result, {"ok": True, "op_id": "op-9"})
        session.editor.set_scene_assets.assert_called_once_with([{"asset_id": "table_000"}])

    def test_set_scene_assets_rejects_ambiguous_placement(self):
        session = self.build_pose_edit_session()
        editor = session.editor
        editor.base_scene_usd = Path("scene.usda")
        editor.robot_usd = Path("robot.usda")
        editor.overlay_state = default_overlay_state("C:/assets")
        editor.scene_revision = "rev-1"
        editor.overlay_path = Path("overlay.json")
        editor.composed_scene_usd = Path("overlay_scene.usda")
        editor.rebuilding_scene = False

        with mock.patch.object(
            EDITOR_COMMANDS_MODULE,
            "resolve_asset_record",
            return_value={"asset_id": "table_000", "url": "objects/table_000/Aligned.usda", "description": {}},
        ):
            with self.assertRaises(HTTPException) as err:
                session.editor.set_scene_assets([{
                    "asset_id": "table_000",
                    "motion": "static",
                    "placement": {},
                }])

        self.assertEqual(err.exception.status_code, 400)
        self.assertIn("placement.kind", err.exception.detail)

    def test_set_scene_assets_rejects_unsupported_motion(self):
        session = self.build_pose_edit_session()
        editor = session.editor
        editor.base_scene_usd = Path("scene.usda")
        editor.robot_usd = Path("robot.usda")
        editor.overlay_state = default_overlay_state("C:/assets")
        editor.scene_revision = "rev-1"
        editor.overlay_path = Path("overlay.json")
        editor.composed_scene_usd = Path("overlay_scene.usda")
        editor.rebuilding_scene = False

        with self.assertRaises(HTTPException) as err:
            session.editor.set_scene_assets([{
                "asset_id": "table_000",
                "motion": "frozen",
                "placement": {"kind": "floor_at_xy", "xy": [0.0, 0.0], "z_offset": 0.0, "yaw": 0.0},
            }])

        self.assertEqual(err.exception.status_code, 400)
        self.assertIn("Unsupported motion", err.exception.detail)

    def test_set_scene_assets_rejects_empty_assets(self):
        session = self.build_pose_edit_session()
        editor = session.editor
        editor.base_scene_usd = Path("scene.usda")
        editor.robot_usd = Path("robot.usda")
        editor.overlay_state = default_overlay_state("C:/assets")
        editor.scene_revision = "rev-1"
        editor.overlay_path = Path("overlay.json")
        editor.composed_scene_usd = Path("overlay_scene.usda")
        editor.rebuilding_scene = False

        with self.assertRaises(HTTPException) as err:
            session.editor.set_scene_assets([])

        self.assertEqual(err.exception.status_code, 400)
        self.assertIn("assets", err.exception.detail)

    def test_set_scene_assets_rejects_unknown_asset_id(self):
        session = self.build_pose_edit_session()
        editor = session.editor
        editor.base_scene_usd = Path("scene.usda")
        editor.robot_usd = Path("robot.usda")
        editor.overlay_state = default_overlay_state("C:/assets")
        editor.scene_revision = "rev-1"
        editor.overlay_path = Path("overlay.json")
        editor.composed_scene_usd = Path("overlay_scene.usda")
        editor.rebuilding_scene = False

        with mock.patch.object(
            EDITOR_COMMANDS_MODULE,
            "resolve_asset_record",
            side_effect=KeyError("missing_table"),
        ):
            with self.assertRaises(HTTPException) as err:
                session.editor.set_scene_assets([{
                    "asset_id": "missing_table",
                    "motion": "static",
                    "placement": {"kind": "floor_at_xy", "xy": [0.0, 0.0], "z_offset": 0.0, "yaw": 0.0},
                }])

        self.assertEqual(err.exception.status_code, 404)
        self.assertIn("missing_table", err.exception.detail)

    def test_set_scene_assets_rejects_missing_assets_root(self):
        session = self.build_pose_edit_session()
        editor = session.editor
        editor.base_scene_usd = Path("scene.usda")
        editor.robot_usd = Path("robot.usda")
        editor.overlay_state = default_overlay_state("C:/missing-assets")
        editor.scene_revision = "rev-1"
        editor.overlay_path = Path("overlay.json")
        editor.composed_scene_usd = Path("overlay_scene.usda")
        editor.rebuilding_scene = False

        with mock.patch.object(
            EDITOR_COMMANDS_MODULE,
            "resolve_asset_record",
            side_effect=FileNotFoundError("GenieSim assets entry not found: C:/missing-assets/__init__.py"),
        ):
            with self.assertRaises(HTTPException) as err:
                session.editor.set_scene_assets([{
                    "asset_id": "benchmark_table_000",
                    "motion": "static",
                    "placement": {"kind": "floor_at_xy", "xy": [0.0, 0.0], "z_offset": 0.0, "yaw": 0.0},
                }])

        self.assertEqual(err.exception.status_code, 409)
        self.assertIn("assets entry", err.exception.detail)

    def test_set_scene_assets_authoritatively_replaces_instances_and_pose_overrides(self):
        session = self.build_pose_edit_session()
        editor = session.editor
        editor.base_scene_usd = Path("scene.usda")
        editor.robot_usd = Path("robot.usda")
        editor.overlay_state = default_overlay_state("C:/assets")
        editor.overlay_state["instances"] = [{
            "id": "crate_000_01",
            "asset_id": "crate_000",
            "url": "objects/crate_000/Aligned.usda",
            "motion": "dynamic",
            "placement": {"kind": "world_pose", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
        }]
        editor.overlay_state["pose_overrides"] = {
            "Scene_crate_000_01": {"pos": [1.0, 2.0, 3.0], "quat": [1.0, 0.0, 0.0, 0.0]},
        }
        editor.scene_revision = "rev-1"
        editor.overlay_path = Path("overlay.json")
        editor.composed_scene_usd = Path("overlay_scene.usda")
        editor.rebuilding_scene = False
        captured = {}

        with mock.patch.object(
            EDITOR_COMMANDS_MODULE,
            "resolve_asset_record",
            return_value={"asset_id": "table_000", "url": "objects/table_000/Aligned.usda", "description": {}},
        ):
            with mock.patch.object(editor, "_start_overlay_operation", side_effect=lambda next_overlay, payload: captured.update({"state": next_overlay, "payload": payload}) or {"ok": True, "op_id": "op-2"}):
                result = session.editor.set_scene_assets([{
                    "asset_id": "table_000",
                    "motion": "static",
                    "placement": {"kind": "floor_at_xy", "xy": [0.0, 0.0], "z_offset": 0.0, "yaw": 0.0},
                }])

        self.assertEqual(result, {"ok": True, "op_id": "op-2"})
        self.assertEqual(captured["state"]["pose_overrides"], {})
        self.assertEqual(captured["state"]["instances"], [{
            "id": "table_000_01",
            "asset_id": "table_000",
            "url": "objects/table_000/Aligned.usda",
            "motion": "static",
            "placement": {"kind": "floor_at_xy", "xy": [0.0, 0.0], "z_offset": 0.0, "yaw": 0.0},
        }])

    def test_set_body_pose_persists_pose_override_without_changing_scene_revision(self):
        session = self.build_freejoint_pose_edit_session()
        session.editor.overlay_state = default_overlay_state("C:/assets")
        session.editor.overlay_path = Path("overlay.json")
        session.editor.scene_revision = "rev-1"

        with mock.patch.object(EDITOR_SESSION_MODULE, "save_overlay_state") as save_overlay:
            session.call_once("set_body_pose", ("Scene_Crate", [4.0, 5.0, 6.0], [1.0, 0.0, 0.0, 0.0]))

        self.assertEqual(session.editor.scene_revision, "rev-1")
        self.assertEqual(session.editor.overlay_state["pose_overrides"]["Scene_Crate"]["pos"], [4.0, 5.0, 6.0])
        save_overlay.assert_called_once()

    def test_set_scene_assets_rejects_broken_existing_overlay_asset(self):
        session = self.build_pose_edit_session()
        editor = session.editor
        editor.base_scene_usd = Path("scene.usda")
        editor.robot_usd = Path("robot.usda")
        editor.overlay_state = default_overlay_state("C:/assets")
        editor.overlay_state["instances"] = [{
            "id": "Crate",
            "asset_id": "crate_000",
            "url": "objects/crate_000/Aligned.usda",
            "motion": "static",
            "placement": {"kind": "floor_at_xy", "xy": [0.0, 0.0], "z_offset": 0.0, "yaw": 0.0},
        }]
        editor.scene_revision = "rev-1"
        editor.overlay_path = Path("overlay.json")
        editor.composed_scene_usd = Path("overlay_scene.usda")
        editor.rebuilding_scene = False

        with mock.patch.object(
            EDITOR_COMMANDS_MODULE,
            "resolve_asset_record",
            return_value={"asset_id": "benchmark_table_000", "url": "objects/benchmark/table/benchmark_table_000/Aligned.usda", "description": {}},
        ):
            with mock.patch.object(
                EDITOR_SESSION_MODULE,
                "asset_local_bounds",
                side_effect=RuntimeError("Failed to open asset stage: C:/assets/objects/crate_000/Aligned.usda"),
            ):
                with self.assertRaises(HTTPException) as err:
                    session.editor.set_scene_assets([{
                        "asset_id": "benchmark_table_000",
                        "motion": "static",
                        "placement": {"kind": "floor_at_xy", "xy": [0.0, 0.0], "z_offset": 0.0, "yaw": 0.0},
                    }])

        self.assertEqual(err.exception.status_code, 409)
        self.assertIn("Crate", err.exception.detail)

    def test_remove_asset_from_scene_delegates_to_editor(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.editor = mock.Mock(
            remove_asset_from_scene=mock.Mock(return_value={"ok": True, "instance_id": "table_000_01"}),
        )

        result = MODULE.ZapdosSession.call_once(session, "remove_asset_from_scene", ("table_000_01",))

        self.assertEqual(result, {"ok": True, "instance_id": "table_000_01"})
        session.editor.remove_asset_from_scene.assert_called_once_with("table_000_01")

    def test_swap_runtime_bundle_closes_previous_physics_after_successful_swap(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.sess = "sess-1"
        old_physics = SimpleNamespace(
            data=SimpleNamespace(qpos=[1.0, 2.0], ctrl=[3.0]),
            close=mock.Mock(),
        )
        old_renderer = SimpleNamespace(close=mock.Mock())
        session.physics = old_physics
        session.renderer = old_renderer
        session.bundle = SimpleNamespace(cameras=[])
        editor = EDITOR_SESSION_MODULE.ZapdosEditor.__new__(EDITOR_SESSION_MODULE.ZapdosEditor)
        editor.session = session
        session.editor = editor

        new_physics = SimpleNamespace(
            model=object(),
            data=SimpleNamespace(qpos=[0.0, 0.0, 0.0], ctrl=[0.0, 0.0]),
            editable_body_names=set(),
            set_body_pose=mock.Mock(),
            close=mock.Mock(),
        )
        new_renderer = SimpleNamespace(close=mock.Mock())

        with tempfile.TemporaryDirectory() as tmp:
            body_map_path = Path(tmp) / "render_scene_body_map.json"
            body_map_path.write_text("{}", encoding="utf-8")
            bundle = SimpleNamespace(body_map_json=body_map_path, cameras=[])

            with mock.patch.object(SESSION_MODULE, "ZapdosPhysics", return_value=new_physics):
                    with mock.patch.object(SESSION_MODULE, "IsaacRenderer", return_value=new_renderer):
                        with mock.patch.object(mujoco, "mj_forward"):
                            editor._swap_runtime_bundle(bundle, {"pose_overrides": {}})

        old_renderer.close.assert_called_once_with(stop_remote=False)
        old_physics.close.assert_called_once_with()
        self.assertIs(session.physics, new_physics)
        self.assertIs(getattr(session.renderer, "backend", session.renderer), new_renderer)
        self.assertEqual(new_physics.data.qpos[:2], [1.0, 2.0])
        self.assertEqual(new_physics.data.ctrl[:1], [3.0])

    def test_swap_runtime_bundle_reuses_existing_renderer_when_reload_succeeds(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.sess = "sess-1"
        old_physics = SimpleNamespace(
            data=SimpleNamespace(qpos=[1.0, 2.0], ctrl=[3.0]),
            close=mock.Mock(),
        )
        old_renderer = SimpleNamespace(
            reload_scene=mock.Mock(),
            close=mock.Mock(),
        )
        session.physics = old_physics
        session.renderer = old_renderer
        session.bundle = SimpleNamespace(cameras=[])
        editor = EDITOR_SESSION_MODULE.ZapdosEditor.__new__(EDITOR_SESSION_MODULE.ZapdosEditor)
        editor.session = session
        session.editor = editor

        new_physics = SimpleNamespace(
            model=object(),
            data=SimpleNamespace(qpos=[0.0, 0.0], ctrl=[0.0]),
            editable_body_names=set(),
            set_body_pose=mock.Mock(),
            close=mock.Mock(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            body_map_path = Path(tmp) / "render_scene_body_map.json"
            body_map_path.write_text("{}", encoding="utf-8")
            bundle = SimpleNamespace(
                body_map_json=body_map_path,
                cameras=[SimpleNamespace(name="head_camera")],
            )

            with mock.patch.object(SESSION_MODULE, "ZapdosPhysics", return_value=new_physics):
                with mock.patch.object(SESSION_MODULE, "IsaacRenderer") as renderer_cls:
                    with mock.patch.object(mujoco, "mj_forward"):
                        editor._swap_runtime_bundle(bundle, {"pose_overrides": {}})

        renderer_cls.assert_not_called()
        old_renderer.reload_scene.assert_called_once_with(bundle)
        old_renderer.close.assert_not_called()
        old_physics.close.assert_called_once_with()
        self.assertIs(session.renderer, old_renderer)
        self.assertIs(session.physics, new_physics)
        self.assertEqual(session.renderer.camera_index, {"head_camera": 0})
        self.assertEqual(session.renderer.last_frame_index, {"head_camera": -1})

    def test_swap_runtime_bundle_falls_back_to_new_renderer_when_reload_fails(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.sess = "sess-1"
        old_physics = SimpleNamespace(
            data=SimpleNamespace(qpos=[1.0], ctrl=[2.0]),
            close=mock.Mock(),
        )
        old_renderer = SimpleNamespace(
            reload_scene=mock.Mock(side_effect=RuntimeError("reload failed")),
            close=mock.Mock(),
        )
        session.physics = old_physics
        session.renderer = old_renderer
        session.bundle = SimpleNamespace(cameras=[])
        editor = EDITOR_SESSION_MODULE.ZapdosEditor.__new__(EDITOR_SESSION_MODULE.ZapdosEditor)
        editor.session = session
        session.editor = editor

        new_physics = SimpleNamespace(
            model=object(),
            data=SimpleNamespace(qpos=[0.0], ctrl=[0.0]),
            editable_body_names=set(),
            set_body_pose=mock.Mock(),
            close=mock.Mock(),
        )
        new_renderer = SimpleNamespace(close=mock.Mock())

        with tempfile.TemporaryDirectory() as tmp:
            body_map_path = Path(tmp) / "render_scene_body_map.json"
            body_map_path.write_text("{}", encoding="utf-8")
            bundle = SimpleNamespace(
                body_map_json=body_map_path,
                cameras=[SimpleNamespace(name="head_camera")],
            )

            with mock.patch.object(SESSION_MODULE, "ZapdosPhysics", return_value=new_physics):
                with mock.patch.object(SESSION_MODULE, "IsaacRenderer", return_value=new_renderer):
                    with mock.patch.object(mujoco, "mj_forward"):
                        editor._swap_runtime_bundle(bundle, {"pose_overrides": {}})

        old_renderer.reload_scene.assert_called_once_with(bundle)
        old_renderer.close.assert_called_once_with(stop_remote=False)
        old_physics.close.assert_called_once_with()
        self.assertIs(getattr(session.renderer, "backend", session.renderer), new_renderer)
        self.assertIs(session.physics, new_physics)

    def test_swap_runtime_bundle_replays_pose_overrides_for_movable_robot_roots(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.sess = "sess-1"
        old_physics = SimpleNamespace(
            data=SimpleNamespace(qpos=[1.0], ctrl=[2.0]),
            close=mock.Mock(),
        )
        old_renderer = SimpleNamespace(close=mock.Mock())
        session.physics = old_physics
        session.renderer = old_renderer
        session.bundle = SimpleNamespace(cameras=[])
        editor = EDITOR_SESSION_MODULE.ZapdosEditor.__new__(EDITOR_SESSION_MODULE.ZapdosEditor)
        editor.session = session
        session.editor = editor

        new_physics = SimpleNamespace(
            model=object(),
            data=SimpleNamespace(qpos=[0.0], ctrl=[0.0]),
            editable_body_names=set(),
            movable_body_names={"Root_base_link"},
            set_body_pose=mock.Mock(),
            close=mock.Mock(),
        )
        new_renderer = SimpleNamespace(close=mock.Mock())

        with tempfile.TemporaryDirectory() as tmp:
            body_map_path = Path(tmp) / "render_scene_body_map.json"
            body_map_path.write_text("{}", encoding="utf-8")
            bundle = SimpleNamespace(
                body_map_json=body_map_path,
                cameras=[SimpleNamespace(name="head_camera")],
            )
            overlay_state = {
                "pose_overrides": {
                    "Root_base_link": {"pos": [1.0, 2.0, 3.0], "quat": [1.0, 0.0, 0.0, 0.0]},
                    "Arm_link": {"pos": [9.0, 9.0, 9.0], "quat": [1.0, 0.0, 0.0, 0.0]},
                },
            }

            with mock.patch.object(SESSION_MODULE, "ZapdosPhysics", return_value=new_physics):
                with mock.patch.object(SESSION_MODULE, "IsaacRenderer", return_value=new_renderer):
                    with mock.patch.object(mujoco, "mj_forward"):
                        editor._swap_runtime_bundle(bundle, overlay_state)

        new_physics.set_body_pose.assert_called_once_with(
            "Root_base_link",
            [1.0, 2.0, 3.0],
            [1.0, 0.0, 0.0, 0.0],
        )

    def test_swap_runtime_bundle_closes_new_physics_when_renderer_creation_fails(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        old_physics = SimpleNamespace(
            data=SimpleNamespace(qpos=[1.0], ctrl=[2.0]),
            close=mock.Mock(),
        )
        old_renderer = SimpleNamespace(close=mock.Mock())
        session.sess = "sess-1"
        session.physics = old_physics
        session.renderer = old_renderer
        session.bundle = SimpleNamespace(cameras=[])
        editor = EDITOR_SESSION_MODULE.ZapdosEditor.__new__(EDITOR_SESSION_MODULE.ZapdosEditor)
        editor.session = session
        session.editor = editor

        new_physics = SimpleNamespace(
            model=object(),
            data=SimpleNamespace(qpos=[0.0], ctrl=[0.0]),
            editable_body_names=set(),
            set_body_pose=mock.Mock(),
            close=mock.Mock(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            body_map_path = Path(tmp) / "render_scene_body_map.json"
            body_map_path.write_text("{}", encoding="utf-8")
            bundle = SimpleNamespace(body_map_json=body_map_path, cameras=[])

            with mock.patch.object(SESSION_MODULE, "ZapdosPhysics", return_value=new_physics):
                with mock.patch.object(SESSION_MODULE, "IsaacRenderer", side_effect=RuntimeError("renderer failed")):
                    with mock.patch.object(mujoco, "mj_forward"):
                        with self.assertRaises(RuntimeError):
                            editor._swap_runtime_bundle(bundle, {"pose_overrides": {}})

        old_physics.close.assert_not_called()
        old_renderer.close.assert_not_called()
        new_physics.close.assert_called_once_with()
        self.assertIs(session.physics, old_physics)
        self.assertIs(session.renderer, old_renderer)

    def test_swap_runtime_bundle_keeps_old_state_when_reload_and_restart_both_fail(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.sess = "sess-1"
        old_physics = SimpleNamespace(
            data=SimpleNamespace(qpos=[1.0], ctrl=[2.0]),
            close=mock.Mock(),
        )
        old_renderer = SimpleNamespace(
            reload_scene=mock.Mock(side_effect=RuntimeError("reload failed")),
            close=mock.Mock(),
        )
        session.physics = old_physics
        session.renderer = old_renderer
        session.bundle = SimpleNamespace(cameras=[])
        editor = EDITOR_SESSION_MODULE.ZapdosEditor.__new__(EDITOR_SESSION_MODULE.ZapdosEditor)
        editor.session = session
        session.editor = editor

        new_physics = SimpleNamespace(
            model=object(),
            data=SimpleNamespace(qpos=[0.0], ctrl=[0.0]),
            editable_body_names=set(),
            set_body_pose=mock.Mock(),
            close=mock.Mock(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            body_map_path = Path(tmp) / "render_scene_body_map.json"
            body_map_path.write_text("{}", encoding="utf-8")
            bundle = SimpleNamespace(
                body_map_json=body_map_path,
                cameras=[SimpleNamespace(name="head_camera")],
            )

            with mock.patch.object(SESSION_MODULE, "ZapdosPhysics", return_value=new_physics):
                with mock.patch.object(SESSION_MODULE, "IsaacRenderer", side_effect=RuntimeError("renderer failed")):
                    with mock.patch.object(mujoco, "mj_forward"):
                        with self.assertRaises(RuntimeError):
                            editor._swap_runtime_bundle(bundle, {"pose_overrides": {}})

        old_renderer.reload_scene.assert_called_once_with(bundle)
        old_renderer.close.assert_not_called()
        old_physics.close.assert_not_called()
        new_physics.close.assert_called_once_with()
        self.assertIs(session.physics, old_physics)
        self.assertIs(session.renderer, old_renderer)

    def test_require_session_future_rejects_missing_runtime_session(self):
        with self.assertRaises(HTTPException) as err:
            MODULE._require_session_future("sess-1")

        self.assertEqual(err.exception.status_code, 409)
        self.assertEqual(err.exception.detail, "Session not initialized")

    async def test_require_session_future_evicts_inactive_session(self):
        future: asyncio.Future[object] = asyncio.Future()
        future.set_result(SimpleNamespace(is_active=lambda: False))
        MODULE.sessions["sess-1"] = future

        with self.assertRaises(HTTPException) as err:
            MODULE._require_session_future("sess-1")

        self.assertEqual(err.exception.status_code, 409)
        self.assertEqual(err.exception.detail, "Session expired")
        self.assertNotIn("sess-1", MODULE.sessions)

    def test_call_once_dispatches_save_camera_override(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.save_camera_override = mock.Mock(return_value={"ok": True, "saved": 1, "path": "config.json"})

        result = MODULE.ZapdosSession.call_once(session, "save_camera_override", ())

        self.assertEqual(result["saved"], 1)
        session.save_camera_override.assert_called_once_with()

    def test_call_once_dispatches_set_scene_assets(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.editor = mock.Mock(
            set_scene_assets=mock.Mock(return_value={"ok": True, "op_id": "op-1"}),
        )

        result = MODULE.ZapdosSession.call_once(session, "set_scene_assets", ([{"asset_id": "table_000"}],))

        self.assertEqual(result, {"ok": True, "op_id": "op-1"})
        session.editor.set_scene_assets.assert_called_once_with([{"asset_id": "table_000"}])

    def test_call_once_dispatches_manipulation_runtime_methods(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.runtime = mock.Mock(
            list_scene_objects=mock.Mock(return_value={"items": [{"body": "Scene_Crate"}], "scene_revision": "rev-1"}),
            grab_apple=mock.Mock(return_value={"ok": True, "target_body": "Scene_apple_1", "scene_revision": "rev-1"}),
            pick_object=mock.Mock(return_value={"ok": True, "target_body": "Scene_Crate", "scene_revision": "rev-1"}),
        )

        listed = MODULE.ZapdosSession.call_once(session, "list_scene_objects", ())
        grabbed = MODULE.ZapdosSession.call_once(session, "grab_apple", ())
        picked = MODULE.ZapdosSession.call_once(session, "pick_object", ({"target_query": "crate"},))

        self.assertEqual(listed["items"][0]["body"], "Scene_Crate")
        self.assertEqual(grabbed["target_body"], "Scene_apple_1")
        self.assertEqual(picked["target_body"], "Scene_Crate")
        session.runtime.list_scene_objects.assert_called_once_with()
        session.runtime.grab_apple.assert_called_once_with()
        session.runtime.pick_object.assert_called_once_with({"target_query": "crate"})

    def test_manipulation_runtime_executes_grounded_pick_plan(self):
        from utils.zapdos.manipulation.runtime import ManipulationRuntime

        session = SimpleNamespace(
            editor=SimpleNamespace(
                scene_revision="rev-1",
                overlay_state={},
                list_scene_bodies=mock.Mock(return_value={"items": []}),
            ),
            bundle=SimpleNamespace(scene_usd=Path("scene.usda"), robot_usd=Path("robot.usda")),
            physics=mock.Mock(),
        )
        executor = mock.Mock(
            current_pose=mock.Mock(return_value={"position": [0.2, 0.1, 0.4], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}),
            execute=mock.Mock(return_value={"ok": True, "target_body": "Scene_Crate"}),
        )
        catalog_loader = mock.Mock(return_value=[{"body": "Scene_Crate", "label": "crate", "motion": "dynamic"}])
        grounder = mock.Mock(return_value={"target": {"body": "Scene_Crate", "label": "crate", "motion": "dynamic"}, "support": None})
        planner = mock.Mock(return_value={"kind": "pick", "target_body": "Scene_Crate"})

        runtime = ManipulationRuntime(
            session,
            catalog_loader=catalog_loader,
            grounding_fn=grounder,
            planning_fn=planner,
            executor=executor,
        )
        result = runtime.pick_object({"target_query": "crate"})

        self.assertEqual(result["target_body"], "Scene_Crate")
        self.assertEqual(result["scene_revision"], "rev-1")
        session.editor.list_scene_bodies.assert_called_once_with()
        catalog_loader.assert_called_once_with({"items": []}, {})
        grounder.assert_called_once_with(catalog_loader.return_value, target_query="crate", support_query=None)
        planner.assert_called_once_with(
            grounder.return_value["target"],
            support=grounder.return_value["support"],
            scene_objects=catalog_loader.return_value,
            arm="left",
            start_pose={"position": [0.2, 0.1, 0.4], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
        )
        executor.execute.assert_called_once_with({"kind": "pick", "target_body": "Scene_Crate", "arm": "left"})

    def test_manipulation_runtime_executes_arm_only_grab_apple_plan(self):
        from utils.zapdos.manipulation.runtime import ManipulationRuntime

        target = {
            "body": "Scene_apple_1",
            "label": "apple",
            "motion": "dynamic",
            "position": [0.5, 0.0, 0.83],
            "world_aabb": {"min": [0.454, -0.046, 0.784], "max": [0.546, 0.046, 0.876]},
        }
        support = {
            "body": "table_body",
            "label": "benchmark table",
            "motion": "static",
            "position": [0.5, 0.0, 0.75],
            "world_aabb": {"min": [0.1, -0.3, 0.7], "max": [0.9, 0.3, 0.8]},
        }
        session = SimpleNamespace(
            editor=SimpleNamespace(
                scene_revision="rev-1",
                overlay_state={},
                list_scene_bodies=mock.Mock(return_value={"items": []}),
            ),
            bundle=SimpleNamespace(scene_usd=Path("scene.usda"), robot_usd=Path("robot.usda")),
            physics=mock.Mock(),
        )
        executor = mock.Mock(
            current_pose=mock.Mock(return_value={"position": [-0.068666, 0.251999, 0.72023], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}),
            execute=mock.Mock(return_value={"ok": True, "target_body": "Scene_apple_1"}),
        )
        catalog_loader = mock.Mock(return_value=[target, support])
        grounder = mock.Mock(return_value={"target": target, "support": support})

        runtime = ManipulationRuntime(
            session,
            catalog_loader=catalog_loader,
            grounding_fn=grounder,
            executor=executor,
        )
        result = runtime.grab_apple()

        self.assertEqual(result["target_body"], "Scene_apple_1")
        self.assertEqual(result["scene_revision"], "rev-1")
        grounder.assert_called_once_with(catalog_loader.return_value, target_query="apple", support_query="benchmark table")
        executor.execute.assert_called_once()
        plan = executor.execute.call_args.args[0]
        self.assertEqual(plan["arm"], "left")
        self.assertEqual(plan["target_body"], "Scene_apple_1")
        self.assertEqual(plan["attach_tolerance"], 0.11)
        self.assertEqual(plan["grasp_tolerance"], 0.16)
        self.assertEqual([stage["name"] for stage in plan["stages"]], [
            "descend_to_grasp",
            "close_gripper",
            "retreat",
        ])
        self.assertTrue(plan["stages"][0]["position_only"])
        self.assertFalse(plan["stages"][0].get("include_torso", False))
        self.assertEqual(plan["stages"][0]["pose"]["position"], [0.48, 0.06, 0.82])
        self.assertEqual(plan["stages"][2]["pose"]["position"], [0.4, 0.18, 0.92])

    def test_manipulation_runtime_passes_start_pose_and_scene_objects_to_planner(self):
        from utils.zapdos.manipulation.runtime import ManipulationRuntime

        session = SimpleNamespace(
            editor=SimpleNamespace(
                scene_revision="rev-1",
                overlay_state={},
                list_scene_bodies=mock.Mock(return_value={"items": []}),
            ),
            bundle=SimpleNamespace(scene_usd=Path("scene.usda"), robot_usd=Path("robot.usda")),
            physics=mock.Mock(name="physics"),
        )
        catalog_loader = mock.Mock(return_value=[{
            "body": "Scene_Crate",
            "label": "crate",
            "asset_id": None,
            "motion": "dynamic",
            "tags": ["crate"],
            "support_body": None,
            "position": [0.2, 0.0, 0.2],
            "matrix": None,
            "top_z": 0.25,
            "bounds_min": [-0.05, -0.05, -0.05],
            "bounds_max": [0.05, 0.05, 0.05],
            "world_aabb": {"min": [0.15, -0.05, 0.15], "max": [0.25, 0.05, 0.25]},
        }])
        grounder = mock.Mock(return_value={"target": catalog_loader.return_value[0], "support": None})
        planner_call: dict[str, object] = {}

        def planner(target, *, support, scene_objects, arm, start_pose):
            planner_call.update(
                target=target,
                support=support,
                scene_objects=scene_objects,
                arm=arm,
                start_pose=start_pose,
            )
            return {"kind": "pick", "target_body": "Scene_Crate", "stages": []}

        executor = mock.Mock(
            current_pose=mock.Mock(return_value={"position": [0.2, 0.1, 0.4], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}),
            execute=mock.Mock(return_value={"ok": True, "target_body": "Scene_Crate"}),
        )

        runtime = ManipulationRuntime(
            session,
            catalog_loader=catalog_loader,
            grounding_fn=grounder,
            planning_fn=planner,
            executor=executor,
        )
        runtime.pick_object({"target_query": "crate"})

        self.assertEqual(planner_call, {
            "target": grounder.return_value["target"],
            "support": grounder.return_value["support"],
            "scene_objects": catalog_loader.return_value,
            "arm": "left",
            "start_pose": {"position": [0.2, 0.1, 0.4], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
        })

    def test_pick_plan_keeps_stages_optional_for_unstaged_default_planner(self):
        from utils.zapdos.manipulation.types import PickPlan

        self.assertNotIn("stages", PickPlan.__required_keys__)

    def test_manipulation_runtime_rebinds_executor_to_current_session_state(self):
        from utils.zapdos.manipulation.runtime import ManipulationRuntime

        session = SimpleNamespace(
            editor=SimpleNamespace(
                scene_revision="rev-2",
                overlay_state={},
                list_scene_bodies=mock.Mock(return_value={"items": []}),
            ),
            bundle=SimpleNamespace(scene_usd=Path("next-scene.usda"), robot_usd=Path("robot.usda")),
            physics=mock.Mock(name="physics"),
        )
        executor = SimpleNamespace(
            physics="stale-physics",
            bundle="stale-bundle",
            current_pose=mock.Mock(return_value={"position": [0.0, 0.0, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}),
            execute=mock.Mock(return_value={"ok": True, "target_body": "Scene_Crate"}),
        )
        runtime = ManipulationRuntime(
            session,
            catalog_loader=mock.Mock(return_value=[{"body": "Scene_Crate", "label": "crate", "motion": "dynamic"}]),
            grounding_fn=mock.Mock(return_value={"target": {"body": "Scene_Crate", "label": "crate", "motion": "dynamic"}, "support": None}),
            planning_fn=mock.Mock(return_value={"kind": "pick", "target_body": "Scene_Crate"}),
            executor=executor,
        )

        runtime.pick_object({"target_query": "crate"})

        self.assertIs(executor.physics, session.physics)
        self.assertIs(executor.bundle, session.bundle)

    def test_run_overlay_rebuild_background_queues_completion_without_nested_session_call(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.sess = "sess-1"
        session.bundle = SimpleNamespace(cameras=[])
        editor = self.attach_editor(session)
        editor.overlay_completions = queue.Queue()
        session.call = mock.Mock()
        prepared = PreparedOverlayRebuild(
            bundle=SimpleNamespace(),
            next_overlay=default_overlay_state("C:/assets"),
            previous_overlay=default_overlay_state("C:/assets"),
            previous_revision="rev-1",
            next_revision="rev-2",
        )

        with mock.patch.object(editor, "_prepare_overlay_rebuild", return_value=prepared):
            editor._run_overlay_rebuild_background(
                "op-1",
                default_overlay_state("C:/assets"),
                {},
                default_overlay_state("C:/assets"),
                "rev-1",
            )

        completion = editor.overlay_completions.get_nowait()
        self.assertEqual(completion.op_id, "op-1")
        self.assertIs(completion.prepared, prepared)
        self.assertIsNone(completion.error)
        session.call.assert_not_called()

    def test_drain_overlay_completions_resolves_scene_rebuild_job_future(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.sess = "sess-1"
        session.bundle = SimpleNamespace(cameras=[])
        editor = self.attach_editor(session)
        editor.overlay_completions = queue.Queue()
        editor.scene_rebuild_jobs = {
            "op-1": SceneRebuildJob(
                future=ConcurrentFuture(),
                success_payload={"ok": True, "items": []},
                events=queue.Queue(),
            ),
        }
        editor.scene_rebuild_jobs_lock = threading.Lock()
        prepared = PreparedOverlayRebuild(
            bundle=SimpleNamespace(),
            next_overlay=default_overlay_state("C:/assets"),
            previous_overlay=default_overlay_state("C:/assets"),
            previous_revision="rev-1",
            next_revision="rev-2",
        )
        editor.overlay_completions.put(OverlayRebuildCompletion(op_id="op-1", prepared=prepared))

        with mock.patch.object(editor, "_apply_prepared_overlay_rebuild", return_value="rev-2"):
            editor.drain_completions()

        self.assertEqual(
            editor.scene_rebuild_jobs["op-1"].future.result(timeout=1),
            {"ok": True, "items": [], "scene_revision": "rev-2"},
        )

    def test_step_once_drains_editor_completions(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.editor = mock.Mock(drain_completions=mock.Mock())
        session.physics = mock.Mock(
            apply_joint_command=mock.Mock(),
            step=mock.Mock(),
        )
        session.command_msgs = queue.Queue()

        with mock.patch.object(Session, "step_once", return_value={"ok": True}) as base_step:
            result = MODULE.ZapdosSession.step_once(session)

        self.assertEqual(result, {"ok": True})
        session.editor.drain_completions.assert_called_once_with()
        session.physics.apply_joint_command.assert_called_once_with(None)
        session.physics.step.assert_called_once_with()
        base_step.assert_called_once_with()

    async def test_stream_scene_rebuild_job_emits_done_event_and_discards_job(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.sess = "sess-1"
        session.bundle = SimpleNamespace(cameras=[])
        editor = self.attach_editor(session)
        future = ConcurrentFuture()
        future.set_result({"ok": True, "items": [], "scene_revision": "rev-2"})
        editor.scene_rebuild_jobs = {
            "op-1": SceneRebuildJob(
                future=future,
                success_payload={"ok": True, "items": []},
                events=queue.Queue(),
            ),
        }
        editor.scene_rebuild_jobs_lock = threading.Lock()

        events = [chunk async for chunk in MODULE._stream_scene_rebuild_job(session, "op-1")]

        self.assertEqual(events, [
            'event: started\ndata: {"op_id": "op-1"}\n\n',
            'event: done\ndata: {"ok": true, "items": [], "scene_revision": "rev-2"}\n\n',
        ])
        self.assertNotIn("op-1", editor.scene_rebuild_jobs)

    async def test_stream_scene_rebuild_job_emits_started_event_before_completion(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.sess = "sess-1"
        session.bundle = SimpleNamespace(cameras=[])
        editor = self.attach_editor(session)
        future = ConcurrentFuture()
        editor.scene_rebuild_jobs = {
            "op-1": SceneRebuildJob(
                future=future,
                success_payload={"ok": True, "items": []},
                events=queue.Queue(),
            ),
        }
        editor.scene_rebuild_jobs_lock = threading.Lock()
        stream = MODULE._stream_scene_rebuild_job(session, "op-1")

        started = await asyncio.wait_for(anext(stream), timeout=0.1)
        future.set_result({"ok": True, "items": [], "scene_revision": "rev-2"})
        done = await asyncio.wait_for(anext(stream), timeout=1.0)

        self.assertEqual(started, 'event: started\ndata: {"op_id": "op-1"}\n\n')
        self.assertEqual(done, 'event: done\ndata: {"ok": true, "items": [], "scene_revision": "rev-2"}\n\n')
        with self.assertRaises(StopAsyncIteration):
            await anext(stream)
        self.assertNotIn("op-1", editor.scene_rebuild_jobs)

    async def test_stream_scene_rebuild_job_emits_failed_event_when_background_prepare_errors_without_drain(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.sess = "sess-1"
        session.bundle = SimpleNamespace(cameras=[])
        editor = self.attach_editor(session)
        editor.overlay_completions = queue.Queue()
        editor.rebuilding_scene = True
        editor.scene_rebuild_jobs = {
            "op-1": SceneRebuildJob(
                future=ConcurrentFuture(),
                success_payload={"ok": True, "items": []},
                events=queue.Queue(),
            ),
        }
        editor.scene_rebuild_jobs_lock = threading.Lock()
        stream = MODULE._stream_scene_rebuild_job(session, "op-1")

        started = await asyncio.wait_for(anext(stream), timeout=0.1)
        with mock.patch.object(editor, "_prepare_overlay_rebuild", side_effect=RuntimeError("subprocess crashed")):
            editor._run_overlay_rebuild_background(
                "op-1",
                default_overlay_state("C:/assets"),
                {},
                default_overlay_state("C:/assets"),
                "rev-1",
            )
        progress = await asyncio.wait_for(anext(stream), timeout=0.2)
        failed = await asyncio.wait_for(anext(stream), timeout=0.2)

        self.assertEqual(started, 'event: started\ndata: {"op_id": "op-1"}\n\n')
        self.assertEqual(progress, 'event: progress\ndata: {"stage": "prepare_overlay_rebuild.started"}\n\n')
        self.assertIn('"detail": "subprocess crashed"', failed)
        self.assertFalse(editor.rebuilding_scene)
        with self.assertRaises(StopAsyncIteration):
            await anext(stream)
        self.assertNotIn("op-1", editor.scene_rebuild_jobs)

    def test_save_camera_override_delegates_to_renderer(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.renderer = SimpleNamespace(
            save_camera_override=mock.Mock(return_value={"ok": True, "saved": 1, "path": "camera.json"}),
        )

        result = MODULE.ZapdosSession.save_camera_override(session)

        self.assertEqual(result["saved"], 1)
        session.renderer.save_camera_override.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
