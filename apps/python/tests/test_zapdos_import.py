from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import queue
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.parse import urlencode

import mujoco  # type: ignore
from fastapi import Request
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

import utils.zapdos.zapdos_scene_operations as SCENE_OPS_MODULE
import utils.zapdos.zapdos_session as SESSION_MODULE
from utils.zapdos.rl_bundle import ensure_render_bundle

MODULE_PATH = REPO_ROOT / "apps" / "python" / "api" / "zapdos" / "{session}" / "{action}.py"
SPEC = importlib.util.spec_from_file_location("zapdos_session_action_import", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class ZapdosImportTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        MODULE.sessions.clear()

    def test_action_module_reexports_split_runtime_symbols(self):
        self.assertEqual(MODULE.ZapdosSession.__module__, "utils.zapdos.zapdos_session")
        self.assertEqual(MODULE._stream_scene_operation.__module__, "utils.zapdos.zapdos_scene_operations")

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

    def build_physics_session(self, xml: str, body_map: dict[str, str]):
        with tempfile.TemporaryDirectory() as tmp:
            xml_path = Path(tmp) / "scene.xml"
            xml_path.write_text(xml.strip(), encoding="utf-8")
            physics = MODULE.ZapdosPhysics("sess-1", SimpleNamespace(mjcf=xml_path), body_map)
            session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
            session.sess = "sess-1"
            session.physics = physics
            session.bundle = SimpleNamespace(cameras=[])
            session.msgs = queue.Queue(maxsize=64)
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
        physics = MODULE.ZapdosPhysics("sess-1", bundle, {})

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

            with mock.patch.object(MODULE.Session, "__init__", new=fake_session_init):
                with mock.patch.object(SESSION_MODULE, "IsaacRenderer", return_value=SimpleNamespace(wait_ready=mock.AsyncMock(), read=mock.Mock(return_value=None), close=mock.Mock())):
                    with mock.patch.object(
                        MODULE.asyncio,
                        "run_coroutine_threadsafe",
                        side_effect=lambda coro, loop: coro.close(),
                    ):
                        session = MODULE.ZapdosSession("sess-1", bundle)

        self.assertIsInstance(session.physics, MODULE.ZapdosPhysics)
        self.assertEqual(session.physics.body_map, {"Scene_Crate": "Crate"})
        self.assertEqual(session.physics.editable_body_names, {"Scene_Crate"})

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
                with mock.patch.object(MODULE.Session, "__init__", new=fake_session_init):
                    with mock.patch.object(SESSION_MODULE, "IsaacRenderer", return_value=SimpleNamespace(wait_ready=mock.AsyncMock(), read=mock.Mock(return_value=None), close=mock.Mock())):
                        with mock.patch.object(
                            MODULE.asyncio,
                            "run_coroutine_threadsafe",
                            side_effect=lambda coro, loop: coro.close(),
                        ):
                            session = MODULE.ZapdosSession("sess-1", bundle)
            rewritten_overlay = json.loads(overlay_path.read_text(encoding="utf-8"))

        self.assertEqual(session.overlay_state["instances"], [])
        self.assertEqual(session.overlay_state["pose_overrides"], {})
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
        self.assertFalse(bodies["RobotLink"]["editable"])
        self.assertEqual(bodies["Scene_Crate"]["label"], "Crate")
        attached = [mesh for mesh in payload["meshes"] if mesh.get("body") == "Scene_Crate"]
        self.assertTrue(attached)
        self.assertIn("localMatrix", attached[0])
        self.assertNotIn("matrix", attached[0])
        static_mesh = next(mesh for mesh in payload["meshes"] if mesh.get("body") is None)
        self.assertIn("matrix", static_mesh)
        self.assertNotIn("localMatrix", static_mesh)

    def test_get_visual_attaches_descendant_scene_mesh_to_editable_ancestor(self):
        session = self.build_nested_pose_edit_session()

        payload = session.call_once("get_visual", ())

        mesh = next(item for item in payload["meshes"] if item["name"] == "geom-0")
        self.assertEqual(mesh["body"], "Scene_Crate")
        self.assertIn("localMatrix", mesh)

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

    def test_set_body_pose_rejects_robot_and_unknown_bodies(self):
        session = self.build_pose_edit_session()

        with self.assertRaises(MODULE.HTTPException) as robot_error:
            session.call_once("set_body_pose", ("RobotLink", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]))
        self.assertEqual(robot_error.exception.status_code, 403)

        with self.assertRaises(MODULE.HTTPException) as missing_error:
            session.call_once("set_body_pose", ("MissingBody", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]))
        self.assertEqual(missing_error.exception.status_code, 404)

    def test_set_body_pose_rejects_while_scene_rebuild_is_running(self):
        session = self.build_pose_edit_session()
        session.rebuilding_scene = True

        with self.assertRaises(MODULE.HTTPException) as err:
            session.call_once("set_body_pose", ("Scene_Crate", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]))

        self.assertEqual(err.exception.status_code, 409)
        self.assertIn("Scene rebuild already in progress", err.exception.detail)

    def test_set_scene_assets_returns_operation_id(self):
        session = self.build_pose_edit_session()
        session.base_scene_usd = Path("scene.usda")
        session.robot_usd = Path("robot.usda")
        session.overlay_state = MODULE.default_overlay_state("C:/assets")
        session.scene_revision = "rev-1"
        session.overlay_path = Path("overlay.json")
        session.composed_scene_usd = Path("overlay_scene.usda")
        session.rebuilding_scene = False

        with mock.patch.object(
            SCENE_OPS_MODULE,
            "resolve_asset_record",
            return_value={"asset_id": "table_000", "url": "objects/table_000/Aligned.usda", "description": {}},
        ):
            with mock.patch.object(session, "_start_overlay_operation", return_value={"ok": True, "op_id": "op-1"}) as start_op:
                result = MODULE.ZapdosSession.set_scene_assets(
                    session,
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

    def test_set_scene_assets_rejects_ambiguous_placement(self):
        session = self.build_pose_edit_session()
        session.base_scene_usd = Path("scene.usda")
        session.robot_usd = Path("robot.usda")
        session.overlay_state = MODULE.default_overlay_state("C:/assets")
        session.scene_revision = "rev-1"
        session.overlay_path = Path("overlay.json")
        session.composed_scene_usd = Path("overlay_scene.usda")
        session.rebuilding_scene = False

        with mock.patch.object(
            SCENE_OPS_MODULE,
            "resolve_asset_record",
            return_value={"asset_id": "table_000", "url": "objects/table_000/Aligned.usda", "description": {}},
        ):
            with self.assertRaises(MODULE.HTTPException) as err:
                MODULE.ZapdosSession.set_scene_assets(session, [{
                    "asset_id": "table_000",
                    "motion": "static",
                    "placement": {},
                }])

        self.assertEqual(err.exception.status_code, 400)
        self.assertIn("placement.kind", err.exception.detail)

    def test_set_scene_assets_rejects_unsupported_motion(self):
        session = self.build_pose_edit_session()
        session.base_scene_usd = Path("scene.usda")
        session.robot_usd = Path("robot.usda")
        session.overlay_state = MODULE.default_overlay_state("C:/assets")
        session.scene_revision = "rev-1"
        session.overlay_path = Path("overlay.json")
        session.composed_scene_usd = Path("overlay_scene.usda")
        session.rebuilding_scene = False

        with self.assertRaises(MODULE.HTTPException) as err:
            MODULE.ZapdosSession.set_scene_assets(session, [{
                "asset_id": "table_000",
                "motion": "frozen",
                "placement": {"kind": "floor_at_xy", "xy": [0.0, 0.0], "z_offset": 0.0, "yaw": 0.0},
            }])

        self.assertEqual(err.exception.status_code, 400)
        self.assertIn("Unsupported motion", err.exception.detail)

    def test_set_scene_assets_rejects_empty_assets(self):
        session = self.build_pose_edit_session()
        session.base_scene_usd = Path("scene.usda")
        session.robot_usd = Path("robot.usda")
        session.overlay_state = MODULE.default_overlay_state("C:/assets")
        session.scene_revision = "rev-1"
        session.overlay_path = Path("overlay.json")
        session.composed_scene_usd = Path("overlay_scene.usda")
        session.rebuilding_scene = False

        with self.assertRaises(MODULE.HTTPException) as err:
            MODULE.ZapdosSession.set_scene_assets(session, [])

        self.assertEqual(err.exception.status_code, 400)
        self.assertIn("assets", err.exception.detail)

    def test_set_scene_assets_authoritatively_replaces_instances_and_pose_overrides(self):
        session = self.build_pose_edit_session()
        session.base_scene_usd = Path("scene.usda")
        session.robot_usd = Path("robot.usda")
        session.overlay_state = MODULE.default_overlay_state("C:/assets")
        session.overlay_state["instances"] = [{
            "id": "crate_000_01",
            "asset_id": "crate_000",
            "url": "objects/crate_000/Aligned.usda",
            "motion": "dynamic",
            "placement": {"kind": "world_pose", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
        }]
        session.overlay_state["pose_overrides"] = {
            "Scene_crate_000_01": {"pos": [1.0, 2.0, 3.0], "quat": [1.0, 0.0, 0.0, 0.0]},
        }
        session.scene_revision = "rev-1"
        session.overlay_path = Path("overlay.json")
        session.composed_scene_usd = Path("overlay_scene.usda")
        session.rebuilding_scene = False
        captured = {}

        with mock.patch.object(
            SCENE_OPS_MODULE,
            "resolve_asset_record",
            return_value={"asset_id": "table_000", "url": "objects/table_000/Aligned.usda", "description": {}},
        ):
            with mock.patch.object(session, "_start_overlay_operation", side_effect=lambda next_overlay, payload: captured.update({"state": next_overlay, "payload": payload}) or {"ok": True, "op_id": "op-2"}):
                result = MODULE.ZapdosSession.set_scene_assets(session, [{
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
        session.overlay_state = MODULE.default_overlay_state("C:/assets")
        session.overlay_path = Path("overlay.json")
        session.scene_revision = "rev-1"

        with mock.patch.object(SESSION_MODULE, "save_overlay_state") as save_overlay:
            session.call_once("set_body_pose", ("Scene_Crate", [4.0, 5.0, 6.0], [1.0, 0.0, 0.0, 0.0]))

        self.assertEqual(session.scene_revision, "rev-1")
        self.assertEqual(session.overlay_state["pose_overrides"]["Scene_Crate"]["pos"], [4.0, 5.0, 6.0])
        save_overlay.assert_called_once()

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
        session.camera_index = {}
        session.last_frame_index = {}

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
                        with mock.patch.object(MODULE.mujoco, "mj_forward"):
                            session._swap_runtime_bundle(bundle, {"pose_overrides": {}})

        old_renderer.close.assert_called_once_with(stop_remote=False)
        old_physics.close.assert_called_once_with()
        self.assertIs(session.physics, new_physics)
        self.assertIs(session.renderer, new_renderer)
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
        session.camera_index = {}
        session.last_frame_index = {}

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
                    with mock.patch.object(MODULE.mujoco, "mj_forward"):
                        session._swap_runtime_bundle(bundle, {"pose_overrides": {}})

        renderer_cls.assert_not_called()
        old_renderer.reload_scene.assert_called_once_with(bundle)
        old_renderer.close.assert_not_called()
        old_physics.close.assert_called_once_with()
        self.assertIs(session.renderer, old_renderer)
        self.assertIs(session.physics, new_physics)
        self.assertEqual(session.camera_index, {"head_camera": 0})
        self.assertEqual(session.last_frame_index, {"head_camera": -1})

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
        session.camera_index = {}
        session.last_frame_index = {}

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
                    with mock.patch.object(MODULE.mujoco, "mj_forward"):
                        session._swap_runtime_bundle(bundle, {"pose_overrides": {}})

        old_renderer.reload_scene.assert_called_once_with(bundle)
        old_renderer.close.assert_called_once_with(stop_remote=False)
        old_physics.close.assert_called_once_with()
        self.assertIs(session.renderer, new_renderer)
        self.assertIs(session.physics, new_physics)

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
        session.camera_index = {}
        session.last_frame_index = {}

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
                    with mock.patch.object(MODULE.mujoco, "mj_forward"):
                        with self.assertRaises(RuntimeError):
                            session._swap_runtime_bundle(bundle, {"pose_overrides": {}})

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
        session.camera_index = {}
        session.last_frame_index = {}

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
                    with mock.patch.object(MODULE.mujoco, "mj_forward"):
                        with self.assertRaises(RuntimeError):
                            session._swap_runtime_bundle(bundle, {"pose_overrides": {}})

        old_renderer.reload_scene.assert_called_once_with(bundle)
        old_renderer.close.assert_not_called()
        old_physics.close.assert_not_called()
        new_physics.close.assert_called_once_with()
        self.assertIs(session.physics, old_physics)
        self.assertIs(session.renderer, old_renderer)

    def test_require_session_future_rejects_missing_runtime_session(self):
        with self.assertRaises(MODULE.HTTPException) as err:
            MODULE._require_session_future("sess-1")

        self.assertEqual(err.exception.status_code, 409)
        self.assertEqual(err.exception.detail, "Session not initialized")

    async def test_require_session_future_evicts_inactive_session(self):
        future: asyncio.Future[object] = asyncio.Future()
        future.set_result(SimpleNamespace(is_active=lambda: False))
        MODULE.sessions["sess-1"] = future

        with self.assertRaises(MODULE.HTTPException) as err:
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
        session.set_scene_assets = mock.Mock(return_value={"ok": True, "op_id": "op-1"})

        result = MODULE.ZapdosSession.call_once(session, "set_scene_assets", ([{"asset_id": "table_000"}],))

        self.assertEqual(result, {"ok": True, "op_id": "op-1"})
        session.set_scene_assets.assert_called_once_with([{"asset_id": "table_000"}])

    def test_run_overlay_rebuild_background_queues_completion_without_nested_session_call(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.overlay_completions = queue.Queue()
        session.call = mock.Mock()
        prepared = MODULE.PreparedOverlayRebuild(
            bundle=SimpleNamespace(),
            next_overlay=MODULE.default_overlay_state("C:/assets"),
            previous_overlay=MODULE.default_overlay_state("C:/assets"),
            previous_revision="rev-1",
            next_revision="rev-2",
        )

        with mock.patch.object(session, "_prepare_overlay_rebuild", return_value=prepared):
            session._run_overlay_rebuild_background(
                "op-1",
                MODULE.default_overlay_state("C:/assets"),
                {},
                MODULE.default_overlay_state("C:/assets"),
                "rev-1",
            )

        completion = session.overlay_completions.get_nowait()
        self.assertEqual(completion.op_id, "op-1")
        self.assertIs(completion.prepared, prepared)
        self.assertIsNone(completion.error)
        session.call.assert_not_called()

    def test_drain_overlay_completions_resolves_scene_operation_future(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.overlay_completions = queue.Queue()
        session.scene_operations = {
            "op-1": MODULE.SceneOperation(
                future=MODULE.ConcurrentFuture(),
                success_payload={"ok": True, "items": []},
                events=queue.Queue(),
            ),
        }
        session.scene_operations_lock = threading.Lock()
        prepared = MODULE.PreparedOverlayRebuild(
            bundle=SimpleNamespace(),
            next_overlay=MODULE.default_overlay_state("C:/assets"),
            previous_overlay=MODULE.default_overlay_state("C:/assets"),
            previous_revision="rev-1",
            next_revision="rev-2",
        )
        session.overlay_completions.put(MODULE.OverlayRebuildCompletion(op_id="op-1", prepared=prepared))

        with mock.patch.object(session, "_apply_prepared_overlay_rebuild", return_value="rev-2"):
            session._drain_overlay_completions()

        self.assertEqual(
            session.scene_operations["op-1"].future.result(timeout=1),
            {"ok": True, "items": [], "scene_revision": "rev-2"},
        )

    async def test_stream_scene_operation_emits_done_event_and_discards_operation(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        future = MODULE.ConcurrentFuture()
        future.set_result({"ok": True, "items": [], "scene_revision": "rev-2"})
        session.scene_operations = {
            "op-1": MODULE.SceneOperation(
                future=future,
                success_payload={"ok": True, "items": []},
                events=queue.Queue(),
            ),
        }
        session.scene_operations_lock = threading.Lock()

        events = [chunk async for chunk in MODULE._stream_scene_operation(session, "op-1")]

        self.assertEqual(events, [
            'event: started\ndata: {"op_id": "op-1"}\n\n',
            'event: done\ndata: {"ok": true, "items": [], "scene_revision": "rev-2"}\n\n',
        ])
        self.assertNotIn("op-1", session.scene_operations)

    async def test_stream_scene_operation_emits_started_event_before_completion(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        future = MODULE.ConcurrentFuture()
        session.scene_operations = {
            "op-1": MODULE.SceneOperation(
                future=future,
                success_payload={"ok": True, "items": []},
                events=queue.Queue(),
            ),
        }
        session.scene_operations_lock = threading.Lock()
        stream = MODULE._stream_scene_operation(session, "op-1")

        started = await asyncio.wait_for(anext(stream), timeout=0.1)
        future.set_result({"ok": True, "items": [], "scene_revision": "rev-2"})
        done = await asyncio.wait_for(anext(stream), timeout=1.0)

        self.assertEqual(started, 'event: started\ndata: {"op_id": "op-1"}\n\n')
        self.assertEqual(done, 'event: done\ndata: {"ok": true, "items": [], "scene_revision": "rev-2"}\n\n')
        with self.assertRaises(StopAsyncIteration):
            await anext(stream)
        self.assertNotIn("op-1", session.scene_operations)

    async def test_stream_scene_operation_emits_failed_event_when_background_prepare_errors_without_drain(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.overlay_completions = queue.Queue()
        session.rebuilding_scene = True
        session.scene_operations = {
            "op-1": MODULE.SceneOperation(
                future=MODULE.ConcurrentFuture(),
                success_payload={"ok": True, "items": []},
                events=queue.Queue(),
            ),
        }
        session.scene_operations_lock = threading.Lock()
        stream = MODULE._stream_scene_operation(session, "op-1")

        started = await asyncio.wait_for(anext(stream), timeout=0.1)
        with mock.patch.object(session, "_prepare_overlay_rebuild", side_effect=RuntimeError("subprocess crashed")):
            session._run_overlay_rebuild_background(
                "op-1",
                MODULE.default_overlay_state("C:/assets"),
                {},
                MODULE.default_overlay_state("C:/assets"),
                "rev-1",
            )
        failed = await asyncio.wait_for(anext(stream), timeout=0.2)

        self.assertEqual(started, 'event: started\ndata: {"op_id": "op-1"}\n\n')
        self.assertEqual(failed, 'event: failed\ndata: {"detail": "subprocess crashed"}\n\n')
        self.assertFalse(session.rebuilding_scene)
        with self.assertRaises(StopAsyncIteration):
            await anext(stream)
        self.assertNotIn("op-1", session.scene_operations)

    def test_save_camera_override_persists_renderer_snapshot(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.renderer = SimpleNamespace(snapshot_cameras=mock.Mock(return_value=[{
            "name": "head_camera",
            "parent_prim": "/MyRobot/zed_link",
            "pos": [0.1, 0.2, 0.3],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "fovy": 60.0,
            "horizontal_aperture": 30.0,
            "vertical_aperture": 20.0,
            "clipping_range": [0.2, 80.0],
        }]))

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"USERPROFILE": tmp}, clear=False):
                result = MODULE.ZapdosSession.save_camera_override(session)
                payload = json.loads((Path(tmp) / ".glass-beaker" / "config.json").read_text(encoding="utf-8"))

        self.assertEqual(result["saved"], 1)
        self.assertEqual(payload["override"]["camera"]["/MyRobot/zed_link"]["head_camera"]["fovy"], 60.0)


if __name__ == "__main__":
    unittest.main()

