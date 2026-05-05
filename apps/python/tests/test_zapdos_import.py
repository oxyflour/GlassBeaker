from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import queue
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.parse import urlencode

import mujoco  # type: ignore
from fastapi import Request
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.rl_bundle import ensure_render_bundle

MODULE_PATH = REPO_ROOT / "apps" / "python" / "api" / "zapdos" / "{session}" / "{action}.py"
SPEC = importlib.util.spec_from_file_location("zapdos_session_action_import", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class ZapdosImportTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        MODULE.sessions.clear()

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
                with mock.patch.object(MODULE, "IsaacRenderer", return_value=SimpleNamespace(wait_ready=mock.AsyncMock(), read=mock.Mock(return_value=None), close=mock.Mock())):
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

            with mock.patch.object(MODULE, "REPO_ROOT", root):
                with mock.patch.object(MODULE.Session, "__init__", new=fake_session_init):
                    with mock.patch.object(MODULE, "IsaacRenderer", return_value=SimpleNamespace(wait_ready=mock.AsyncMock(), read=mock.Mock(return_value=None), close=mock.Mock())):
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

    def test_add_asset_to_scene_returns_body_and_revision(self):
        session = self.build_pose_edit_session()
        session.base_scene_usd = Path("scene.usda")
        session.robot_usd = Path("robot.usda")
        session.overlay_state = MODULE.default_overlay_state("C:/assets")
        session.scene_revision = "rev-1"
        session.overlay_path = Path("overlay.json")
        session.composed_scene_usd = Path("overlay_scene.usda")
        session.rebuilding_scene = False

        with mock.patch.object(
            MODULE,
            "resolve_asset_record",
            return_value={"asset_id": "table_000", "url": "objects/table_000/Aligned.usda", "description": {}},
        ):
            with mock.patch.object(session, "_rebuild_overlay_runtime", return_value="rev-2"):
                result = MODULE.ZapdosSession.add_asset_to_scene(
                    session,
                    "table_000",
                    "static",
                    {"kind": "floor_at_xy", "xy": [0.0, 0.0], "z_offset": 0.0, "yaw": 0.0},
                )

        self.assertEqual(result["scene_revision"], "rev-2")
        self.assertEqual(result["body"], "Scene_table_000_01")

    def test_add_asset_to_scene_rejects_ambiguous_placement(self):
        session = self.build_pose_edit_session()
        session.base_scene_usd = Path("scene.usda")
        session.robot_usd = Path("robot.usda")
        session.overlay_state = MODULE.default_overlay_state("C:/assets")
        session.scene_revision = "rev-1"
        session.overlay_path = Path("overlay.json")
        session.composed_scene_usd = Path("overlay_scene.usda")
        session.rebuilding_scene = False

        with mock.patch.object(
            MODULE,
            "resolve_asset_record",
            return_value={"asset_id": "table_000", "url": "objects/table_000/Aligned.usda", "description": {}},
        ):
            with self.assertRaises(MODULE.HTTPException) as err:
                MODULE.ZapdosSession.add_asset_to_scene(session, "table_000", "static", {})

        self.assertEqual(err.exception.status_code, 400)
        self.assertIn("placement.kind", err.exception.detail)

    def test_set_body_pose_persists_pose_override_without_changing_scene_revision(self):
        session = self.build_freejoint_pose_edit_session()
        session.overlay_state = MODULE.default_overlay_state("C:/assets")
        session.overlay_path = Path("overlay.json")
        session.scene_revision = "rev-1"

        with mock.patch.object(MODULE, "save_overlay_state") as save_overlay:
            session.call_once("set_body_pose", ("Scene_Crate", [4.0, 5.0, 6.0], [1.0, 0.0, 0.0, 0.0]))

        self.assertEqual(session.scene_revision, "rev-1")
        self.assertEqual(session.overlay_state["pose_overrides"]["Scene_Crate"]["pos"], [4.0, 5.0, 6.0])
        save_overlay.assert_called_once()

    def test_failed_rebuild_restores_previous_overlay_state(self):
        session = self.build_pose_edit_session()
        session.base_scene_usd = Path("scene.usda")
        session.robot_usd = Path("robot.usda")
        session.overlay_state = MODULE.default_overlay_state("C:/assets")
        session.overlay_path = Path("overlay.json")
        session.scene_revision = "rev-1"
        session.rebuilding_scene = False

        with mock.patch.object(MODULE, "save_overlay_state"):
            with mock.patch.object(session, "_build_support_infos", return_value={}):
                with mock.patch.object(MODULE, "write_overlay_scene", side_effect=RuntimeError("bundle exploded")):
                    with self.assertRaises(RuntimeError):
                        session._rebuild_overlay_runtime(
                            lambda state: state["instances"].append({
                                "id": "table_000_01",
                                "asset_id": "table_000",
                                "url": "objects/table_000/Aligned.usda",
                                "motion": "static",
                                "placement": {
                                    "kind": "floor_at_xy",
                                    "xy": [0.0, 0.0],
                                    "z_offset": 0.0,
                                    "yaw": 0.0,
                                },
                            })
                        )

        self.assertEqual(session.overlay_state["instances"], [])
        self.assertEqual(session.scene_revision, "rev-1")

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
