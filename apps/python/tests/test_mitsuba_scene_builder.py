from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pxr import Gf, Usd, UsdGeom

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.zapdos.bundle.camera_specs import RenderCamera
from utils.zapdos.renderer.mitsuba_scene import build_mitsuba_scene_dict


def _camera() -> RenderCamera:
    return RenderCamera(
        name="main",
        prim="/SceneRender/main",
        topic="/env_0/main/image_raw",
        frame_id="main",
        body=None,
        pos=[0.0, -3.0, 1.0],
        quat=[1.0, 0.0, 0.0, 0.0],
        fovy=45.0,
    )


class MitsubaSceneBuilderTest(unittest.TestCase):
    def test_builds_scene_dict_and_mesh_files_from_usd_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            usd_path = root / "scene.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            mesh = UsdGeom.Mesh.Define(stage, "/World/Triangle")
            mesh.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)])
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
            stage.GetRootLayer().Save()
            bundle = SimpleNamespace(render_scene_usda=usd_path, cameras=[_camera()])
            scene, snapshots = build_mitsuba_scene_dict(bundle, root / "meshes", 64, 48, spp=2)

            mesh_entries = [value for value in scene.values() if isinstance(value, dict) and value.get("type") == "ply"]

            self.assertEqual(scene["type"], "scene")
            self.assertEqual(scene["integrator"]["type"], "direct")
            self.assertIn("sensor_main", scene)
            self.assertEqual(scene["sensor_main"]["film"]["width"], 64)
            self.assertEqual(scene["sensor_main"]["sampler"]["sample_count"], 2)
            self.assertTrue(mesh_entries)
            self.assertTrue(Path(mesh_entries[0]["filename"]).exists())
            self.assertEqual(snapshots[0]["name"], "main")

    def test_empty_usd_scene_uses_fallback_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            usd_path = root / "empty.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.Xform.Define(stage, "/World")
            stage.GetRootLayer().Save()
            bundle = SimpleNamespace(render_scene_usda=usd_path, cameras=[_camera()])

            scene, _ = build_mitsuba_scene_dict(bundle, root / "meshes", 32, 24, spp=1)

            self.assertIn("fallback_sphere", scene)
            self.assertEqual(scene["fallback_sphere"]["type"], "sphere")

    def test_camera_sensor_uses_composed_usd_world_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            usd_path = root / "camera.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            parent = UsdGeom.Xform.Define(stage, "/RenderScene/MyRobot/Link")
            UsdGeom.Xformable(parent.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
            UsdGeom.Camera.Define(stage, "/RenderScene/MyRobot/Link/main")
            stage.GetRootLayer().Save()
            camera = RenderCamera(
                **{
                    **_camera().to_json(),
                    "prim": "/MyRobot/Link/main",
                    "pos": [0.0, 0.0, 0.0],
                    "quat": [1.0, 0.0, 0.0, 0.0],
                }
            )
            bundle = SimpleNamespace(render_scene_usda=usd_path, cameras=[camera])

            scene, _ = build_mitsuba_scene_dict(bundle, root / "meshes", 32, 24, spp=1)
            look_at = scene["sensor_main"]["to_world_look_at"]

            self.assertEqual(look_at["origin"], [1.0, 2.0, 3.0])
            self.assertEqual(look_at["target"], [1.0, 2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
