from __future__ import annotations

import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

from PIL import Image
from pxr import Sdf, Usd, UsdGeom, UsdShade

from utils.usd_to_mjcf import USDToMJCFConverter


class USDToMJCFCachingTest(unittest.TestCase):
    def define_triangle_mesh(self, stage, path: str):
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr([(-1, 0, 0), (1, 0, 0), (0, 1, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.vertex,
        )
        primvar.Set([(0, 0), (1, 0), (0, 1)])
        return mesh

    def bind_texture_material(self, stage, mesh, material_path: str, texture_asset_path: str):
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(texture_asset_path)
        )
        material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        UsdShade.MaterialBindingAPI(mesh).Bind(material)

    def test_duplicate_mesh_prims_share_one_exported_obj(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scene_path = root / "duplicate_mesh.usda"
            output_xml = root / "duplicate_mesh.xml"
            stage = Usd.Stage.CreateNew(str(scene_path))
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            self.define_triangle_mesh(stage, "/World/Visual")
            self.define_triangle_mesh(stage, "/World/Collision")
            stage.GetRootLayer().Save()

            USDToMJCFConverter(scene_path, output_xml, model_name="duplicate_mesh").convert()

            mesh_assets = ET.parse(output_xml).getroot().findall("./asset/mesh")
            self.assertEqual(len(mesh_assets), 1)
            self.assertEqual(len(list((root / "meshes").glob("*.obj"))), 1)

    def test_warm_cache_skips_mesh_and_texture_rebuilds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "cache"
            scene_path = root / "scene.usda"
            first_output = root / "first" / "scene.xml"
            second_output = root / "second" / "scene.xml"
            texture_path = root / "diffuse.jpg"
            Image.new("RGB", (2, 2), (255, 0, 0)).save(texture_path, format="JPEG")

            stage = Usd.Stage.CreateNew(str(scene_path))
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            mesh = self.define_triangle_mesh(stage, "/World/Tri")
            self.bind_texture_material(stage, mesh, "/World/Looks/TestMat", "./diffuse.jpg")
            stage.GetRootLayer().Save()

            with mock.patch.dict(os.environ, {"USD_TO_MJCF_CACHE_DIR": str(cache_dir)}):
                USDToMJCFConverter(scene_path, first_output, model_name="first").convert()

                converter = USDToMJCFConverter(scene_path, second_output, model_name="second")
                with mock.patch.object(
                    converter,
                    "write_obj_mesh",
                    wraps=converter.write_obj_mesh,
                ) as write_obj_mesh:
                    with mock.patch("utils.usd_to_mjcf.Image.open", wraps=Image.open) as image_open:
                        converter.convert()

                self.assertEqual(write_obj_mesh.call_count, 0)
                self.assertEqual(image_open.call_count, 0)


if __name__ == "__main__":
    unittest.main()
