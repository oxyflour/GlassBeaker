from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco  # type: ignore
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from utils.zapdos.usd_to_mjcf import USDToMJCFConverter


class USDToMJCFMaterialsTest(unittest.TestCase):
    def bind_mdl_texture_material(
        self,
        stage,
        mesh,
        material_path: str,
        texture_asset_path: str,
    ):
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(texture_asset_path)
        )
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.5, 0.5, 0.5)
        )
        material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        UsdShade.MaterialBindingAPI(mesh).Bind(material)

    def test_mdl_diffuse_texture_is_exported_as_mjcf_material(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scene_path = root / "scene.usda"
            output_xml = root / "scene.xml"
            texture_path = root / "diffuse.png"

            Image.new("RGB", (1, 1), (255, 0, 0)).save(texture_path)

            stage = Usd.Stage.CreateNew(str(scene_path))
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            mesh = UsdGeom.Mesh.Define(stage, "/World/Tri")
            mesh.CreatePointsAttr([(-1, 0, 0), (1, 0, 0), (0, 1, 0)])
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
            primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                "st",
                Sdf.ValueTypeNames.TexCoord2fArray,
                UsdGeom.Tokens.vertex,
            )
            primvar.Set([(0, 0), (1, 0), (0, 1)])

            self.bind_mdl_texture_material(
                stage,
                mesh,
                "/World/Looks/TestMat",
                "./diffuse.png",
            )
            stage.GetRootLayer().Save()

            USDToMJCFConverter(scene_path, output_xml, model_name="mdl_texture").convert()

            root_xml = ET.parse(output_xml).getroot()
            geom = root_xml.find(".//geom")
            texture = root_xml.find("./asset/texture")
            material_xml = root_xml.find("./asset/material")

            self.assertIsNotNone(geom)
            self.assertEqual(geom.attrib.get("material"), "World_Looks_TestMat")
            self.assertNotIn("rgba", geom.attrib)
            self.assertIsNotNone(texture)
            self.assertEqual(texture.attrib.get("file"), "textures/diffuse.png")
            self.assertIsNotNone(material_xml)
            self.assertEqual(material_xml.attrib.get("texture"), "diffuse")
            self.assertTrue((root / "textures" / "diffuse.png").exists())

    def test_external_textures_with_same_stem_get_unique_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scene_root = root / "scene"
            scene_root.mkdir()
            asset_a = root / "assets" / "a"
            asset_b = root / "assets" / "b"
            asset_a.mkdir(parents=True)
            asset_b.mkdir(parents=True)
            Image.new("RGB", (1, 1), (255, 0, 0)).save(asset_a / "diffuse.png")
            Image.new("RGB", (1, 1), (0, 255, 0)).save(asset_b / "diffuse.png")

            scene_path = scene_root / "scene.usda"
            output_xml = scene_root / "scene.xml"
            stage = Usd.Stage.CreateNew(str(scene_path))
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            meshes = [
                UsdGeom.Mesh.Define(stage, "/World/TriA"),
                UsdGeom.Mesh.Define(stage, "/World/TriB"),
            ]
            points = [
                [(-1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
                [(2, 0, 0), (4, 0, 0), (3, 1, 0), (3, 0, 1)],
            ]
            texture_paths = ["../assets/a/diffuse.png", "../assets/b/diffuse.png"]
            for mesh, mesh_points, material_path, texture_path in zip(
                meshes,
                points,
                ["/World/Looks/MatA", "/World/Looks/MatB"],
                texture_paths,
            ):
                mesh.CreatePointsAttr(mesh_points)
                mesh.CreateFaceVertexCountsAttr([3, 3, 3, 3])
                mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 1, 3, 1, 2, 3, 0, 2, 3])
                primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                    "st",
                    Sdf.ValueTypeNames.TexCoord2fArray,
                    UsdGeom.Tokens.vertex,
                )
                primvar.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
                self.bind_mdl_texture_material(stage, mesh, material_path, texture_path)
            stage.GetRootLayer().Save()

            USDToMJCFConverter(scene_path, output_xml, model_name="duplicate_textures").convert()

            root_xml = ET.parse(output_xml).getroot()
            texture_names = [node.attrib["name"] for node in root_xml.findall("./asset/texture")]
            self.assertEqual(len(texture_names), 2)
            self.assertEqual(len(set(texture_names)), 2)
            self.assertEqual(len(list((scene_root / "textures").glob("*.png"))), 2)
            mujoco.MjModel.from_xml_path(str(output_xml))  # type: ignore


if __name__ == "__main__":
    unittest.main()

