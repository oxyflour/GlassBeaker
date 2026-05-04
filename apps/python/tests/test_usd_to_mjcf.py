from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco  # type: ignore
from pxr import Usd, UsdGeom, UsdPhysics

from utils.usd_to_mjcf import USDToMJCFConverter

REPO_ROOT = Path(__file__).resolve().parents[3]
ROBOT_USD = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"


class USDToMJCFTest(unittest.TestCase):
    def test_mass_only_body_defaults_inertial_position(self):
        scene_usda = """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Xform "Body"
    {
        float physics:mass = 2
        def Cube "Visual"
        {
            double size = 1
        }
    }
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_path = Path(tmpdir) / "mass_only.usda"
            output_xml = Path(tmpdir) / "mass_only.xml"
            scene_path.write_text(scene_usda, encoding="utf-8")

            USDToMJCFConverter(scene_path, output_xml, model_name="mass_only").convert()

            model = mujoco.MjModel.from_xml_path(str(output_xml))  # type: ignore
            self.assertGreater(model.nbody, 0)

    def test_r1pro_excludes_base_contacts_for_wheels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_xml = Path(tmpdir) / "r1pro.xml"
            converter = USDToMJCFConverter(ROBOT_USD, output_xml, model_name="r1pro_test")

            converter.convert()

            root = ET.parse(output_xml).getroot()
            excludes = {
                tuple(sorted((node.attrib["body1"], node.attrib["body2"])))
                for node in root.findall("./contact/exclude")
            }

            expected = {
                tuple(sorted((
                    "Root_r1_pro_with_gripper_base_link",
                    f"Root_r1_pro_with_gripper_wheel_motor_link{index}",
                )))
                for index in range(1, 4)
            }
            self.assertTrue(expected.issubset(excludes))

    def test_r1pro_excludes_gripper_finger_self_contacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_xml = Path(tmpdir) / "r1pro.xml"
            converter = USDToMJCFConverter(ROBOT_USD, output_xml, model_name="r1pro_test")

            converter.convert()

            root = ET.parse(output_xml).getroot()
            excludes = {
                tuple(sorted((node.attrib["body1"], node.attrib["body2"])))
                for node in root.findall("./contact/exclude")
            }

            expected = {
                tuple(sorted((
                    "Root_r1_pro_with_gripper_left_gripper_finger_link1",
                    "Root_r1_pro_with_gripper_left_gripper_finger_link2",
                ))),
                tuple(sorted((
                    "Root_r1_pro_with_gripper_right_gripper_finger_link1",
                    "Root_r1_pro_with_gripper_right_gripper_finger_link2",
                ))),
            }
            self.assertTrue(expected.issubset(excludes))

    def test_duplicate_joint_leaf_names_become_unique_mjcf_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_path = Path(tmpdir) / "duplicate_joint_names.usda"
            output_xml = Path(tmpdir) / "duplicate_joint_names.xml"
            stage = Usd.Stage.CreateNew(str(scene_path))
            stage.SetMetadata("metersPerUnit", 1.0)
            UsdGeom.SetStageUpAxis(stage, "Z")
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            base = UsdGeom.Xform.Define(stage, "/World/Base")
            slider_a = UsdGeom.Xform.Define(stage, "/World/SliderA")
            slider_b = UsdGeom.Xform.Define(stage, "/World/SliderB")
            for prim in (base.GetPrim(), slider_a.GetPrim(), slider_b.GetPrim()):
                UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(1.0)
            for slider in (slider_a, slider_b):
                joint = UsdPhysics.PrismaticJoint.Define(stage, f"{slider.GetPath()}/PrismaticJoint")
                joint.CreateBody0Rel().SetTargets([base.GetPath()])
                joint.CreateBody1Rel().SetTargets([slider.GetPath()])
                joint.CreateAxisAttr("Y")
                joint.CreateLowerLimitAttr(0.0)
                joint.CreateUpperLimitAttr(0.25)
            stage.GetRootLayer().Save()

            USDToMJCFConverter(scene_path, output_xml, model_name="duplicate_joint_names").convert()

            joint_names = [node.attrib["name"] for node in ET.parse(output_xml).getroot().findall(".//joint")]
            self.assertEqual(len(joint_names), 2)
            self.assertEqual(len(set(joint_names)), 2)
            self.assertTrue(all(name.startswith("PrismaticJoint") for name in joint_names))
            mujoco.MjModel.from_xml_path(str(output_xml))  # type: ignore

    def test_invisible_cube_is_not_exported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_path = Path(tmpdir) / "invisible_cube.usda"
            output_xml = Path(tmpdir) / "invisible_cube.xml"
            stage = Usd.Stage.CreateNew(str(scene_path))
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            cube = UsdGeom.Cube.Define(stage, "/World/HiddenCube")
            cube.CreateSizeAttr(1.0)
            UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
            stage.GetRootLayer().Save()

            USDToMJCFConverter(scene_path, output_xml, model_name="invisible_cube").convert()

            root = ET.parse(output_xml).getroot()
            self.assertEqual(root.findall(".//geom"), [])

    def test_cube_scale_changes_emitted_box_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_path = Path(tmpdir) / "scaled_cube.usda"
            output_xml = Path(tmpdir) / "scaled_cube.xml"
            stage = Usd.Stage.CreateNew(str(scene_path))
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            cube = UsdGeom.Cube.Define(stage, "/World/ScaledCube")
            cube.CreateSizeAttr(1.0)
            UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set((2.0, 4.0, 6.0))
            stage.GetRootLayer().Save()

            USDToMJCFConverter(scene_path, output_xml, model_name="scaled_cube").convert()

            geom = ET.parse(output_xml).getroot().find(".//geom")
            self.assertIsNotNone(geom)
            self.assertEqual(geom.attrib["type"], "box")
            self.assertEqual(geom.attrib["size"], "1 2 3")

    def test_parent_scale_changes_emitted_box_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_path = Path(tmpdir) / "parent_scaled_cube.usda"
            output_xml = Path(tmpdir) / "parent_scaled_cube.xml"
            stage = Usd.Stage.CreateNew(str(scene_path))
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            parent = UsdGeom.Xform.Define(stage, "/World/Parent")
            UsdGeom.Xformable(parent.GetPrim()).AddScaleOp().Set((2.0, 4.0, 6.0))
            cube = UsdGeom.Cube.Define(stage, "/World/Parent/ScaledCube")
            cube.CreateSizeAttr(1.0)
            stage.GetRootLayer().Save()

            USDToMJCFConverter(scene_path, output_xml, model_name="parent_scaled_cube").convert()

            geom = ET.parse(output_xml).getroot().find(".//geom")
            self.assertIsNotNone(geom)
            self.assertEqual(geom.attrib["type"], "box")
            self.assertEqual(geom.attrib["size"], "1 2 3")


if __name__ == "__main__":
    unittest.main()
