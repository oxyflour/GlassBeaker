from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco  # type: ignore
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from utils.zapdos.usd_to_mjcf import USDToMJCFConverter, fmt_f, sanitize_name

REPO_ROOT = Path(__file__).resolve().parents[3]
ROBOT_USD = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"


class USDToMJCFTest(unittest.TestCase):
    def test_force_body_paths_emit_scene_object_as_top_level_body(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_path = Path(tmpdir) / "scene_object.usda"
            output_xml = Path(tmpdir) / "scene_object.xml"
            stage = Usd.Stage.CreateNew(str(scene_path))
            stage.SetMetadata("metersPerUnit", 1.0)
            UsdGeom.SetStageUpAxis(stage, "Z")
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            crate = UsdGeom.Xform.Define(stage, "/World/Crate")
            UsdGeom.Xformable(crate.GetPrim()).AddTranslateOp().Set((1.0, 2.0, 3.0))
            visual = UsdGeom.Cube.Define(stage, "/World/Crate/Visual")
            visual.CreateSizeAttr(0.5)
            stage.GetRootLayer().Save()

            USDToMJCFConverter(
                scene_path,
                output_xml,
                model_name="scene_object",
                force_body_paths={"/World/Crate"},
            ).convert()

            body_name = sanitize_name("/World/Crate")
            body = ET.parse(output_xml).getroot().find(f"./worldbody/body[@name='{body_name}']")
            self.assertIsNotNone(body)
            self.assertIsNotNone(body.find("./geom"))
            model = mujoco.MjModel.from_xml_path(str(output_xml))  # type: ignore
            self.assertGreater(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name),  # type: ignore
                0,
            )

    def test_force_body_paths_keep_y_up_scene_objects_at_world_level(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_path = Path(tmpdir) / "scene_object_y_up.usda"
            output_xml = Path(tmpdir) / "scene_object_y_up.xml"
            stage = Usd.Stage.CreateNew(str(scene_path))
            stage.SetMetadata("metersPerUnit", 1.0)
            UsdGeom.SetStageUpAxis(stage, "Y")
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            crate = UsdGeom.Xform.Define(stage, "/World/Crate")
            UsdGeom.Cube.Define(stage, "/World/Crate/Visual").CreateSizeAttr(0.5)
            stage.GetRootLayer().Save()

            USDToMJCFConverter(
                scene_path,
                output_xml,
                model_name="scene_object_y_up",
                force_body_paths={"/World/Crate"},
            ).convert()

            root = ET.parse(output_xml).getroot()
            body_name = sanitize_name("/World/Crate")
            self.assertIsNotNone(root.find(f"./worldbody/body[@name='{body_name}']"))
            wrapper = root.find("./worldbody/body[@name='usd_stage_root']")
            self.assertIsNotNone(wrapper)
            self.assertIsNone(wrapper.find(f"./body[@name='{body_name}']"))

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

    def test_slide_joint_force_servo_kp_is_capped_for_stability(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_path = Path(tmpdir) / "slider.usda"
            output_xml = Path(tmpdir) / "slider.xml"
            stage = Usd.Stage.CreateNew(str(scene_path))
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            base = UsdGeom.Xform.Define(stage, "/World/Base")
            slider = UsdGeom.Xform.Define(stage, "/World/Slider")
            for prim in (base.GetPrim(), slider.GetPrim()):
                UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(1.0)

            joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/Slider/PrismaticJoint")
            joint.CreateBody0Rel().SetTargets([base.GetPath()])
            joint.CreateBody1Rel().SetTargets([slider.GetPath()])
            joint.CreateAxisAttr("Y")
            joint.CreateLowerLimitAttr(0.0)
            joint.CreateUpperLimitAttr(0.05)
            joint_prim = joint.GetPrim()
            joint_prim.CreateAttribute(
                "drive:linear:physics:damping",
                Sdf.ValueTypeNames.Float,
            ).Set(20.0)
            joint_prim.CreateAttribute(
                "drive:linear:physics:maxForce",
                Sdf.ValueTypeNames.Float,
            ).Set(100.0)
            joint_prim.CreateAttribute(
                "drive:linear:physics:stiffness",
                Sdf.ValueTypeNames.Float,
            ).Set(0.0)
            joint_prim.CreateAttribute(
                "drive:linear:physics:targetPosition",
                Sdf.ValueTypeNames.Float,
            ).Set(0.0)
            stage.GetRootLayer().Save()

            USDToMJCFConverter(scene_path, output_xml, model_name="slider").convert()

            actuator = ET.parse(output_xml).getroot().find("./actuator/position")
            self.assertIsNotNone(actuator)
            self.assertEqual(actuator.attrib["kp"], fmt_f(1e4))
            mujoco.MjModel.from_xml_path(str(output_xml))  # type: ignore

    def test_explicit_collision_enabled_visual_geom_keeps_contact_bits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_path = Path(tmpdir) / "visual_collision.usda"
            output_xml = Path(tmpdir) / "visual_collision.xml"
            stage = Usd.Stage.CreateNew(str(scene_path))
            stage.SetMetadata("metersPerUnit", 1.0)
            UsdGeom.SetStageUpAxis(stage, "Z")
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            crate = UsdGeom.Xform.Define(stage, "/World/Crate")
            UsdPhysics.MassAPI.Apply(crate.GetPrim()).CreateMassAttr(1.0)
            body = UsdGeom.Xform.Define(stage, "/World/Crate/body")
            visual = UsdGeom.Cube.Define(stage, "/World/Crate/body/visual")
            visual.CreateSizeAttr(0.5)
            UsdPhysics.CollisionAPI.Apply(visual.GetPrim()).CreateCollisionEnabledAttr(True)
            stage.GetRootLayer().Save()

            USDToMJCFConverter(
                scene_path,
                output_xml,
                model_name="visual_collision",
                force_body_paths={"/World/Crate"},
            ).convert()

            model = mujoco.MjModel.from_xml_path(str(output_xml))  # type: ignore
            geom_name = sanitize_name("/World/Crate/body/visual") + "_geom"
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)  # type: ignore
            self.assertGreaterEqual(geom_id, 0)
            self.assertGreater(int(model.geom_contype[geom_id]), 0)
            self.assertGreater(int(model.geom_conaffinity[geom_id]), 0)

    def test_dynamic_body_with_explicit_visual_collision_stops_on_floor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_path = Path(tmpdir) / "visual_collision_gravity.usda"
            output_xml = Path(tmpdir) / "visual_collision_gravity.xml"
            stage = Usd.Stage.CreateNew(str(scene_path))
            stage.SetMetadata("metersPerUnit", 1.0)
            UsdGeom.SetStageUpAxis(stage, "Z")
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            ground = UsdGeom.Cube.Define(stage, "/World/Ground")
            ground.CreateSizeAttr(2.0)
            UsdGeom.Xformable(ground.GetPrim()).AddTranslateOp().Set((0.0, 0.0, -1.0))
            crate = UsdGeom.Xform.Define(stage, "/World/Crate")
            UsdGeom.Xformable(crate.GetPrim()).AddTranslateOp().Set((0.0, 0.0, 1.0))
            UsdPhysics.MassAPI.Apply(crate.GetPrim()).CreateMassAttr(1.0)
            body = UsdGeom.Xform.Define(stage, "/World/Crate/body")
            visual = UsdGeom.Cube.Define(stage, "/World/Crate/body/visual")
            visual.CreateSizeAttr(0.5)
            UsdPhysics.CollisionAPI.Apply(visual.GetPrim()).CreateCollisionEnabledAttr(True)
            stage.GetRootLayer().Save()

            USDToMJCFConverter(
                scene_path,
                output_xml,
                model_name="visual_collision_gravity",
                force_body_paths={"/World/Crate"},
            ).convert()

            model = mujoco.MjModel.from_xml_path(str(output_xml))  # type: ignore
            data = mujoco.MjData(model)  # type: ignore
            body_name = sanitize_name("/World/Crate")
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)  # type: ignore
            mujoco.mj_forward(model, data)  # type: ignore
            for _ in range(300):
                mujoco.mj_step(model, data)  # type: ignore
            self.assertGreater(float(data.xpos[body_id][2]), 0.2)


if __name__ == "__main__":
    unittest.main()

