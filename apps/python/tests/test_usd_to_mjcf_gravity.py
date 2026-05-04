from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco  # type: ignore

from utils.usd_to_mjcf import USDToMJCFConverter, sanitize_name


class USDToMJCFGravityTest(unittest.TestCase):
    def test_force_body_path_with_entity_mass_gets_freejoint_and_gravity(self):
        scene_usda = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{
    def Xform "Crate"
    {
        double3 xformOp:translate = (0, 0, 1)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        over "entity"
        {
            float physics:mass = 1
        }

        def Cube "Visual"
        {
            double size = 0.5
        }
    }
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_path = Path(tmpdir) / "scene.usda"
            output_xml = Path(tmpdir) / "scene.xml"
            scene_path.write_text(scene_usda, encoding="utf-8")

            USDToMJCFConverter(
                scene_path,
                output_xml,
                model_name="gravity_scene_object",
                force_body_paths={"/World/Crate"},
            ).convert()

            body_name = sanitize_name("/World/Crate")
            body = ET.parse(output_xml).getroot().find(f"./worldbody/body[@name='{body_name}']")
            self.assertIsNotNone(body)
            self.assertIsNotNone(body.find("./freejoint"))
            inertial = body.find("./inertial")
            self.assertIsNotNone(inertial)
            self.assertEqual(inertial.attrib["mass"], "1")

            model = mujoco.MjModel.from_xml_path(str(output_xml))  # type: ignore
            data = mujoco.MjData(model)  # type: ignore
            mujoco.mj_forward(model, data)  # type: ignore
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)  # type: ignore
            initial_z = float(data.xpos[body_id][2])
            for _ in range(100):
                mujoco.mj_step(model, data)  # type: ignore
            self.assertLess(float(data.xpos[body_id][2]), initial_z - 1e-3)

    def test_force_body_path_with_entity_kinematic_enabled_stays_static(self):
        scene_usda = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{
    def Xform "Table"
    {
        over "entity"
        {
            bool physics:kinematicEnabled = 1
        }

        def Cube "Visual"
        {
            double size = 1
        }
    }
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_path = Path(tmpdir) / "scene.usda"
            output_xml = Path(tmpdir) / "scene.xml"
            scene_path.write_text(scene_usda, encoding="utf-8")

            USDToMJCFConverter(
                scene_path,
                output_xml,
                model_name="kinematic_scene_object",
                force_body_paths={"/World/Table"},
            ).convert()

            body_name = sanitize_name("/World/Table")
            body = ET.parse(output_xml).getroot().find(f"./worldbody/body[@name='{body_name}']")
            self.assertIsNotNone(body)
            self.assertIsNone(body.find("./freejoint"))


if __name__ == "__main__":
    unittest.main()
