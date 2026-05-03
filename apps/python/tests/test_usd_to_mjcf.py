from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from utils.usd_to_mjcf import USDToMJCFConverter

REPO_ROOT = Path(__file__).resolve().parents[3]
ROBOT_USD = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"


class USDToMJCFTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
