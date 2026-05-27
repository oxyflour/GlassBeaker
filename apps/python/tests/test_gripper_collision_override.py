from __future__ import annotations

import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.zapdos.gripper_collision_override import (  # noqa: E402
    apply_gripper_collision_overrides,
    parse_gripper_collision_overrides,
)
from utils.zapdos.bundle.bundle_builder import _build_sim_scene_mjcf  # noqa: E402


class GripperCollisionOverrideTest(unittest.TestCase):
    def test_default_r1pro_override_uses_lower_tip_pad_not_full_visual_obb(self) -> None:
        payload = json.loads((REPO_ROOT / "apps" / "desktop" / "config.json").read_text(encoding="utf-8"))
        overrides = parse_gripper_collision_overrides(payload["override"], "r1pro")
        by_name = {spec.geom_name: spec for spec in overrides}
        finger1 = by_name["Root_r1_pro_with_gripper_left_gripper_finger_link1_collisions_left_gripper_finger_link1_Mesh_geom"]
        finger2 = by_name["Root_r1_pro_with_gripper_left_gripper_finger_link2_collisions_left_gripper_finger_link2_Mesh_geom"]
        finger1_inner = by_name["Root_r1_pro_with_gripper_left_gripper_finger_link1_collisions_left_gripper_finger_link1_Mesh_geom_inner_pad"]

        self.assertLess(finger1.pos[2], 0.0)
        self.assertLess(finger1.size[2], 0.02)
        self.assertLess(finger1.size[0], 0.012)
        self.assertAlmostEqual(finger1.pos[1], -finger2.pos[1], places=6)
        self.assertAlmostEqual(finger1.pos[2], finger2.pos[2], places=6)
        self.assertGreater(finger1_inner.pos[2], 0.0)
        self.assertGreater(finger1_inner.pos[1], 0.0)
        self.assertLess(finger1_inner.size[2], 0.02)

    def test_apply_replaces_named_finger_collision_mesh_with_box(self) -> None:
        overrides = parse_gripper_collision_overrides(
            {
                "gripper_collision": {
                    "r1pro": {
                        "left": {
                            "finger1": {
                                "type": "box",
                                "pos": [0.0, -0.035, -0.03],
                                "quat": [0.943837, 0.283566, 0.101737, 0.135687],
                                "size": [0.012, 0.004, 0.02],
                            }
                        }
                    }
                }
            },
            "r1pro",
        )
        with tempfile.TemporaryDirectory() as tmp:
            mjcf = Path(tmp) / "scene.xml"
            mjcf.write_text(
                """
                <mujoco>
                  <asset>
                    <mesh name="finger_mesh" file="finger.obj" />
                  </asset>
                  <worldbody>
                    <body name="Root_r1_pro_with_gripper_left_gripper_finger_link1">
                      <geom name="Root_r1_pro_with_gripper_left_gripper_finger_link1_visuals_left_gripper_finger_link1_Mesh_geom"
                            type="mesh" mesh="finger_mesh" contype="0" conaffinity="0" />
                      <geom name="Root_r1_pro_with_gripper_left_gripper_finger_link1_collisions_left_gripper_finger_link1_Mesh_geom"
                            type="mesh" mesh="finger_mesh" material="finger_mat" />
                    </body>
                  </worldbody>
                </mujoco>
                """,
                encoding="utf-8",
            )

            apply_gripper_collision_overrides(mjcf, overrides)

            geom = ET.parse(mjcf).getroot().find(".//geom[@name='Root_r1_pro_with_gripper_left_gripper_finger_link1_collisions_left_gripper_finger_link1_Mesh_geom']")

        self.assertIsNotNone(geom)
        self.assertEqual(geom.attrib["type"], "box")
        self.assertEqual(geom.attrib["pos"], "0 -0.035 -0.03")
        self.assertEqual(geom.attrib["quat"], "0.943837 0.283566 0.101737 0.135687")
        self.assertEqual(geom.attrib["size"], "0.012 0.004 0.02")
        self.assertNotIn("mesh", geom.attrib)
        self.assertNotIn("material", geom.attrib)

    def test_apply_splits_nonconvex_finger_collision_into_compound_boxes(self) -> None:
        overrides = parse_gripper_collision_overrides(
            {
                "gripper_collision": {
                    "r1pro": {
                        "left": {
                            "finger1": {
                                "geoms": [
                                    {
                                        "name": "tip",
                                        "type": "box",
                                        "pos": [0.0, -0.009, -0.035],
                                        "size": [0.007, 0.004, 0.006],
                                    },
                                    {
                                        "name": "inner_pad",
                                        "type": "box",
                                        "pos": [0.0, 0.002, 0.023],
                                        "size": [0.014, 0.002, 0.011],
                                    },
                                ],
                            }
                        }
                    }
                }
            },
            "r1pro",
        )
        with tempfile.TemporaryDirectory() as tmp:
            mjcf = Path(tmp) / "scene.xml"
            mjcf.write_text(
                """
                <mujoco>
                  <worldbody>
                    <body name="Root_r1_pro_with_gripper_left_gripper_finger_link1">
                      <geom name="Root_r1_pro_with_gripper_left_gripper_finger_link1_collisions_left_gripper_finger_link1_Mesh_geom"
                            type="mesh" mesh="finger_mesh" material="finger_mat" />
                    </body>
                  </worldbody>
                </mujoco>
                """,
                encoding="utf-8",
            )

            apply_gripper_collision_overrides(mjcf, overrides)

            root = ET.parse(mjcf).getroot()
            primary = root.find(".//geom[@name='Root_r1_pro_with_gripper_left_gripper_finger_link1_collisions_left_gripper_finger_link1_Mesh_geom']")
            extra = root.find(".//geom[@name='Root_r1_pro_with_gripper_left_gripper_finger_link1_collisions_left_gripper_finger_link1_Mesh_geom_inner_pad']")

        self.assertIsNotNone(primary)
        self.assertIsNotNone(extra)
        self.assertEqual(primary.attrib["type"], "box")
        self.assertEqual(primary.attrib["pos"], "0 -0.009 -0.035")
        self.assertEqual(extra.attrib["type"], "box")
        self.assertEqual(extra.attrib["pos"], "0 0.002 0.023")
        self.assertNotIn("mesh", primary.attrib)
        self.assertNotIn("mesh", extra.attrib)

    def test_parse_rejects_non_numeric_size(self) -> None:
        with self.assertRaises(RuntimeError) as err:
            parse_gripper_collision_overrides(
                {
                    "gripper_collision": {
                        "r1pro": {
                            "left": {
                                "finger1": {
                                    "type": "box",
                                    "size": [0.01, True, 0.02],
                                }
                            }
                        }
                    }
                },
                "r1pro",
            )

        self.assertIn("size", str(err.exception))

    def test_build_sim_scene_applies_gripper_collision_override(self) -> None:
        overrides = parse_gripper_collision_overrides(
            {
                "gripper_collision": {
                    "r1pro": {
                        "left": {
                            "finger1": {
                                "type": "box",
                                "pos": [0.0, -0.035, -0.03],
                                "size": [0.012, 0.004, 0.02],
                            }
                        }
                    }
                }
            },
            "r1pro",
        )

        class FakeConverter:
            def __init__(self, _stage_path, output_xml, *_args, **_kwargs) -> None:
                self.output_xml = Path(output_xml)

            def convert(self) -> None:
                self.output_xml.write_text(
                    """
                    <mujoco>
                      <worldbody>
                        <body name="Root_r1_pro_with_gripper_left_gripper_finger_link1">
                          <geom name="Root_r1_pro_with_gripper_left_gripper_finger_link1_collisions_left_gripper_finger_link1_Mesh_geom"
                                type="mesh" mesh="finger_mesh" material="finger_mat" />
                        </body>
                      </worldbody>
                    </mujoco>
                    """,
                    encoding="utf-8",
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            descriptor = SimpleNamespace(
                physics_input=root / "robot.usda",
                visual_usd=root / "robot.usda",
            )
            scene_usd = root / "scene.usda"
            scene_usd.write_text("#usda 1.0\n", encoding="utf-8")
            mjcf = root / "sim_scene.xml"

            with mock.patch("utils.zapdos.bundle.bundle_builder.load_joint_drive_overrides", return_value={}):
                with mock.patch("utils.zapdos.bundle.bundle_builder.load_gripper_collision_overrides", return_value=overrides):
                    with mock.patch("utils.zapdos.bundle.bundle_builder.build_sim_scene", return_value=object()):
                        with mock.patch("utils.zapdos.bundle.bundle_builder.USDToMJCFConverter", FakeConverter):
                            _build_sim_scene_mjcf(
                                descriptor,
                                scene_usd,
                                root / "sim_scene.usda",
                                mjcf,
                                "Z",
                                1.0,
                                None,
                                set(),
                            )

            geom = ET.parse(mjcf).getroot().find(".//geom[@name='Root_r1_pro_with_gripper_left_gripper_finger_link1_collisions_left_gripper_finger_link1_Mesh_geom']")

        self.assertIsNotNone(geom)
        self.assertEqual(geom.attrib["type"], "box")
        self.assertNotIn("mesh", geom.attrib)


if __name__ == "__main__":
    unittest.main()
