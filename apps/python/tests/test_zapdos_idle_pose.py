from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.zapdos.bundle.bundle_builder import ensure_render_bundle
from utils.zapdos.bundle.render_bundle import DEFAULT_SCENE_USD
from utils.zapdos.physics.mujoco_physics import MujocoPhysics

R1PRO_USD = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"
MOZ1_USD = REPO_ROOT / "deps" / "spirit01_model" / "USD" / "Moz1_robot_only.usda"


@lru_cache(maxsize=1)
def r1pro_bundle():
    return ensure_render_bundle(R1PRO_USD, DEFAULT_SCENE_USD)


class ZapdosIdlePoseTest(unittest.TestCase):
    def load_robot_model(self):
        importlib.invalidate_caches()
        spec = importlib.util.find_spec("utils.zapdos.robot_model")
        self.assertIsNotNone(spec, "utils.zapdos.robot_model is missing")
        return importlib.import_module("utils.zapdos.robot_model")

    def build_physics(
        self,
        default_payload: dict[str, object],
        *,
        user_payload: dict[str, object] | None = None,
        bundle=None,
        body_map: dict[str, str] | None = None,
    ) -> MujocoPhysics:
        with tempfile.TemporaryDirectory() as tmp:
            default_path = Path(tmp) / "desktop-config.json"
            default_path.write_text(json.dumps(default_payload), encoding="utf-8")
            if user_payload is not None:
                user_path = Path(tmp) / ".glass-beaker" / "config.json"
                user_path.parent.mkdir(parents=True)
                user_path.write_text(json.dumps(user_payload), encoding="utf-8")
            resolved_bundle = bundle or r1pro_bundle()
            resolved_body_map = body_map or json.loads(resolved_bundle.body_map_json.read_text(encoding="utf-8"))
            with mock.patch("utils.user_config.default_config_path", return_value=default_path):
                with mock.patch.dict(os.environ, {
                    "DEBUG_MUJOCO_VIEWER": "",
                    "USERPROFILE": tmp if user_payload is not None else "",
                }, clear=False):
                    return MujocoPhysics("sess-idle", resolved_bundle, resolved_body_map)

    def test_robot_model_codec_matches_known_usd_paths(self):
        module = self.load_robot_model()

        self.assertEqual(module.DEFAULT_ROBOT_MODEL_KEY, "r1pro")
        self.assertEqual(module.get_robot_model_key_from_usd(R1PRO_USD), "r1pro")
        self.assertEqual(module.get_robot_model_key_from_usd(MOZ1_USD), "moz1")
        self.assertIsNone(module.get_robot_model_key_from_usd(REPO_ROOT / "deps" / "unknown.usda"))

    def test_mujoco_physics_applies_idle_pose_on_startup(self):
        physics = self.build_physics({
            "override": {
                "position": {
                    "r1pro": {
                        "torso_joint1": 0.04,
                        "left_arm_joint1": 0.21,
                        "right_arm_joint1": -0.19,
                    }
                }
            }
        })
        try:
            joints = dict(zip(physics.joint_state_msg()["name"], physics.joint_state_msg()["position"]))
        finally:
            physics.close()

        self.assertAlmostEqual(joints["torso_joint1"], 0.04)
        self.assertAlmostEqual(joints["left_arm_joint1"], 0.21)
        self.assertAlmostEqual(joints["right_arm_joint1"], -0.19)

    def test_mujoco_physics_prefers_user_idle_pose_over_repo_default(self):
        physics = self.build_physics(
            {
                "override": {
                    "position": {
                        "r1pro": {
                            "left_arm_joint1": 0.11,
                        }
                    }
                }
            },
            user_payload={
                "override": {
                    "position": {
                        "r1pro": {
                            "left_arm_joint1": 0.29,
                        }
                    }
                }
            },
        )
        try:
            joints = dict(zip(physics.joint_state_msg()["name"], physics.joint_state_msg()["position"]))
        finally:
            physics.close()

        self.assertAlmostEqual(joints["left_arm_joint1"], 0.29)

    def test_mujoco_physics_ignores_unknown_idle_pose_joint_names(self):
        physics = self.build_physics({
            "override": {
                "position": {
                    "r1pro": {
                        "missing_joint": 1.23,
                        "left_arm_joint2": 0.17,
                    }
                }
            }
        })
        try:
            joints = dict(zip(physics.joint_state_msg()["name"], physics.joint_state_msg()["position"]))
        finally:
            physics.close()

        self.assertAlmostEqual(joints["left_arm_joint2"], 0.17)
        self.assertNotIn("missing_joint", joints)

    def test_render_bundle_applies_joint_drive_override_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            default_path = Path(tmp) / "desktop-config.json"
            default_path.write_text(json.dumps({
                "override": {
                    "joint_drive": {
                        "r1pro": {
                            "torso_joint1": {
                                "damping": 80.0,
                                "kp": 180.0,
                                "forcerange": [-150.0, 150.0],
                            }
                        }
                    }
                }
            }), encoding="utf-8")
            with mock.patch("utils.user_config.default_config_path", return_value=default_path):
                with mock.patch.dict(os.environ, {"DEBUG_MUJOCO_VIEWER": "", "USERPROFILE": ""}, clear=False):
                    bundle = ensure_render_bundle(R1PRO_USD, DEFAULT_SCENE_USD)

        root = ET.parse(bundle.mjcf).getroot()
        joint_xml = root.find(".//joint[@name='torso_joint1']")
        self.assertIsNotNone(joint_xml)
        self.assertEqual(joint_xml.attrib.get("damping"), "80")
        self.assertEqual(joint_xml.attrib.get("actuatorfrcrange"), "-150 150")
        actuator = root.find("./actuator/position[@joint='torso_joint1']")
        self.assertIsNotNone(actuator)
        self.assertEqual(actuator.attrib.get("kp"), "180")
        self.assertEqual(actuator.attrib.get("forcerange"), "-150 150")

    def test_mujoco_physics_rejects_non_object_idle_pose_override(self):
        with self.assertRaisesRegex(RuntimeError, "override.position.r1pro must be a JSON object"):
            physics = self.build_physics({"override": {"position": {"r1pro": 1}}})
            physics.close()

    def test_mujoco_physics_rejects_non_numeric_idle_pose_value(self):
        with self.assertRaisesRegex(RuntimeError, "override.position.r1pro.left_arm_joint1 must be numeric"):
            physics = self.build_physics({
                "override": {
                    "position": {
                        "r1pro": {
                            "left_arm_joint1": "bad",
                        }
                    }
                }
            })
            physics.close()

    def test_mujoco_physics_rejects_non_scalar_joint_idle_pose_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml_path = Path(tmp) / "ball-joint.xml"
            xml_path.write_text("""<mujoco model="ball_joint_idle_pose">
  <worldbody>
    <body name="Root_base_link">
      <joint name="left_arm_joint1" type="ball"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
""", encoding="utf-8")
            bundle = type("Bundle", (), {"mjcf": xml_path, "robot_usd": R1PRO_USD})()
            with self.assertRaisesRegex(RuntimeError, "override.position.r1pro.left_arm_joint1 must target a 1-DOF joint"):
                physics = self.build_physics(
                    {
                        "override": {
                            "position": {
                                "r1pro": {
                                    "left_arm_joint1": 0.25,
                                }
                            }
                        }
                    },
                    bundle=bundle,
                    body_map={"Root_base_link": "MyRobot/Root_base_link"},
                )
                physics.close()


if __name__ == "__main__":
    unittest.main()
