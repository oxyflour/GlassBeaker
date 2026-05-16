from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.user_config import read_raw_user_config, read_user_config


class UserConfigTest(unittest.TestCase):
    def test_read_user_config_deep_merges_default_and_user_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default_path = root / "desktop-config.json"
            default_path.write_text(json.dumps({
                "override": {
                    "position": {
                        "r1pro": {
                            "left_arm_joint1": 0.12,
                            "left_arm_joint2": 0.48,
                        }
                    }
                },
                "keep": {"default_only": True},
            }), encoding="utf-8")
            user_path = root / ".glass-beaker" / "config.json"
            user_path.parent.mkdir(parents=True)
            user_path.write_text(json.dumps({
                "override": {
                    "position": {
                        "r1pro": {
                            "left_arm_joint2": 0.52,
                        }
                    }
                },
                "keep": {"user_only": True},
            }), encoding="utf-8")

            with mock.patch("utils.user_config.default_config_path", return_value=default_path):
                with mock.patch.dict(os.environ, {"USERPROFILE": tmp}, clear=False):
                    payload = read_user_config()

        self.assertEqual(payload["override"]["position"]["r1pro"]["left_arm_joint1"], 0.12)
        self.assertEqual(payload["override"]["position"]["r1pro"]["left_arm_joint2"], 0.52)
        self.assertTrue(payload["keep"]["default_only"])
        self.assertTrue(payload["keep"]["user_only"])

    def test_read_user_config_replaces_scalars_and_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default_path = root / "desktop-config.json"
            default_path.write_text(json.dumps({
                "override": {"camera": {"order": ["head_camera", "wrist_camera"]}},
                "mode": "default",
            }), encoding="utf-8")
            user_path = root / ".glass-beaker" / "config.json"
            user_path.parent.mkdir(parents=True)
            user_path.write_text(json.dumps({
                "override": {"camera": {"order": ["head_camera"]}},
                "mode": "user",
            }), encoding="utf-8")

            with mock.patch("utils.user_config.default_config_path", return_value=default_path):
                with mock.patch.dict(os.environ, {"USERPROFILE": tmp}, clear=False):
                    payload = read_user_config()

        self.assertEqual(payload["override"]["camera"]["order"], ["head_camera"])
        self.assertEqual(payload["mode"], "user")

    def test_read_user_config_uses_default_payload_when_userprofile_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            default_path = Path(tmp) / "desktop-config.json"
            default_path.write_text(json.dumps({"override": {"position": {"r1pro": {"left_arm_joint1": 0.12}}}}), encoding="utf-8")

            with mock.patch("utils.user_config.default_config_path", return_value=default_path):
                with mock.patch.dict(os.environ, {}, clear=True):
                    payload = read_user_config()

        self.assertEqual(payload["override"]["position"]["r1pro"]["left_arm_joint1"], 0.12)

    def test_read_raw_user_config_returns_only_user_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default_path = root / "desktop-config.json"
            default_path.write_text(json.dumps({"override": {"position": {"r1pro": {"left_arm_joint1": 0.12}}}}), encoding="utf-8")
            user_path = root / ".glass-beaker" / "config.json"
            user_path.parent.mkdir(parents=True)
            user_path.write_text(json.dumps({"keep": {"value": 1}}), encoding="utf-8")

            with mock.patch("utils.user_config.default_config_path", return_value=default_path):
                with mock.patch.dict(os.environ, {"USERPROFILE": tmp}, clear=False):
                    payload = read_raw_user_config()

        self.assertEqual(payload, {"keep": {"value": 1}})


if __name__ == "__main__":
    unittest.main()
