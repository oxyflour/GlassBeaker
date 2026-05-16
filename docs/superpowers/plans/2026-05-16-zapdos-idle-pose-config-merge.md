# Zapdos Idle Pose Config Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the repo default config with the user config, keep writeback scoped to the user file, and apply `override.position[robot_name]` as the robot idle joint pose during Zapdos physics initialization.

**Architecture:** Add a default-config reader plus recursive merge logic in `user_config.py`, then split read paths into merged reads and raw user-file reads. Keep camera override save on the raw user path, add a small robot-model codec in Python, and apply the merged idle joint positions in `MujocoPhysics` before the first `mj_forward` so the whole runtime starts from the configured neutral pose.

**Tech Stack:** Python 3.12, MuJoCo, existing Zapdos runtime modules, JSON config files, `unittest`, `uv`

---

## File Structure

- `apps/desktop/config.json`
  Repo-owned default config containing the default `override.position.r1pro` idle pose.
- `apps/python/utils/user_config.py`
  Default-config path resolution, raw user-config reading, merged effective-config reading, and deep merge logic.
- `apps/python/utils/camera_override.py`
  Read merged config for runtime application, but save camera overrides back into the raw user config only.
- `apps/python/utils/zapdos/robot_model.py`
  Python-side explicit mapping between robot model key and known robot USD path.
- `apps/python/utils/zapdos/physics/mujoco_physics.py`
  Apply the configured idle joint positions during MuJoCo initialization.
- `apps/python/tests/test_user_config.py`
  Focused config merge and validation coverage.
- `apps/python/tests/test_camera_override.py`
  Regression coverage for raw-user writeback.
- `apps/python/tests/test_zapdos_idle_pose.py`
  Runtime coverage that the configured idle pose reaches MuJoCo joint state.

### Task 1: Add merged config loading with raw-user readback

**Files:**
- Create: `apps/python/tests/test_user_config.py`
- Modify: `apps/python/utils/user_config.py`

- [ ] **Step 1: Write the failing config merge tests**

```python
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
                "override": {"position": {"r1pro": {"left_arm_joint1": 0.12, "left_arm_joint2": 0.48}}},
                "keep": {"default_only": True},
            }), encoding="utf-8")
            user_path = root / ".glass-beaker" / "config.json"
            user_path.parent.mkdir(parents=True)
            user_path.write_text(json.dumps({
                "override": {"position": {"r1pro": {"left_arm_joint2": 0.52}}},
                "keep": {"user_only": True},
            }), encoding="utf-8")

            with mock.patch("utils.user_config.default_config_path", return_value=default_path):
                with mock.patch.dict(os.environ, {"USERPROFILE": tmp}, clear=False):
                    payload = read_user_config()

        self.assertEqual(payload["override"]["position"]["r1pro"]["left_arm_joint1"], 0.12)
        self.assertEqual(payload["override"]["position"]["r1pro"]["left_arm_joint2"], 0.52)
        self.assertTrue(payload["keep"]["default_only"])
        self.assertTrue(payload["keep"]["user_only"])

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
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_user_config -v`

Expected: FAIL with import errors or missing merged/raw config behavior.

- [ ] **Step 3: Implement default-path resolution, raw reads, and deep merge**

```python
REPO_ROOT = Path(__file__).resolve().parents[4]


def default_config_path() -> Path:
    return REPO_ROOT / "apps" / "desktop" / "config.json"


def read_raw_user_config() -> dict[str, Any]:
    path = user_config_path()
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object.")
    return payload


def read_user_config() -> dict[str, Any]:
    default_payload = _read_json_object(default_config_path(), missing_ok=True)
    try:
        user_payload = read_raw_user_config()
    except RuntimeError:
        if os.environ.get("USERPROFILE", "").strip():
            raise
        user_payload = {}
    return _deep_merge(default_payload, user_payload)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged
```

- [ ] **Step 4: Re-run the focused config tests**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_user_config -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/python/utils/user_config.py apps/python/tests/test_user_config.py
git commit -m "feat: merge default and user config"
```

### Task 2: Keep camera override save scoped to the raw user file

**Files:**
- Modify: `apps/python/utils/camera_override.py`
- Modify: `apps/python/tests/test_camera_override.py`

- [ ] **Step 1: Write the failing raw-writeback regression**

```python
    def test_save_camera_overrides_does_not_copy_repo_defaults_into_user_file(self):
        snapshot = [{
            "name": "head_camera",
            "parent_prim": "/MyRobot/zed_link",
            "pos": [0.1, 0.2, 0.3],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "fovy": 60.0,
            "horizontal_aperture": 30.0,
            "vertical_aperture": 20.0,
            "clipping_range": [0.2, 80.0],
        }]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default_path = root / "desktop-config.json"
            default_path.write_text(json.dumps({
                "override": {"position": {"r1pro": {"left_arm_joint1": 0.12}}}
            }), encoding="utf-8")
            user_path = root / ".glass-beaker" / "config.json"
            user_path.parent.mkdir(parents=True)
            user_path.write_text("{}", encoding="utf-8")

            with mock.patch("utils.user_config.default_config_path", return_value=default_path):
                with mock.patch.dict(os.environ, {"USERPROFILE": tmp}, clear=False):
                    save_camera_overrides(snapshot)

            payload = json.loads(user_path.read_text(encoding="utf-8"))

        self.assertNotIn("position", payload.get("override", {}))
        self.assertIn("camera", payload["override"])
```

- [ ] **Step 2: Run the focused regression to verify it fails**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_camera_override.CameraOverrideTest.test_save_camera_overrides_does_not_copy_repo_defaults_into_user_file -v`

Expected: FAIL because save currently starts from the merged config payload.

- [ ] **Step 3: Switch save paths to the raw user config reader**

```python
from utils.user_config import read_raw_user_config, read_user_config, write_user_config


def save_camera_overrides(snapshot: list[dict[str, Any]]) -> tuple[Path, int]:
    payload = read_raw_user_config()
    override = payload.setdefault("override", {})
    if not isinstance(override, dict):
        raise RuntimeError("override must be a JSON object.")
    camera = override.setdefault("camera", {})
    if not isinstance(camera, dict):
        raise RuntimeError("override.camera must be a JSON object.")
    ...
```

- [ ] **Step 4: Re-run camera override tests**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_camera_override -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/python/utils/camera_override.py apps/python/tests/test_camera_override.py
git commit -m "fix: keep camera override writes user-scoped"
```

### Task 3: Add the default config file and robot model mapping

**Files:**
- Create: `apps/desktop/config.json`
- Create: `apps/python/utils/zapdos/robot_model.py`

- [ ] **Step 1: Write the default config and robot model codec**

```json
{
  "override": {
    "position": {
      "r1pro": {
        "torso_joint1": 0.0,
        "torso_joint2": 0.0,
        "torso_joint3": 0.0,
        "torso_joint4": 0.0,
        "left_arm_joint1": 0.12,
        "left_arm_joint2": 0.48,
        "left_arm_joint3": -0.18,
        "left_arm_joint4": -1.12,
        "left_arm_joint5": 0.04,
        "left_arm_joint6": 0.72,
        "left_arm_joint7": 0.08,
        "right_arm_joint1": -0.12,
        "right_arm_joint2": -0.48,
        "right_arm_joint3": 0.18,
        "right_arm_joint4": 1.12,
        "right_arm_joint5": -0.04,
        "right_arm_joint6": -0.72,
        "right_arm_joint7": -0.08
      }
    }
  }
}
```

```python
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_ROBOT_MODEL_KEY = "r1pro"
ROBOT_USD_BY_KEY = {
    "r1pro": (REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda").resolve(),
    "moz1": (REPO_ROOT / "deps" / "spirit01_model" / "USD" / "Moz1_robot_only.usda").resolve(),
}


def get_robot_model_key_from_usd(robot_usd: Path) -> str | None:
    resolved = Path(robot_usd).resolve()
    for key, value in ROBOT_USD_BY_KEY.items():
        if value == resolved:
            return key
    return None
```

- [ ] **Step 2: Stage the new files and sanity-check JSON loading**

Run: `uv run --project apps/python python -c "from utils.user_config import read_user_config; print(read_user_config().get('override', {}).get('position', {}).keys())"`

Expected: Prints a dictionary view containing `r1pro` when run with the repo default config.

- [ ] **Step 3: Commit**

```bash
git add apps/desktop/config.json apps/python/utils/zapdos/robot_model.py
git commit -m "feat: add default zapdos idle pose config"
```

### Task 4: Apply idle joint positions during MuJoCo initialization

**Files:**
- Create: `apps/python/tests/test_zapdos_idle_pose.py`
- Modify: `apps/python/utils/zapdos/physics/mujoco_physics.py`

- [ ] **Step 1: Write the failing physics-init tests**

```python
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

from utils.zapdos.bundle.bundle_builder import ensure_render_bundle
from utils.zapdos.bundle.render_bundle import DEFAULT_SCENE_USD
from utils.zapdos.physics.mujoco_physics import MujocoPhysics

R1PRO_USD = REPO_ROOT / "deps" / "galaxea" / "object" / "r1pro" / "r1pro.usda"


class ZapdosIdlePoseTest(unittest.TestCase):
    def test_mujoco_physics_applies_idle_pose_from_merged_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default_path = root / "desktop-config.json"
            default_path.write_text(json.dumps({
                "override": {
                    "position": {
                        "r1pro": {"left_arm_joint1": 0.33, "right_arm_joint1": -0.27}
                    }
                }
            }), encoding="utf-8")
            bundle = ensure_render_bundle(R1PRO_USD, DEFAULT_SCENE_USD)

            with mock.patch("utils.user_config.default_config_path", return_value=default_path):
                physics = MujocoPhysics("sess-idle", bundle, json.loads(bundle.body_map_json.read_text(encoding="utf-8")))

        joints = dict(zip(physics.joint_state_msg()["name"], physics.joint_state_msg()["position"]))
        self.assertAlmostEqual(joints["left_arm_joint1"], 0.33)
        self.assertAlmostEqual(joints["right_arm_joint1"], -0.27)
```

- [ ] **Step 2: Run the focused physics test to verify it fails**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_idle_pose -v`

Expected: FAIL because MuJoCo still starts from the model default qpos.

- [ ] **Step 3: Apply idle joint positions before the first `mj_forward`**

```python
from utils.user_config import read_user_config
from utils.zapdos.robot_model import get_robot_model_key_from_usd


class MujocoPhysics:
    def __init__(self, sess: str, bundle: Any, body_map: dict[str, str]) -> None:
        self.sess = sess
        self.model = mujoco.MjModel.from_xml_path(str(bundle.mjcf))  # type: ignore
        self.data = mujoco.MjData(self.model)  # type: ignore
        self._apply_idle_pose(bundle)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data) if os.environ.get("DEBUG_MUJOCO_VIEWER") else None
        ...
        mujoco.mj_forward(self.model, self.data)  # type: ignore

    def _apply_idle_pose(self, bundle: Any) -> None:
        robot_key = get_robot_model_key_from_usd(Path(bundle.robot_usd))
        if robot_key is None:
            return
        position = read_user_config().get("override", {}).get("position", {}).get(robot_key)
        if position is None:
            return
        if not isinstance(position, dict):
            raise RuntimeError(f"override.position.{robot_key} must be a JSON object.")
        for joint_name, value in position.items():
            if not isinstance(value, (int, float)):
                raise RuntimeError(f"override.position.{robot_key}.{joint_name} must be numeric.")
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)  # type: ignore
            if joint_id < 0:
                continue
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            self.data.qpos[qpos_adr] = float(value)
```

- [ ] **Step 4: Re-run focused idle pose tests**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_idle_pose -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/python/utils/zapdos/physics/mujoco_physics.py apps/python/tests/test_zapdos_idle_pose.py
git commit -m "feat: apply zapdos idle joint pose on startup"
```

### Task 5: Run combined regression coverage

**Files:**
- Modify: none

- [ ] **Step 1: Run the focused regression suite**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_user_config apps.python.tests.test_camera_override apps.python.tests.test_zapdos_idle_pose apps.python.tests.test_rl_bundle -v`

Expected: PASS.

- [ ] **Step 2: Run the broader Zapdos import smoke if focused tests pass**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_zapdos_import -v`

Expected: PASS with no config-loading regressions.

- [ ] **Step 3: Commit the verification state**

```bash
git status --short
```

Expected: No unexpected files beyond the intended implementation set.
