# Zapdos Camera Override Save Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the Zapdos page save the current IsaacSim camera calibration into `%USERPROFILE%\.glass-beaker\config.json` under `override.camera`, and apply those saved values the next time a render bundle is built.

**Architecture:** Add two small Python helpers: one for `%USERPROFILE%` config I/O and one for camera override parsing/application. Extend the local Isaac renderer wrapper with a file-based snapshot request driven by an environment variable so the Python backend can ask the running renderer for the current camera prim state without modifying `deps/genie_sim`. Wire a new Zapdos `call` method to persist that snapshot, then add a small web button that triggers the save and reports success or failure.

**Tech Stack:** Python 3.12, FastAPI, unittest, MuJoCo, USD, Next.js client components, TypeScript node:test

---

### Task 1: Add User Config And Camera Override Helpers

**Files:**
- Create: `apps/python/utils/user_config.py`
- Create: `apps/python/utils/camera_override.py`
- Create: `apps/python/tests/test_camera_override.py`
- Modify: `apps/python/utils/rl_cameras.py`
- Modify: `apps/python/tests/test_rl_cameras.py`

- [ ] **Step 1: Write the failing tests**

Create `apps/python/tests/test_camera_override.py` with:

```python
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils.camera_override import apply_camera_overrides, load_camera_overrides, save_camera_overrides
from utils.rl_cameras import RenderCamera
from utils.user_config import read_user_config


class CameraOverrideTest(unittest.TestCase):
    def test_read_user_config_rejects_non_object_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / ".glass-beaker" / "config.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("[]", encoding="utf-8")

            with mock.patch.dict(os.environ, {"USERPROFILE": tmp}, clear=False):
                with self.assertRaises(RuntimeError) as err:
                    read_user_config()

        self.assertIn("must contain a JSON object", str(err.exception))

    def test_save_camera_overrides_merges_existing_config(self):
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
            config_path = Path(tmp) / ".glass-beaker" / "config.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps({"keep": {"value": 1}}, indent=2), encoding="utf-8")

            with mock.patch.dict(os.environ, {"USERPROFILE": tmp}, clear=False):
                written_path, saved = save_camera_overrides(snapshot)

            payload = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(written_path, config_path)
        self.assertEqual(saved, 1)
        self.assertEqual(payload["keep"]["value"], 1)
        self.assertEqual(payload["override"]["camera"]["/MyRobot/zed_link"]["head_camera"]["fovy"], 60.0)

    def test_apply_camera_overrides_updates_only_matching_camera(self):
        cameras = [
            RenderCamera(
                name="head_camera",
                prim="/MyRobot/zed_link/head_camera",
                topic="/env_0/head_camera/image_raw",
                frame_id="head_camera",
                body="Root_r1_pro_with_gripper_zed_link",
                pos=[0.0, 0.0, 0.0],
                quat=[1.0, 0.0, 0.0, 0.0],
                fovy=45.0,
            ),
            RenderCamera(
                name="left_wrist_camera",
                prim="/MyRobot/left_link/left_wrist_camera",
                topic="/env_0/left_wrist_camera/image_raw",
                frame_id="left_wrist_camera",
                body="Root_r1_pro_with_gripper_left_realsense_link",
                pos=[0.0, 0.0, 0.0],
                quat=[1.0, 0.0, 0.0, 0.0],
                fovy=45.0,
            ),
        ]
        overrides = load_camera_overrides({
            "override": {
                "camera": {
                    "/MyRobot/zed_link": {
                        "head_camera": {
                            "pos": [0.4, 0.5, 0.6],
                            "quat": [0.0, 0.0, 1.0, 0.0],
                            "fovy": 55.0,
                            "horizontal_aperture": 31.0,
                            "vertical_aperture": 21.0,
                            "clipping_range": [0.3, 90.0],
                        }
                    }
                }
            }
        })

        updated = apply_camera_overrides(cameras, overrides)

        self.assertEqual(updated[0].pos, [0.4, 0.5, 0.6])
        self.assertEqual(updated[0].fovy, 55.0)
        self.assertEqual(updated[0].horizontal_aperture, 31.0)
        self.assertEqual(updated[1].pos, [0.0, 0.0, 0.0])
        self.assertEqual(updated[1].fovy, 45.0)


if __name__ == "__main__":
    unittest.main()
```

Append this test to `apps/python/tests/test_rl_cameras.py`:

```python
    def test_render_camera_defaults_include_aperture_and_clipping(self):
        camera = RenderCamera(
            name="head_camera",
            prim="/MyRobot/zed_link/head_camera",
            topic="/env_0/head_camera/image_raw",
            frame_id="head_camera",
            body="Root_r1_pro_with_gripper_zed_link",
            pos=[0.0, 0.0, 0.0],
            quat=[1.0, 0.0, 0.0, 0.0],
            fovy=45.0,
        )

        self.assertEqual(camera.horizontal_aperture, 32.0)
        self.assertEqual(camera.vertical_aperture, 24.0)
        self.assertEqual(camera.clipping_range, [0.01, 100.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_camera_override apps.python.tests.test_rl_cameras`

Expected: FAIL because `utils.camera_override` and `utils.user_config` do not exist yet, and `RenderCamera` does not yet expose aperture or clipping defaults.

- [ ] **Step 3: Write the minimal implementation**

Create `apps/python/utils/user_config.py`:

```python
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def user_config_path() -> Path:
    root = os.environ.get("USERPROFILE", "").strip()
    if not root:
        raise RuntimeError("USERPROFILE is not set.")
    return Path(root).resolve() / ".glass-beaker" / "config.json"


def read_user_config() -> dict[str, Any]:
    path = user_config_path()
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object.")
    return payload


def write_user_config(payload: dict[str, Any]) -> Path:
    path = user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
```

Create `apps/python/utils/camera_override.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from utils.rl_cameras import RenderCamera
from utils.user_config import read_user_config, write_user_config


@dataclass(frozen=True)
class CameraOverride:
    parent_prim: str
    name: str
    pos: list[float]
    quat: list[float]
    fovy: float
    horizontal_aperture: float
    vertical_aperture: float
    clipping_range: list[float]


def load_camera_overrides(payload: dict[str, Any] | None = None) -> dict[tuple[str, str], CameraOverride]:
    data = read_user_config() if payload is None else payload
    override = data.get("override", {})
    if not isinstance(override, dict):
        raise RuntimeError("override must be a JSON object.")
    camera = override.get("camera", {})
    if not isinstance(camera, dict):
        raise RuntimeError("override.camera must be a JSON object.")
    loaded: dict[tuple[str, str], CameraOverride] = {}
    for parent_prim, entries in camera.items():
        if not isinstance(entries, dict):
            raise RuntimeError(f"override.camera.{parent_prim} must be a JSON object.")
        for name, spec in entries.items():
            if not isinstance(spec, dict):
                raise RuntimeError(f"override.camera.{parent_prim}.{name} must be a JSON object.")
            loaded[(str(parent_prim), str(name))] = CameraOverride(
                parent_prim=str(parent_prim),
                name=str(name),
                pos=[float(value) for value in spec["pos"]],
                quat=[float(value) for value in spec["quat"]],
                fovy=float(spec["fovy"]),
                horizontal_aperture=float(spec["horizontal_aperture"]),
                vertical_aperture=float(spec["vertical_aperture"]),
                clipping_range=[float(value) for value in spec["clipping_range"]],
            )
    return loaded


def apply_camera_overrides(
    cameras: list[RenderCamera],
    overrides: dict[tuple[str, str], CameraOverride] | None = None,
) -> list[RenderCamera]:
    resolved = load_camera_overrides() if overrides is None else overrides
    updated: list[RenderCamera] = []
    for camera in cameras:
        key = (PurePosixPath(camera.prim).parent.as_posix(), camera.name)
        spec = resolved.get(key)
        if spec is None:
            updated.append(camera)
            continue
        updated.append(RenderCamera(
            name=camera.name,
            prim=camera.prim,
            topic=camera.topic,
            frame_id=camera.frame_id,
            body=camera.body,
            pos=list(spec.pos),
            quat=list(spec.quat),
            fovy=spec.fovy,
            horizontal_aperture=spec.horizontal_aperture,
            vertical_aperture=spec.vertical_aperture,
            clipping_range=list(spec.clipping_range),
        ))
    return updated


def save_camera_overrides(snapshot: list[dict[str, Any]]) -> tuple[Path, int]:
    payload = read_user_config()
    override = payload.setdefault("override", {})
    if not isinstance(override, dict):
        raise RuntimeError("override must be a JSON object.")
    camera = override.setdefault("camera", {})
    if not isinstance(camera, dict):
        raise RuntimeError("override.camera must be a JSON object.")
    saved = 0
    for item in snapshot:
        parent_prim = str(item["parent_prim"])
        name = str(item["name"])
        group = camera.setdefault(parent_prim, {})
        if not isinstance(group, dict):
            raise RuntimeError(f"override.camera.{parent_prim} must be a JSON object.")
        group[name] = {
            "pos": [float(value) for value in item["pos"]],
            "quat": [float(value) for value in item["quat"]],
            "fovy": float(item["fovy"]),
            "horizontal_aperture": float(item["horizontal_aperture"]),
            "vertical_aperture": float(item["vertical_aperture"]),
            "clipping_range": [float(value) for value in item["clipping_range"]],
        }
        saved += 1
    return write_user_config(payload), saved
```

In `apps/python/utils/rl_cameras.py`, change the dataclass and focal helpers to:

```python
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class RenderCamera:
    name: str
    prim: str
    topic: str
    frame_id: str
    body: str | None
    pos: list[float]
    quat: list[float]
    fovy: float
    horizontal_aperture: float = CAMERA_HORIZONTAL_APERTURE
    vertical_aperture: float = CAMERA_VERTICAL_APERTURE
    clipping_range: list[float] = field(default_factory=lambda: list(CAMERA_CLIPPING_RANGE))

    def to_json(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "RenderCamera":
        return cls(
            name=str(data["name"]),
            prim=str(data["prim"]),
            topic=str(data["topic"]),
            frame_id=str(data["frame_id"]),
            body=(str(data["body"]) if data.get("body") is not None else None),
            pos=[float(value) for value in data["pos"]],
            quat=[float(value) for value in data["quat"]],
            fovy=float(data["fovy"]),
            horizontal_aperture=float(data.get("horizontal_aperture", CAMERA_HORIZONTAL_APERTURE)),
            vertical_aperture=float(data.get("vertical_aperture", CAMERA_VERTICAL_APERTURE)),
            clipping_range=[float(value) for value in data.get("clipping_range", CAMERA_CLIPPING_RANGE)],
        )


def focal_length_from_fovy(fovy: float, vertical_aperture: float = CAMERA_VERTICAL_APERTURE) -> float:
    radians = math.radians(float(fovy))
    return 0.5 * float(vertical_aperture) / math.tan(radians * 0.5)


def fovy_from_focal_length(focal_length: float, vertical_aperture: float) -> float:
    return math.degrees(2.0 * math.atan(float(vertical_aperture) * 0.5 / float(focal_length)))
```

In `build_render_cameras()` in the same file, set the new per-camera fields explicitly:

```python
        cameras.append(RenderCamera(
            name=name,
            prim=prim,
            topic=image_topic(name),
            frame_id=name,
            body=body,
            pos=[float(value) for value in model.cam_pos[cam_id]],
            quat=[float(value) for value in model.cam_quat[cam_id]],
            fovy=float(model.cam_fovy[cam_id]),
            horizontal_aperture=float(CAMERA_HORIZONTAL_APERTURE),
            vertical_aperture=float(CAMERA_VERTICAL_APERTURE),
            clipping_range=[float(value) for value in CAMERA_CLIPPING_RANGE],
        ))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_camera_override apps.python.tests.test_rl_cameras`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/python/utils/user_config.py apps/python/utils/camera_override.py apps/python/utils/rl_cameras.py apps/python/tests/test_camera_override.py apps/python/tests/test_rl_cameras.py
git commit -m "Add camera override helpers"
```

### Task 2: Apply Saved Overrides During Bundle Generation

**Files:**
- Modify: `apps/python/utils/rl_bundle.py`
- Modify: `apps/python/utils/rl_bundle_stage.py`
- Modify: `apps/python/tests/test_rl_bundle.py`

- [ ] **Step 1: Write the failing tests**

Append these tests to `apps/python/tests/test_rl_bundle.py`:

```python
    def test_bundle_key_changes_when_camera_override_config_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / ".glass-beaker" / "config.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps({"override": {"camera": {}}}, indent=2), encoding="utf-8")
            with mock.patch.dict(os.environ, {"USERPROFILE": tmp}, clear=False):
                before = MODULE._bundle_key(ROBOT_USD.resolve(), DEFAULT_SCENE_USD.resolve())
                config_path.write_text(json.dumps({
                    "override": {
                        "camera": {
                            "/MyRobot/Root_r1_pro_with_gripper_zed_link": {
                                "head_camera": {
                                    "pos": [0.1, 0.2, 0.3],
                                    "quat": [1.0, 0.0, 0.0, 0.0],
                                    "fovy": 60.0,
                                    "horizontal_aperture": 30.0,
                                    "vertical_aperture": 20.0,
                                    "clipping_range": [0.2, 80.0],
                                }
                            }
                        }
                    }
                }, indent=2), encoding="utf-8")
                after = MODULE._bundle_key(ROBOT_USD.resolve(), DEFAULT_SCENE_USD.resolve())

        self.assertNotEqual(before, after)

    def test_ensure_render_bundle_writes_overridden_camera_values_to_usd(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / ".glass-beaker" / "config.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps({
                "override": {
                    "camera": {
                        "/MyRobot/Root_r1_pro_with_gripper_zed_link": {
                            "head_camera": {
                                "pos": [0.1, 0.2, 0.3],
                                "quat": [1.0, 0.0, 0.0, 0.0],
                                "fovy": 60.0,
                                "horizontal_aperture": 30.0,
                                "vertical_aperture": 20.0,
                                "clipping_range": [0.2, 80.0],
                            }
                        }
                    }
                }
            }, indent=2), encoding="utf-8")

            with mock.patch.dict(os.environ, {"USERPROFILE": tmp}, clear=False):
                bundle = MODULE.ensure_render_bundle(ROBOT_USD, DEFAULT_SCENE_USD)

        stage = Usd.Stage.Open(str(bundle.robot_wrapper_usda))
        camera_prim = stage.GetPrimAtPath("/MyRobot/Root_r1_pro_with_gripper_zed_link/head_camera")
        camera = UsdGeom.Camera(camera_prim)
        self.assertEqual(float(camera.GetHorizontalApertureAttr().Get()), 30.0)
        self.assertEqual(float(camera.GetVerticalApertureAttr().Get()), 20.0)
        self.assertEqual(tuple(camera.GetClippingRangeAttr().Get()), (0.2, 80.0))
        self.assertEqual(tuple(camera_prim.GetAttribute("xformOp:translate").Get()), (0.1, 0.2, 0.3))
```

Also add these imports at the top of that test file:

```python
import os
import tempfile
from unittest import mock
import apps.python.utils.rl_bundle as MODULE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_rl_bundle`

Expected: FAIL because bundle keys ignore the user config and `_define_camera()` still writes shared aperture and clipping constants.

- [ ] **Step 3: Write the minimal implementation**

In `apps/python/utils/rl_bundle.py`, add the imports:

```python
from utils.camera_override import apply_camera_overrides
from utils.user_config import read_user_config
```

In `ensure_render_bundle()`, replace the camera build block with:

```python
    cameras = build_render_cameras(model, {body: f"MyRobot/{body}" for body in robot_bodies})
    cameras = apply_camera_overrides(cameras)
```

Replace `_bundle_key()` with:

```python
def _bundle_key(robot_usd: Path, scene_usd: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"bundle-v{BUNDLE_VERSION}".encode("utf-8"))
    try:
        overrides = read_user_config().get("override", {})
    except RuntimeError:
        overrides = {}
    digest.update(json.dumps(overrides, sort_keys=True).encode("utf-8"))
    for path in (robot_usd, scene_usd):
        stat = path.stat()
        digest.update(str(path).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
    return digest.hexdigest()[:16]
```

In `apps/python/utils/rl_bundle_stage.py`, replace `_define_camera()` with:

```python
def _define_camera(stage: Usd.Stage, path: str, spec: RenderCamera) -> None:
    camera = UsdGeom.Camera.Define(stage, path)
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*spec.pos))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(*spec.quat))
    camera.CreateFocalLengthAttr(float(focal_length_from_fovy(spec.fovy, spec.vertical_aperture)))
    camera.CreateHorizontalApertureAttr(float(spec.horizontal_aperture))
    camera.CreateVerticalApertureAttr(float(spec.vertical_aperture))
    camera.CreateClippingRangeAttr(Gf.Vec2f(*spec.clipping_range))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_camera_override apps.python.tests.test_rl_bundle`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/python/utils/rl_bundle.py apps/python/utils/rl_bundle_stage.py apps/python/tests/test_rl_bundle.py
git commit -m "Apply saved camera overrides to render bundles"
```

### Task 3: Add Renderer Camera Snapshot IPC

**Files:**
- Create: `apps/python/utils/renderer_ipc.py`
- Modify: `apps/python/utils/sim_env.py`
- Modify: `apps/isaac/rl_renderer_entry.py`
- Modify: `apps/python/tests/test_sim_env_renderer.py`

- [ ] **Step 1: Write the failing tests**

Add these imports to `apps/python/tests/test_sim_env_renderer.py`:

```python
import json
import tempfile
import threading
import time
```

Append these tests to the existing `IsaacRendererReadTest` class:

```python
    def test_snapshot_cameras_reads_renderer_response_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            renderer = IsaacRenderer.__new__(IsaacRenderer)
            renderer._running = True
            renderer.proc_id = "renderer"
            renderer.proc = None
            renderer.control_dir = Path(tmp)
            renderer.log_path = Path(tmp) / "renderer.log"
            renderer._refresh_process_state = lambda: True

            def respond() -> None:
                request_path = renderer.control_dir / "request.json"
                response_path = renderer.control_dir / "response.json"
                deadline = time.time() + 2.0
                while time.time() < deadline and not request_path.exists():
                    time.sleep(0.01)
                payload = json.loads(request_path.read_text(encoding="utf-8"))
                response_path.write_text(json.dumps({
                    "id": payload["id"],
                    "ok": True,
                    "cameras": [{"name": "head_camera", "parent_prim": "/MyRobot/zed_link"}],
                }), encoding="utf-8")

            thread = threading.Thread(target=respond, daemon=True)
            thread.start()

            cameras = renderer.snapshot_cameras(timeout=2.0)

        self.assertEqual(cameras[0]["name"], "head_camera")
        self.assertEqual(cameras[0]["parent_prim"], "/MyRobot/zed_link")

    def test_snapshot_cameras_raises_renderer_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            renderer = IsaacRenderer.__new__(IsaacRenderer)
            renderer._running = True
            renderer.proc_id = "renderer"
            renderer.proc = None
            renderer.control_dir = Path(tmp)
            renderer.log_path = Path(tmp) / "renderer.log"
            renderer._refresh_process_state = lambda: True

            def respond() -> None:
                request_path = renderer.control_dir / "request.json"
                response_path = renderer.control_dir / "response.json"
                deadline = time.time() + 2.0
                while time.time() < deadline and not request_path.exists():
                    time.sleep(0.01)
                payload = json.loads(request_path.read_text(encoding="utf-8"))
                response_path.write_text(json.dumps({
                    "id": payload["id"],
                    "ok": False,
                    "error": "snapshot failed",
                }), encoding="utf-8")

            thread = threading.Thread(target=respond, daemon=True)
            thread.start()

            with self.assertRaises(RuntimeError) as err:
                renderer.snapshot_cameras(timeout=2.0)

        self.assertIn("snapshot failed", str(err.exception))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_sim_env_renderer`

Expected: FAIL because `IsaacRenderer.snapshot_cameras()` and the control file paths do not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Create `apps/python/utils/renderer_ipc.py`:

```python
from __future__ import annotations

from pathlib import Path


def request_path(control_dir: Path) -> Path:
    return control_dir / "request.json"


def response_path(control_dir: Path) -> Path:
    return control_dir / "response.json"
```

In `apps/python/utils/sim_env.py`, add this field in `IsaacRenderer.__init__()` right after `self.log_path`:

```python
        self.control_dir = REPO_ROOT / "apps" / "python" / "tmp" / f"renderer_{tag}_ipc"
```

In `_spawn()`, add the control dir setup and environment variable:

```python
        self.control_dir.mkdir(parents=True, exist_ok=True)
        env["GB_RENDERER_CONTROL_DIR"] = str(self.control_dir)
```

Still in `apps/python/utils/sim_env.py`, add:

```python
from utils.renderer_ipc import request_path, response_path
```

And add this method to `IsaacRenderer`:

```python
    def snapshot_cameras(self, timeout: float = 5.0) -> list[dict[str, Any]]:
        req_path = request_path(self.control_dir)
        res_path = response_path(self.control_dir)
        req_id = str(time.time_ns())
        res_path.unlink(missing_ok=True)
        req_path.write_text(json.dumps({"id": req_id, "op": "snapshot_cameras"}), encoding="utf-8")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if res_path.exists():
                payload = json.loads(res_path.read_text(encoding="utf-8"))
                if payload.get("id") != req_id:
                    time.sleep(0.02)
                    continue
                res_path.unlink(missing_ok=True)
                req_path.unlink(missing_ok=True)
                if not payload.get("ok"):
                    raise RuntimeError(str(payload.get("error") or "renderer snapshot failed"))
                cameras = payload.get("cameras")
                if not isinstance(cameras, list):
                    raise RuntimeError("renderer snapshot returned invalid cameras")
                return cameras
            if not self._refresh_process_state():
                raise RuntimeError(_tail(self.log_path) or "renderer exited while waiting for snapshot")
            time.sleep(0.02)
        raise TimeoutError(f"renderer snapshot did not complete in {timeout:.1f}s")
```

In `apps/isaac/rl_renderer_entry.py`, add imports:

```python
import json
import os
from pathlib import Path, PurePosixPath

PYTHON_ROOT = REPO_ROOT / "apps" / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from utils.renderer_ipc import request_path, response_path
from utils.rl_cameras import fovy_from_focal_length
```

Add these methods to `LocalRLRenderer`:

```python
    def _camera_snapshot(self) -> list[dict[str, object]]:
        env_root = "/World/envs/env_0"
        cameras: list[dict[str, object]] = []
        for camera in getattr(self, "_camera_list", []):
            prim_path = env_root + str(camera["prim"])
            prim = self.stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise RuntimeError(f"Camera prim not found: {prim_path}")
            usd_camera = upstream.UsdGeom.Camera(prim)
            translate = prim.GetAttribute("xformOp:translate").Get()
            orient = prim.GetAttribute("xformOp:orient").Get()
            focal_length = float(usd_camera.GetFocalLengthAttr().Get())
            vertical_aperture = float(usd_camera.GetVerticalApertureAttr().Get())
            cameras.append({
                "name": str(camera["name"]),
                "prim": str(camera["prim"]),
                "parent_prim": PurePosixPath(str(camera["prim"])).parent.as_posix(),
                "pos": [float(translate[0]), float(translate[1]), float(translate[2])],
                "quat": [float(orient.GetReal()), float(orient.GetImaginary()[0]), float(orient.GetImaginary()[1]), float(orient.GetImaginary()[2])],
                "focal_length": focal_length,
                "horizontal_aperture": float(usd_camera.GetHorizontalApertureAttr().Get()),
                "vertical_aperture": vertical_aperture,
                "clipping_range": [float(value) for value in usd_camera.GetClippingRangeAttr().Get()],
                "fovy": fovy_from_focal_length(focal_length, vertical_aperture),
            })
        return cameras

    def _service_control_request(self) -> None:
        control_dir = os.environ.get("GB_RENDERER_CONTROL_DIR", "").strip()
        if not control_dir:
            return
        req_path = request_path(Path(control_dir))
        res_path = response_path(Path(control_dir))
        if not req_path.exists():
            return
        request = json.loads(req_path.read_text(encoding="utf-8"))
        try:
            if request.get("op") != "snapshot_cameras":
                raise RuntimeError(f"Unsupported renderer op: {request.get('op')}")
            payload = {"id": request.get("id"), "ok": True, "cameras": self._camera_snapshot()}
        except Exception as err:
            payload = {"id": request.get("id"), "ok": False, "error": str(err)}
        res_path.write_text(json.dumps(payload), encoding="utf-8")
        req_path.unlink(missing_ok=True)
```

Replace `run()` in the same class with:

```python
    def run(self) -> None:
        while upstream.simulation_app.is_running():
            self._service_control_request()
            self._ros_executor.spin_once(timeout_sec=0.0)
            self.world.step(render=True)
        self._ros_executor.shutdown(timeout_sec=2.0)
        for sub in self.env_subscribers:
            sub.destroy_node()
        upstream.rclpy.shutdown()
        if self.shm:
            self.shm.close()
            self.shm.unlink()
        self.world.stop()
        upstream.simulation_app.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_sim_env_renderer`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/python/utils/renderer_ipc.py apps/python/utils/sim_env.py apps/isaac/rl_renderer_entry.py apps/python/tests/test_sim_env_renderer.py
git commit -m "Add renderer camera snapshot IPC"
```

### Task 4: Persist Renderer Snapshot Through Zapdos

**Files:**
- Modify: `apps/python/api/zapdos/{session}/{action}.py`
- Modify: `apps/python/tests/test_zapdos_import.py`

- [ ] **Step 1: Write the failing tests**

Add these imports to `apps/python/tests/test_zapdos_import.py`:

```python
import json
import os
```

Append these tests to `ZapdosImportTest`:

```python
    def test_call_once_dispatches_save_camera_override(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.save_camera_override = mock.Mock(return_value={"ok": True, "saved": 1, "path": "config.json"})

        result = MODULE.ZapdosSession.call_once(session, "save_camera_override", ())

        self.assertEqual(result["saved"], 1)
        session.save_camera_override.assert_called_once_with()

    def test_save_camera_override_persists_renderer_snapshot(self):
        session = MODULE.ZapdosSession.__new__(MODULE.ZapdosSession)
        session.renderer = SimpleNamespace(snapshot_cameras=mock.Mock(return_value=[{
            "name": "head_camera",
            "parent_prim": "/MyRobot/zed_link",
            "pos": [0.1, 0.2, 0.3],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "fovy": 60.0,
            "horizontal_aperture": 30.0,
            "vertical_aperture": 20.0,
            "clipping_range": [0.2, 80.0],
        }]))

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"USERPROFILE": tmp}, clear=False):
                result = MODULE.ZapdosSession.save_camera_override(session)
                payload = json.loads((Path(tmp) / ".glass-beaker" / "config.json").read_text(encoding="utf-8"))

        self.assertEqual(result["saved"], 1)
        self.assertEqual(payload["override"]["camera"]["/MyRobot/zed_link"]["head_camera"]["fovy"], 60.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_zapdos_import`

Expected: FAIL because `save_camera_override()` is not defined and `call_once()` does not route that method.

- [ ] **Step 3: Write the minimal implementation**

In `apps/python/api/zapdos/{session}/{action}.py`, add:

```python
from utils.camera_override import save_camera_overrides
```

Add this method to `ZapdosSession`:

```python
    def save_camera_override(self) -> dict[str, object]:
        snapshot = self.renderer.snapshot_cameras()
        path, saved = save_camera_overrides(snapshot)
        return {"ok": True, "saved": saved, "path": str(path)}
```

Add this branch to `call_once()`:

```python
        if method == "save_camera_override":
            return self.save_camera_override()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_zapdos_import`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/python/api/zapdos/{session}/{action}.py apps/python/tests/test_zapdos_import.py
git commit -m "Add Zapdos camera override save call"
```

### Task 5: Add Zapdos Save Button

**Files:**
- Create: `apps/web/components/zapdos/camera-override-save.ts`
- Create: `apps/web/components/zapdos/camera-override-save.test.ts`
- Create: `apps/web/components/zapdos/CameraOverrideSaveButton.tsx`
- Modify: `apps/web/app/demo/zapdos/page.tsx`

- [ ] **Step 1: Write the failing tests**

Create `apps/web/components/zapdos/camera-override-save.test.ts`:

```typescript
import assert from "node:assert/strict";
import test from "node:test";

type CameraOverrideSaveModule = typeof import("./camera-override-save");

test("createSaveCameraOverrideRequest posts an empty argument array", async () => {
  const { createSaveCameraOverrideRequest } = await loadModule<CameraOverrideSaveModule>("./camera-override-save.ts");

  assert.deepEqual(createSaveCameraOverrideRequest(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify([]),
  });
});

test("saveCameraOverride posts to the zapdos save route and returns a success message", async () => {
  const { createSaveCameraOverrideRequest, saveCameraOverride } = await loadModule<CameraOverrideSaveModule>("./camera-override-save.ts");
  const calls: Array<{ input: unknown; init: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: unknown, init?: unknown) => {
    calls.push({ input, init });
    return new Response(JSON.stringify({ ok: true, saved: 3, path: "C:/Users/me/.glass-beaker/config.json" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    const message = await saveCameraOverride("sess-1");
    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/call/save_camera_override");
    assert.deepEqual(calls[0]?.init, createSaveCameraOverrideRequest());
    assert.equal(message, "Saved 3 camera overrides to C:/Users/me/.glass-beaker/config.json");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("saveCameraOverride throws backend text for a failed save", async () => {
  const { saveCameraOverride } = await loadModule<CameraOverrideSaveModule>("./camera-override-save.ts");
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => new Response("Session expired", { status: 409 })) as typeof fetch;

  try {
    await assert.rejects(() => saveCameraOverride("sess-1"), /Session expired/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

async function loadModule<TModule>(specifier: string): Promise<TModule> {
  const loaded = await import(specifier);
  return (loaded.default ?? loaded["module.exports"] ?? loaded) as TModule;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm --filter glassbeaker-web test -- camera-override-save`

Expected: FAIL because the helper module does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Create `apps/web/components/zapdos/camera-override-save.ts`:

```typescript
export interface SaveCameraOverrideResponse {
  ok: boolean;
  saved: number;
  path: string;
}

export function createSaveCameraOverrideRequest(): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify([]),
  };
}

export async function saveCameraOverride(sess: string): Promise<string> {
  const response = await fetch(`/python/zapdos/${sess}/call/save_camera_override`, createSaveCameraOverrideRequest());
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const payload = await response.json() as SaveCameraOverrideResponse;
  return `Saved ${payload.saved} camera overrides to ${payload.path}`;
}
```

Create `apps/web/components/zapdos/CameraOverrideSaveButton.tsx`:

```tsx
'use client'

import { useState } from "react";

import { saveCameraOverride } from "./camera-override-save";

export function CameraOverrideSaveButton({ sess }: { sess: string }) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function handleClick() {
    setBusy(true);
    setMessage("");
    setError("");
    try {
      setMessage(await saveCameraOverride(sess));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Save failed");
    } finally {
      setBusy(false);
    }
  }

  return <div className="rounded-md bg-black/60 px-3 py-2 text-white backdrop-blur-sm">
    <button
      className="rounded border border-white/20 bg-black/40 px-3 py-1 text-sm disabled:opacity-60"
      disabled={ busy || !sess }
      onClick={ () => void handleClick() }>
      { busy ? "Saving..." : "Save camera override" }
    </button>
    { message ? <div className="mt-2 max-w-80 text-xs text-emerald-200">{ message }</div> : null }
    { error ? <div className="mt-2 max-w-80 text-xs text-red-200">{ error }</div> : null }
  </div>;
}
```

In `apps/web/app/demo/zapdos/page.tsx`, add this import:

```tsx
import { CameraOverrideSaveButton } from "../../../components/zapdos/CameraOverrideSaveButton"
```

And add this JSX beside `SpaceMouseModeSelect`:

```tsx
            <CameraOverrideSaveButton sess={ sess } />
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm --filter glassbeaker-web test -- camera-override-save spacemouse-mode zapdos-import`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/components/zapdos/camera-override-save.ts apps/web/components/zapdos/camera-override-save.test.ts apps/web/components/zapdos/CameraOverrideSaveButton.tsx apps/web/app/demo/zapdos/page.tsx
git commit -m "Add Zapdos camera override save button"
```

### Task 6: End-To-End Verification

**Files:**
- Modify: `apps/python/utils/user_config.py`
- Modify: `apps/python/utils/camera_override.py`
- Modify: `apps/python/utils/rl_cameras.py`
- Modify: `apps/python/utils/rl_bundle.py`
- Modify: `apps/python/utils/rl_bundle_stage.py`
- Create: `apps/python/utils/renderer_ipc.py`
- Modify: `apps/python/utils/sim_env.py`
- Modify: `apps/isaac/rl_renderer_entry.py`
- Modify: `apps/python/api/zapdos/{session}/{action}.py`
- Create: `apps/web/components/zapdos/camera-override-save.ts`
- Create: `apps/web/components/zapdos/camera-override-save.test.ts`
- Create: `apps/web/components/zapdos/CameraOverrideSaveButton.tsx`
- Modify: `apps/web/app/demo/zapdos/page.tsx`

- [ ] **Step 1: Run the focused Python suites**

Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_camera_override apps.python.tests.test_rl_bundle apps.python.tests.test_sim_env_renderer apps.python.tests.test_zapdos_import`

Expected: PASS

- [ ] **Step 2: Run the focused web suite**

Run: `pnpm --filter glassbeaker-web test -- camera-override-save spacemouse-mode zapdos-import`

Expected: PASS

- [ ] **Step 3: Manual verification with Zapdos**

Run:

```powershell
$env:DEBUG_ISAAC_SHOW='1'
uv run --project apps/python --python 3.12 python -m uvicorn main:app --app-dir apps/python --host 127.0.0.1 --port 8000
```

Expected:
- Open `/demo/zapdos`
- Adjust a camera in IsaacSim
- Click `Save camera override`
- See a success message with `%USERPROFILE%\.glass-beaker\config.json`
- Confirm the saved JSON contains `override.camera`

- [ ] **Step 4: Verify the saved override is used on a fresh bundle**

Run:

```powershell
Remove-Item -Recurse -Force apps\python\tmp\rl_bundles
uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_rl_bundle
```

Expected: PASS, and a fresh Zapdos session shows the saved camera calibration after restart.
