from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI

from utils.mount import load_module, mount_routes


class MountRoutesTest(unittest.TestCase):
    def test_mount_routes_use_nested_resource_paths(self):
        app = FastAPI()
        mount_routes(app, "api")

        paths = {route.path for route in app.routes} # type: ignore

        self.assertIn("/api/chinatsu/hello", paths)
        self.assertIn("/api/nanami/session/create", paths)
        self.assertIn("/api/nanami/session/destroy", paths)
        self.assertIn("/api/nanami/robot/load", paths)
        self.assertIn("/api/nanami/state/update", paths)
        self.assertIn("/api/nanami/export/start", paths)
        self.assertIn("/api/nanami/events/stream", paths)
        self.assertIn("/api/nanami/preview/stream", paths)
        self.assertNotIn("/api/nanami/create_session", paths)

    def test_load_module_resolves_future_annotations_in_pydantic_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            api_root = root / "api"
            api_root.mkdir()
            module_path = api_root / "sample.py"
            module_path.write_text(
                "\n".join(
                    [
                        "from __future__ import annotations",
                        "",
                        "from pydantic import BaseModel",
                        "",
                        "class Child(BaseModel):",
                        "    value: str",
                        "",
                        "class Body(BaseModel):",
                        "    child: Child | None = None",
                        "",
                        "async def create(body: Body):",
                        "    return {'ok': body.child.value if body.child else None}",
                    ]
                ),
                encoding="utf-8",
            )
            module = load_module(module_path)

            if module:
                payload = module.Body.model_validate({"child": {"value": "ok"}})
                self.assertEqual(payload.child.value, "ok")

if __name__ == "__main__":
    unittest.main()
