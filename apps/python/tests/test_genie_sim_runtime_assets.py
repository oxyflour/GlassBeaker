from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

import utils.genie_sim_runtime as runtime

SCENE_CODE = "\n".join(
    [
        "from helper import *",
        "",
        "@register()",
        "def root_scene() -> Shape:",
        '    return library_call("usd", oid="table_001", keywords=["table"])',
    ]
)


class GenieSimRuntimeAssetsTest(unittest.TestCase):
    def make_assets_root(self, tmp: str) -> Path:
        assets_root = Path(tmp) / "GenieSimAssets"
        aligned = assets_root / "objects" / "table_001"
        aligned.mkdir(parents=True)
        (aligned / "Aligned.usda").write_text(
            "#usda 1.0\n(\n    defaultPrim = \"Asset\"\n)\n\ndef Xform \"Asset\"\n{\n}\n",
            encoding="utf-8",
        )
        (assets_root / "__init__.py").write_text(
            "\n".join(
                [
                    "from pathlib import Path",
                    "ASSETS_PATH = Path(__file__).parent",
                    "ASSETS_INDEX = {",
                    "  'table_001': {",
                    "    'description': {'semantic_name': ['table'], 'full_description': ['A white table']},",
                    "    'url': 'objects/table_001/Aligned.usda',",
                    "    'shapes': [",
                    "      {'name': 'origin', 'type': 'cube', 'size': [1.2, 0.8, 0.78], 'position': [0, 0, 0.39], 'quaternion': [0, 0, 0, 1]},",
                    "    ],",
                    "  },",
                    "}",
                    "ASSETS_INDEX_HASH = 'fake-hash'",
                ]
            ),
            encoding="utf-8",
        )
        return assets_root

    def test_execute_scene_code_supports_builtin_usd_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_root = self.make_assets_root(tmp)
            scene_usda_path = Path(tmp) / "scene.usda"
            scene_usda_path.write_text("#usda 1.0\n", encoding="utf-8")
            runtime._prepare_runtime_cached.cache_clear()
            sys.modules.pop("helper", None)

            with mock.patch.object(runtime, "persist_scene_output", return_value={"sceneUsdaPath": scene_usda_path.as_posix()}):
                result = runtime.execute_scene_code(SCENE_CODE, assets_root=assets_root)

            self.assertEqual(result["objects"][0]["assetId"], "table_001")
            self.assertTrue(Path(result["sceneUsdaPath"]).exists())


if __name__ == "__main__":
    unittest.main()
