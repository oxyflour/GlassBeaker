from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

import utils.genie_sim.genie_sim_runtime as runtime

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

    def test_load_assets_module_caches_by_root_and_rebinds_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = getattr(runtime, "_load_assets_module_cached", None)
            if cache is not None:
                cache.cache_clear()
            sys.modules.pop("assets", None)
            sys.modules.pop("geniesim.assets", None)

            def make_counted_assets_root(name: str, asset_id: str) -> Path:
                assets_root = Path(tmp) / name
                aligned = assets_root / "objects" / asset_id
                aligned.mkdir(parents=True)
                (aligned / "Aligned.usda").write_text("#usda 1.0\n", encoding="utf-8")
                counter_path = assets_root / "counter.txt"
                (assets_root / "__init__.py").write_text(
                    "\n".join(
                        [
                            "from pathlib import Path",
                            "ASSETS_PATH = Path(__file__).parent",
                            "COUNTER_PATH = ASSETS_PATH / 'counter.txt'",
                            "count = int(COUNTER_PATH.read_text(encoding='utf-8')) if COUNTER_PATH.exists() else 0",
                            "COUNTER_PATH.write_text(str(count + 1), encoding='utf-8')",
                            f"ASSETS_INDEX = {{'{asset_id}': {{'url': 'objects/{asset_id}/Aligned.usda', 'description': {{}}}}}}",
                            f"ASSETS_INDEX_HASH = '{asset_id}-hash'",
                        ]
                    ),
                    encoding="utf-8",
                )
                return assets_root

            assets_root_a = make_counted_assets_root("GenieSimAssetsA", "table_001")
            assets_root_b = make_counted_assets_root("GenieSimAssetsB", "box_002")

            module_a = runtime.load_assets_module(assets_root_a)
            module_b = runtime.load_assets_module(assets_root_b)
            module_a_again = runtime.load_assets_module(assets_root_a)

            self.assertIs(module_a_again, module_a)
            self.assertEqual((assets_root_a / "counter.txt").read_text(encoding="utf-8"), "1")
            self.assertEqual((assets_root_b / "counter.txt").read_text(encoding="utf-8"), "1")
            self.assertIn("table_001", module_a_again.ASSETS_INDEX)
            self.assertIs(sys.modules["assets"], module_a_again)
            self.assertIs(sys.modules["geniesim.assets"], module_a_again)


if __name__ == "__main__":
    unittest.main()
