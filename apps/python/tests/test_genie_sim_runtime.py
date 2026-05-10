from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from pxr import Usd, UsdGeom

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

import utils.genie_sim.genie_sim_runtime as runtime
from utils.genie_sim import load_assets_module, search_assets

SCENE_CODE = "\n".join(
    [
        "def root_scene():",
        "    return [{",
        "        'type': 'cube',",
        "        'to_world': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.78], [0, 0, 0, 1]],",
        "        'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [1, 0, 0]}},",
        "        'info': {'stack': [('root', 'stack-1')], 'info': {'id': 'table_001', 'name': 'origin', 'keywords': ['table']}},",
        "    }]",
    ]
)


class GenieSimRuntimeTest(unittest.TestCase):
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
                    "  },",
                    "}",
                    "ASSETS_INDEX_HASH = 'fake-hash'",
                ]
            ),
            encoding="utf-8",
        )
        return assets_root

    def fake_helper(self):
        scene_data = eval(
            SCENE_CODE.split("return ", 1)[1],
            {"__builtins__": {}},
            {},
        )
        object_id = runtime._scene_object_id(scene_data[0])
        layout_info = {
            "scene_id": "table_scene",
            "seed": 7,
            "layout": {
                object_id: {
                    "usd": "table_001",
                    "xyz": [0.0, 0.0, 0.78],
                    "xyzw": [0.0, 0.0, 0.0, 1.0],
                    "keywords": ["table"],
                    "description": {"semantic_name": ["table"]},
                }
            },
            "relations": {"graph": {"nodes": [], "links": []}},
        }

        class _FakeHelper:
            @staticmethod
            def gen_scene_layout_info(_scene_data):
                return layout_info, None

            @staticmethod
            def get_object_info(_shapes):
                return {"size": [1.2, 0.8, 0.78]}

        return _FakeHelper()

    def library_modules(self) -> dict[str, object]:
        library = types.SimpleNamespace(clear=lambda: None)
        geniesim = types.ModuleType("geniesim")
        generator = types.ModuleType("geniesim.generator")
        scene_language = types.ModuleType("geniesim.generator.scene_language")
        dsl_utils = types.ModuleType("geniesim.generator.scene_language.dsl_utils")
        dsl_utils.library = library
        return {
            "geniesim": geniesim,
            "geniesim.generator": generator,
            "geniesim.generator.scene_language": scene_language,
            "geniesim.generator.scene_language.dsl_utils": dsl_utils,
        }

    def test_load_assets_module_aliases_external_assets_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_root = self.make_assets_root(tmp)

            module = load_assets_module(assets_root)

            self.assertEqual(module.ASSETS_PATH, assets_root)
            self.assertIn("geniesim.assets", sys.modules)
            self.assertIs(sys.modules["geniesim.assets"], module)

    def test_search_assets_prefers_exact_id_then_semantic_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_root = Path(tmp) / "GenieSimAssets"
            assets_root.mkdir()
            (assets_root / "__init__.py").write_text(
                "\n".join(
                    [
                        "from pathlib import Path",
                        "ASSETS_PATH = Path(__file__).parent",
                        "ASSETS_INDEX = {",
                        "  'table_001': {'description': {'semantic_name': ['white table'], 'full_description': ['A dining table']}, 'url': 'objects/table_001/Aligned.usda'},",
                        "  'box_002': {'description': {'semantic_name': ['red storage box'], 'full_description': ['A small red box']}, 'url': 'objects/box_002/Aligned.usda'},",
                        "}",
                        "ASSETS_INDEX_HASH = 'fake-hash'",
                    ]
                ),
                encoding="utf-8",
            )

            module = load_assets_module(assets_root)
            hits = search_assets(module.ASSETS_INDEX, "table_001", top_k=5)
            semantic_hits = search_assets(module.ASSETS_INDEX, "red box", top_k=5)

            self.assertEqual(hits[0]["asset_id"], "table_001")
            self.assertEqual(semantic_hits[0]["asset_id"], "box_002")

    def test_execute_scene_code_returns_scene_usda_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_root = self.make_assets_root(tmp)
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            with mock.patch.dict(sys.modules, self.library_modules(), clear=False):
                with mock.patch.object(runtime, "prepare_runtime", return_value=self.fake_helper()):
                    with mock.patch.object(runtime, "resolve_repo_root", return_value=repo_root):
                        result = runtime.execute_scene_code(SCENE_CODE, assets_root=assets_root)

            scene_dir = Path(result["sceneUsdaPath"]).parent

            self.assertNotIn("bundleId", result)
            self.assertTrue((scene_dir / "scene.usda").exists())
            self.assertFalse((scene_dir / "shape.json").exists())
            self.assertFalse((scene_dir / "manifest.json").exists())

    def test_execute_scene_code_exports_expected_stage_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_root = self.make_assets_root(tmp)
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            with mock.patch.dict(sys.modules, self.library_modules(), clear=False):
                with mock.patch.object(runtime, "prepare_runtime", return_value=self.fake_helper()):
                    with mock.patch.object(runtime, "resolve_repo_root", return_value=repo_root):
                        result = runtime.execute_scene_code(SCENE_CODE, assets_root=assets_root)

            stage = Usd.Stage.Open(result["sceneUsdaPath"])
            self.assertEqual(UsdGeom.GetStageUpAxis(stage), UsdGeom.Tokens.z)
            self.assertEqual(stage.GetMetadata("metersPerUnit"), 1.0)

    def test_execute_scene_code_accepts_known_helper_import_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_root = self.make_assets_root(tmp)
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            for module_name in ("genie_sim_helper", "genie_sim_open", "geniesim_helper", "geniesim_open"):
                code = f"from {module_name} import *\n" + SCENE_CODE
                sys.modules.pop(module_name, None)
                with mock.patch.dict(sys.modules, self.library_modules(), clear=False):
                    with mock.patch.object(runtime, "prepare_runtime", return_value=self.fake_helper()):
                        with mock.patch.object(runtime, "resolve_repo_root", return_value=repo_root):
                            result = runtime.execute_scene_code(code, assets_root=assets_root)

                self.assertNotIn("bundleId", result)
                self.assertEqual(result["description"], "table_scene")

    def test_execute_scene_code_accepts_scene_return_annotation_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_root = self.make_assets_root(tmp)
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            code = SCENE_CODE.replace("def root_scene():", "def root_scene() -> Scene:")
            with mock.patch.dict(sys.modules, self.library_modules(), clear=False):
                with mock.patch.object(runtime, "prepare_runtime", return_value=self.fake_helper()):
                    with mock.patch.object(runtime, "resolve_repo_root", return_value=repo_root):
                        result = runtime.execute_scene_code(code, assets_root=assets_root)

            self.assertNotIn("bundleId", result)
            self.assertEqual(result["description"], "table_scene")

    def test_prepare_runtime_does_not_require_mi_helper_for_scene_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_root = self.make_assets_root(tmp)
            repo_root = Path(tmp) / "repo"
            helper_root = repo_root / "deps" / "genie_sim" / "source"
            helper_root.mkdir(parents=True)
            (helper_root / "helper.py").write_text("RUNTIME_OK = True\n", encoding="utf-8")
            runtime._prepare_runtime_cached.cache_clear()
            sys.modules.pop("helper", None)
            original_import = __import__

            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "geniesim.generator.scene_language.mi_helper":
                    raise ModuleNotFoundError("No module named 'jaxtyping'")
                return original_import(name, globals, locals, fromlist, level)

            with mock.patch.object(runtime, "resolve_repo_root", return_value=repo_root):
                with mock.patch("builtins.__import__", side_effect=guarded_import):
                    helper = runtime.prepare_runtime(assets_root)

            self.assertTrue(helper.RUNTIME_OK)
            runtime._prepare_runtime_cached.cache_clear()


if __name__ == "__main__":
    unittest.main()
