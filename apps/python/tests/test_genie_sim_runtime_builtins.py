from __future__ import annotations

import sys
import tempfile
import types
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


class GenieSimRuntimeBuiltinsTest(unittest.TestCase):
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

    def make_helper(self, library: dict[str, dict[str, object]]) -> types.ModuleType:
        helper = types.ModuleType("helper")

        def usd(*, oid: str, keywords: list[str]):
            return [
                {
                    "type": "cube",
                    "to_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.78], [0, 0, 0, 1]],
                    "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [1, 0, 0]}},
                    "info": {"stack": [], "info": {"id": oid, "name": "origin", "keywords": keywords}},
                }
            ]

        def register(_docstring=None):
            def decorator(func):
                def wrapper(*args, **kwargs):
                    result = func(*args, **kwargs)
                    for shape in result:
                        shape["info"]["stack"].append((func.__name__, "call-1"))
                    return result

                library[func.__name__] = {"__target__": wrapper}
                return wrapper

            return decorator

        def library_call(func_name: str, **kwargs):
            return library[func_name]["__target__"](**kwargs)

        def gen_scene_layout_info(scene_data):
            object_id = runtime._scene_object_id(scene_data[0])
            return (
                {
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
                },
                None,
            )

        helper.register = register
        helper.library_call = library_call
        helper.gen_scene_layout_info = gen_scene_layout_info
        helper.get_object_info = lambda _shape: {"size": [1.2, 0.8, 0.78]}
        library["usd"] = {"__target__": usd}
        return helper

    def library_modules(self, library: dict[str, dict[str, object]]) -> dict[str, object]:
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

    def test_execute_scene_code_preserves_builtin_usd_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_root = self.make_assets_root(tmp)
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            library: dict[str, dict[str, object]] = {}
            helper = self.make_helper(library)
            with mock.patch.dict(sys.modules, self.library_modules(library), clear=False):
                with mock.patch.object(runtime, "prepare_runtime", return_value=helper):
                    with mock.patch.object(runtime, "resolve_repo_root", return_value=repo_root):
                        result = runtime.execute_scene_code(SCENE_CODE, assets_root=assets_root)

            self.assertNotIn("bundleId", result)
            self.assertEqual(result["description"], "table_scene")

    def test_execute_scene_code_does_not_initialize_mi_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_root = self.make_assets_root(tmp)
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            library: dict[str, dict[str, object]] = {}
            helper = self.make_helper(library)
            with mock.patch.dict(sys.modules, self.library_modules(library), clear=False):
                with mock.patch.object(runtime, "prepare_runtime", return_value=helper):
                    with mock.patch.object(runtime, "resolve_repo_root", return_value=repo_root):
                        with mock.patch("utils.genie_sim.genie_sim_runtime.importlib.import_module") as import_module:
                            runtime.execute_scene_code(SCENE_CODE, assets_root=assets_root)

            import_module.assert_not_called()


if __name__ == "__main__":
    unittest.main()
