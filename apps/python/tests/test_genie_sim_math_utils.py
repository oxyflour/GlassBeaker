from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

import utils.genie_sim_runtime as runtime


class GenieSimMathUtilsTest(unittest.TestCase):
    def test_runtime_patch_converts_rotation_matrix_inputs_to_float64(self):
        helper = types.SimpleNamespace()
        math_utils = types.ModuleType("geniesim.generator.scene_language.math_utils")
        captured: dict[str, np.dtype] = {}

        def rotation_matrix(angle, direction, point):
            captured["direction"] = direction.dtype
            captured["point"] = point.dtype
            return np.eye(4)

        math_utils.rotation_matrix = rotation_matrix

        with mock.patch.dict(sys.modules, {"geniesim.generator.scene_language.math_utils": math_utils}, clear=False):
            runtime._patch_rotation_matrix_compat(helper)
            matrix = helper.rotation_matrix(
                0.0,
                np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
                np.asarray((0.0, 0.0, 0.0), dtype=np.float32),
            )

        self.assertEqual(matrix.shape, (4, 4))
        self.assertEqual(captured["direction"], np.dtype("float64"))
        self.assertEqual(captured["point"], np.dtype("float64"))


if __name__ == "__main__":
    unittest.main()
