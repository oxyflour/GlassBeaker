from __future__ import annotations

import importlib
import unittest


class ZapdosPackageNamespaceTest(unittest.TestCase):
    def test_moved_zapdos_modules_are_importable_from_package_namespace(self):
        module_names = [
            "utils.zapdos.isaac_renderer_reload",
            "utils.zapdos.mujoco_tools",
            "utils.zapdos.renderer_ipc",
            "utils.zapdos.rl_bundle",
            "utils.zapdos.rl_bundle_stage",
            "utils.zapdos.rl_cameras",
            "utils.zapdos.scene_objects",
            "utils.zapdos.sim_env",
            "utils.zapdos.usd_asset_cache",
            "utils.zapdos.usd_to_mjcf",
            "utils.zapdos.zapdos_asset_library",
            "utils.zapdos.zapdos_overlay",
            "utils.zapdos.zapdos_overlay_scene",
            "utils.zapdos.zapdos_physics",
            "utils.zapdos.zapdos_scene_visuals",
        ]

        for module_name in module_names:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
