from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nanami.github_cache import ensure_r1_asset_cache
from nanami.models import SourceConfig


URDF_SAMPLE = """<robot name="demo">
  <link name="base_link"><visual><geometry><mesh filename="package://demo/meshes/base.STL" /></geometry></visual></link>
</robot>"""

ASCII_STL = """solid demo
facet normal 0 0 1
 outer loop
  vertex 0 0 0
  vertex 1 0 0
  vertex 0 1 0
 endloop
endfacet
endsolid demo
"""


class NanamiCacheTest(unittest.TestCase):
    def test_ensure_r1_asset_cache_uses_local_robot_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            robot_root = Path(tmp) / "R1"
            (robot_root / "urdf").mkdir(parents=True)
            (robot_root / "meshes").mkdir(parents=True)
            (robot_root / "package.xml").write_text("<package><name>demo</name></package>", encoding="utf-8")
            (robot_root / "urdf" / "demo.urdf").write_text(URDF_SAMPLE, encoding="utf-8")
            (robot_root / "meshes" / "base.STL").write_text(ASCII_STL, encoding="utf-8")

            bundle = ensure_r1_asset_cache(SourceConfig(robot_path=str(robot_root)))
            cached = ensure_r1_asset_cache(SourceConfig(robot_path=str(robot_root)))

        self.assertFalse(bundle.cache_hit)
        self.assertTrue(cached.cache_hit)
        manifest = bundle.manifest
        self.assertEqual(manifest.robot_id, "demo")
        mesh_obj_path = manifest.links[0].mesh_obj_path or ''
        self.assertTrue(mesh_obj_path.endswith("base.obj"))
        self.assertTrue(Path(mesh_obj_path).exists())
        self.assertEqual(cached.manifest.resolved_commit, manifest.resolved_commit)
        self.assertTrue(manifest.resolved_commit.startswith("local:"))
        self.assertEqual(bundle.asset_key, cached.asset_key)


if __name__ == "__main__":
    unittest.main()
