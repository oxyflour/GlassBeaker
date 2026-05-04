# USD To MJCF Performance Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut `USDToMJCFConverter` wall time by removing duplicate mesh export work and reusing mesh/texture artifacts across bundle builds.

**Architecture:** Current profiles show `build_geom_for_prim()` spends almost all time in `export_mesh_prim()`, especially `write_obj_mesh()` and `get_mesh_texcoords()`. Implement two cache layers first: content-addressed mesh dedupe inside one conversion, then persistent mesh/texture cache shared across conversions. Only prototype parallel mesh writing if the cache work still leaves cold conversions above the target.

**Tech Stack:** Python 3.12, `pxr`, `numpy`, `Pillow`, `unittest`, `uv`

---

### Task 1: Lock The Baseline With Regression Tests
**Files:**
- Modify: `apps/python/tests/test_usd_to_mjcf.py`
- Reference: `apps/python/utils/usd_to_mjcf.py:653-679`
- Reference: `apps/python/utils/usd_to_mjcf.py:806-986`
- [ ] **Step 1: Write the failing duplicate-mesh test**
```python
def test_duplicate_mesh_prims_share_one_exported_obj(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        scene_path = Path(tmpdir) / "duplicate_mesh.usda"
        output_xml = Path(tmpdir) / "duplicate_mesh.xml"
        stage = Usd.Stage.CreateNew(str(scene_path))
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        for path in ("/World/Visual", "/World/Collision"):
            mesh = UsdGeom.Mesh.Define(stage, path)
            mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        stage.GetRootLayer().Save()
```
- [ ] **Step 2: Assert current output is wrong for the optimization goal**
```python
        USDToMJCFConverter(scene_path, output_xml, model_name="duplicate_mesh").convert()
        mesh_assets = ET.parse(output_xml).getroot().findall("./asset/mesh")
        self.assertEqual(len(mesh_assets), 1)
        self.assertEqual(len(list(output_xml.parent.glob("meshes/*.obj"))), 1)
```
- [ ] **Step 3: Run the targeted test to confirm the assertion fails**

Run: `uv run python -m unittest tests.test_usd_to_mjcf.USDToMJCFTest.test_duplicate_mesh_prims_share_one_exported_obj -v`

Expected: `FAIL` because the converter emits two mesh assets and two OBJ files today.
- [ ] **Step 4: Commit**

```bash
git add apps/python/tests/test_usd_to_mjcf.py
git commit -m "test: capture usd_to_mjcf mesh cache regressions"
```

### Task 2: Add In-Process Mesh Asset Dedupe
**Files:**
- Modify: `apps/python/utils/usd_to_mjcf.py:341-439`
- Modify: `apps/python/utils/usd_to_mjcf.py:896-986`
- Test: `apps/python/tests/test_usd_to_mjcf.py`
- [ ] **Step 1: Add mesh asset state to the converter**
```python
@dataclass
class MeshAssetData:
    name: str
    file_rel: str
    signature: str

self.mesh_assets: Dict[str, MeshAssetData] = {}
```
- [ ] **Step 2: Add a stable mesh signature helper**
```python
def mesh_signature(self, vertices, face_counts, face_indices, texcoords, face_texcoords) -> str:
    digest = hashlib.sha1()
    for array in (vertices, face_counts, face_indices, texcoords, face_texcoords):
        if array is None:
            digest.update(b"<none>")
            continue
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(fmt_f(self.meters_per_unit).encode("ascii"))
    return digest.hexdigest()
```
- [ ] **Step 3: Reuse an existing mesh asset before writing a second OBJ**
```python
signature = self.mesh_signature(vertices, face_counts, face_indices, texcoords, face_texcoords)
cached = self.mesh_assets.get(signature)
if cached is not None:
    return cached.name, cached.file_rel
mesh_name = sanitize_name(str(prim.GetPath()))
file_rel = (Path("meshes") / f"{mesh_name}.obj").as_posix()
self.write_obj_mesh(out_file, vertices, faces, texcoords, face_texcoords)
self.mesh_assets[signature] = MeshAssetData(mesh_name, file_rel, signature)
```
- [ ] **Step 4: Run the focused tests and the existing converter suite**

Run: `uv run python -m unittest tests.test_usd_to_mjcf -v`

Expected: `OK`
- [ ] **Step 5: Commit**

```bash
git add apps/python/utils/usd_to_mjcf.py apps/python/tests/test_usd_to_mjcf.py
git commit -m "feat: dedupe identical usd mesh exports"
```

### Task 3: Add Persistent Mesh And Texture Cache Across Bundles
**Files:**
- Create: `apps/python/utils/usd_asset_cache.py`
- Modify: `apps/python/utils/usd_to_mjcf.py:407-439`
- Modify: `apps/python/utils/usd_to_mjcf.py:653-679`
- Modify: `apps/python/utils/usd_to_mjcf.py:962-986`
- Test: `apps/python/tests/test_usd_to_mjcf.py`
- [ ] **Step 1: Add a cache helper with versioned content-addressed paths**
```python
CACHE_ROOT = REPO_ROOT / "apps" / "python" / "tmp" / "usd_to_mjcf_cache"

def mesh_cache_path(signature: str) -> Path:
    return CACHE_ROOT / "meshes" / f"{signature}.obj"

def texture_cache_path(signature: str) -> Path:
    return CACHE_ROOT / "textures" / f"{signature}.png"
```
- [ ] **Step 2: Use cache hits to avoid rebuilding OBJ and PNG payloads**
```python
cache_file = mesh_cache_path(signature)
if cache_file.exists():
    materialize_cached_file(cache_file, out_file)
    self.mesh_assets[signature] = MeshAssetData(mesh_name, file_rel, signature)
    return mesh_name, file_rel
self.write_obj_mesh(cache_file, vertices, faces, texcoords, face_texcoords)
materialize_cached_file(cache_file, out_file)
```
- [ ] **Step 3: Cover warm-cache behavior with a test**
```python
with mock.patch.object(USDToMJCFConverter, "write_obj_mesh", wraps=USDToMJCFConverter.write_obj_mesh) as writer:
    USDToMJCFConverter(scene_path, first_xml, model_name="first").convert()
    writer.reset_mock()
    USDToMJCFConverter(scene_path, second_xml, model_name="second").convert()
    writer.assert_not_called()
```
- [ ] **Step 4: Run tests for cold and warm cache flows**

Run: `uv run python -m unittest tests.test_usd_to_mjcf -v`

Expected: `OK`
- [ ] **Step 5: Commit**

```bash
git add apps/python/utils/usd_asset_cache.py apps/python/utils/usd_to_mjcf.py apps/python/tests/test_usd_to_mjcf.py
git commit -m "feat: persist usd conversion mesh cache"
```

### Task 4: Re-Profile And Gate Parallel Writing
**Files:**
- Reference: `apps/python/utils/usd_to_mjcf.py`
- Reference: `apps/python/utils/rl_bundle.py:88-116`
- Optional Create: `apps/python/utils/usd_mesh_writer.py`
- [ ] **Step 1: Re-run the same cold profiles after Tasks 1-3**

Run: `uv run python -m utils.usd_to_mjcf ../../deps/galaxea/object/r1pro/r1pro.usda tmp/perf_check/r1pro.xml`

Expected: wall time materially below the current `~44s` total for `r1pro.usda`, with most duplicate robot OBJ work gone.
- [ ] **Step 2: Decide whether parallelism is still justified**
```python
if cold_robot_seconds <= 25 and cold_scene_seconds <= 45:
    return "stop_after_cache"
return "prototype_process_pool"
```
- [ ] **Step 3: If still too slow, parallelize only pure file generation**
```python
payloads = collect_mesh_payloads_serially(stage)
with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 1, 4)) as pool:
    list(pool.map(write_mesh_payload, payloads))
```
- [ ] **Step 4: Verify identical XML and mesh counts on the parallel path**

Run: `uv run python -m unittest tests.test_usd_to_mjcf -v`

Expected: `OK`
- [ ] **Step 5: Commit**

```bash
git add apps/python/utils/usd_to_mjcf.py apps/python/utils/usd_mesh_writer.py apps/python/tests/test_usd_to_mjcf.py
git commit -m "feat: parallelize usd mesh file generation"
```
