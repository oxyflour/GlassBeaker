# Genie Sim Mitsuba Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist each generated Genie Sim scene as a frozen local bundle with `scene.usda`, then let the preview panel trigger Mitsuba CUDA renders from that bundle without re-executing scene code.

**Architecture:** Keep `execute_scene_code()` as the single scene-generation path, but split bundle persistence and Mitsuba rendering into focused helper modules so the runtime stays composable and under the repo's file-size limit. Extend the FastAPI router with bundle render/artifact endpoints, then add a dedicated render panel on the web page that manages render state independently from the existing Three.js preview.

**Tech Stack:** Python 3.12, FastAPI, `unittest`, Pixar USD (`pxr`), Mitsuba 3, Next.js, React 19, `tsx --test`

---

### Task 1: Bundle Persistence Runtime

**Files:**
- Create: `apps/python/utils/genie_sim_bundle.py`
- Modify: `apps/python/utils/genie_sim_runtime.py`
- Modify: `apps/python/tests/test_genie_sim_runtime.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_execute_scene_code_returns_bundle_fields(self):
    result = runtime.execute_scene_code(SCENE_CODE, assets_root=self.assets_root)
    self.assertIn("bundleId", result)
    self.assertTrue(Path(result["sceneUsdaPath"]).exists())

def test_execute_scene_code_writes_shape_and_manifest(self):
    result = runtime.execute_scene_code(SCENE_CODE, assets_root=self.assets_root)
    bundle_dir = Path(result["sceneUsdaPath"]).parent
    self.assertTrue((bundle_dir / "shape.json").exists())
    self.assertTrue((bundle_dir / "manifest.json").exists())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_genie_sim_runtime -v`
Expected: FAIL because bundle metadata and artifact files do not exist yet.

- [ ] **Step 3: Write the minimal persistence implementation**

```python
bundle_dir = create_bundle_dir(resolve_repo_root())
shape_path = write_shape_payload(bundle_dir, layout_info, scene_data)
scene_usda_path = export_scene_usda(bundle_dir, layout_info)
manifest_path = write_manifest(bundle_dir, manifest)
```

- [ ] **Step 4: Re-run the runtime tests**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_genie_sim_runtime -v`
Expected: PASS for new bundle fields and artifact creation.

### Task 2: USDA Metadata And Frozen Render Helpers

**Files:**
- Create: `apps/python/utils/genie_sim_mitsuba.py`
- Modify: `apps/python/utils/genie_sim_bundle.py`
- Modify: `apps/python/tests/test_genie_sim_runtime.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_exported_usda_has_expected_stage_metadata(self):
    result = runtime.execute_scene_code(SCENE_CODE, assets_root=self.assets_root)
    stage = Usd.Stage.Open(result["sceneUsdaPath"])
    self.assertEqual(stage.GetMetadata("upAxis"), "Z")
    self.assertEqual(stage.GetMetadata("metersPerUnit"), 1.0)

def test_render_bundle_rejects_missing_bundle_id(self):
    with self.assertRaises(FileNotFoundError):
        runtime.render_scene_bundle("missing-bundle")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_genie_sim_runtime -v`
Expected: FAIL because USDA metadata is not exported and bundle render lookup does not exist.

- [ ] **Step 3: Implement frozen render helpers**

```python
def render_scene_bundle(bundle_id: str) -> dict[str, Any]:
    manifest = load_manifest(bundle_id)
    shape = load_shape_payload(bundle_id)
    return render_shape_bundle(manifest, shape, variant="cuda_ad_rgb")
```

- [ ] **Step 4: Re-run the runtime tests**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_genie_sim_runtime -v`
Expected: PASS with missing-bundle lookup covered and USDA metadata asserted.

### Task 3: Genie Sim API Surface

**Files:**
- Modify: `apps/python/api/genie_sim.py`
- Create: `apps/python/tests/test_genie_sim_api.py`

- [ ] **Step 1: Write the failing API tests**

```python
def test_execute_returns_bundle_fields(self):
    response = client.post("/api/genie_sim/execute", json={"code": "..."})
    self.assertEqual(response.status_code, 200)
    self.assertIn("bundleId", response.json())

def test_render_returns_404_for_unknown_bundle(self):
    response = client.post("/api/genie_sim/render", json={"bundle_id": "missing"})
    self.assertEqual(response.status_code, 404)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_genie_sim_api -v`
Expected: FAIL because `/render` and artifact serving do not exist yet.

- [ ] **Step 3: Implement the router changes**

```python
@router.post("/render")
async def render(body: RenderRequest) -> dict: ...

@router.get("/artifacts/{bundle_id}/{name}")
async def artifact(bundle_id: str, name: str) -> FileResponse: ...
```

- [ ] **Step 4: Re-run the API tests**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_genie_sim_api -v`
Expected: PASS for bundle fields, stubbed render success, missing bundle `404`, and invalid artifact rejection.

### Task 4: Web Render State And Panel

**Files:**
- Create: `apps/web/components/genie-sim/render-panel.tsx`
- Create: `apps/web/components/genie-sim/render-panel.test.tsx`
- Modify: `apps/web/components/genie-sim/scene-types.ts`
- Modify: `apps/web/components/genie-sim/scene-state.ts`
- Modify: `apps/web/components/genie-sim/index.ts`
- Modify: `apps/web/app/demo/agent-genie-sim/page.tsx`

- [ ] **Step 1: Write the failing UI tests**

```tsx
test("render button is disabled without a bundle id", () => {
  const html = renderToStaticMarkup(<RenderPanel scene={null} renderStatus="idle" />)
  assert.match(html, /disabled/)
})

test("render error is shown when status is error", () => {
  const html = renderToStaticMarkup(<RenderPanel scene={scene} renderStatus="error" renderError="CUDA unavailable" />)
  assert.match(html, /CUDA unavailable/)
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm --dir apps/web test -- components/genie-sim/render-panel.test.tsx`
Expected: FAIL because the render panel component and render-specific scene state do not exist.

- [ ] **Step 3: Implement the render panel and state reset flow**

```tsx
type SceneRenderResult = { primaryImageUrl: string; views: { name: string; url: string }[] }
const [renderStatus, setRenderStatus] = useState<RenderStatus>("idle")
setSceneData(data) // also clears old renderResult and renderError
```

- [ ] **Step 4: Re-run the UI tests**

Run: `pnpm --dir apps/web test -- components/genie-sim/render-panel.test.tsx`
Expected: PASS for disabled, loading, error, and success render states.

### Task 5: Regression Verification

**Files:**
- Test: `apps/python/tests/test_genie_sim_runtime.py`
- Test: `apps/python/tests/test_genie_sim_api.py`
- Test: `apps/web/components/genie-sim/render-panel.test.tsx`

- [ ] **Step 1: Run backend Genie Sim tests**

Run: `uv run --project apps/python python -m unittest apps.python.tests.test_genie_sim_runtime apps.python.tests.test_genie_sim_api -v`
Expected: PASS

- [ ] **Step 2: Run frontend Genie Sim tests**

Run: `pnpm --dir apps/web test -- components/genie-sim/render-panel.test.tsx components/genie-sim/scene-math.test.ts`
Expected: PASS

- [ ] **Step 3: Manual smoke**

Verify in `/demo/agent-genie-sim`:
- the generated scene shows a `sceneUsdaPath`
- the bundle directory contains `scene.usda`, `shape.json`, `manifest.json`
- clicking `Render` shows a Mitsuba image without clearing the Three.js preview
- an unknown bundle or CUDA failure surfaces a short readable error
