# Kokoro VMF Noise Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add adaptive vMF lobe width, weighted vMF mixtures, and footprint-aware patch targets so Kokoro can reduce isolated sparkle while preserving the visible 1024 SPP light bands.

**Architecture:** Keep the current `normal` target as the stable baseline and add new behavior behind explicit CLI/scene/checkpoint metadata flags. First support deterministic adaptive `kappa` in the existing single-lobe BSDF, then add a first-class `normal_mixture` output layout with weighted vMF components, then train that layout from local patch normal distributions. Defaults stay unchanged until render panels show better band continuity and lower noise.

**Tech Stack:** Python 3.12 via `uv --project apps/python`, PyTorch for target generation/training, Mitsuba Python BSDF plugin, `unittest` for regression coverage, JSON/NPZ checkpoint metadata for renderer handoff.

---

## Scope And Acceptance

This plan implements all three requested lines of work:

- Adaptive kappa: a local vMF concentration policy that can lower `kappa` near facet boundaries or use learned patch concentration for mixture components.
- Mixture vMF: a renderer path where the neural BSDF samples and evaluates a weighted sum of vMF lobes instead of one reflected axis.
- Footprint-aware target: a patch normal distribution target so one shaded point can represent a small surface footprint instead of a single microfacet normal.

Acceptance criteria:

- Existing default command still emits the same single-lobe normal-mode scene unless the new flags are passed.
- `--adaptive-kappa-mode facet-boundary` changes the BSDF peak width near known rotated-pyramid facet boundaries and keeps fixed-kappa behavior elsewhere.
- `--target-mode normal_mixture --mixture-component-count 4` exports metadata and output dimensions that Mitsuba can load.
- The mixture BSDF `sample`, `eval`, `pdf`, and `eval_pdf` are internally consistent for scalar Mitsuba tests.
- A smoke render completes with `--target-mode normal_mixture --mixture-patch-radius-m 0.001 --mixture-patch-samples 64 --adaptive-kappa-mode mixture-concentration`.
- Full Kokoro tests pass with `uv run --project apps/python python -m unittest discover packages/kokoro/tests`.

## File Map

- Modify `packages/kokoro/kokoro/brdf.py`: add `normal_mixture` target mode, model output layout, mixture metadata export, and training target wiring.
- Create `packages/kokoro/kokoro/vmf_targets.py`: pure patch-normal clustering and concentration helpers. This is a pure target codec used by training and tests, so extracting it keeps `brdf.py` from growing another dense block.
- Modify `packages/kokoro/kokoro/mitsuba_neural_bsdf.py`: read new metadata, decode mixture outputs, sample/evaluate weighted vMF components, and compute adaptive kappa.
- Modify `packages/kokoro/kokoro/mitsuba_scene.py`: pass new BSDF properties into scene dictionaries.
- Modify `packages/kokoro/run_demo.py`: add CLI flags and wire dataset/export/scene parameters.
- Modify `packages/kokoro/README.md`: document the new modes, recommended commands, and promotion criteria.
- Add or modify tests in `packages/kokoro/tests/test_vmf_targets.py`, `packages/kokoro/tests/test_brdf_surrogate.py`, `packages/kokoro/tests/test_mitsuba_neural_bsdf.py`, `packages/kokoro/tests/test_mitsuba_scene.py`, and `packages/kokoro/tests/test_run_demo.py`.

## Public Flags And Metadata

CLI flags to add:

```text
--adaptive-kappa-mode fixed|facet-boundary|mixture-concentration
--min-lobe-kappa FLOAT
--facet-boundary-width FLOAT
--mixture-component-count INT
--mixture-patch-radius-m FLOAT
--mixture-patch-samples INT
--mixture-clustering-iterations INT
```

Constants to add in `packages/kokoro/run_demo.py`:

```python
DEFAULT_ADAPTIVE_KAPPA_MODE = "fixed"
DEFAULT_MIN_LOBE_KAPPA = 512.0
DEFAULT_FACET_BOUNDARY_WIDTH = 0.18
DEFAULT_MIXTURE_COMPONENT_COUNT = 1
DEFAULT_MIXTURE_PATCH_RADIUS_M = 0.0
DEFAULT_MIXTURE_PATCH_SAMPLES = 1
DEFAULT_MIXTURE_CLUSTERING_ITERATIONS = 6
```

Checkpoint metadata to add when relevant:

```json
{
  "output_layout": "direction" or "normal_mixture",
  "mixture_component_count": 4,
  "mixture_patch_radius_m": 0.001,
  "mixture_patch_samples": 64,
  "mixture_clustering_iterations": 6
}
```

Scene BSDF properties to add:

```json
{
  "adaptive_kappa_mode": "fixed",
  "min_lobe_kappa": 512.0,
  "facet_boundary_width": 0.18
}
```

---

### Task 1: Add Adaptive Kappa Scene Properties

**Files:**
- Modify: `packages/kokoro/tests/test_mitsuba_scene.py`
- Modify: `packages/kokoro/kokoro/mitsuba_scene.py`

- [ ] **Step 1: Write failing scene-builder tests**

Add this test to `MitsubaSceneTest`:

```python
def test_scene_dict_passes_adaptive_kappa_controls_to_neural_bsdf(self) -> None:
    scene = build_kokoro_scene_dict(
        checkpoint_path=Path("packages/kokoro/tmp/kokoro_brdf.npz"),
        hdr_path=REPO_ROOT / "apps" / "web" / "public" / "studio_small_03_1k.hdr",
        adaptive_kappa_mode="facet-boundary",
        min_lobe_kappa=384.0,
        facet_boundary_width=0.12,
    )

    bsdf = scene["surface"]["bsdf"]
    self.assertEqual(bsdf["adaptive_kappa_mode"], "facet-boundary")
    self.assertEqual(bsdf["min_lobe_kappa"], 384.0)
    self.assertEqual(bsdf["facet_boundary_width"], 0.12)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
uv run --project apps/python python -m unittest packages/kokoro/tests/test_mitsuba_scene.py
```

Expected: FAIL with `TypeError: build_kokoro_scene_dict() got an unexpected keyword argument 'adaptive_kappa_mode'`.

- [ ] **Step 3: Add scene builder parameters**

Modify the `build_kokoro_scene_dict` signature in `packages/kokoro/kokoro/mitsuba_scene.py`:

```python
    adaptive_kappa_mode: str = "fixed",
    min_lobe_kappa: float = 512.0,
    facet_boundary_width: float = 0.18,
```

Add these fields to `scene["surface"]["bsdf"]`:

```python
                "adaptive_kappa_mode": adaptive_kappa_mode,
                "min_lobe_kappa": float(min_lobe_kappa),
                "facet_boundary_width": float(facet_boundary_width),
```

Apply the same three keyword arguments to `build_height_field_reference_scene_dict` only if the reference BSDF gains adaptive width later. For Task 1, keep the reference unchanged because the adaptive kappa renderer change is neural-BSDF specific.

- [ ] **Step 4: Run scene tests**

Run:

```powershell
uv run --project apps/python python -m unittest packages/kokoro/tests/test_mitsuba_scene.py
```

Expected: OK.

- [ ] **Step 5: Commit**

```powershell
git add packages/kokoro/tests/test_mitsuba_scene.py packages/kokoro/kokoro/mitsuba_scene.py
git commit -m "feat(kokoro): expose adaptive kappa scene controls"
```

### Task 2: Implement Single-Lobe Facet-Boundary Adaptive Kappa

**Files:**
- Modify: `packages/kokoro/tests/test_mitsuba_neural_bsdf.py`
- Modify: `packages/kokoro/kokoro/mitsuba_neural_bsdf.py`

- [ ] **Step 1: Write failing Mitsuba BSDF test**

Add this test to `MitsubaNeuralBsdfTest`:

```python
def test_facet_boundary_adaptive_kappa_broadens_boundary_pdf(self) -> None:
    import mitsuba as mi

    mi.set_variant("scalar_rgb")
    register_kokoro_bsdf(mi)
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = Path(temp_dir) / "facet_boundary.npz"
        model = KokoroBrdfNet(hidden_dim=5, input_dim=8)
        with torch.no_grad():
            for layer in model.layers:
                layer.weight.zero_()
                layer.bias.zero_()
            model.layers[-1].bias[:] = torch.tensor([0.0, 0.0, 1.0])
        export_surrogate_npz(
            model,
            checkpoint,
            width_m=0.1,
            depth_m=0.1,
            radial_cell_feature_period_m=500e-6,
            radial_cell_facet_features=True,
            include_position_features=True,
            include_incident_features=False,
            target_mode="normal",
        )

        def load_pdf(adaptive_mode: str, point_x: float, point_y: float) -> float:
            scene = mi.load_dict({
                "type": "scene",
                "shape": {
                    "type": "rectangle",
                    "bsdf": {
                        "type": "kokoro_neural_reflector",
                        "checkpoint": str(checkpoint),
                        "lobe_kappa": 2048.0,
                        "min_lobe_kappa": 256.0,
                        "facet_boundary_width": 0.2,
                        "adaptive_kappa_mode": adaptive_mode,
                    },
                },
            })
            si = mi.SurfaceInteraction3f()
            si.p = mi.Point3f(point_x, point_y, 0.0)
            si.wi = mi.Vector3f(0.0, 0.0, 1.0)
            off_peak = mi.Vector3f(0.25, 0.0, 0.9682458)
            return float(scene.shapes()[0].bsdf().pdf(mi.BSDFContext(), si, off_peak, True))

        boundary_pdf = load_pdf("facet-boundary", 125e-6, 125e-6)
        fixed_pdf = load_pdf("fixed", 125e-6, 125e-6)
        interior_pdf = load_pdf("facet-boundary", 225e-6, 25e-6)

    self.assertGreater(boundary_pdf, fixed_pdf * 10.0)
    self.assertLess(interior_pdf, boundary_pdf)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
uv run --project apps/python python -m unittest packages/kokoro/tests/test_mitsuba_neural_bsdf.py
```

Expected: FAIL because `adaptive_kappa_mode` is ignored and `boundary_pdf` is not higher than fixed mode.

- [ ] **Step 3: Read BSDF props**

In `KokoroNeuralReflector.__init__`, add:

```python
self.adaptive_kappa_mode = str(props.get("adaptive_kappa_mode", "fixed"))
self.min_lobe_kappa = float(props.get("min_lobe_kappa", min(self.lobe_kappa, 512.0)))
self.facet_boundary_width = float(props.get("facet_boundary_width", 0.18))
if self.adaptive_kappa_mode not in {"fixed", "facet-boundary", "mixture-concentration"}:
    raise ValueError("adaptive_kappa_mode must be fixed, facet-boundary, or mixture-concentration")
```

- [ ] **Step 4: Replace direct `self.lobe_kappa` use with local kappa**

Change `sample`, `eval`, and `pdf` to ask `_target_lobe` for a kappa:

```python
axis, cone_cos, phase, kappa = self._target_lobe(si, dr)
raw = mi.Frame3f(target).to_world(mi.warp.square_to_von_mises_fisher(sample2, kappa))
```

Update `_pdf_for_lobe` and `_pdf_for_target` signatures to accept `kappa`.

- [ ] **Step 5: Add adaptive kappa helper**

Add this method near `_target_lobe`:

```python
def _local_kappa(self, position_features, concentration, dr):
    if self.adaptive_kappa_mode == "fixed":
        return self.lobe_kappa
    if self.adaptive_kappa_mode == "mixture-concentration":
        tightness = dr.clip(concentration, 0.0, 1.0)
        return self.min_lobe_kappa + (self.lobe_kappa - self.min_lobe_kappa) * tightness * tightness
    radial_start = 2 + (2 if self.local_feature_period_m is not None else 0)
    if self.radial_cell_feature_period_m is None or len(position_features) < radial_start + 2:
        return self.lobe_kappa
    rotated_x = position_features[radial_start]
    rotated_y = position_features[radial_start + 1]
    boundary_distance = dr.abs(dr.abs(rotated_x) - dr.abs(rotated_y))
    width = max(self.facet_boundary_width, 1e-6)
    blend = dr.clip(boundary_distance / width, 0.0, 1.0)
    smooth = blend * blend * (3.0 - 2.0 * blend)
    return self.min_lobe_kappa + (self.lobe_kappa - self.min_lobe_kappa) * smooth
```

Update `_target_lobe`:

```python
kappa = self._local_kappa(position_features, 1.0, dr)
return axis, cone_cos, phase, kappa
```

- [ ] **Step 6: Run BSDF tests**

Run:

```powershell
uv run --project apps/python python -m unittest packages/kokoro/tests/test_mitsuba_neural_bsdf.py
```

Expected: OK.

- [ ] **Step 7: Commit**

```powershell
git add packages/kokoro/tests/test_mitsuba_neural_bsdf.py packages/kokoro/kokoro/mitsuba_neural_bsdf.py
git commit -m "feat(kokoro): add facet-boundary adaptive kappa"
```

### Task 3: Add Patch Normal Mixture Target Codec

**Files:**
- Create: `packages/kokoro/kokoro/vmf_targets.py`
- Create: `packages/kokoro/tests/test_vmf_targets.py`

- [ ] **Step 1: Write failing pure target tests**

Create `packages/kokoro/tests/test_vmf_targets.py`:

```python
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from kokoro.height_field import compile_height_program
from kokoro.vmf_targets import patch_normal_mixture_targets


class VmfTargetsTest(unittest.TestCase):
    def test_patch_normal_mixture_returns_weighted_normal_components(self) -> None:
        program = compile_height_program(
            """
def height(x, y):
    return radial_rotated_pyramid_height(x, y, period_m=500e-6, amplitude_m=150e-6)
"""
        )

        targets = patch_normal_mixture_targets(
            program,
            torch.tensor([0.0, 0.001], dtype=torch.float32),
            torch.tensor([0.0, 0.001], dtype=torch.float32),
            patch_radius_m=0.001,
            patch_sample_count=128,
            component_count=4,
            iteration_count=5,
            seed=11,
        )

        self.assertEqual(targets.shape, (2, 20))
        mixture = targets.reshape(2, 4, 5)
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(mixture[:, :, :3], dim=2), torch.ones((2, 4)), atol=1e-5))
        self.assertTrue(torch.allclose(mixture[:, :, 3].sum(dim=1), torch.ones(2), atol=1e-5))
        self.assertTrue(torch.all(mixture[:, :, 4] >= 0.0))
        self.assertTrue(torch.all(mixture[:, :, 4] <= 1.0))

    def test_patch_normal_mixture_is_deterministic_for_seed(self) -> None:
        program = compile_height_program("def height(x, y):\n    return 0.0005 * torch.sin(800.0 * x)\n")
        x = torch.tensor([0.0, 0.002], dtype=torch.float32)
        y = torch.tensor([0.0, 0.001], dtype=torch.float32)

        first = patch_normal_mixture_targets(
            program,
            x,
            y,
            patch_radius_m=0.001,
            patch_sample_count=96,
            component_count=2,
            iteration_count=4,
            seed=7,
        )
        second = patch_normal_mixture_targets(
            program,
            x,
            y,
            patch_radius_m=0.001,
            patch_sample_count=96,
            component_count=2,
            iteration_count=4,
            seed=7,
        )

        self.assertTrue(torch.allclose(first, second, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
uv run --project apps/python python -m unittest packages/kokoro/tests/test_vmf_targets.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'kokoro.vmf_targets'`.

- [ ] **Step 3: Implement the pure codec**

Create `packages/kokoro/kokoro/vmf_targets.py`:

```python
from __future__ import annotations

import torch

from .height_field import HeightProgram, surface_normals


def patch_normal_mixture_targets(
    program: HeightProgram,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    patch_radius_m: float,
    patch_sample_count: int,
    component_count: int,
    iteration_count: int,
    seed: int = 0,
) -> torch.Tensor:
    if patch_sample_count <= 0:
        raise ValueError("patch_sample_count must be positive")
    if component_count <= 0:
        raise ValueError("component_count must be positive")
    if component_count > patch_sample_count:
        raise ValueError("component_count must not exceed patch_sample_count")
    if patch_radius_m < 0.0:
        raise ValueError("patch_radius_m must be non-negative")

    count = x.shape[0]
    sample_count = int(patch_sample_count)
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))
    if patch_radius_m == 0.0:
        dx = torch.zeros((count, sample_count), dtype=x.dtype, device=x.device)
        dy = torch.zeros_like(dx)
    else:
        radius = float(patch_radius_m)
        dx = (torch.rand((count, sample_count), generator=gen, dtype=x.dtype, device=x.device) * 2.0 - 1.0) * radius
        dy = (torch.rand((count, sample_count), generator=gen, dtype=x.dtype, device=x.device) * 2.0 - 1.0) * radius

    sample_x = (x[:, None] + dx).reshape(-1)
    sample_y = (y[:, None] + dy).reshape(-1)
    normals = surface_normals(program, sample_x, sample_y).reshape(count, sample_count, 3)
    return _cluster_unit_vectors(normals, component_count=int(component_count), iteration_count=int(iteration_count)).reshape(count, -1)


def _cluster_unit_vectors(vectors: torch.Tensor, *, component_count: int, iteration_count: int) -> torch.Tensor:
    count, sample_count, _ = vectors.shape
    seed_indices = torch.linspace(0, sample_count - 1, component_count, device=vectors.device).round().long()
    centers = torch.nn.functional.normalize(vectors[:, seed_indices, :], dim=2)
    assignments = torch.zeros((count, sample_count), dtype=torch.long, device=vectors.device)

    for _ in range(max(1, int(iteration_count))):
        scores = torch.matmul(vectors, centers.transpose(1, 2))
        assignments = torch.argmax(scores, dim=2)
        updated = []
        for component in range(component_count):
            mask = assignments == component
            raw_count = mask.sum(dim=1, keepdim=True)
            safe_count = raw_count.clamp(min=1).to(vectors.dtype)
            mean = (vectors * mask[:, :, None].to(vectors.dtype)).sum(dim=1) / safe_count
            fallback = centers[:, component, :]
            updated.append(torch.where(raw_count > 0, torch.nn.functional.normalize(mean, dim=1), fallback))
        centers = torch.stack(updated, dim=1)

    scores = torch.matmul(vectors, centers.transpose(1, 2))
    assignments = torch.argmax(scores, dim=2)
    components = []
    for component in range(component_count):
        mask = assignments == component
        weight_count = mask.sum(dim=1, keepdim=True).to(vectors.dtype)
        weight = weight_count / float(sample_count)
        weighted = vectors * mask[:, :, None].to(vectors.dtype)
        mean = weighted.sum(dim=1)
        length = torch.linalg.vector_norm(mean, dim=1, keepdim=True)
        normal = torch.where(length > 1e-6, mean / length.clamp(min=1e-6), centers[:, component, :])
        concentration = torch.where(weight_count > 0, (length / weight_count.clamp(min=1.0)).clamp(0.0, 1.0), torch.zeros_like(length))
        components.append(torch.cat([normal, weight, concentration], dim=1))
    result = torch.stack(components, dim=1)
    order = torch.argsort(result[:, :, 3], dim=1, descending=True)
    gather_index = order[:, :, None].expand(-1, -1, 5)
    return torch.gather(result, 1, gather_index)
```

- [ ] **Step 4: Run pure target tests**

Run:

```powershell
uv run --project apps/python python -m unittest packages/kokoro/tests/test_vmf_targets.py
```

Expected: OK.

- [ ] **Step 5: Commit**

```powershell
git add packages/kokoro/kokoro/vmf_targets.py packages/kokoro/tests/test_vmf_targets.py
git commit -m "feat(kokoro): add patch normal mixture targets"
```

### Task 4: Wire Normal Mixture Targets Into BRDF Training

**Files:**
- Modify: `packages/kokoro/tests/test_brdf_surrogate.py`
- Modify: `packages/kokoro/kokoro/brdf.py`

- [ ] **Step 1: Write failing dataset and export tests**

Add these imports in `test_brdf_surrogate.py` if missing:

```python
from kokoro.brdf import BrdfTrainingConfig, KokoroBrdfNet, build_brdf_dataset, export_surrogate_npz, load_npz_surrogate
```

Add tests:

```python
def test_normal_mixture_dataset_targets_patch_normal_distribution(self) -> None:
    program = compile_height_program(
        """
def height(x, y):
    return radial_rotated_pyramid_height(x, y, period_m=500e-6, amplitude_m=150e-6)
"""
    )

    dataset = build_brdf_dataset(
        program,
        sample_count=10,
        width_m=0.10,
        depth_m=0.10,
        seed=8,
        target_mode="normal_mixture",
        mixture_component_count=4,
        mixture_patch_radius_m=0.001,
        mixture_patch_sample_count=64,
        mixture_clustering_iterations=4,
    )

    self.assertEqual(dataset.targets.shape, (10, 20))
    self.assertEqual(dataset.target_mode, "normal_mixture")
    self.assertFalse(dataset.include_incident_features)
    mixture = dataset.targets.reshape(10, 4, 5)
    self.assertTrue(torch.allclose(torch.linalg.vector_norm(mixture[:, :, :3], dim=2), torch.ones((10, 4)), atol=1e-5))
    self.assertTrue(torch.allclose(mixture[:, :, 3].sum(dim=1), torch.ones(10), atol=1e-5))

def test_npz_export_records_normal_mixture_metadata(self) -> None:
    model = KokoroBrdfNet(hidden_dim=4, input_dim=8, output_dim=20, output_layout="normal_mixture", mixture_component_count=4)
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = Path(temp_dir) / "normal_mixture.npz"
        export_surrogate_npz(
            model,
            checkpoint,
            width_m=0.10,
            depth_m=0.10,
            include_incident_features=False,
            target_mode="normal_mixture",
            mixture_component_count=4,
            mixture_patch_radius_m=0.001,
            mixture_patch_sample_count=64,
            mixture_clustering_iterations=4,
        )
        loaded = load_npz_surrogate(checkpoint)

    self.assertEqual(loaded.metadata["target_mode"], "normal_mixture")
    self.assertEqual(loaded.metadata["output_layout"], "normal_mixture")
    self.assertEqual(loaded.metadata["mixture_component_count"], 4)
    self.assertEqual(loaded.metadata["mixture_patch_radius_m"], 0.001)
    self.assertEqual(loaded.metadata["mixture_patch_sample_count"], 64)
    self.assertEqual(loaded.metadata["mixture_clustering_iterations"], 4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
uv run --project apps/python python -m unittest packages/kokoro/tests/test_brdf_surrogate.py
```

Expected: FAIL with `ValueError: target_mode must be 'reflection' or 'normal'` or constructor argument errors.

- [ ] **Step 3: Extend `BrdfDataset` fields**

In `packages/kokoro/kokoro/brdf.py`, add dataclass fields:

```python
    output_layout: str = "direction"
    mixture_component_count: int = 1
    mixture_patch_radius_m: float = 0.0
    mixture_patch_sample_count: int = 1
    mixture_clustering_iterations: int = 6
```

- [ ] **Step 4: Extend `KokoroBrdfNet` layout decoding**

Add constructor parameters:

```python
        output_layout: str = "direction",
        mixture_component_count: int = 1,
```

Add validation:

```python
if output_layout not in {"direction", "normal_mixture"}:
    raise ValueError("output_layout must be 'direction' or 'normal_mixture'")
if int(mixture_component_count) <= 0:
    raise ValueError("mixture_component_count must be positive")
self.output_layout = output_layout
self.mixture_component_count = int(mixture_component_count)
if self.output_layout == "normal_mixture" and self.output_dim != self.mixture_component_count * 5:
    raise ValueError("normal_mixture output_dim must equal mixture_component_count * 5")
```

At the top of `forward`, after `raw = self.layers[-1](x)`, branch:

```python
if self.output_layout == "normal_mixture":
    raw_mix = raw.reshape(raw.shape[0], self.mixture_component_count, 5)
    normals = torch.nn.functional.normalize(raw_mix[:, :, :3], dim=2)
    weights = torch.softmax(raw_mix[:, :, 3], dim=1).unsqueeze(2)
    concentration = torch.sigmoid(raw_mix[:, :, 4:5])
    return torch.cat([normals, weights, concentration], dim=2).reshape(raw.shape[0], self.output_dim)
```

Apply the same output-layout branch in `NpzSurrogate.predict` after the final affine layer. Use NumPy equivalents:

```python
if self.metadata.get("output_layout", "direction") == "normal_mixture":
    component_count = int(self.metadata["mixture_component_count"])
    raw_mix = x.reshape(x.shape[0], component_count, 5)
    normals = raw_mix[:, :, :3]
    normals = normals / np.maximum(np.linalg.norm(normals, axis=2, keepdims=True), 1e-6)
    logits = raw_mix[:, :, 3:4]
    logits = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits) / np.maximum(np.exp(logits).sum(axis=1, keepdims=True), 1e-6)
    concentration = 1.0 / (1.0 + np.exp(-raw_mix[:, :, 4:5]))
    return np.concatenate([normals, weights, concentration], axis=2).reshape(x.shape[0], -1)
```

- [ ] **Step 5: Build normal mixture dataset targets**

Import the new codec:

```python
from .vmf_targets import patch_normal_mixture_targets
```

Extend `build_brdf_dataset` signature:

```python
    mixture_component_count: int = 1,
    mixture_patch_radius_m: float = 0.0,
    mixture_patch_sample_count: int = 1,
    mixture_clustering_iterations: int = 6,
```

Extend validation in both `build_brdf_dataset` and `export_surrogate_npz`:

```python
if target_mode not in {"reflection", "normal", "normal_mixture"}:
    raise ValueError("target_mode must be 'reflection', 'normal', or 'normal_mixture'")
```

Add target branch before `target_mode == "normal"`:

```python
if target_mode == "normal_mixture":
    wo = patch_normal_mixture_targets(
        program,
        surface.positions[:, 0],
        surface.positions[:, 1],
        patch_radius_m=mixture_patch_radius_m,
        patch_sample_count=mixture_patch_sample_count,
        component_count=mixture_component_count,
        iteration_count=mixture_clustering_iterations,
        seed=int(seed) + 71,
    )
elif target_mode == "normal":
    wo = surface.normals
```

Set:

```python
uses_incident = False if target_mode == "normal_mixture" else (target_mode == "reflection" if include_incident_features is None else bool(include_incident_features))
output_layout = "normal_mixture" if target_mode == "normal_mixture" else "direction"
```

Return new metadata fields in `BrdfDataset`.

- [ ] **Step 6: Train with the correct model layout**

In `train_brdf_surrogate`, pass:

```python
        output_layout=dataset.output_layout,
        mixture_component_count=dataset.mixture_component_count,
```

- [ ] **Step 7: Export mixture metadata**

Extend `export_surrogate_npz` signature:

```python
    output_layout: str | None = None,
    mixture_component_count: int = 1,
    mixture_patch_radius_m: float = 0.0,
    mixture_patch_sample_count: int = 1,
    mixture_clustering_iterations: int = 6,
```

Set:

```python
layout = model.output_layout if output_layout is None else output_layout
metadata["output_layout"] = layout
metadata["mixture_component_count"] = int(mixture_component_count)
if layout == "normal_mixture":
    metadata["mixture_patch_radius_m"] = float(mixture_patch_radius_m)
    metadata["mixture_patch_sample_count"] = int(mixture_patch_sample_count)
    metadata["mixture_clustering_iterations"] = int(mixture_clustering_iterations)
```

- [ ] **Step 8: Run BRDF tests**

Run:

```powershell
uv run --project apps/python python -m unittest packages/kokoro/tests/test_brdf_surrogate.py
```

Expected: OK.

- [ ] **Step 9: Commit**

```powershell
git add packages/kokoro/kokoro/brdf.py packages/kokoro/tests/test_brdf_surrogate.py
git commit -m "feat(kokoro): train normal mixture targets"
```

### Task 5: Decode Weighted Normal Mixtures In Mitsuba BSDF

**Files:**
- Modify: `packages/kokoro/tests/test_mitsuba_neural_bsdf.py`
- Modify: `packages/kokoro/kokoro/mitsuba_neural_bsdf.py`

- [ ] **Step 1: Write failing mixture BSDF test**

Add this test:

```python
def test_normal_mixture_checkpoint_samples_weighted_reflection_lobes(self) -> None:
    import mitsuba as mi

    mi.set_variant("scalar_rgb")
    register_kokoro_bsdf(mi)
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = Path(temp_dir) / "normal_mixture.npz"
        model = KokoroBrdfNet(hidden_dim=5, input_dim=2, output_dim=10, output_layout="normal_mixture", mixture_component_count=2)
        with torch.no_grad():
            for layer in model.layers:
                layer.weight.zero_()
                layer.bias.zero_()
            model.layers[-1].bias[:] = torch.tensor([
                0.0, 0.0, 1.0, 2.0, 0.9,
                0.6, 0.0, 0.8, -2.0, 0.9,
            ])
        export_surrogate_npz(
            model,
            checkpoint,
            width_m=0.1,
            depth_m=0.1,
            include_incident_features=False,
            target_mode="normal_mixture",
            mixture_component_count=2,
        )

        scene = mi.load_dict({
            "type": "scene",
            "shape": {
                "type": "rectangle",
                "bsdf": {
                    "type": "kokoro_neural_reflector",
                    "checkpoint": str(checkpoint),
                    "lobe_kappa": 512.0,
                    "min_lobe_kappa": 128.0,
                    "adaptive_kappa_mode": "mixture-concentration",
                },
            },
        })
        bsdf = scene.shapes()[0].bsdf()
        si = mi.SurfaceInteraction3f()
        si.p = mi.Point3f(0.0, 0.0, 0.0)
        si.wi = mi.Vector3f(0.0, 0.0, 1.0)
        ctx = mi.BSDFContext()
        primary = mi.Vector3f(0.0, 0.0, 1.0)
        secondary = mi.Vector3f(0.96, 0.0, 0.28)

        primary_pdf = float(bsdf.pdf(ctx, si, primary, True))
        secondary_pdf = float(bsdf.pdf(ctx, si, secondary, True))
        sample, weight = bsdf.sample(ctx, si, 0.9, mi.Point2f(0.5, 0.5), True)

    self.assertGreater(primary_pdf, secondary_pdf)
    self.assertGreater(float(sample.pdf), 0.0)
    self.assertGreater(float(weight[0]), 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
uv run --project apps/python python -m unittest packages/kokoro/tests/test_mitsuba_neural_bsdf.py
```

Expected: FAIL because the BSDF still decodes only one axis from `raw[:3]`.

- [ ] **Step 3: Read mixture metadata**

In `KokoroNeuralReflector.__init__`, add:

```python
self.output_layout = str(surrogate.metadata.get("output_layout", "direction"))
self.mixture_component_count = int(surrogate.metadata.get("mixture_component_count", 1))
if self.output_layout not in {"direction", "normal_mixture"}:
    raise ValueError("output_layout must be direction or normal_mixture")
```

- [ ] **Step 4: Add normalized mixture decoder**

Add methods:

```python
def _decode_mixture(self, raw, si, dr):
    components = []
    logits = []
    for index in range(self.mixture_component_count):
        offset = index * 5
        norm = dr.rsqrt(dr.maximum(raw[offset] * raw[offset] + raw[offset + 1] * raw[offset + 1] + raw[offset + 2] * raw[offset + 2], 1e-8))
        normal = mi.Vector3f(raw[offset] * norm, raw[offset + 1] * norm, dr.abs(raw[offset + 2] * norm))
        axis = self._reflect(_component(si.wi, 0), _component(si.wi, 1), _component(si.wi, 2), normal, dr)
        logits.append(raw[offset + 3])
        concentration = dr.clip(1.0 / (1.0 + dr.exp(-raw[offset + 4])), 0.0, 1.0)
        components.append((axis, concentration))
    max_logit = logits[0]
    for value in logits[1:]:
        max_logit = dr.maximum(max_logit, value)
    denom = 0.0
    weights = []
    for value in logits:
        weight = dr.exp(value - max_logit)
        weights.append(weight)
        denom = denom + weight
    return [(axis, weight / denom, concentration) for (axis, concentration), weight in zip(components, weights)]
```

Add:

```python
def _single_component(self, si, dr):
    axis, cone_cos, phase, kappa = self._target_lobe(si, dr)
    return [(axis, 1.0, 1.0, cone_cos, phase, kappa)]
```

- [ ] **Step 5: Route `sample`, `eval`, and `pdf` through component lists**

For mixture mode:

```python
def _target_components(self, si, dr):
    if self.output_layout != "normal_mixture":
        return self._single_component(si, dr)
    position_features = self._position_features(si, dr)
    raw = self._eval_mlp(self._features(si, position_features, dr), dr)
    components = []
    for axis, weight, concentration in self._decode_mixture(raw, si, dr):
        kappa = self._local_kappa(position_features, concentration, dr)
        components.append((axis, weight, 1.0, 0.0, kappa))
    return components
```

Update `_pdf_for_components`:

```python
def _pdf_for_components(self, wo, components, dr, active):
    pdf = 0.0
    for axis, weight, cone_cos, phase, kappa in components:
        pdf = pdf + weight * self._pdf_for_lobe(wo, axis, cone_cos, phase, kappa, dr, active)
    return pdf
```

In `sample`, select a component by cumulative weight using `sample1`, then sample vMF around that component. Keep `eval(ctx, si, wo)` equal to `reflectance * pdf(ctx, si, wo)`.

- [ ] **Step 6: Run BSDF tests**

Run:

```powershell
uv run --project apps/python python -m unittest packages/kokoro/tests/test_mitsuba_neural_bsdf.py
```

Expected: OK.

- [ ] **Step 7: Commit**

```powershell
git add packages/kokoro/kokoro/mitsuba_neural_bsdf.py packages/kokoro/tests/test_mitsuba_neural_bsdf.py
git commit -m "feat(kokoro): render weighted normal vmf mixtures"
```

### Task 6: Wire CLI Flags Through `run_demo.py`

**Files:**
- Modify: `packages/kokoro/tests/test_run_demo.py`
- Modify: `packages/kokoro/run_demo.py`

- [ ] **Step 1: Write failing default tests**

Add imports:

```python
    DEFAULT_ADAPTIVE_KAPPA_MODE,
    DEFAULT_FACET_BOUNDARY_WIDTH,
    DEFAULT_MIN_LOBE_KAPPA,
    DEFAULT_MIXTURE_CLUSTERING_ITERATIONS,
    DEFAULT_MIXTURE_COMPONENT_COUNT,
    DEFAULT_MIXTURE_PATCH_RADIUS_M,
    DEFAULT_MIXTURE_PATCH_SAMPLES,
```

Add test:

```python
def test_default_vmf_noise_controls_are_available_but_not_promoted(self) -> None:
    self.assertEqual(DEFAULT_ADAPTIVE_KAPPA_MODE, "fixed")
    self.assertEqual(DEFAULT_MIN_LOBE_KAPPA, 512.0)
    self.assertEqual(DEFAULT_FACET_BOUNDARY_WIDTH, 0.18)
    self.assertEqual(DEFAULT_MIXTURE_COMPONENT_COUNT, 1)
    self.assertEqual(DEFAULT_MIXTURE_PATCH_RADIUS_M, 0.0)
    self.assertEqual(DEFAULT_MIXTURE_PATCH_SAMPLES, 1)
    self.assertEqual(DEFAULT_MIXTURE_CLUSTERING_ITERATIONS, 6)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
uv run --project apps/python python -m unittest packages/kokoro/tests/test_run_demo.py
```

Expected: FAIL with import errors for the new constants.

- [ ] **Step 3: Add constants and parser args**

Add constants from the Public Flags section.

Add parser arguments:

```python
parser.add_argument("--adaptive-kappa-mode", choices=["fixed", "facet-boundary", "mixture-concentration"], default=DEFAULT_ADAPTIVE_KAPPA_MODE)
parser.add_argument("--min-lobe-kappa", type=float, default=DEFAULT_MIN_LOBE_KAPPA)
parser.add_argument("--facet-boundary-width", type=float, default=DEFAULT_FACET_BOUNDARY_WIDTH)
parser.add_argument("--mixture-component-count", type=int, default=DEFAULT_MIXTURE_COMPONENT_COUNT)
parser.add_argument("--mixture-patch-radius-m", type=float, default=DEFAULT_MIXTURE_PATCH_RADIUS_M)
parser.add_argument("--mixture-patch-samples", type=int, default=DEFAULT_MIXTURE_PATCH_SAMPLES)
parser.add_argument("--mixture-clustering-iterations", type=int, default=DEFAULT_MIXTURE_CLUSTERING_ITERATIONS)
```

Extend `--target-mode` choices:

```python
parser.add_argument("--target-mode", choices=["reflection", "normal", "normal_mixture"], default=DEFAULT_TARGET_MODE)
```

- [ ] **Step 4: Wire dataset, export, and scene**

Pass to `build_brdf_dataset`:

```python
        mixture_component_count=args.mixture_component_count,
        mixture_patch_radius_m=args.mixture_patch_radius_m,
        mixture_patch_sample_count=args.mixture_patch_samples,
        mixture_clustering_iterations=args.mixture_clustering_iterations,
```

Pass to `export_surrogate_npz`:

```python
        mixture_component_count=args.mixture_component_count,
        mixture_patch_radius_m=args.mixture_patch_radius_m,
        mixture_patch_sample_count=args.mixture_patch_samples,
        mixture_clustering_iterations=args.mixture_clustering_iterations,
```

Pass to `build_kokoro_scene_dict`:

```python
        adaptive_kappa_mode=args.adaptive_kappa_mode,
        min_lobe_kappa=args.min_lobe_kappa,
        facet_boundary_width=args.facet_boundary_width,
```

For holdout metrics, build the existing angular holdout only when `args.target_mode != "normal_mixture"` because the existing holdout compares one predicted direction against one target. Replace the unconditional holdout block with:

```python
if args.target_mode == "normal_mixture":
    metrics["holdout_sample_count"] = 0
    metrics["holdout_angular_error"] = None
else:
    holdout = build_direction_holdout_dataset(
        program,
        DirectionHoldoutConfig(
            x_count=args.holdout_grid_size,
            y_count=args.holdout_grid_size,
            theta_count=args.holdout_theta_count,
            phi_count=args.holdout_phi_count,
        ),
        width_m=args.width_m,
        depth_m=args.depth_m,
        local_feature_period_m=args.local_feature_period_m,
        position_frequency_count=args.position_frequency_count,
        radial_cell_feature_period_m=radial_cell_feature_period_m,
        radial_cell_feature_max_rotation_rad=args.radial_cell_feature_max_rotation_rad,
        radial_cell_feature_radial_power=args.radial_cell_feature_radial_power,
        radial_cell_facet_features=radial_cell_facet_features,
        include_incident_features=args.target_mode == "reflection",
        target_mode=args.target_mode,
    )
    metrics["holdout_sample_count"] = int(holdout.features.shape[0])
    metrics["holdout_angular_error"] = angular_error_degrees(result.model, holdout.features, holdout.targets)
```

- [ ] **Step 5: Run run_demo tests**

Run:

```powershell
uv run --project apps/python python -m unittest packages/kokoro/tests/test_run_demo.py
```

Expected: OK.

- [ ] **Step 6: Commit**

```powershell
git add packages/kokoro/run_demo.py packages/kokoro/tests/test_run_demo.py
git commit -m "feat(kokoro): wire vmf noise controls into demo"
```

### Task 7: Add Scene Builder Coverage For Adaptive Controls

**Files:**
- Modify: `packages/kokoro/tests/test_mitsuba_scene.py`
- Modify: `packages/kokoro/kokoro/mitsuba_scene.py`

- [ ] **Step 1: Write failing passthrough test**

Add this assertion to `test_scene_dict_passes_adaptive_kappa_controls_to_neural_bsdf` from Task 1 after run_demo flags exist:

```python
self.assertEqual(bsdf["lobe_kappa"], 96.0)
```

Add a reference non-regression test:

```python
def test_reference_scene_keeps_height_field_lobe_kappa_fixed(self) -> None:
    scene = build_height_field_reference_scene_dict(
        height_source="def height(x, y):\n    return x * 0.0\n",
        lobe_kappa=4096.0,
    )

    self.assertEqual(scene["surface"]["bsdf"]["type"], "kokoro_height_field_reflector")
    self.assertEqual(scene["surface"]["bsdf"]["lobe_kappa"], 4096.0)
    self.assertNotIn("adaptive_kappa_mode", scene["surface"]["bsdf"])
```

- [ ] **Step 2: Run tests**

Run:

```powershell
uv run --project apps/python python -m unittest packages/kokoro/tests/test_mitsuba_scene.py
```

Expected: OK after Task 1 implementation.

- [ ] **Step 3: Commit**

```powershell
git add packages/kokoro/tests/test_mitsuba_scene.py
git commit -m "test(kokoro): lock adaptive kappa scene behavior"
```

### Task 8: Add Mixture Smoke Test At The CLI Boundary

**Files:**
- Modify: `packages/kokoro/tests/test_run_demo.py`
- No production code if Tasks 4-6 were complete.

- [ ] **Step 1: Add a subprocess smoke test only if this test file already uses subprocess**

If `test_run_demo.py` does not use subprocess, do not add a slow boundary test there. Instead, add the smoke command to the manual verification section in Task 11. The current file tests constants and path helpers, so keep it fast.

- [ ] **Step 2: Add metadata-level smoke to `test_brdf_surrogate.py`**

Add:

```python
def test_normal_mixture_training_result_exports_loadable_npz(self) -> None:
    program = compile_height_program("def height(x, y):\n    return 0.001 * x\n")
    dataset = build_brdf_dataset(
        program,
        sample_count=16,
        width_m=0.10,
        depth_m=0.10,
        seed=4,
        target_mode="normal_mixture",
        mixture_component_count=2,
        mixture_patch_radius_m=0.0,
        mixture_patch_sample_count=4,
        mixture_clustering_iterations=2,
    )
    result = train_brdf_surrogate(
        dataset,
        BrdfTrainingConfig(hidden_dim=8, epochs=2, batch_size=8, lr=0.01, seed=5),
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = Path(temp_dir) / "trained_normal_mixture.npz"
        export_surrogate_npz(
            result.model,
            checkpoint,
            width_m=0.10,
            depth_m=0.10,
            include_incident_features=False,
            target_mode="normal_mixture",
            mixture_component_count=2,
            mixture_patch_radius_m=0.0,
            mixture_patch_sample_count=4,
            mixture_clustering_iterations=2,
        )
        loaded = load_npz_surrogate(checkpoint)

    self.assertEqual(loaded.metadata["output_layout"], "normal_mixture")
    self.assertEqual(loaded.predict(dataset.features[:3].numpy()).shape, (3, 10))
```

- [ ] **Step 3: Run BRDF tests**

Run:

```powershell
uv run --project apps/python python -m unittest packages/kokoro/tests/test_brdf_surrogate.py
```

Expected: OK.

- [ ] **Step 4: Commit**

```powershell
git add packages/kokoro/tests/test_brdf_surrogate.py
git commit -m "test(kokoro): smoke trained normal mixtures"
```

### Task 9: Document The New Modes

**Files:**
- Modify: `packages/kokoro/README.md`

- [ ] **Step 1: Update the rendering controls section**

Add this paragraph after the current `--reconstruction-filter` explanation:

```markdown
For noise experiments beyond fixed-width lobes, Kokoro exposes three opt-in controls. `--adaptive-kappa-mode facet-boundary` keeps the default single normal target but lowers the vMF concentration near known rotated-pyramid facet boundaries. `--target-mode normal_mixture --mixture-component-count 4` trains the network to output a weighted local normal distribution instead of one normal. `--adaptive-kappa-mode mixture-concentration` maps each learned component's concentration to a per-component vMF width, bounded by `--min-lobe-kappa` and `--lobe-kappa`.
```

- [ ] **Step 2: Add a mixture experiment command**

Add:

```powershell
uv run --project apps/python python packages/kokoro/run_demo.py `
  --output-dir packages/kokoro/tmp/vmf-mixture-k4 `
  --samples 16384 `
  --epochs 240 `
  --target-mode normal_mixture `
  --mixture-component-count 4 `
  --mixture-patch-radius-m 0.001 `
  --mixture-patch-samples 64 `
  --adaptive-kappa-mode mixture-concentration `
  --min-lobe-kappa 512 `
  --lobe-kappa 4096 `
  --spp 1024 `
  --render `
  --reference-render
```

- [ ] **Step 3: Add promotion criteria**

Add:

```markdown
Do not promote mixture defaults from one image. Keep the current fixed single-lobe normal mode until a comparison panel shows less isolated sparkle, a continuous light band at 1024 SPP, and no unacceptable loss of band sharpness against the height-field reference. Keep every accepted panel under `packages/kokoro/tmp/<case>`.
```

- [ ] **Step 4: Commit**

```powershell
git add packages/kokoro/README.md
git commit -m "docs(kokoro): document vmf noise experiments"
```

### Task 10: Full Automated Verification

**Files:**
- No code edits.

- [ ] **Step 1: Run all Kokoro tests**

Run:

```powershell
uv run --project apps/python python -m unittest discover packages/kokoro/tests
```

Expected: all tests OK. The existing `jitc_llvm_init(): LLVM API initialization failed` message can appear as long as the command exits with code 0.

- [ ] **Step 2: Run the new pure target test directly**

Run:

```powershell
uv run --project apps/python python -m unittest packages/kokoro/tests/test_vmf_targets.py
```

Expected: OK.

- [ ] **Step 3: Commit if any verification-only fixes were needed**

```powershell
git status --short
git add packages/kokoro
git commit -m "fix(kokoro): stabilize vmf mixture tests"
```

Skip the commit command when `git status --short` shows no uncommitted files from verification fixes.

### Task 11: Manual Smoke And Render Evidence

**Files:**
- Generated outputs under `packages/kokoro/tmp`.

- [ ] **Step 1: Single-lobe adaptive kappa smoke**

Run:

```powershell
uv run --project apps/python python packages/kokoro/run_demo.py `
  --output-dir packages/kokoro/tmp/adaptive-kappa-smoke `
  --samples 512 `
  --epochs 2 `
  --holdout-grid-size 4 `
  --holdout-theta-count 2 `
  --holdout-phi-count 4 `
  --height-map-size 128 `
  --film-width 64 `
  --film-height 48 `
  --spp 16 `
  --adaptive-kappa-mode facet-boundary `
  --min-lobe-kappa 512 `
  --render
```

Expected: command exits 0 and writes `packages/kokoro/tmp/adaptive-kappa-smoke/kokoro_scene.json` with `"adaptive_kappa_mode": "facet-boundary"`.

- [ ] **Step 2: Mixture target smoke**

Run:

```powershell
uv run --project apps/python python packages/kokoro/run_demo.py `
  --output-dir packages/kokoro/tmp/vmf-mixture-smoke `
  --samples 512 `
  --epochs 2 `
  --target-mode normal_mixture `
  --mixture-component-count 4 `
  --mixture-patch-radius-m 0.001 `
  --mixture-patch-samples 64 `
  --mixture-clustering-iterations 4 `
  --height-map-size 128 `
  --film-width 64 `
  --film-height 48 `
  --spp 16 `
  --adaptive-kappa-mode mixture-concentration `
  --min-lobe-kappa 512 `
  --render
```

Expected: command exits 0 and `kokoro_brdf.npz` metadata contains `"output_layout": "normal_mixture"` and `"mixture_component_count": 4`.

- [ ] **Step 3: Evidence render panel**

Run these three full comparisons:

```powershell
uv run --project apps/python python packages/kokoro/run_demo.py `
  --output-dir packages/kokoro/tmp/baseline-single-fixed `
  --samples 16384 `
  --epochs 240 `
  --spp 1024 `
  --render `
  --reference-render
```

```powershell
uv run --project apps/python python packages/kokoro/run_demo.py `
  --output-dir packages/kokoro/tmp/single-adaptive-kappa `
  --samples 16384 `
  --epochs 240 `
  --spp 1024 `
  --adaptive-kappa-mode facet-boundary `
  --min-lobe-kappa 512 `
  --render `
  --reference-render
```

```powershell
uv run --project apps/python python packages/kokoro/run_demo.py `
  --output-dir packages/kokoro/tmp/vmf-mixture-k4 `
  --samples 16384 `
  --epochs 240 `
  --target-mode normal_mixture `
  --mixture-component-count 4 `
  --mixture-patch-radius-m 0.001 `
  --mixture-patch-samples 64 `
  --adaptive-kappa-mode mixture-concentration `
  --min-lobe-kappa 512 `
  --lobe-kappa 4096 `
  --spp 1024 `
  --render `
  --reference-render
```

Expected: all commands exit 0. Compare `kokoro_render.png` and `kokoro_height_reference.png` from the three output directories. Accept the new mode only if sparkle is lower and the light band remains visible.

### Task 12: Decide Default Promotion

**Files:**
- Modify only if evidence supports promotion: `packages/kokoro/run_demo.py`, `packages/kokoro/tests/test_run_demo.py`, `packages/kokoro/README.md`.

- [ ] **Step 1: Keep defaults unchanged when evidence is mixed**

If the mixture or adaptive renders reduce sparkle but visibly smear the band, leave:

```python
DEFAULT_TARGET_MODE = "normal"
DEFAULT_ADAPTIVE_KAPPA_MODE = "fixed"
DEFAULT_MIXTURE_COMPONENT_COUNT = 1
DEFAULT_MIXTURE_PATCH_RADIUS_M = 0.0
DEFAULT_MIXTURE_PATCH_SAMPLES = 1
```

Record the best command in README as an experiment.

- [ ] **Step 2: Promote adaptive kappa only if it wins cleanly**

When `single-adaptive-kappa` has lower visible sparkle and comparable band sharpness, update defaults:

```python
DEFAULT_ADAPTIVE_KAPPA_MODE = "facet-boundary"
DEFAULT_MIN_LOBE_KAPPA = 512.0
DEFAULT_FACET_BOUNDARY_WIDTH = 0.18
```

Update `test_default_vmf_noise_controls_are_available_but_not_promoted` into:

```python
def test_default_uses_facet_boundary_adaptive_kappa_after_noise_sweep(self) -> None:
    self.assertEqual(DEFAULT_ADAPTIVE_KAPPA_MODE, "facet-boundary")
    self.assertEqual(DEFAULT_MIN_LOBE_KAPPA, 512.0)
    self.assertEqual(DEFAULT_FACET_BOUNDARY_WIDTH, 0.18)
```

- [ ] **Step 3: Promote mixture mode only with stronger evidence**

When `vmf-mixture-k4` keeps the band coherent and clearly reduces sparkle, update defaults:

```python
DEFAULT_TARGET_MODE = "normal_mixture"
DEFAULT_MIXTURE_COMPONENT_COUNT = 4
DEFAULT_MIXTURE_PATCH_RADIUS_M = 0.001
DEFAULT_MIXTURE_PATCH_SAMPLES = 64
DEFAULT_ADAPTIVE_KAPPA_MODE = "mixture-concentration"
DEFAULT_MIN_LOBE_KAPPA = 512.0
DEFAULT_LOBE_KAPPA = 4096.0
```

Add a README sentence with the exact output directory used as evidence.

- [ ] **Step 4: Verify after any default promotion**

Run:

```powershell
uv run --project apps/python python -m unittest discover packages/kokoro/tests
```

Expected: all tests OK.

- [ ] **Step 5: Commit default promotion**

```powershell
git add packages/kokoro/run_demo.py packages/kokoro/tests/test_run_demo.py packages/kokoro/README.md
git commit -m "feat(kokoro): promote vmf noise defaults"
```

Use this commit only when Step 2 or Step 3 changes defaults.

---

## Self-Review

- Spec coverage: Adaptive kappa is covered by Tasks 1, 2, 6, 7, 9, 11, and 12. Mixture vMF is covered by Tasks 4, 5, 6, 8, 9, 11, and 12. Footprint-aware targets are covered by Tasks 3, 4, 8, 9, and 11.
- Red-flag scan: No incomplete-task markers are intentionally left in this plan. Each code-changing task includes concrete test code, implementation snippets, commands, and expected outcomes.
- Type consistency: The plan consistently uses `target_mode="normal_mixture"`, `output_layout="normal_mixture"`, `mixture_component_count`, `mixture_patch_radius_m`, `mixture_patch_sample_count`, `mixture_clustering_iterations`, `adaptive_kappa_mode`, `min_lobe_kappa`, and `facet_boundary_width`.
