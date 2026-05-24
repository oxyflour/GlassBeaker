# Kokoro

Kokoro is a small height-field neural BRDF prototype.

It takes Python source that defines `height(x, y)`, randomly samples the microstructure to train a tiny neural BRDF, exports the network as `npz`, and renders an equivalent 10 cm x 10 cm Mitsuba plane. The default scene is lit by a point light above the surface; pass `--light-source hdr` to use the bundled studio HDR environment instead.

The default height field is a square-pyramid lattice with 500 um cells whose pyramid orientation rotates around each cell center. The rotation angle increases with the cell center's distance from the 10 cm x 10 cm region center, so this baseline is not globally periodic. The default surrogate now learns the local normal field from macro `x/y` plus default-specific cell/facet features, and the Mitsuba BSDF computes the reflected direction analytically from the runtime incident direction. Reflection-direction and patch-averaged ring targets remain available as diagnostics.

## Run

Run commands from the repo root with the shared Python environment in `apps/python`:

```powershell
uv run --project apps/python python packages/kokoro/run_demo.py `
  --samples 16384 `
  --epochs 240
```

Outputs:

- `packages/kokoro/tmp/kokoro_height.png`: normalized grayscale preview of the sampled height field.
- `packages/kokoro/tmp/kokoro_brdf.npz`: exported MLP weights and surface bounds.
- `packages/kokoro/tmp/kokoro_scene.json`: serialized Mitsuba scene with an equivalent plane using `kokoro_neural_reflector`.
- `packages/kokoro/tmp/metrics.json`: initial/final training loss plus dense holdout angular error.

The default neural material is a `tanh` MLP with three 128-wide hidden layers trained in `--target-mode normal`.
For the built-in default height source, `run_demo.py` appends default-specific radial rotated cell features and two facet-slope features for the known 500 um lattice. These features are not enabled automatically for custom height sources.
Patch averaging is disabled by default; pass `--average-patch-radius-m 0.001 --average-patch-samples 32` when one rendered surface point should represent a larger local microstructure region.
Height previews default to 4096x4096; pass `--height-map-size` to change the PNG resolution.
The main scene uses a point light above the surface by default. Pass `--light-source hdr` to replace that point light with `apps/web/public/studio_small_03_1k.hdr`; use `--hdr-path` for a different HDR file and `--env-scale` to tune its brightness. The neural render defaults to `--film-width 384 --film-height 288 --spp 1024 --lobe-kappa 2048 --sampler-type ldsampler --reconstruction-filter box`, matching the best current 1024 SPP band/noise trade-off. The optional inspection area light is disabled by default; pass `--inspection-light-scale 0.2` or similar when you need extra fill light.

`--reconstruction-filter` is Mitsuba's film reconstruction filter, not a post-process applied after image output. During rendering, subpixel samples are splatted into film pixels through this filter. `box` keeps each sample inside its pixel footprint, preserving sharp bands; wider filters such as `tent` or `gaussian` can reduce jagged per-pixel variation but also spread bright samples into neighboring pixels and visibly blur or thicken the band.

## MLP Definition

`run_demo.py` defaults to this network:

- Input dimension: `8` with the built-in default height source, default radial cell features, facet-slope features, and `--target-mode normal`.
- Hidden body: `--hidden-layers 3`, `--hidden-dim 128`, `tanh` activation.
- Hidden-layer limit: `--hidden-layers` accepts `1..5`.
- Output dimension: `3` normal components in normal mode, `3` point reflection components in reflection mode, or `6` when patch averaging is enabled.

The default input vector is:

```text
[
  x_norm,
  y_norm,
  radial_cell_x,
  radial_cell_y,
  sin(radial_cell_rotation),
  cos(radial_cell_rotation),
  radial_cell_facet_slope_x,
  radial_cell_facet_slope_y,
]
```

Definitions:

- `x_norm = x / (width_m * 0.5)` and `y_norm = y / (depth_m * 0.5)`, so the default 10 cm plane maps to roughly `[-1, 1]`.
- `radial_cell_x/y` are the default height field's local cell coordinates after applying the same radial cell rotation used by `radial_rotated_pyramid_height`.
- `radial_cell_facet_slope_x/y` are default-specific signed facet-slope hints derived from the dominant local pyramid facet. They make the hard facet boundary explicit instead of asking the MLP to infer a discontinuous normal from smooth coordinates.
- `--position-frequency-count 0` is the default for normal mode. Fourier bands can still be enabled for reflection-mode diagnostics.
- In `--target-mode normal`, incident direction is not an MLP input. The Mitsuba BSDF reads `si.wi` at render time and reflects it about the predicted normal. In `--target-mode reflection`, training and rendering use the previous `[wi_z, wi_x, wi_y]` incident features.
- `--local-feature-period-m p` appends two explicit global-axis local phase inputs, `remainder(x, p) / (p * 0.5) - 1` and `remainder(y, p) / (p * 0.5) - 1`.
- `--radial-cell-feature-period-m p` appends the four default-specific radial rotated cell features. `--disable-radial-cell-facet-features` removes the two facet-slope features. Pass `0` to disable radial cell features, or leave it unset for custom height sources.

The current default is still an approximation, not a solved BRDF. On the 1024 SPP diagnostic run in `packages/kokoro/tmp/spp1024-normal-facetonly-tanh128x3`, the exported normal-mode MLP reached holdout angular error around mean `2.10 deg`, p95 `8.75 deg`, and produced visible light bands. The follow-up noise pass uses `packages/kokoro/tmp/noise-reduced-default/noise_reduced_default_panel.png` to compare `lobe_kappa=2048` with low-discrepancy sampling against the previous sharper render. The reference render is a noisy finite-sample estimate of a light band, so reference-image absolute error is a diagnostic only, not the optimization target.

Every default run also evaluates a deterministic direction holdout grid after training. The default holdout has `32 x 32` surface positions, `5` incident polar samples, and `8` incident azimuth samples, and `metrics.json` records its sample count plus angular mean/p95/p99/max.

Footprint size is not currently part of the input. Adding it as a constant would not help: the network would see the same value at every sample, and the Mitsuba BSDF currently does not pass a true per-hit ray footprint. It should only become an input after training samples variable footprint radii and rendering provides the same quantity, preferably as a log-scaled normalized radius such as `log2(footprint_radius_m / width_m)`.

Optional render:

```powershell
uv run --project apps/python python packages/kokoro/run_demo.py `
  --samples 16384 `
  --epochs 240 `
  --render
```

This registers the Python BSDF plugin and writes `packages/kokoro/tmp/kokoro_render.png`.
To render with the studio HDR environment instead of the point light, add `--light-source hdr`.
The default Mitsuba variant is `cuda_ad_rgb`, matching the existing renderer backend in `apps/python`.
For custom loading, pass the JSON through `kokoro.mitsuba_scene.prepare_mitsuba_scene_dict(scene, mi)` before `mi.load_dict(scene)`.

Embedded height-field reference:

```powershell
uv run --project apps/python python packages/kokoro/run_demo.py `
  --samples 16384 `
  --epochs 240 `
  --render `
  --reference-render
```

This also writes `packages/kokoro/tmp/kokoro_height_reference.png` using `kokoro_height_field_reflector`, a Mitsuba BSDF that executes the same `height(x, y)` source through DrJit and computes finite-difference normals directly in the render.

Ring diagnostic:

```powershell
uv run --project apps/python python packages/kokoro/run_demo.py `
  --samples 4096 `
  --epochs 260 `
  --hidden-dim 128 `
  --average-patch-samples 512 `
  --film-width 512 `
  --film-height 512 `
  --spp 1024 `
  --ring-diagnostic
```

This writes `packages/kokoro/tmp/kokoro_ring_diagnostic.png` with a top-down camera and a point light above the equivalent plane.

Orbit video:

```powershell
uv run --project apps/python python packages/kokoro/run_demo.py `
  --samples 16384 `
  --epochs 240 `
  --render `
  --reference-render `
  --video
```

This writes `packages/kokoro/tmp/kokoro_orbit.mp4` using H.264 encoding through `imageio-ffmpeg`.
With `--reference-render`, it also writes `packages/kokoro/tmp/kokoro_orbit_reference.mp4` using the embedded height-field reference BSDF.
The default orbit is 120 frames at 24 fps, so one rotation takes 5 seconds.

## Validation Renders

Run the material validation from the repo root:

```powershell
uv run --project apps/python python packages/kokoro/validate_material.py `
  --samples 4096 `
  --epochs 200 `
  --hidden-dim 96 `
  --film-width 320 `
  --film-height 240 `
  --spp 64 `
  --ply-grid 96 `
  --lobe-kappa 4096
```

This writes `packages/kokoro/tmp/validation/validation_metrics.json` and two comparison sets:

- `flat_neural.png` vs `flat_mirror.png`: `z = 0` neural material against a smooth mirror.
- `pyramid_neural.png` vs `pyramid_ply.png`: 5 cm period square-pyramid neural material against a local PLY reference.
- `flat_absdiff.png` and `pyramid_absdiff.png`: 4x amplified absolute-difference images.

## Tuning Plan

The tuning target is a coherent low-sample neural light band that matches the reference distribution, not a pointwise fit to the reference image noise. Keep every sweep output in its own `packages/kokoro/tmp/<case>` directory with the checkpoint, metadata, metrics, neural render, reference render, and comparison panel.

1. Establish a fixed baseline.

   Run one default neural/reference comparison with fixed seeds, `--samples 16384 --epochs 240 --hidden-layers 3 --hidden-dim 128 --target-mode normal --position-frequency-count 0 --render --reference-render`. Record final loss, checkpoint metadata, holdout angular mean/p95/p99/max, neural/reference renders, and a visual comparison panel. Treat image absolute error as a smoke metric only because the reference contains sampling noise.

2. Separate representation error from render sensitivity.

   Use the dense holdout angular metrics written by `run_demo.py` to compare predicted normals against direct height-field normals. Track angular mean, p95, p99, and worst-case error. Then render 1024 SPP panels while sweeping `--lobe-kappa 1024`, `2048`, `4096`, and `8192`, and keep `--sampler-type ldsampler --reconstruction-filter box` as the noise baseline. Prefer the sharpest lobe that keeps the band visible without turning normal error into isolated sparkle.

3. Sweep MLP capacity within the current cap.

   Test `--hidden-layers 2`, `3`, and `5` with `--hidden-dim 64` and `128`. Current normal-mode evidence favors three 128-wide `tanh` layers. Do not increase beyond five layers until the feature and target sweeps below are exhausted.

4. Sweep position encoding.

   Keep `--position-frequency-count 0` as the normal-mode baseline. Test `2`, `4`, and `8` only if the facet-slope model misses radial variation; Fourier bands previously helped reflection-mode direction fitting but also encouraged smooth bright blobs.

5. Tune sample count and optimizer budget.

   For the best architecture and frequency count, sweep `--samples 8192`, `16384`, and `32768`; then sweep `--epochs 160`, `240`, and `400`. Increase batch size only if training becomes unstable or too slow. Prefer the smallest setting whose angular p95 and visible band structure stop improving materially.

6. Decide whether patch averaging belongs in the target.

   Compare point targets against `--average-patch-radius-m 0.0005`, `0.001`, and `0.002` with `--average-patch-samples 64`, `256`, and `512`. Patch targets should only stay enabled if the reference being matched also represents an area footprint instead of a point-normal finite difference.

7. Add footprint size only after the renderer can supply it.

   If patch averaging is needed, add a second experiment where every training sample draws a random footprint radius and appends a normalized log footprint feature. The same feature must be supplied in Mitsuba from ray differentials or an explicit material footprint parameter. Acceptance criteria: changing footprint size changes the learned lobe width/cone in the expected direction, and a fixed-footprint run still produces the low-sample light band.

8. Validate on holdout height fields.

   After the default radial rotated pyramid gives a stable low-sample light band, repeat the best settings on a flat plane, a non-rotated pyramid lattice, and at least one smooth sinusoidal field. A setting that only works on the default source is not a robust material surrogate.

9. Promote defaults only with evidence.

   Update `run_demo.py` defaults only when a sweep improves low-sample band visibility without unacceptable directional error or runtime. For every promoted default, keep the command, metrics, and representative panels in `packages/kokoro/tmp` so later changes can be compared against the same baseline.

## Custom Height Field

Create a UTF-8 Python file that defines `height(x, y)`. `x` and `y` are Torch tensors in meters.

```python
def height(x, y):
    return radial_rotated_pyramid_height(
        x,
        y,
        period_m=500e-6,
        amplitude_m=150e-6,
        max_rotation_rad=2.0 * math.pi / 4.0,
    )
```

Run it with:

```powershell
uv run --project apps/python python packages/kokoro/run_demo.py `
  --height-source path/to/height.py `
  --output-dir packages/kokoro/tmp/custom
```
