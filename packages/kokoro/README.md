# Kokoro

Kokoro is a small height-field neural BRDF prototype.

It takes Python source that defines `height(x, y)`, randomly samples the microstructure to train a tiny neural BRDF, exports the network as `npz`, and renders an equivalent 10 cm x 10 cm Mitsuba plane lit by a point light above the surface.

The default height field is a square-pyramid lattice with 500 um cells whose pyramid orientation rotates around each cell center. The rotation angle increases with the cell center's distance from the 10 cm x 10 cm region center, so this baseline is not globally periodic. The BRDF surrogate keeps macro `x/y` as material inputs. By default each target is a point reflection direction; when patch averaging is enabled, the target becomes `axis + cone cosine + fourfold phase` so the neural material can keep an oriented ring lobe instead of collapsing it into one mean reflection direction.

## Run

Run commands from the repo root with the shared Python environment in `apps/python`:

```powershell
uv run --project apps/python python packages/kokoro/run_demo.py `
  --samples 8192 `
  --epochs 240
```

Outputs:

- `packages/kokoro/tmp/kokoro_height.png`: normalized grayscale preview of the sampled height field.
- `packages/kokoro/tmp/kokoro_brdf.npz`: exported MLP weights and surface bounds.
- `packages/kokoro/tmp/kokoro_scene.json`: serialized Mitsuba scene with an equivalent plane using `kokoro_neural_reflector`.
- `packages/kokoro/tmp/metrics.json`: initial and final training loss.

The default neural material is a `sine` MLP with five 128-wide hidden layers and `--omega-0 4.0`.
Position input uses generic multiscale sine/cosine bands by default through `--position-frequency-count 9`; it does not feed the known 500 um cell phase unless you explicitly pass `--local-feature-period-m`.
Patch averaging is disabled by default; pass `--average-patch-radius-m 0.001 --average-patch-samples 32` when one rendered surface point should represent a larger local microstructure region.
Height previews default to 4096x4096; pass `--height-map-size` to change the PNG resolution.
The main scene uses a point light above the surface. The optional inspection area light is disabled by default; pass `--inspection-light-scale 0.2` or similar when you need extra fill light.

## MLP Definition

`run_demo.py` defaults to this network:

- Input dimension: `41` with the default position encoding.
- Hidden body: `--hidden-layers 5`, `--hidden-dim 128`, `sine(omega_0 * x)` activation.
- Hidden-layer limit: `--hidden-layers` accepts `1..5`.
- Output dimension: `3` for point reflection targets, or `6` when patch averaging is enabled.

The default input vector is:

```text
[
  x_norm,
  y_norm,
  sin(2^0 * pi * x_norm), cos(2^0 * pi * x_norm),
  sin(2^0 * pi * y_norm), cos(2^0 * pi * y_norm),
  ...
  sin(2^8 * pi * x_norm), cos(2^8 * pi * x_norm),
  sin(2^8 * pi * y_norm), cos(2^8 * pi * y_norm),
  wi_z,
  wi_x,
  wi_y,
]
```

Definitions:

- `x_norm = x / (width_m * 0.5)` and `y_norm = y / (depth_m * 0.5)`, so the default 10 cm plane maps to roughly `[-1, 1]`.
- `--position-frequency-count 9` adds four features per frequency band, for `2 + 4 * 9 + 3 = 41` inputs.
- `wi` is the local incoming direction above the equivalent plane. Training builds it from sampled `theta/phi`; the Mitsuba BSDF reads it from `si.wi` and uses the same `[wi_z, wi_x, wi_y]` order.
- `--local-feature-period-m p` appends two explicit local phase inputs, `remainder(x, p) / (p * 0.5) - 1` and `remainder(y, p) / (p * 0.5) - 1`. This is off by default because it hardcodes known cell pitch instead of making the model infer periodic structure from generic position features.

Footprint size is not currently part of the input. Adding it as a constant would not help: the network would see the same value at every sample, and the Mitsuba BSDF currently does not pass a true per-hit ray footprint. It should only become an input after training samples variable footprint radii and rendering provides the same quantity, preferably as a log-scaled normalized radius such as `log2(footprint_radius_m / width_m)`.

Optional render:

```powershell
uv run --project apps/python python packages/kokoro/run_demo.py `
  --samples 8192 `
  --epochs 240 `
  --render
```

This registers the Python BSDF plugin and writes `packages/kokoro/tmp/kokoro_render.png`.
The default Mitsuba variant is `cuda_ad_rgb`, matching the existing renderer backend in `apps/python`.
For custom loading, pass the JSON through `kokoro.mitsuba_scene.prepare_mitsuba_scene_dict(scene, mi)` before `mi.load_dict(scene)`.

Embedded height-field reference:

```powershell
uv run --project apps/python python packages/kokoro/run_demo.py `
  --samples 8192 `
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
  --samples 8192 `
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

The tuning target is agreement with the embedded height-field reference render, not just low training loss. Keep every sweep output in its own `packages/kokoro/tmp/<case>` directory with the checkpoint, metadata, metrics, neural render, reference render, and diff panel.

1. Establish a fixed baseline.

   Run one default neural/reference comparison with fixed seeds, `--samples 8192 --epochs 240 --hidden-layers 5 --hidden-dim 128 --position-frequency-count 9 --render --reference-render`. Record final loss, checkpoint metadata, image mean absolute error, max error, and a visual diff panel.

2. Separate representation error from render sensitivity.

   Before changing render settings, evaluate the exported MLP on a dense grid of `x/y/wi` samples and compare predicted directions against direct height-field normals. Track angular mean, p95, p99, and worst-case error. Then render with `--lobe-kappa 512`, `1024`, `2048`, and `4096` to see whether visible error is caused by direction error or by an overly sharp lobe.

3. Sweep MLP capacity within the current cap.

   Test `--hidden-layers 2`, `3`, and `5` with `--hidden-dim 64` and `128`. Keep the default cap at five 128-wide layers unless a smaller model matches the reference within the same angular and image thresholds. Do not increase beyond five layers until the feature and target sweeps below are exhausted.

4. Sweep position encoding.

   Test `--position-frequency-count 7`, `8`, `9`, and `10`. The default 500 um pitch maps to a normalized period of `0.01`, so the useful Fourier bands should bracket angular frequencies around `200 * pi`; counts `8..10` are the main region to check. Use `--local-feature-period-m 500e-6` only as a diagnostic upper bound, not as the default solution.

5. Tune sample count and optimizer budget.

   For the best architecture and frequency count, sweep `--samples 8192`, `16384`, and `32768`; then sweep `--epochs 240`, `400`, and `640`. Increase batch size only if training becomes unstable or too slow. Prefer the smallest setting whose angular p95 and image mean error stop improving materially.

6. Decide whether patch averaging belongs in the target.

   Compare point targets against `--average-patch-radius-m 0.0005`, `0.001`, and `0.002` with `--average-patch-samples 64`, `256`, and `512`. Patch targets should only stay enabled if the reference being matched also represents an area footprint instead of a point-normal finite difference.

7. Add footprint size only after the renderer can supply it.

   If patch averaging is needed, add a second experiment where every training sample draws a random footprint radius and appends a normalized log footprint feature. The same feature must be supplied in Mitsuba from ray differentials or an explicit material footprint parameter. Acceptance criteria: changing footprint size changes the learned lobe width/cone in the expected direction, and a fixed-footprint run still matches the current reference.

8. Validate on holdout height fields.

   After the default radial rotated pyramid matches the reference, repeat the best settings on a flat plane, a non-rotated pyramid lattice, and at least one smooth sinusoidal field. A setting that only works on the default source is not a robust material surrogate.

9. Promote defaults only with evidence.

   Update `run_demo.py` defaults only when a sweep shows lower reference image error and acceptable runtime. For every promoted default, keep the command, metrics, and representative panels in `packages/kokoro/tmp` so later changes can be compared against the same baseline.

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
