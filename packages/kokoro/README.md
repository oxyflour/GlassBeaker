# Kokoro

Kokoro is a small height-field neural BRDF prototype.

It takes Python source that defines `height(x, y)`, randomly samples the microstructure to train a tiny neural BRDF, exports the network as `npz`, and renders an equivalent 10 cm x 10 cm Mitsuba plane under `apps/web/public/studio_small_03_1k.hdr`.

The default height field is a repeated square pyramid lattice with 500 um period. For periodic structures, the training features use the local cell phase instead of the global `x/y` position; this is controlled by `--feature-period-m`.

## Run

Run commands from the repo root with the shared Python environment in `apps/python`:

```powershell
uv run --project apps/python python packages/kokoro/run_demo.py `
  --samples 1024 `
  --epochs 80
```

Outputs:

- `packages/kokoro/tmp/kokoro_brdf.npz`: exported MLP weights and surface bounds.
- `packages/kokoro/tmp/kokoro_scene.json`: serialized Mitsuba scene with an equivalent plane using `kokoro_neural_reflector`.
- `packages/kokoro/tmp/metrics.json`: initial and final training loss.

Optional render:

```powershell
uv run --project apps/python python packages/kokoro/run_demo.py `
  --samples 1024 `
  --epochs 80 `
  --render
```

This registers the Python BSDF plugin and writes `packages/kokoro/tmp/kokoro_render.png`.
The default Mitsuba variant is `cuda_ad_rgb`, matching the existing renderer backend in `apps/python`.
For custom loading, pass the JSON through `kokoro.mitsuba_scene.prepare_mitsuba_scene_dict(scene, mi)` before `mi.load_dict(scene)`.

Orbit video:

```powershell
uv run --project apps/python python packages/kokoro/run_demo.py `
  --samples 1024 `
  --epochs 80 `
  --render `
  --video
```

This writes `packages/kokoro/tmp/kokoro_orbit.avi` using a pure-Python MJPEG AVI writer.
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

## Custom Height Field

Create a UTF-8 Python file that defines `height(x, y)`. `x` and `y` are Torch tensors in meters.

```python
def height(x, y):
    return pyramid_height(x, y, period_m=500e-6, amplitude_m=150e-6)
```

Run it with:

```powershell
uv run --project apps/python python packages/kokoro/run_demo.py `
  --height-source path/to/height.py `
  --feature-period-m 0.0005 `
  --output-dir packages/kokoro/tmp/custom
```
