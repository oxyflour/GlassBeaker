---
name: nijika-exe
description: Use this skill when the task is to run nijika.exe in this repository, reproduce or inspect FITD/FDFD results, run native .json or .njk project inputs, convert .cst files to .njk, or export .ffs farfield files from CST or FITD outputs.
---

# Nijika CLI

Use this skill for local CLI work against the `nijika.exe` binary in this repository.

## Quick Start

- Work from the repository root.
- Prefer an existing binary in this order:
  - `build\Debug\nijika.exe`
  - `build\Release\nijika.exe`
- Assume the binary already exists in the workspace.
- `nijika.exe` does not support `--help`. Read `src/main.cc` and `src/nijika.cu` for command and flag behavior.

## Main Commands

### Start a run

Use explicit `start` even though it is the default command:

```powershell
.\build\Debug\nijika.exe start path\to\case.cst --verbose
```

Useful flags:

- `--fitd-use-cst-matrix`
- `--fitd-core-data-type fp32|bf16|cp64`
- `--fitd-loop-steps N`
- `--fitd-flow-steps N`
- `--fitd-activated-port-list 1,2`
- `--fitd-farfield-freq 1880000000`
- `--fitd-dump-farfield-freq 1880000000`
- `--fitd-dump-farfield-steps N`
- `--fitd-check-energy-interval N`
- `--fitd-check-energy-threshold X`
- `--fitd-dump-slice-x IDX`
- `--fitd-dump-slice-y IDX`
- `--fitd-dump-slice-z IDX`
- `--fitd-dump-slice-range A:B:C`
- `--fitd-dump-mesh-obj name`
- `--fitd-dump-port-obj name`
- `--solver fdfd`

Example used successfully in this repository:

```powershell
.\build\Debug\nijika.exe start tmp\ffs\monopole-sweep.cst --verbose --fitd-use-cst-matrix --fitd-farfield-freq 1880000000 --fitd-dump-farfield-freq 1880000000
```

### Start from JSON

`.json` is a first-class input format:

```powershell
.\build\Debug\nijika.exe start path\to\case.json --verbose
```

Typical JSON runs also add grid and FITD controls:

```powershell
.\build\Debug\nijika.exe start path\to\case.json --verbose --mesh-hex-build-grid --mesh-hex-grid-padding-cells 2 --mesh-hex-cell-per-wavelength 20 --fitd-loop-steps 3000
```

You can also load extra JSON on top of an existing `.json` or `.njk` input:

```powershell
.\build\Debug\nijika.exe start path\to\case.njk --load-json path\to\override.json
```

`--load-json` behavior from `src/nijika.cu` and `src/utils/project.cc`:

- Scalar fields such as `freq`, `units`, `fitd.timestep`, `fitd.loop_steps`, `fitd.cfl_factor`, `hex.grid`, and `pml` are replaced when present.
- Array/object-list fields such as `ports`, `solids`, and `materials` are appended, not cleared first.
- Do not treat `--load-json` as a strict merge-patch override for lists.

### Convert input

Use `convert` when you need a persistent converted model:

```powershell
.\build\Debug\nijika.exe convert path\to\case.cst path\to\case.njk
```

If you run `start` on a `.cst` without `--fitd-use-cst-matrix`, `nijika.exe` creates a temporary `.njk...` file and deletes it after the run. Use `convert` if you need to inspect or reuse the converted file.

### Export a CST farfield `.ffs`

Use `convert` with an `.ffs` output and exactly one target frequency:

```powershell
.\build\Debug\nijika.exe convert tmp\ffs\monopole-sweep.cst tmp\ffs\monopole-sweep[f=1.88].ffs --fitd-dump-farfield-freq 1880000000
```

Notes:

- The output must end with `.ffs`.
- `src/main.cc` requires exactly one farfield frequency for `.ffs` export.
- If the output name includes `[f=1.88]`, the code can infer GHz from the filename, but passing `--fitd-dump-farfield-freq` is safer.

## JSON Input Format

`start path\to\case.json` loads the file through `utils::project_t::load(json, root)`.

Supported top-level keys:

- `freq`: `{ "min": number, "max": number }`
- `units`: `{ "geometry": number, "time": number, "frequency": number }`
- `fitd`: `data_type`, `loop_steps`, `flow_steps`, `port_list`, `timestep`, `cfl_factor`
- `pml`: `{ "lower": ..., "upper": ... }`
- `hex.grid` (optional): `{ "x": [...], "y": [...], "z": [...] }`
- `mesh`: either a relative mesh file path string, or an inline object with `verts` and `faces`
- `ports`: list of lumped-port segments
- `materials`: list of material definitions
- `solids`: list mapping solid names to material names
- `planewave`: optional plane-wave source definition

Minimal schema example:

```json
{
  "freq": {
    "min": 0.0,
    "max": 5000000000.0
  },
  "units": {
    "geometry": 0.001,
    "time": 1e-9,
    "frequency": 1000000000.0
  },
  "fitd": {
    "data_type": "fp32",
    "loop_steps": 3000,
    "timestep": 0.0
  },
  "pml": {
    "lower": {
      "x": [4, 0.005],
      "y": [4, 0.005],
      "z": [4, 0.005]
    },
    "upper": {
      "x": [4, 0.005],
      "y": [4, 0.005],
      "z": [4, 0.005]
    }
  },
  "hex": {
    "grid": {
      "x": [-0.01, 0.0, 0.01],
      "y": [-0.01, 0.0, 0.01],
      "z": [-0.02, 0.0, 0.02]
    }
  },
  "materials": [
    {
      "name": "Vacuum",
      "type": "Normal",
      "eps": { "x": 1.0, "y": 1.0, "z": 1.0 },
      "tand": { "value": 0.0, "freq": 0.0 }
    },
    {
      "name": "PEC",
      "type": "PEC",
      "eps": { "x": 1.0, "y": 1.0, "z": 1.0 },
      "tand": { "value": 0.0, "freq": 0.0 }
    }
  ],
  "solids": [
    {
      "name": "component1:solid1",
      "material": "PEC"
    }
  ],
  "mesh": {
    "verts": [
      [-0.005, -0.005, 0.0],
      [0.005, -0.005, 0.0],
      [0.005, 0.005, 0.0],
      [-0.005, 0.005, 0.0]
    ],
    "faces": [
      [0, 1, 2, 0],
      [0, 2, 3, 0]
    ]
  },
  "ports": [
    {
      "num": 1,
      "label": "port1",
      "impedance": 50.0,
      "positions": [
        {
          "from": { "x": 0.0, "y": 0.0, "z": 0.001 },
          "to": { "x": 0.0, "y": 0.0, "z": 0.0 }
        }
      ]
    }
  ],
  "planewave": {
    "dir": "x",
    "idx": 0,
    "k_dir": { "x": 0.0, "y": 0.0, "z": 0.0 },
    "e_dir": { "x": 0.0, "y": 0.0, "z": 0.0 }
  }
}
```

Important format details:

- Use `fitd.timestep`, not `fitd.dt`.
- `mesh.faces` accepts either `[a, b, c]` or `[a, b, c, solid_index]`.
- `mesh` may also be a relative file path string; it is resolved relative to the JSON file directory.
- `pml.lower` and `pml.upper` accept either a scalar layer count or per-axis objects. Per-axis values may be integers or `[layer, depth]`.
- `hex.grid` is optional. If it is omitted or empty, the solver rebuilds the grid from geometry and frequency settings automatically.
- Pass `--mesh-hex-build-grid` only when you want to force a rebuild even though the JSON already contains a grid.

## Output Layout

`nijika.exe` writes outputs into a sibling directory with the input extension removed.

Example:

- Input: `tmp\ffs\monopole-sweep.cst`
- Output directory: `tmp\ffs\monopole-sweep\`

Common files:

- `run.log`
- `grid.json`
- `i1.txt`
- `Port 1 [1].txt`
- `S1,1.cst.txt`
- `Tot. Efficiency [1].txt`
- `port1.ffs`
- `port1.ffs.ntff.txt`
- `port1.ffs.ntff-face.txt`

## Working Rules

- Use explicit relative or absolute paths.
- Inspect `run.log` first when a run fails or stops early.
- For farfield investigations, compare `portN.ffs` against a CST-exported `.ffs` at the same frequency.
- Do not probe `--help`; it is unsupported and returns an error.
- For the full flag list or behavior changes, inspect `src/nijika.cu` and `src/main.cc`.

## Key Code References

- `src/main.cc`: `start` vs `convert`, `.ffs` export rules
- `src/nijika.cu`: CLI flags, output directory, runtime entry
- `src/fitd/solver.cu`: FITD output paths and farfield prefixes
- `include/fitd/solver/farfield.h`: `.ffs` and NTFF dump behavior
