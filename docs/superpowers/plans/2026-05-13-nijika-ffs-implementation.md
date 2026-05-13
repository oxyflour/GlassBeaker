# Nijika FFS Surrogate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an FFS-capable structured geometry surrogate that can train, predict, export `.ffs`, and optimize cut/nib geometry with either legacy `rez` or new `farfield` efficiency.

**Architecture:** Keep the current `structured_pair_spectral_head` path intact and add a new multi-head structured model that predicts both normalized `S` and compressed FFS coefficients. Use a deterministic FFS IO/codec layer plus a pure-torch farfield efficiency helper so prediction export and optimization share one canonical tensor contract.

**Tech Stack:** Python, PyTorch, NumPy, existing `packages/chinatsu/farfield.py` parser contract, Nijika baseline training/inference CLI, `unittest`

---

### Task 1: FFS IO Contract And Round-Trip Coverage

**Files:**
- Create: `packages/nijika/baseline/ffs_io.py`
- Test: `packages/nijika/tests/test_ffs_io.py`
- Reference: `packages/chinatsu/farfield.py`

- [ ] **Step 1: Write a failing round-trip test for one port/frequency sample**

Write `test_round_trip_parses_with_chinatsu` in `packages/nijika/tests/test_ffs_io.py` that:
- loads one sample `.ffs` from `tmp/dataset-v3-ffs`
- parses it into a canonical tensor plus metadata
- writes it back to a temp path
- reparses it through `packages/chinatsu/farfield.py::load_ffs`
- asserts equal frequency, angle grid, and field values within tolerance

Run: `python -m pytest packages/nijika/tests/test_ffs_io.py -q`
Expected: FAIL because `baseline.ffs_io` does not exist yet

- [ ] **Step 2: Implement the minimal parser/exporter**

Add `packages/nijika/baseline/ffs_io.py` with:
- `FfsMetadata`
- `load_ffs_sample(path: Path) -> tuple[FfsMetadata, np.ndarray]`
- `load_ffs_group(paths: list[Path]) -> tuple[FfsMetadata, np.ndarray]`
- `write_ffs_sample(path: Path, metadata: FfsMetadata, field: np.ndarray) -> None`

Use canonical field shape `(freq_count, angle_count, 4)` with channels:
- `Re(E_theta)`
- `Im(E_theta)`
- `Re(E_phi)`
- `Im(E_phi)`

- [ ] **Step 3: Re-run the FFS IO test**

Run: `python -m pytest packages/nijika/tests/test_ffs_io.py -q`
Expected: PASS

### Task 2: Deterministic FFS Codec And Dataset Wiring

**Files:**
- Create: `packages/nijika/baseline/ffs_codec.py`
- Modify: `packages/nijika/baseline/data.py`
- Test: `packages/nijika/tests/test_ffs_codec.py`

- [ ] **Step 1: Write failing codec and dataset tests**

Add tests that:
- fit a codec on small synthetic FFS tensors
- round-trip encode/decode within a fixed error threshold
- load one tiny FFS fixture dataset and expose `record.ffs` plus `bundle.ffs_metadata`

Run: `python -m pytest packages/nijika/tests/test_ffs_codec.py -q`
Expected: FAIL because the codec and dataset fields are missing

- [ ] **Step 2: Implement codec primitives**

Add `packages/nijika/baseline/ffs_codec.py` with:
- `FfsCodecConfig`
- `FfsCodecState`
- `fit_ffs_codec(fields: np.ndarray, rank: int) -> FfsCodecState`
- `encode_ffs(fields: np.ndarray, state: FfsCodecState) -> np.ndarray`
- `decode_ffs(coeffs: np.ndarray, state: FfsCodecState) -> np.ndarray`

Keep the first version linear and deterministic.

- [ ] **Step 3: Extend dataset records without breaking S-only callers**

Update `packages/nijika/baseline/data.py` so:
- `SampleRecord` optionally carries `ffs` and `ffs_coeff`
- `DatasetBundle` optionally carries `ffs_metadata` and `ffs_codec_state`
- `load_dataset(..., include_ffs: bool = False, ffs_rank: int | None = None)` loads FFS only when requested
- `stack_records` includes FFS tensors only when present

- [ ] **Step 4: Re-run codec tests**

Run: `python -m pytest packages/nijika/tests/test_ffs_codec.py -q`
Expected: PASS

### Task 3: Multi-Head Structured Model And Loss Plumbing

**Files:**
- Modify: `packages/nijika/baseline/structured_spectral_model.py`
- Modify: `packages/nijika/baseline/model.py`
- Modify: `packages/nijika/baseline/training_utils.py`
- Test: `packages/nijika/tests/test_structured_ffs_model.py`

- [ ] **Step 1: Write failing model contract tests**

Add tests that create `structured_pair_spectral_ffs_head` and assert:
- forward output contains `s_pred` and `ffs_coeff_pred`
- `s_pred` shape matches `(batch, freq_bins, port_count * port_count * 2)`
- `ffs_coeff_pred` shape matches `(batch, coeff_dim)`

Run: `python -m pytest packages/nijika/tests/test_structured_ffs_model.py -q`
Expected: FAIL because the model kind is unsupported

- [ ] **Step 2: Refactor the structured model into shared encoder plus two heads**

In `packages/nijika/baseline/structured_spectral_model.py`:
- keep the existing `StructuredSpectralPredictor` behavior unchanged
- add a new FFS-capable predictor that shares the encoder trunk and returns a dict

In `packages/nijika/baseline/model.py`:
- register `structured_pair_spectral_ffs_head`

- [ ] **Step 3: Extend forward/loss evaluation helpers**

Update `packages/nijika/baseline/training_utils.py` so:
- `forward_model` can return either a tensor or a dict payload
- `composite_loss` accepts optional FFS targets and codec decode inputs
- `evaluate` still reports existing S metrics and optionally adds FFS metrics

- [ ] **Step 4: Re-run model tests**

Run: `python -m pytest packages/nijika/tests/test_structured_ffs_model.py -q`
Expected: PASS

### Task 4: Train, Predict, Analyze, And Checkpoint Metadata

**Files:**
- Modify: `packages/nijika/baseline/train.py`
- Modify: `packages/nijika/baseline/predict.py`
- Modify: `packages/nijika/baseline/analyze.py`
- Modify: `packages/nijika/baseline/metrics.py`
- Modify: `packages/nijika/tests/test_train_regression.py`

- [ ] **Step 1: Write failing regression coverage**

Extend regression tests to cover:
- a tiny FFS-enabled training smoke run
- checkpoint metadata containing FFS codec and export metadata
- prediction writing `.ffs` artifacts for one sample

Run: `python -m pytest packages/nijika/tests/test_train_regression.py -q`
Expected: FAIL because training and prediction do not understand FFS metadata

- [ ] **Step 2: Wire FFS-aware training and checkpoint save/load**

Update `packages/nijika/baseline/train.py` so the new model kind:
- requests `include_ffs=True`
- fits or restores the codec on the training split
- stores codec state and FFS metadata in the checkpoint

Update `packages/nijika/baseline/predict.py` so the new checkpoint path:
- decodes `ffs_coeff_pred`
- exports per-port, per-frequency `.ffs`
- keeps current `S` artifacts unchanged

Update `packages/nijika/baseline/analyze.py` and `metrics.py` so they report first-pass FFS metrics.

- [ ] **Step 3: Re-run the regression tests**

Run: `python -m pytest packages/nijika/tests/test_train_regression.py -q`
Expected: PASS

### Task 5: Farfield Efficiency Mode In The Optimizer

**Files:**
- Create: `packages/nijika/optimizer_torch_farfield.py`
- Modify: `packages/nijika/optimizer_runner.py`
- Modify: `packages/nijika/optimize_baseline.py`
- Modify: `packages/nijika/tests/test_optimize_baseline.py`

- [ ] **Step 1: Write a failing farfield optimization regression**

Extend `packages/nijika/tests/test_optimize_baseline.py` with a toy surrogate that returns:
- `s_pred`
- synthetic `ffs_coeff_pred` or decoded farfield tensors

Assert `optimize_model(..., efficiency_mode="farfield")`:
- reduces loss
- writes the same optimization artifacts as `rez`
- errors clearly when the band misses the FFS frequency grid

Run: `python -m pytest packages/nijika/tests/test_optimize_baseline.py -q`
Expected: FAIL because `efficiency_mode="farfield"` is unsupported

- [ ] **Step 2: Implement pure-torch farfield efficiency helpers**

Add `packages/nijika/optimizer_torch_farfield.py` with helpers that:
- derive current weights from complex `S`
- linearly combine per-port farfield basis tensors
- integrate power over phi/theta
- return differentiable efficiency over the selected frequency mask

- [ ] **Step 3: Wire the optimizer CLI and runtime branch**

Update `packages/nijika/optimize_baseline.py`:
- add `--efficiency-mode rez|farfield`
- validate checkpoint metadata for farfield mode

Update `packages/nijika/optimizer_runner.py`:
- decode predicted FFS when present
- branch between legacy `rez` loss and new farfield loss
- keep default behavior unchanged

- [ ] **Step 4: Re-run the optimizer tests**

Run: `python -m pytest packages/nijika/tests/test_optimize_baseline.py -q`
Expected: PASS

### Task 6: Final Verification

**Files:**
- No new files

- [ ] **Step 1: Run focused coverage**

Run:
- `python -m pytest packages/nijika/tests/test_ffs_io.py -q`
- `python -m pytest packages/nijika/tests/test_ffs_codec.py -q`
- `python -m pytest packages/nijika/tests/test_structured_ffs_model.py -q`
- `python -m pytest packages/nijika/tests/test_train_regression.py -q`
- `python -m pytest packages/nijika/tests/test_optimize_baseline.py -q`

Expected: all PASS

- [ ] **Step 2: Run one end-to-end smoke command**

Run a tiny FFS-capable train/predict flow against `tmp/dataset-v3-ffs` using the new model kind and confirm:
- checkpoint writes FFS metadata
- prediction exports `.ffs`
- optimizer starts in `farfield` mode without shape or metadata errors

