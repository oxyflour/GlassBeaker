# Nijika FFS Prediction And Efficiency Optimization Design

## Goal

Extend Nijika so a geometry-only surrogate can predict both `S` parameters and farfield basis data from cut/nib geometry, export valid `.ffs` files, and drive the existing cut/nib optimization loop with either:

- legacy `Re(Z)`-based efficiency
- new farfield-based total efficiency

The first version must preserve the current `S`-only workflow and add a new `FFS`-capable model path without regressing existing training, prediction, or optimization commands.

## User Decisions

- Keep existing `S` capability: yes
- Prediction output must support real `.ffs` export: yes
- Optimization must support both `rez` and `farfield` modes: yes
- First version scope: only the structured geometry model path

## Non-Goals

- No first-pass support for graph, temporal, or transolver models
- No attempt to predict arbitrary farfield frequencies beyond the frequencies present in the dataset
- No replacement of the existing `S`-only checkpoint format for old models

## Current State

### Data

`baseline.data` currently loads:

- geometry features
- sampled mesh points
- port geometry
- interpolated `S*.cst.txt` curves
- optional temporal traces

It does not load `.ffs` files or any farfield targets.

### Models

Current baselines predict only flattened complex `S` matrices over frequency. The recommended geometry model is `structured_pair_spectral_head`, implemented by `StructuredSpectralPredictor`.

### Optimization

`optimize_baseline.py -> optimizer_runner.py` currently:

1. rebuilds differentiable cut/nib geometry inputs
2. runs the surrogate to predict `S`
3. computes efficiency from `S`
4. optimizes cut/nib distances plus soft role assignments

There is an `optimizer_farfield.py` bridge, but it depends on existing `.ffs` exports rather than surrogate-predicted farfield data and is not wired into the current optimization CLI.

## Design Summary

Use a new multi-head geometry model that shares the current structured geometry encoder and predicts:

- `S` outputs, using the current spectral objective
- compressed `FFS` coefficients, using a learned dataset codec

At inference time, decode the predicted `FFS` coefficients back into farfield tensors and export them as CST-compatible `.ffs` files. At optimization time, add a pure-torch farfield efficiency path that consumes predicted `S` plus decoded farfield tensors and remains differentiable with respect to cut/nib variables.

## Why This Design

### Rejected Option: Direct Full-Grid FFS Prediction

Directly predicting every field sample in every `.ffs` file would make the target much larger than the current `S` target. With only 200 farfield samples, that target size is too expensive and too brittle for a first pass.

### Rejected Option: Separate S And FFS Models

Two independent models would duplicate geometry encoding, add inference overhead, and weaken consistency between the predicted network response and predicted radiation pattern.

### Chosen Option: Shared Backbone + Compressed FFS Head

This keeps the current geometry modeling path, preserves `S` quality, and makes the first version tractable on the available dataset while still allowing real `.ffs` export and differentiable efficiency optimization.

## Architecture

### 1. Dataset And Codec Layer

Add farfield target support to the baseline data pipeline.

Each sample in `tmp/dataset-v3-ffs` contains:

- `antenna_000.json`
- `S*.cst.txt`
- `N-[f=...].ffs` files for each port and farfield frequency

The data loader must parse `.ffs` files into a canonical tensor with shape:

- `port_count x farfield_freq_count x angle_count x 4`

where the last dimension is:

- `Re(E_theta)`
- `Im(E_theta)`
- `Re(E_phi)`
- `Im(E_phi)`

The loader must also retain the metadata needed to reconstruct valid `.ffs` files:

- farfield frequency grid
- phi/theta grid
- CST header metadata shared across the dataset export shape

Add an FFS codec stage:

1. flatten each sample's farfield tensor
2. fit a low-rank basis on the training split only
3. represent each sample by `ffs_coefficients`
4. store basis metadata in the checkpoint

The codec must be deterministic and checkpointed. The first version can use a linear basis transform so decode remains simple and stable.

### 2. Model Layer

Add a new model kind:

- `structured_pair_spectral_ffs_head`

This model reuses the current structured geometry encoder and splits into two heads:

- existing `S` spectral decoder
- new `FFS` coefficient decoder

The shared encoder should stay in the structured spectral model module because the two heads depend on the same geometry latent. This is a justified refactor because it removes real duplication and defines a stable shared contract.

The model output becomes structured rather than a single flat tensor:

- `s_pred`
- `ffs_coeff_pred`

Old model kinds keep their current output contract.

### 3. Training Layer

Training for the new model kind becomes multi-task:

- current `S` composite loss
- coefficient reconstruction loss in codec space
- decoded field reconstruction loss in physical field space
- optional radiated power consistency loss over the farfield frequencies

The `S` loss remains the anchor objective. The FFS losses are additive and configurable.

Checkpoints for the new model kind must include:

- normal `S` metadata
- FFS codec basis
- farfield frequency grid
- angle grid
- FFS output metadata needed for export

Old checkpoints remain readable without FFS metadata.

### 4. Prediction And Export Layer

Prediction must remain backward-compatible.

For existing checkpoints:

- behavior stays unchanged

For `structured_pair_spectral_ffs_head` checkpoints:

- predict `S`
- predict FFS coefficients
- decode to farfield tensors
- export CST-compatible `.ffs` files per port and per farfield frequency

Prediction artifacts become:

- current `S` artifacts
- `predicted_ffs/` directory or equivalent per-sample export location
- FFS summary metadata in JSON

The exported files must be readable by the current `packages/chinatsu/farfield.py` parser without modification.

### 5. Analysis Layer

Add FFS-aware evaluation to `analyze_baseline.py`.

Required first-pass metrics:

- coefficient RMSE
- decoded field RMSE
- decoded field magnitude MAE
- radiated power error at each farfield frequency
- efficiency error under a fixed role assignment

Existing `S` metrics remain unchanged and continue to be reported.

### 6. Optimization Layer

Add:

- `--efficiency-mode rez`
- `--efficiency-mode farfield`

Behavior:

- `rez`: keep current `S`-only path
- `farfield`: run the new `S + decoded FFS` path

The farfield optimization path must not depend on writing temporary `.ffs` files during the optimization loop. It must operate on decoded torch tensors so gradients can flow from efficiency back into the model input features.

The farfield efficiency calculation should mirror the existing `chinatsu.farfield` logic:

1. get per-port current weights from predicted `S`
2. linearly combine predicted per-port farfield basis tensors
3. compute field power density
4. integrate over phi/theta
5. divide by stimulated power to obtain efficiency

The tensor path should be implemented in Nijika, not by routing through scikit-rf objects that detach tensors from autograd.

Band selection rules:

- `rez` keeps the current frequency-mask behavior
- `farfield` only optimizes over the intersection of the requested band and the farfield frequency grid
- if no farfield frequencies fall in-band, fail with a clear error

## File-Level Changes

### New Or Extended Data Files

- `packages/nijika/baseline/data.py`
  Add FFS loading, dataset record fields, and codec-aware stacking support.
- `packages/nijika/baseline/ffs_codec.py`
  New codec module for fit/encode/decode/export metadata.
- `packages/nijika/baseline/ffs_io.py`
  New loader/exporter for canonical FFS tensors and CST-compatible `.ffs` writing.

### Model Files

- `packages/nijika/baseline/structured_spectral_model.py`
  Refactor shared encoder and add the multi-head FFS variant.
- `packages/nijika/baseline/model.py`
  Register `structured_pair_spectral_ffs_head`.

### Training / Inference / Analysis

- `packages/nijika/baseline/train.py`
- `packages/nijika/baseline/training_utils.py`
- `packages/nijika/baseline/predict.py`
- `packages/nijika/baseline/analyze.py`
- `packages/nijika/baseline/metrics.py`

### Optimization

- `packages/nijika/optimizer_runner.py`
  Add branching by efficiency mode and integrate decoded FFS prediction.
- `packages/nijika/optimize_baseline.py`
  Add CLI flag and checkpoint validation.
- `packages/nijika/optimizer_farfield.py`
  Either narrow it to reusable helper logic or keep it for non-optimization bridges; it should not remain the core gradient path.
- `packages/nijika/optimizer_torch_farfield.py`
  New pure-torch farfield efficiency implementation.

### Tests

- `packages/nijika/tests/test_ffs_io.py`
- `packages/nijika/tests/test_ffs_codec.py`
- `packages/nijika/tests/test_structured_ffs_model.py`
- `packages/nijika/tests/test_optimize_baseline.py`
- existing regression tests updated where CLI choices or checkpoint metadata expand

## Data Flow

### Training

1. load sample geometry and `S`
2. load sample FFS tensors
3. fit or load FFS codec
4. encode FFS tensors to coefficients
5. train shared geometry model to predict `S + FFS coefficients`
6. decode coefficients during loss computation for field-space losses

### Prediction

1. load checkpoint
2. build geometry features
3. predict `S + FFS coefficients`
4. decode FFS coefficients to farfield tensors
5. export `.ffs` files
6. write normal `S` prediction artifacts

### Optimization

1. parameterize cut/nib distances
2. rebuild differentiable model inputs
3. predict `S + FFS coefficients`
4. decode FFS coefficients
5. compute selected efficiency objective
6. backprop into cut/nib variables
7. emit ranked candidate outputs

## Backward Compatibility

- Existing `S`-only checkpoints keep working without modification.
- Existing training commands keep working unchanged.
- Existing prediction and analysis commands keep working for old checkpoints.
- Existing optimization defaults stay on `rez` mode.

## Error Handling

- Fail fast if a requested FFS-capable workflow is used with a checkpoint lacking FFS metadata.
- Fail fast if dataset samples are missing any expected `.ffs` file for the configured farfield frequencies.
- Fail fast if FFS angle grids are inconsistent across samples.
- Fail fast if `farfield` optimization is requested outside the available farfield frequency support.

## Testing Requirements

Minimum required coverage for this feature:

1. FFS parse/export round-trip
   The exported `.ffs` files must parse back through `packages/chinatsu/farfield.py`.
2. Codec round-trip
   Encode/decode error must stay below a defined regression threshold on fixture data.
3. Model output contract
   The new model kind must emit both `S` and FFS coefficient outputs with expected shapes.
4. Training smoke
   A tiny run on a tiny FFS dataset must complete and write checkpoint metadata.
5. Prediction smoke
   Prediction must export `.ffs` files plus the existing `S` artifacts.
6. Optimization regression
   `--efficiency-mode farfield` must decrease loss on a toy surrogate and emit candidate outputs.
7. Backward compatibility
   Existing `S`-only tests must continue to pass.

## Risks

- 200 farfield samples may be too small for a high-capacity FFS target, which is why the codec is required in v1.
- Predicted `S` and predicted FFS can still become physically inconsistent; this is reduced, not eliminated, by the shared backbone and field-space losses.
- Exported `.ffs` files need strict formatting compatibility with the CST-style parser. This should be validated with round-trip tests, not assumed.

## Rollout Plan

Phase 1:

- FFS IO
- codec
- dataset integration
- unit tests

Phase 2:

- new structured multi-head model
- training and checkpoint format
- prediction export

Phase 3:

- farfield analysis metrics
- torch farfield optimization path
- CLI integration

## Acceptance Criteria

This design is complete when all of the following are true:

- a new checkpoint can predict `S` and export valid `.ffs`
- old checkpoints still work unchanged
- `optimize_baseline.py --efficiency-mode farfield` runs end-to-end from cut/nib inputs
- farfield efficiency is computed from surrogate-predicted FFS, not ground-truth files
- the optimization path remains differentiable and covered by regression tests
