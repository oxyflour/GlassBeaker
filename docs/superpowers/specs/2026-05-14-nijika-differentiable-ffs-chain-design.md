# Nijika Fully Differentiable FFS Chain Design

## Status
This document supplements [2026-05-13-nijika-ffs-optimization-design.md](./2026-05-13-nijika-ffs-optimization-design.md).

The May 13 design established:
- `S + FFS` multi-head prediction
- `.ffs` export
- `rez` and `farfield` optimization modes

This supplement tightens one requirement that remained only partially satisfied in the first pass:
- the training and optimization main path for the FFS-capable structured model must remain fully differentiable inside PyTorch

## Goal
Make the FFS-capable structured geometry surrogate fully differentiable across the main computational path:

`cut/nib geometry -> structured model -> FFS coefficients -> decoded FFS tensor -> field/power/efficiency losses`

This applies to:
- training
- cut/nib optimization in `farfield` mode

## Accepted Boundary
The accepted definition of "full chain differentiable" is:
- differentiable: `cut/nib -> model -> decoded FFS tensor -> efficiency or training loss`
- not required to be differentiable: writing `.ffs` text files to disk

`.ffs` export remains a side effect for interoperability and inspection. It is not part of the autograd graph.

## Scope
This design only covers:
- `structured_pair_spectral_ffs_head`

This design does not cover:
- graph optimizer input paths that currently detach to NumPy
- graph, temporal, or transolver FFS heads
- replacing `.ffs` export with a differentiable file format

## Current Gaps
The first-pass FFS implementation already supports FFS training targets, checkpoint metadata, `.ffs` export, and `farfield` optimization. The remaining problem is where the pipeline crosses into NumPy or stops too early in coefficient space.

1. `baseline/ffs_codec.py` is NumPy-only.
   Training fits, encodes, and decodes through NumPy arrays rather than a torch-native runtime codec.
2. `baseline/train.py` optimizes only coefficient MSE for the FFS head.
   It does not decode in-graph and supervise field-space or power-space quantities.
3. `optimizer_runner.py` rebuilds decode math ad hoc from checkpoint arrays.
   The decode is torch-compatible there, but it is not shared with training or prediction as one canonical runtime codec contract.
4. `baseline/predict.py` converts coefficients to NumPy before decode and export.
   This is acceptable for export, but it should be a final side branch, not the only decode path.

## User Decisions
- Keep existing `S` capability: yes
- `.ffs` export must remain supported: yes
- Optimization must support both `rez` and `farfield`: yes
- Full differentiability applies to training and optimization main path: yes
- First implementation scope stays on the structured geometry model only: yes

## Considered Approaches
### Option 1: Fixed Linear Codec With Torch Runtime Decode
Fit a linear codec offline as before, but add a torch-native runtime codec used by training, prediction, and optimization.

Pros:
- minimal change to current checkpoint format
- stable on a 200-sample dataset
- easy to verify against the existing NumPy codec

Cons:
- basis remains fixed rather than learned jointly

### Option 2: Direct Full-Grid FFS Prediction
Predict the entire farfield grid directly without any codec.

Pros:
- simple conceptual pipeline
- no codec state to checkpoint

Cons:
- much larger output head
- higher sample complexity
- worse first-pass stability on the current dataset size

### Option 3: Learned Neural Decoder Or Autoencoder
Train a separate decoder and let the geometry model predict latent codes.

Pros:
- more expressive than a fixed linear codec

Cons:
- adds a second training problem
- introduces more moving parts than needed for the current goal

### Recommendation
Choose Option 1. It preserves the current first-pass system shape while making the actual optimization and training chain fully differentiable where it matters.

## Design Summary
Keep the current linear FFS codec for offline fitting and checkpoint compatibility, but split the codec into two layers:
- offline fitting and reference encode/decode utilities
- torch-native runtime codec used inside the model loss and optimizer path

The differentiable main path becomes:
1. model predicts `s_pred` and `ffs_coeff_pred`
2. torch runtime codec decodes `ffs_coeff_pred` into canonical FFS tensors
3. training computes field-space and power-space losses from decoded tensors
4. optimization computes farfield efficiency from the same decoded tensors

Prediction export also uses the torch runtime codec first, and only converts to NumPy at the final write-to-disk boundary.

## Architecture
### 1. Codec Layer
`baseline/ffs_codec.py` should provide both:
- offline state containers for fitting and checkpoint serialization
- a `TorchFfsCodec` runtime module

`TorchFfsCodec` requirements:
- stores `mean` and `basis` as non-trainable buffers
- accepts batched coefficient tensors
- decodes to canonical FFS tensor shape with pure torch math
- optionally encodes canonical fields back into coefficient space for runtime checks or auxiliary losses

The offline fitted basis remains deterministic and is still learned on the training split only.

### 2. Training Layer
Training for `structured_pair_spectral_ffs_head` should supervise three FFS quantities:
- coefficient loss in codec space
- decoded field loss in canonical field space
- decoded radiated power loss on the farfield frequencies

Recommended first-pass loss form:
- `L = L_s + w_coeff * L_coeff + w_field * L_field + w_power * L_power`

Where:
- `L_s` is the existing `S` composite loss
- `L_coeff` is MSE between predicted and target codec coefficients
- `L_field` is MSE between decoded predicted field and target field tensor
- `L_power` is MSE between integrated decoded power and dataset header power

The default weights should favor `S` as the anchor objective and keep the FFS losses additive and configurable.

### 3. Optimization Layer
`optimizer_runner.py` should stop hand-assembling decode tensors from checkpoint arrays and instead instantiate the same `TorchFfsCodec` runtime object used in training.

The `farfield` optimization path must stay entirely in torch:
1. differentiable cut/nib variables rebuild structured inputs
2. surrogate predicts `s_pred` and `ffs_coeff_pred`
3. `TorchFfsCodec.decode()` reconstructs canonical FFS tensors
4. decoded tensors reshape into per-port farfield basis
5. current weights derive from predicted `S`
6. basis fields combine linearly
7. angular integration yields efficiency
8. gradients flow back into geometry variables

This shared runtime codec removes contract drift between training and optimization.

### 4. Prediction And Export Layer
Prediction keeps two separate responsibilities:
- runtime decode for internal consistency
- text export for external interoperability

Required behavior:
1. decode `ffs_coeff_pred` with `TorchFfsCodec`
2. keep the decoded torch tensor available for internal consumers
3. only at export time call `detach().cpu().numpy()`
4. write CST-compatible `.ffs` files

This keeps the non-differentiable boundary narrow and explicit.

### 5. Checkpoint Contract
The FFS-capable checkpoint format stays backward-compatible with the first-pass design.

Required stored state:
- `ffs_codec.field_shape`
- `ffs_codec.rank`
- `ffs_codec.mean`
- `ffs_codec.basis`
- `ffs_metadata` needed for decode interpretation and `.ffs` export

No checkpoint change is required for old `S`-only models.

## Non-Goals
- no attempt to make filesystem writes differentiable
- no extension of this work to graph-based optimizer input paths yet
- no learned nonlinear FFS codec in this phase
- no redesign of the existing `S` objective

## File-Level Impact
Primary files:
- `packages/nijika/baseline/ffs_codec.py`
- `packages/nijika/baseline/train.py`
- `packages/nijika/baseline/training_utils.py`
- `packages/nijika/baseline/predict.py`
- `packages/nijika/optimizer_runner.py`

Tests to extend:
- `packages/nijika/tests/test_ffs_codec.py`
- `packages/nijika/tests/test_ffs_train_predict.py`
- `packages/nijika/tests/test_optimize_baseline.py`
- `packages/nijika/tests/test_optimizer_torch_farfield.py`

## Testing Requirements
Minimum required regression coverage:
1. Torch codec parity.
   Torch decode must match the existing NumPy decode within tolerance.
2. Torch codec gradient flow.
   Decoded field tensors must backpropagate into coefficient inputs.
3. Training loss gradient coverage.
   Field loss and power loss must backpropagate into `ffs_coeff_pred`.
4. Farfield optimizer gradient coverage.
   `efficiency_mode="farfield"` must produce gradients with respect to cut/nib optimization variables.
5. Export boundary coverage.
   Prediction must still export valid `.ffs` files, but only after the final `detach().cpu().numpy()` boundary.
6. Backward compatibility.
   Existing `S`-only training, prediction, and optimization tests must keep passing.

## Risks
- A fixed linear basis can still limit FFS fidelity if the true field manifold is too nonlinear.
- Field loss and power loss can over-regularize the head if weights are not tuned carefully.
- Header power values and integrated decoded power can differ by convention if the dataset export is inconsistent; tests must lock this down.

## Acceptance Criteria
This design is complete when all of the following are true:
- `structured_pair_spectral_ffs_head` training decodes FFS tensors inside torch
- FFS training uses field-space and power-space supervision in addition to coefficient loss
- `optimize_baseline.py --efficiency-mode farfield` uses the same torch runtime decode path as training
- cut/nib optimization gradients flow through decoded FFS tensors into geometry variables
- `.ffs` export still works, but the only non-differentiable boundary is the final file-writing side branch

## Next Step
After this spec is approved, write a focused implementation plan for the torch-native codec and differentiable FFS loss path instead of reusing the broader May 13 rollout plan unchanged.
