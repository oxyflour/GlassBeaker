Differentiable Feed/Open/Ground + Distance Optimization V1

Saved from the prior planning session so a new session can continue directly from here.

## Summary

- Build V1 around the existing role-agnostic staged-loss surrogate, not a new role-conditioned model.
- Keep the current fixed phone shape and current `3 nibs -> 3 candidate ports` data distribution.
- Optimize only `cut.distance` and `nib.distance` continuously.
- Optimize port roles jointly with geometry using a soft relaxation: exactly one soft `feed`, and every non-feed port gets a soft `open/ground` termination.
- Revalidate by enumerating the final discrete role assignments for the optimized geometry, taking `Final + TopK`, and sending those candidates back to real simulation.

## Key Changes

- Add a new Python optimization entrypoint, e.g. `packages/nijika/optimize_baseline.py`, that:
  - Loads an existing staged-loss checkpoint and one existing antenna config JSON.
  - Builds differentiable geometry variables for every active cut/nib distance.
  - Builds differentiable role variables: `feed_logits[3]` with `softmax`, plus `termination_logits[3]` with `sigmoid`.
  - Recomputes surrogate inputs each step from updated `cuts`, `nibs`, and regenerated `ports`; keep `frame`, `geom`, and point-cloud-independent model inputs otherwise unchanged.
- Use bounded distance parameterization:
  - For each cut/nib, store an unconstrained optimization variable and map it through `tanh` into the legal physical range `[-(crossSize-width)/2, +(crossSize-width)/2]`.
  - Recompute port placement from the same geometry math as `getAntennaFeedPlacement`, so port coordinates move consistently with nib distance.
- Add a torch-complex objective module for differentiable role scoring:
  - Surrogate still predicts the full complex `S(f)` matrix for the 3 candidate ports.
  - Convert `S -> Y` in torch.
  - For each candidate feed port, compute loaded input admittance with Schur-complement reduction under soft non-feed terminations.
  - Model `open` as near-zero admittance and `ground` as large admittance, with finite clipped defaults for numerical stability.
  - Compute per-feed differentiable score from three terms over a configurable frequency mask:
    - Matching term from loaded feed reflection.
    - Isolation term from raw feed-to-other-port coupling magnitudes.
    - Soft bandwidth term from the fraction of frequencies satisfying the matching threshold, approximated with sigmoids.
  - Use default combined weights `matching:isolation:bandwidth = 5:3:2`.
  - Global loss is the `softmax(feed_logits)` weighted average of per-feed losses.
- Add a config rebuild path for simulation candidates:
  - For optimization-time surrogate calls, only update the analytic input tensors.
  - For final simulation export, regenerate a fresh antenna JSON with updated mesh and ports from the optimized distances, reusing the existing antenna-builder logic rather than patching the stale embedded mesh.
- Add discrete hardening and TopK selection:
  - After optimization converges, keep the optimized geometry fixed.
  - Enumerate all discrete role assignments under the V1 rule: exactly 1 feed and each remaining port independently `open` or `ground` (`3 * 2^2 = 12` candidates).
  - Score all 12 with the same surrogate objective.
  - Export the best candidate plus the next best `K-1` candidates for re-simulation.
- Keep training untouched in V1:
  - Reuse the current staged-loss `structured_pair_spectral_head` checkpoint as the default surrogate.
  - Do not retrain a role-conditioned model in this milestone.
  - Do not support 2-feed or variable port-count optimization yet.

## Public Interfaces

- New optimization CLI:
  - Inputs: `--model-path`, `--config-path`, `--output-dir`, `--band-min`, `--band-max`, `--steps`, `--lr`, `--top-k`, `--match-weight`, `--isolation-weight`, `--bandwidth-weight`, `--match-threshold-db`, `--ground-admittance`, `--open-admittance`.
  - Outputs:
    - Optimization trace JSON with stepwise loss, role probabilities, and bounded distances.
    - Final optimized soft solution JSON.
    - Enumerated hard-candidate ranking JSON.
    - Rebuilt simulation-ready config JSONs for `Final + TopK`.
- New rebuild helper interface:
  - Input: original config + overridden `cut/nib` distances.
  - Output: fully regenerated simulation JSON with updated mesh, ports, and `antennaConfig`.

## Test Plan

- Unit-test the bounded distance mapping so every optimized cut/nib stays inside its legal geometric range.
- Unit-test port regeneration against the current `getAntennaFeedPlacement` behavior for representative nib positions.
- Unit-test the torch network-reduction math against a small analytic 2-port/3-port case and a scikit-rf reference.
- Unit-test role relaxation:
  - `feed_probs` always sum to 1.
  - Non-feed termination weights remain in `[0, 1]`.
  - Hard enumeration produces exactly 12 candidates for the V1 3-port rule.
- Smoke-test the optimizer on one existing antenna config using a staged-loss checkpoint:
  - Loss decreases over a short run.
  - Distances move but remain bounded.
  - Candidate artifacts are emitted without mutating repo-tracked data.
- End-to-end validation:
  - Run optimizer on one sample.
  - Re-simulate `Final + TopK`.
  - Compare the same weighted objective before/after.
  - Success criterion: at least one re-simulated candidate beats the original config on the exact same code objective.

## Assumptions And Defaults

- V1 stays on the current fixed phone dimensions; width/height generalization is explicitly out of scope here.
- V1 assumes current dataset semantics: 3 candidate ports derived from 3 nibs.
- V1 supports exactly 1 feed during optimization; `1 or 2 feeds` is deferred to a later extension.
- Non-feed ports are optimized as soft `open/ground` terminations, not as additional measured ports.
- The objective is configurable by frequency band; if no band is provided, default to the full trained frequency grid.
- Use the existing staged-loss surrogate as the default model because it is the current strongest baseline and already ignores mesh point clouds, which keeps differentiable geometry updates cheap and stable.
