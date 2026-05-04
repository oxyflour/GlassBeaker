# Differentiable Feed Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Nijika V1 optimizer that jointly adjusts cut/nib distances and soft port roles against the existing staged-loss surrogate, then exports hardened simulation candidates.

**Architecture:** Keep training untouched and add a separate inference-time optimizer pipeline under `packages/nijika`. Split the feature into focused modules: geometry rebuild helpers, differentiable role/objective math, and a CLI optimizer loop that loads an existing checkpoint and emits JSON artifacts.

**Tech Stack:** Python 3.12, `uv`, PyTorch, NumPy, scikit-rf (tests/reference only), existing `packages/nijika/baseline` utilities.

---

### Task 1: Geometry Rebuild Helpers

**Files:**
- Create: `packages/nijika/optimizer_geometry.py`
- Create: `packages/nijika/tests/test_optimizer_geometry.py`
- Reference: `apps/web/components/nijika/antenna-builder.ts`
- Reference: `packages/nijika/baseline/antenna_features.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_bounded_distance_stays_within_cross_range():
    ...

def test_regenerate_ports_matches_feed_placement_reference():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest packages/nijika/tests/test_optimizer_geometry.py -v`
Expected: FAIL because `optimizer_geometry` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def bounded_distance(raw, cross_size, span_width):
    limit = max((cross_size - span_width) / 2.0, 0.0)
    return limit * torch.tanh(raw)

def regenerate_ports(config):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest packages/nijika/tests/test_optimizer_geometry.py -v`
Expected: PASS

### Task 2: Differentiable Role Objective

**Files:**
- Create: `packages/nijika/optimizer_objective.py`
- Create: `packages/nijika/tests/test_optimizer_objective.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_role_relaxation_has_one_feed_distribution():
    ...

def test_enumerate_hard_assignments_returns_twelve_candidates():
    ...

def test_loaded_input_admittance_matches_reference_network():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest packages/nijika/tests/test_optimizer_objective.py -v`
Expected: FAIL because `optimizer_objective` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def feed_probabilities(feed_logits):
    return torch.softmax(feed_logits, dim=-1)

def termination_probabilities(termination_logits):
    return torch.sigmoid(termination_logits)

def enumerate_role_assignments(port_count):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest packages/nijika/tests/test_optimizer_objective.py -v`
Expected: PASS

### Task 3: Optimizer CLI and Artifact Export

**Files:**
- Create: `packages/nijika/optimizer_runner.py`
- Create: `packages/nijika/optimize_baseline.py`
- Create: `packages/nijika/tests/test_optimize_baseline.py`
- Modify: `packages/nijika/pyproject.toml`

- [ ] **Step 1: Write the failing tests**

```python
def test_optimizer_cli_emits_trace_and_candidate_artifacts():
    ...

def test_short_optimization_run_keeps_distances_bounded():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest packages/nijika/tests/test_optimize_baseline.py -v`
Expected: FAIL because CLI and runner do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def optimize_config(model_path, config_path, output_dir, ...):
    ...
    return result

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest packages/nijika/tests/test_optimize_baseline.py -v`
Expected: PASS

### Task 4: Focused Nijika Verification

**Files:**
- Verify: `packages/nijika/tests/test_optimizer_geometry.py`
- Verify: `packages/nijika/tests/test_optimizer_objective.py`
- Verify: `packages/nijika/tests/test_optimize_baseline.py`

- [ ] **Step 1: Run the focused Nijika suite**

Run: `uv run python -m unittest discover -s packages/nijika/tests -v`
Expected: PASS

- [ ] **Step 2: Smoke-check the entrypoint**

Run: `uv run python packages/nijika/optimize_baseline.py --help`
Expected: PASS with CLI usage output.
