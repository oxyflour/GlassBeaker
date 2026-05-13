# Nijika Differentiable FFS Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `structured_pair_spectral_ffs_head` training and `farfield` optimizer path fully differentiable inside torch while keeping `.ffs` export as a final non-differentiable side branch.

**Architecture:** Keep the existing linear FFS codec fit on the training split, but add a torch-native runtime codec used everywhere after fitting. Extend dataset records to preserve per-port radiated/stimulated power targets, supervise decoded fields and radiated power during training, and route prediction plus optimization through the same runtime decode path.

**Tech Stack:** Python, PyTorch, NumPy, `unittest`, existing Nijika baseline CLI, existing `optimizer_torch_farfield.py` integration helpers

---

## File Structure

- `packages/nijika/baseline/ffs_codec.py`
  Offline linear codec state plus `TorchFfsCodec` runtime decode and payload conversion.
- `packages/nijika/baseline/data.py`
  Dataset record fields for per-port FFS power targets and tensor stacking.
- `packages/nijika/baseline/training_utils.py`
  Shared FFS auxiliary loss computation in torch.
- `packages/nijika/baseline/train.py`
  FFS-capable training loop, runtime codec construction, and checkpoint metadata.
- `packages/nijika/baseline/predict.py`
  Runtime torch decode first, then final `detach().cpu().numpy()` export.
- `packages/nijika/optimizer_torch_farfield.py`
  Shared decoded-field reshape and radiated-power integration helpers.
- `packages/nijika/optimizer_runner.py`
  Replace ad hoc checkpoint decode with shared runtime codec.
- `packages/nijika/tests/test_data_ffs.py`
  Dataset regression for per-port FFS power tensors.
- `packages/nijika/tests/test_ffs_codec.py`
  Runtime torch codec parity and gradient-flow tests.
- `packages/nijika/tests/test_ffs_train_predict.py`
  FFS auxiliary loss and end-to-end train/predict regressions.
- `packages/nijika/tests/test_optimize_baseline.py`
  Geometry-to-FFS farfield optimization gradient regression.
- `packages/nijika/tests/test_optimizer_torch_farfield.py`
  Shared radiated-power helper regressions.

### Task 1: Runtime Torch Codec And Per-Port FFS Targets

**Files:**
- Modify: `packages/nijika/baseline/ffs_codec.py`
- Modify: `packages/nijika/baseline/data.py`
- Modify: `packages/nijika/tests/test_ffs_codec.py`
- Modify: `packages/nijika/tests/test_data_ffs.py`

- [ ] **Step 1: Write the failing runtime codec and dataset target tests**

Add these tests to `packages/nijika/tests/test_ffs_codec.py`:

```python
import torch

from baseline.ffs_codec import TorchFfsCodec, decode_ffs, encode_ffs, fit_ffs_codec

    def test_torch_decode_matches_numpy_decode(self):
        fields = _synthetic_fields()
        state = fit_ffs_codec(fields, rank=3)
        coeffs = encode_ffs(fields, state)

        codec = TorchFfsCodec.from_state(state, dtype=torch.float64)
        decoded = codec.decode(torch.tensor(coeffs, dtype=torch.float64))

        torch.testing.assert_close(decoded, torch.tensor(decode_ffs(coeffs, state), dtype=torch.float64))

    def test_torch_decode_backpropagates_to_coefficients(self):
        fields = _synthetic_fields()
        state = fit_ffs_codec(fields, rank=3)
        coeffs = torch.tensor(encode_ffs(fields[:2], state), dtype=torch.float64, requires_grad=True)

        codec = TorchFfsCodec.from_state(state, dtype=torch.float64)
        loss = codec.decode(coeffs).pow(2).mean()
        loss.backward()

        self.assertIsNotNone(coeffs.grad)
        self.assertGreater(float(coeffs.grad.abs().max()), 1e-12)
```

Add this regression to `packages/nijika/tests/test_data_ffs.py`:

```python
            self.assertIn("ffs_radiated_power", stacked)
            self.assertIn("ffs_stimulated_power", stacked)
            self.assertEqual(stacked["ffs_radiated_power"].shape[:3], (2, bundle.port_count, len(bundle.ffs_metadata.frequencies_hz)))
            self.assertEqual(stacked["ffs_stimulated_power"].shape, stacked["ffs_radiated_power"].shape)
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
uv run --project apps/python python -m unittest packages/nijika/tests/test_ffs_codec.py
uv run --project apps/python python -m unittest packages/nijika/tests/test_data_ffs.py
```

Expected:
- `test_ffs_codec.py` fails with `ImportError` or `AttributeError` for `TorchFfsCodec`
- `test_data_ffs.py` fails because `ffs_radiated_power` and `ffs_stimulated_power` are not stacked

- [ ] **Step 3: Implement the runtime codec and per-port power tensors**

Update `packages/nijika/baseline/ffs_codec.py` with a torch runtime codec and payload helpers:

```python
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class TorchFfsCodec(nn.Module):
    def __init__(self, *, field_shape: tuple[int, ...], mean: torch.Tensor, basis: torch.Tensor) -> None:
        super().__init__()
        self.field_shape = tuple(int(size) for size in field_shape)
        self.flat_dim = int(np.prod(self.field_shape))
        self.rank = int(basis.shape[0])
        self.register_buffer("mean", mean.reshape(self.flat_dim))
        self.register_buffer("basis", basis.reshape(self.rank, self.flat_dim))

    @classmethod
    def from_state(
        cls,
        state: FfsCodecState,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> "TorchFfsCodec":
        return cls(
            field_shape=state.config.field_shape,
            mean=torch.tensor(state.mean, dtype=dtype, device=device),
            basis=torch.tensor(state.basis, dtype=dtype, device=device),
        )

    def encode(self, fields: torch.Tensor) -> torch.Tensor:
        flat = fields.reshape(fields.shape[0], -1)
        return (flat - self.mean) @ self.basis.T

    def decode(self, coeffs: torch.Tensor) -> torch.Tensor:
        flat = coeffs @ self.basis + self.mean
        return flat.reshape(coeffs.shape[0], *self.field_shape)


def codec_state_from_payload(payload: dict[str, object]) -> FfsCodecState:
    config = FfsCodecConfig(
        field_shape=tuple(int(size) for size in payload["field_shape"]),
        flat_dim=int(payload["flat_dim"]),
        rank=int(payload["rank"]),
    )
    return FfsCodecState(
        config=config,
        mean=np.asarray(payload["mean"], dtype=np.float64),
        basis=np.asarray(payload["basis"], dtype=np.float64),
    )
```

Update `packages/nijika/baseline/data.py` so `_load_sample_ffs()` preserves per-port power targets instead of dropping them after the first port:

```python
@dataclass
class SampleRecord:
    name: str
    points: np.ndarray
    ports: np.ndarray
    geom: np.ndarray
    frame: np.ndarray
    cuts: np.ndarray
    nibs: np.ndarray
    graph: dict[str, np.ndarray] | None
    target: np.ndarray
    temporal: np.ndarray | None = None
    ffs: np.ndarray | None = None
    ffs_coeff: np.ndarray | None = None
    ffs_radiated_power: np.ndarray | None = None
    ffs_stimulated_power: np.ndarray | None = None


def _load_sample_ffs(sample_dir: Path, port_count: int) -> tuple[FfsMetadata, np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[int, list[tuple[int, Path]]] = {port: [] for port in range(1, port_count + 1)}
    for path in sample_dir.glob("*.ffs"):
        match = _FFS_FILE_RE.match(path.name)
        if match is None:
            continue
        port = int(match.group("port"))
        if port in grouped:
            grouped[port].append((int(match.group("freq")), path))
    radiated_parts = []
    stimulated_parts = []
    sample_metadata = None
    port_fields = []
    for port in range(1, port_count + 1):
        ordered_paths = [path for _, path in sorted(grouped[port], key=lambda item: item[0])]
        metadata, field = load_ffs_group(ordered_paths)
        if sample_metadata is None:
            sample_metadata = metadata
        elif not _same_ffs_layout(sample_metadata, metadata):
            raise ValueError(f"Inconsistent FFS layout across ports in {sample_dir}")
        port_fields.append(field.astype(np.float32, copy=False))
        radiated_parts.append(metadata.radiated_power_w.astype(np.float32, copy=False))
        stimulated_parts.append(metadata.stimulated_power_w.astype(np.float32, copy=False))
    assert sample_metadata is not None
    return (
        sample_metadata,
        np.stack(port_fields, axis=0),
        np.stack(radiated_parts, axis=0),
        np.stack(stimulated_parts, axis=0),
    )
```

Also stack the new arrays in `stack_records()`:

```python
    if records and records[0].ffs_radiated_power is not None:
        stacked["ffs_radiated_power"] = torch.tensor(
            np.stack([record.ffs_radiated_power for record in records]),
            dtype=torch.float32,
        )
    if records and records[0].ffs_stimulated_power is not None:
        stacked["ffs_stimulated_power"] = torch.tensor(
            np.stack([record.ffs_stimulated_power for record in records]),
            dtype=torch.float32,
        )
```

- [ ] **Step 4: Re-run the focused tests and verify they pass**

Run:

```bash
uv run --project apps/python python -m unittest packages/nijika/tests/test_ffs_codec.py
uv run --project apps/python python -m unittest packages/nijika/tests/test_data_ffs.py
```

Expected:
- both suites pass

- [ ] **Step 5: Commit**

```bash
git -C C:/Users/oxyfl/.config/superpowers/worktrees/GlassBeaker/nijika-dev add packages/nijika/baseline/ffs_codec.py packages/nijika/baseline/data.py packages/nijika/tests/test_ffs_codec.py packages/nijika/tests/test_data_ffs.py
git -C C:/Users/oxyfl/.config/superpowers/worktrees/GlassBeaker/nijika-dev commit -m "feat: add torch ffs codec runtime"
```

### Task 2: Differentiable Field And Power Losses In Training

**Files:**
- Modify: `packages/nijika/optimizer_torch_farfield.py`
- Modify: `packages/nijika/baseline/training_utils.py`
- Modify: `packages/nijika/baseline/train.py`
- Modify: `packages/nijika/tests/test_optimizer_torch_farfield.py`
- Modify: `packages/nijika/tests/test_ffs_train_predict.py`

- [ ] **Step 1: Write the failing farfield power and training-loss tests**

Extend `packages/nijika/tests/test_optimizer_torch_farfield.py` with a power integration regression:

```python
from optimizer_torch_farfield import integrate_decoded_ffs_power

    def test_integrate_decoded_ffs_power_matches_basis_path(self):
        decoded = torch.zeros((1, 2, 3, 4, 4), dtype=torch.float64)
        decoded[..., 0] = 1.0
        decoded[..., 2] = 0.5
        phi = torch.tensor([0.0, torch.pi], dtype=torch.float64)
        theta = torch.tensor([0.0, torch.pi / 2.0], dtype=torch.float64)

        power = integrate_decoded_ffs_power(
            decoded,
            phi=phi,
            theta=theta,
            phi_count=2,
            theta_count=2,
            has_phi_closure=False,
        )

        self.assertEqual(tuple(power.shape), (1, 2, 3))
        self.assertTrue(torch.isfinite(power).all().item())
```

Add a direct auxiliary-loss gradient regression to `packages/nijika/tests/test_ffs_train_predict.py`:

```python
from baseline.ffs_codec import TorchFfsCodec, encode_ffs, fit_ffs_codec
from baseline.training_utils import ffs_aux_loss

    def test_ffs_aux_loss_backpropagates_to_predicted_coefficients(self) -> None:
        fields = torch.tensor(_toy_ffs_fields()[None, ...], dtype=torch.float32)
        state = fit_ffs_codec(fields.numpy(), rank=8)
        codec = TorchFfsCodec.from_state(state)
        target_coeff = torch.tensor(encode_ffs(fields.numpy(), state), dtype=torch.float32)
        pred_coeff = (target_coeff + 0.05).clone().detach().requires_grad_(True)
        target_radiated = torch.ones((1, fields.shape[1], fields.shape[2]), dtype=torch.float32)
        phi = torch.tensor([0.0, torch.pi], dtype=torch.float32)
        theta = torch.tensor([0.0, torch.pi / 2.0], dtype=torch.float32)

        loss, parts = ffs_aux_loss(
            pred_coeff=pred_coeff,
            target_coeff=target_coeff,
            target_field=fields,
            target_radiated_power=target_radiated,
            codec=codec,
            phi=phi,
            theta=theta,
            phi_count=2,
            theta_count=2,
            has_phi_closure=False,
            loss_weights={"coeff": 1.0, "field": 1.0, "power": 1.0},
        )
        loss.backward()

        self.assertIn("ffs_field_loss", parts)
        self.assertIn("ffs_power_loss", parts)
        self.assertIsNotNone(pred_coeff.grad)
        self.assertGreater(float(pred_coeff.grad.abs().max()), 1e-12)
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
uv run --project apps/python python -m unittest packages/nijika/tests/test_optimizer_torch_farfield.py
uv run --project apps/python python -m unittest packages/nijika/tests/test_ffs_train_predict.py
```

Expected:
- `test_optimizer_torch_farfield.py` fails because `integrate_decoded_ffs_power` does not exist
- `test_ffs_train_predict.py` fails because `ffs_aux_loss` does not exist

- [ ] **Step 3: Implement shared decoded-field power helpers and training auxiliary loss**

In `packages/nijika/optimizer_torch_farfield.py`, add a decoded-field reshape helper and a radiated-power integrator:

```python
def decoded_ffs_to_basis(
    decoded: torch.Tensor,
    *,
    phi_count: int,
    theta_count: int,
    has_phi_closure: bool,
) -> torch.Tensor:
    batch, port_count, freq_count, angle_count, channel_count = decoded.shape
    if channel_count != 4 or angle_count != phi_count * theta_count:
        raise ValueError("decoded FFS tensor shape does not match metadata")
    grid = decoded.view(batch, port_count, freq_count, phi_count, theta_count, 4)
    basis = torch.stack(
        [
            torch.complex(grid[..., 0], grid[..., 1]),
            torch.complex(grid[..., 2], grid[..., 3]),
        ],
        dim=3,
    )
    if has_phi_closure:
        basis = basis[..., :-1, :]
    return basis


def integrate_decoded_ffs_power(
    decoded: torch.Tensor,
    *,
    phi: torch.Tensor,
    theta: torch.Tensor,
    phi_count: int,
    theta_count: int,
    has_phi_closure: bool,
) -> torch.Tensor:
    basis = decoded_ffs_to_basis(
        decoded,
        phi_count=phi_count,
        theta_count=theta_count,
        has_phi_closure=has_phi_closure,
    )
    flat = basis.reshape(-1, 2, basis.shape[-2], basis.shape[-1])
    power = integrate_farfield_efficiency(flat, phi, theta)
    return power.reshape(decoded.shape[0], decoded.shape[1], decoded.shape[2])
```

In `packages/nijika/baseline/training_utils.py`, add a reusable FFS auxiliary loss:

```python
import torch.nn.functional as F

from optimizer_torch_farfield import integrate_decoded_ffs_power


def ffs_aux_loss(
    *,
    pred_coeff: torch.Tensor,
    target_coeff: torch.Tensor,
    target_field: torch.Tensor,
    target_radiated_power: torch.Tensor,
    codec,
    phi: torch.Tensor,
    theta: torch.Tensor,
    phi_count: int,
    theta_count: int,
    has_phi_closure: bool,
    loss_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    decoded = codec.decode(pred_coeff)
    coeff_loss = F.mse_loss(pred_coeff, target_coeff)
    field_loss = F.mse_loss(decoded, target_field)
    power_loss = F.mse_loss(
        integrate_decoded_ffs_power(
            decoded,
            phi=phi,
            theta=theta,
            phi_count=phi_count,
            theta_count=theta_count,
            has_phi_closure=has_phi_closure,
        ),
        target_radiated_power,
    )
    total = (
        loss_weights["coeff"] * coeff_loss
        + loss_weights["field"] * field_loss
        + loss_weights["power"] * power_loss
    )
    return total, {
        "ffs_coeff_loss": coeff_loss.detach(),
        "ffs_field_loss": field_loss.detach(),
        "ffs_power_loss": power_loss.detach(),
    }
```

In `packages/nijika/baseline/train.py`, wire the new targets and weights:

```python
def build_dataset(
    tensors: dict[str, torch.Tensor],
    target: torch.Tensor,
    use_graph: bool,
    has_temporal: bool = False,
    ffs_coeff: torch.Tensor | None = None,
    ffs_field: torch.Tensor | None = None,
    ffs_radiated_power: torch.Tensor | None = None,
) -> TensorDataset:
    items = [tensors["points"], tensors["ports"], tensors["geom"], tensors["frame"], tensors["cuts"], tensors["nibs"], target]
    if ffs_coeff is not None:
        items.extend([ffs_coeff, ffs_field, ffs_radiated_power])
    if has_temporal:
        items.append(tensors["temporal"])
    if use_graph:
        items.extend(tensors[key] for key in GRAPH_KEYS)
    return TensorDataset(*items)


    parser.add_argument("--ffs-field-loss-weight", type=float, default=1.0)
    parser.add_argument("--ffs-power-loss-weight", type=float, default=0.25)
    train_loader = DataLoader(
        build_dataset(
            train_tensors,
            train_target,
            use_graph,
            has_temporal,
            train_ffs_coeff,
            train_tensors.get("ffs"),
            train_tensors.get("ffs_radiated_power"),
        ),
        batch_size=min(args.batch_size, len(train_records)),
        shuffle=True,
    )
    runtime_codec = TorchFfsCodec.from_state(ffs_codec_state, device=device) if ffs_codec_state is not None else None
    angle_grid = bundle.ffs_metadata.angles_deg.reshape(bundle.ffs_metadata.phi_count, bundle.ffs_metadata.theta_count, 2)
    phi_values = np.deg2rad(angle_grid[:, 0, 0])
    theta_values = np.deg2rad(angle_grid[0, :, 1])
    has_phi_closure = bool(
        bundle.ffs_metadata.phi_count > 1
        and np.isclose(phi_values[-1], phi_values[0] + 2.0 * np.pi, atol=1e-9, rtol=0.0)
    )
    if has_phi_closure:
        phi_values = phi_values[:-1]
    phi = torch.tensor(phi_values, dtype=torch.float32, device=device)
    theta = torch.tensor(theta_values, dtype=torch.float32, device=device)
    points, ports, geom, frame, cuts, nibs, target, *rest = batch
    ffs_coeff_target = None
    ffs_field_target = None
    ffs_radiated_target = None
    if use_ffs:
        ffs_coeff_target, ffs_field_target, ffs_radiated_target, *rest = rest
    if use_ffs:
        aux_loss, aux_parts = ffs_aux_loss(
            pred_coeff=output["ffs_coeff_pred"],
            target_coeff=ffs_coeff_target.to(device),
            target_field=ffs_field_target.to(device),
            target_radiated_power=ffs_radiated_target.to(device),
            codec=runtime_codec,
            phi=phi,
            theta=theta,
            phi_count=bundle.ffs_metadata.phi_count,
            theta_count=bundle.ffs_metadata.theta_count,
            has_phi_closure=has_phi_closure,
            loss_weights={
                "coeff": args.ffs_loss_weight,
                "field": args.ffs_field_loss_weight,
                "power": args.ffs_power_loss_weight,
            },
        )
        loss = loss + aux_loss
```

- [ ] **Step 4: Re-run the focused tests and verify they pass**

Run:

```bash
uv run --project apps/python python -m unittest packages/nijika/tests/test_optimizer_torch_farfield.py
uv run --project apps/python python -m unittest packages/nijika/tests/test_ffs_train_predict.py
```

Expected:
- both suites pass

- [ ] **Step 5: Commit**

```bash
git -C C:/Users/oxyfl/.config/superpowers/worktrees/GlassBeaker/nijika-dev add packages/nijika/optimizer_torch_farfield.py packages/nijika/baseline/training_utils.py packages/nijika/baseline/train.py packages/nijika/tests/test_optimizer_torch_farfield.py packages/nijika/tests/test_ffs_train_predict.py
git -C C:/Users/oxyfl/.config/superpowers/worktrees/GlassBeaker/nijika-dev commit -m "feat: add differentiable ffs training losses"
```

### Task 3: Shared Runtime Decode In Prediction And Optimizer

**Files:**
- Modify: `packages/nijika/baseline/predict.py`
- Modify: `packages/nijika/optimizer_runner.py`
- Modify: `packages/nijika/tests/test_ffs_train_predict.py`

- [ ] **Step 1: Write the failing runtime-decode regressions**

Extend `packages/nijika/tests/test_ffs_train_predict.py` so the exported `.ffs` file is parsed back and carries nonzero radiated power:

```python
from baseline.ffs_io import load_ffs_sample

            metadata, field = load_ffs_sample(ffs_dir / "port_01.ffs")
            self.assertEqual(field.shape[0], len(metadata.frequencies_hz))
            self.assertTrue((metadata.radiated_power_w > 0.0).all())
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
uv run --project apps/python python -m unittest packages/nijika/tests/test_ffs_train_predict.py
```

Expected:
- prediction export test fails because radiated power headers are still zero

- [ ] **Step 3: Replace ad hoc decode with shared runtime codec and narrow the export boundary**

In `packages/nijika/baseline/predict.py`, replace NumPy decode with torch decode first:

```python
from baseline.ffs_codec import TorchFfsCodec, codec_state_from_payload
from optimizer_torch_farfield import integrate_decoded_ffs_power


def _torch_ffs_codec(checkpoint: dict[str, object], *, device: torch.device) -> TorchFfsCodec | None:
    payload = checkpoint.get("ffs_codec")
    if payload is None:
        return None
    state = codec_state_from_payload(payload)
    return TorchFfsCodec.from_state(state, device=device)


def _angle_grid_tensors(metadata: FfsMetadata, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, bool]:
    grid = metadata.angles_deg.reshape(metadata.phi_count, metadata.theta_count, 2)
    phi_values = np.deg2rad(grid[:, 0, 0])
    theta_values = np.deg2rad(grid[0, :, 1])
    has_phi_closure = bool(
        metadata.phi_count > 1 and np.isclose(phi_values[-1], phi_values[0] + 2.0 * np.pi, atol=1e-9, rtol=0.0)
    )
    if has_phi_closure:
        phi_values = phi_values[:-1]
    return (
        torch.tensor(phi_values, dtype=dtype, device=device),
        torch.tensor(theta_values, dtype=dtype, device=device),
        has_phi_closure,
    )


def _export_predicted_ffs(
    *,
    output_dir: Path,
    sample_name: str,
    checkpoint: dict[str, object],
    coeff_pred: torch.Tensor,
) -> Path | None:
    codec = _torch_ffs_codec(checkpoint, device=coeff_pred.device)
    metadata = _ffs_metadata(checkpoint)
    if codec is None or metadata is None:
        return None
    phi, theta, has_phi_closure = _angle_grid_tensors(metadata, device=coeff_pred.device, dtype=coeff_pred.dtype)
    decoded = codec.decode(coeff_pred)[0]
    radiated = integrate_decoded_ffs_power(
        decoded.unsqueeze(0),
        phi=phi,
        theta=theta,
        phi_count=metadata.phi_count,
        theta_count=metadata.theta_count,
        has_phi_closure=has_phi_closure,
    )[0].detach().cpu().numpy()
    ffs_dir = output_dir / f"{sample_name}_predicted_ffs"
    for port_idx in range(decoded.shape[0]):
        header_for_port = FfsMetadata(
            frequencies_hz=metadata.frequencies_hz.copy(),
            angles_deg=metadata.angles_deg.copy(),
            radiated_power_w=radiated[port_idx],
            accepted_power_w=np.zeros_like(radiated[port_idx]),
            stimulated_power_w=np.maximum(metadata.stimulated_power_w, 1e-9),
            position_m=metadata.position_m.copy(),
            z_axis=metadata.z_axis.copy(),
            x_axis=metadata.x_axis.copy(),
            phi_count=metadata.phi_count,
            theta_count=metadata.theta_count,
        )
        write_ffs_sample(
            ffs_dir / f"port_{port_idx + 1:02d}.ffs",
            header_for_port,
            decoded[port_idx].detach().cpu().numpy(),
        )
    return ffs_dir
```

In `packages/nijika/optimizer_runner.py`, replace `_decode_ffs_coefficients()` with the same runtime codec contract:

```python
from baseline.ffs_codec import TorchFfsCodec, codec_state_from_payload


def _runtime_ffs_codec(checkpoint: dict[str, Any], device: torch.device, dtype: torch.dtype) -> TorchFfsCodec:
    codec = checkpoint.get("ffs_codec")
    if codec is None:
        raise ValueError("Checkpoint does not contain FFS codec metadata")
    state = codec_state_from_payload(codec)
    return TorchFfsCodec.from_state(state, dtype=dtype, device=device)


def _decode_ffs_coefficients(coeff_pred: torch.Tensor, checkpoint: dict[str, Any]) -> torch.Tensor:
    return _runtime_ffs_codec(checkpoint, coeff_pred.device, coeff_pred.dtype).decode(coeff_pred)
```

Only the final `.ffs` write path should call `detach().cpu().numpy()`.

- [ ] **Step 4: Re-run the focused tests and verify they pass**

Run:

```bash
uv run --project apps/python python -m unittest packages/nijika/tests/test_ffs_train_predict.py
```

Expected:
- the train/predict suite passes

- [ ] **Step 5: Commit**

```bash
git -C C:/Users/oxyfl/.config/superpowers/worktrees/GlassBeaker/nijika-dev add packages/nijika/baseline/predict.py packages/nijika/optimizer_runner.py packages/nijika/tests/test_ffs_train_predict.py
git -C C:/Users/oxyfl/.config/superpowers/worktrees/GlassBeaker/nijika-dev commit -m "refactor: share torch ffs decode path"
```

### Task 4: Farfield Optimization Regression Through FFS-Dependent Geometry

**Files:**
- Modify: `packages/nijika/tests/test_optimize_baseline.py`
- Modify: `packages/nijika/optimizer_runner.py`

- [ ] **Step 1: Write a failing regression where only the FFS path depends on geometry**

Add this toy surrogate to `packages/nijika/tests/test_optimize_baseline.py`:

```python
class _GeometryDrivenFfsSurrogate(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "base_coeff",
            torch.tensor(_toy_ffs_fields().reshape(-1), dtype=torch.float32),
            persistent=False,
        )

    def forward_with_aux(
        self,
        points: torch.Tensor,
        ports: torch.Tensor,
        geom: torch.Tensor,
        frame: torch.Tensor,
        cuts: torch.Tensor,
        nibs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del points, ports, geom, frame
        batch = cuts.size(0)
        s_pred = torch.zeros((batch, 3, 18), dtype=cuts.dtype, device=cuts.device)
        scale = 1.0 + 0.02 * cuts[..., 5].sum(dim=1, keepdim=True) + 0.01 * nibs[..., 5].sum(dim=1, keepdim=True)
        coeff = self.base_coeff.unsqueeze(0).expand(batch, -1) * scale
        return {"s_pred": s_pred, "ffs_coeff_pred": coeff}


    def test_farfield_mode_updates_geometry_through_ffs_path(self):
        checkpoint = {
            "freq_grid": [1.0e9, 1.5e9, 2.0e9],
            "port_count": 3,
            "sample_points": 8,
            "ffs_codec": _identity_codec(_toy_ffs_fields()),
            "ffs_metadata": _toy_ffs_metadata(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = optimize_model(
                model=_GeometryDrivenFfsSurrogate(),
                checkpoint=checkpoint,
                config=_toy_config(),
                output_dir=Path(tmp),
                steps=8,
                lr=0.1,
                top_k=1,
                efficiency_mode="farfield",
            )

        self.assertGreater(result["trace"][0]["loss"], result["trace"][-1]["loss"])
        self.assertNotAlmostEqual(result["trace"][0]["cut_distances"][0], result["trace"][-1]["cut_distances"][0], places=4)
```

- [ ] **Step 2: Run the focused optimizer test and verify it fails**

Run:

```bash
uv run --project apps/python python -m unittest packages/nijika/tests/test_optimize_baseline.py
```

Expected:
- the new geometry-through-FFS regression fails until the optimizer path uses the shared torch decode and keeps gradients intact

- [ ] **Step 3: Fix any remaining gradient breaks in `optimizer_runner.py`**

Keep the implementation changes minimal and inline. The only acceptable fixes here are the ones needed to preserve autograd through:

```python
model_inputs -> forward_with_aux -> ffs_coeff_pred -> runtime_codec.decode() -> decoded basis -> farfield power integration -> loss.backward()
```

If a fix is needed, it should look like this:

```python
decoded = runtime_codec.decode(aux["ffs_coeff_pred"])
basis_view = decoded_ffs_to_basis(
    decoded,
    phi_count=int(checkpoint["ffs_metadata"]["phi_count"]),
    theta_count=int(checkpoint["ffs_metadata"]["theta_count"]),
    has_phi_closure=has_phi_closure,
)[0][:, farfield_mask]
```

Do not reintroduce `torch.tensor(...)` reconstruction from Python lists inside the hot path after this task.

- [ ] **Step 4: Re-run the optimizer test and verify it passes**

Run:

```bash
uv run --project apps/python python -m unittest packages/nijika/tests/test_optimize_baseline.py
```

Expected:
- the full optimizer suite passes, including the geometry-through-FFS regression

- [ ] **Step 5: Commit**

```bash
git -C C:/Users/oxyfl/.config/superpowers/worktrees/GlassBeaker/nijika-dev add packages/nijika/tests/test_optimize_baseline.py packages/nijika/optimizer_runner.py
git -C C:/Users/oxyfl/.config/superpowers/worktrees/GlassBeaker/nijika-dev commit -m "test: prove farfield gradients reach geometry"
```

### Task 5: Final Verification

**Files:**
- No new files

- [ ] **Step 1: Run the focused regression suite**

Run:

```bash
uv run --project apps/python python -m unittest packages/nijika/tests/test_data_ffs.py
uv run --project apps/python python -m unittest packages/nijika/tests/test_ffs_codec.py
uv run --project apps/python python -m unittest packages/nijika/tests/test_optimizer_torch_farfield.py
uv run --project apps/python python -m unittest packages/nijika/tests/test_optimize_baseline.py
uv run --project apps/python python -m unittest packages/nijika/tests/test_ffs_train_predict.py
uv run --project apps/python python -m unittest packages/nijika/tests/test_train_regression.py
```

Expected:
- all suites pass

- [ ] **Step 2: Run one end-to-end smoke command**

Run:

```bash
uv run --project apps/python python packages/nijika/baseline/train.py --dataset-root tmp/dataset-v3-ffs --output-dir tmp/nijika-diff-ffs-smoke --epochs 1 --batch-size 1 --points 8 --freq-bins 11 --hidden-dim 16 --lr 1e-3 --model-kind structured_pair_spectral_ffs_head --ffs-rank 8 --ffs-loss-weight 1.0 --ffs-field-loss-weight 1.0 --ffs-power-loss-weight 0.25
uv run --project apps/python python packages/nijika/baseline/predict.py --dataset-root tmp/dataset-v3-ffs --model-path tmp/nijika-diff-ffs-smoke/baseline_model.pt --sample-name antenna_000 --output-dir tmp/nijika-diff-ffs-predict
uv run --project apps/python python packages/nijika/optimize_baseline.py --model-path tmp/nijika-diff-ffs-smoke/baseline_model.pt --config-path tmp/dataset-v3-ffs/antenna_000.json --output-dir tmp/nijika-diff-ffs-optimize --steps 2 --efficiency-mode farfield
```

Expected:
- training writes a checkpoint with `ffs_codec` and `ffs_metadata`
- prediction exports `.ffs` files without shape or metadata errors
- optimizer starts and writes `optimization_trace.json` in `farfield` mode

- [ ] **Step 3: Commit final cleanup if needed**

```bash
git -C C:/Users/oxyfl/.config/superpowers/worktrees/GlassBeaker/nijika-dev status --short
```

Expected:
- only intentional implementation changes remain
