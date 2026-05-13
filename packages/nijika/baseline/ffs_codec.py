from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class FfsCodecConfig:
    field_shape: tuple[int, ...]
    flat_dim: int
    rank: int


@dataclass(frozen=True)
class FfsCodecState:
    config: FfsCodecConfig
    mean: np.ndarray
    basis: np.ndarray


class TorchFfsCodec(nn.Module):
    def __init__(self, field_shape: tuple[int, ...], flat_dim: int, rank: int, mean: torch.Tensor, basis: torch.Tensor):
        super().__init__()
        self.field_shape = tuple(int(size) for size in field_shape)
        self.flat_dim = int(flat_dim)
        self.rank = int(rank)
        self.register_buffer("mean", mean.reshape(self.flat_dim))
        self.register_buffer("basis", basis.reshape(self.rank, self.flat_dim))

    @classmethod
    def from_state(
        cls,
        state: FfsCodecState,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> TorchFfsCodec:
        mean = torch.as_tensor(state.mean, dtype=dtype, device=device)
        basis = torch.as_tensor(state.basis, dtype=dtype, device=device)
        return cls(
            field_shape=state.config.field_shape,
            flat_dim=state.config.flat_dim,
            rank=state.config.rank,
            mean=mean,
            basis=basis,
        )

    def encode(self, fields: torch.Tensor) -> torch.Tensor:
        if fields.ndim < 2:
            raise ValueError("fields must include a batch axis and at least one field axis")
        if tuple(fields.shape[1:]) != self.field_shape:
            raise ValueError("field shape does not match codec state")
        matrix = fields.reshape(fields.shape[0], self.flat_dim).to(dtype=self.mean.dtype, device=self.mean.device)
        return (matrix - self.mean) @ self.basis.T

    def decode(self, coeffs: torch.Tensor) -> torch.Tensor:
        coeff_tensor = coeffs.to(dtype=self.mean.dtype, device=self.mean.device)
        if coeff_tensor.ndim == 1:
            coeff_tensor = coeff_tensor.unsqueeze(0)
        if coeff_tensor.shape[-1] != self.rank:
            raise ValueError("coefficient shape does not match codec rank")
        flat = coeff_tensor @ self.basis + self.mean
        return flat.reshape(coeff_tensor.shape[0], *self.field_shape)


def fit_ffs_codec(fields: np.ndarray, rank: int) -> FfsCodecState:
    matrix, field_shape = _flatten_fields(fields)
    if rank < 1:
        raise ValueError("rank must be at least 1")
    actual_rank = min(rank, matrix.shape[0], matrix.shape[1])
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:actual_rank].copy()
    config = FfsCodecConfig(field_shape=field_shape, flat_dim=matrix.shape[1], rank=actual_rank)
    return FfsCodecState(config=config, mean=mean, basis=basis)


def encode_ffs(fields: np.ndarray, state: FfsCodecState) -> np.ndarray:
    matrix, field_shape = _flatten_fields(fields)
    if field_shape != state.config.field_shape:
        raise ValueError("field shape does not match codec state")
    centered = matrix - state.mean
    return centered @ state.basis.T


def decode_ffs(coeffs: np.ndarray, state: FfsCodecState) -> np.ndarray:
    coeff_array = np.asarray(coeffs, dtype=np.float64)
    if coeff_array.ndim == 1:
        coeff_array = coeff_array[np.newaxis, :]
    if coeff_array.shape[-1] != state.config.rank:
        raise ValueError("coefficient shape does not match codec rank")
    flat = coeff_array @ state.basis + state.mean
    return flat.reshape(coeff_array.shape[0], *state.config.field_shape)


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


def _flatten_fields(fields: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    array = np.asarray(fields, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError("fields must include a batch axis and at least one field axis")
    field_shape = tuple(int(size) for size in array.shape[1:])
    return array.reshape(array.shape[0], -1), field_shape
