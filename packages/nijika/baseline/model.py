from __future__ import annotations

import math
from typing import Any, Sequence
import torch
from torch import nn
from baseline.structured_pole_model import StructuredPoleResiduePredictor
from baseline.structured_spectral_model import StructuredSpectralPredictor
from baseline.structured_model import StructuredAntennaPredictor

def _pair_indices(port_count: int) -> list[tuple[int, int]]:
    return [(row, col) for row in range(port_count) for col in range(row, port_count)]

class PointNetLite(nn.Module):
    def __init__(self, point_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(point_dim, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.mlp(points).max(dim=1).values

class LegacySpectrumPredictor(nn.Module):
    def __init__(self, freq_bins: int, port_count: int, hidden_dim: int = 128):
        super().__init__()
        self.freq_bins = freq_bins
        self.port_count = port_count
        self.point_encoder = PointNetLite(hidden_dim=hidden_dim)
        input_dim = hidden_dim + port_count * 6 + 6
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, freq_bins * port_count * port_count * 2),
        )

    def forward(self, points: torch.Tensor, ports: torch.Tensor, geom: torch.Tensor, frame: torch.Tensor | None = None, cuts: torch.Tensor | None = None, nibs: torch.Tensor | None = None) -> torch.Tensor:
        point_latent = self.point_encoder(points)
        features = torch.cat([point_latent, ports.flatten(start_dim=1), geom], dim=1)
        output = self.head(self.trunk(features))
        return output.view(-1, self.freq_bins, self.port_count * self.port_count * 2)

class SpectrumPredictor(nn.Module):
    def __init__(self, freq_grid: Sequence[float] | torch.Tensor, port_count: int, hidden_dim: int = 128, dropout: float = 0.1, freq_bands: int = 8):
        super().__init__()
        self.port_count = port_count
        self.pairs = _pair_indices(port_count)
        self.point_encoder = PointNetLite(hidden_dim=hidden_dim)
        self.port_encoder = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.port_interaction = nn.TransformerEncoder(layer, num_layers=2)
        self.global_context = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        freq_dim = 1 + freq_bands * 2
        self.freq_encoder = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.register_buffer("freq_grid", torch.as_tensor(freq_grid, dtype=torch.float32), persistent=False)
        self.freq_bands = freq_bands

    def _port_features(self, ports: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        start = ports[..., :3]
        end = ports[..., 3:]
        center = (start + end) * 0.5
        delta = end - start
        scale = geom[:, None, 3:].clamp_min(1e-4)
        origin = geom[:, None, :3]
        start_local = (start - origin) / scale
        end_local = (end - origin) / scale
        center_local = (center - origin) / scale
        delta_local = delta / scale
        length = torch.linalg.vector_norm(delta_local, dim=-1, keepdim=True)
        scale_feat = scale.expand(-1, ports.size(1), -1)
        return torch.cat([start_local, end_local, center_local, delta_local, length, scale_feat], dim=-1)

    def _freq_features(self) -> torch.Tensor:
        freq = self.freq_grid
        freq = (freq - freq.min()) / (freq.max() - freq.min() + 1e-6)
        freq = freq * 2.0 - 1.0
        bands = (2.0 ** torch.arange(self.freq_bands, device=freq.device, dtype=freq.dtype)) * math.pi
        angles = freq.unsqueeze(-1) * bands
        return torch.cat([freq.unsqueeze(-1), torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, points: torch.Tensor, ports: torch.Tensor, geom: torch.Tensor, frame: torch.Tensor | None = None, cuts: torch.Tensor | None = None, nibs: torch.Tensor | None = None) -> torch.Tensor:
        scale = geom[:, None, 3:].clamp_min(1e-4)
        centered_points = (points - geom[:, None, :3]) / scale
        point_latent = self.point_encoder(centered_points)
        global_latent = self.global_context(torch.cat([point_latent, geom[:, 3:]], dim=1))
        port_tokens = self.port_encoder(self._port_features(ports, geom)) + global_latent.unsqueeze(1)
        port_tokens = self.port_interaction(port_tokens)
        pair_tokens = []
        for row, col in self.pairs:
            row_token = port_tokens[:, row]
            col_token = port_tokens[:, col]
            pair_tokens.append(torch.cat([row_token, col_token, torch.abs(row_token - col_token), row_token * col_token, global_latent], dim=1))
        pair_latent = self.pair_mlp(torch.stack(pair_tokens, dim=1))
        freq_latent = self.freq_encoder(self._freq_features())
        pair_latent = pair_latent.unsqueeze(1).expand(-1, freq_latent.size(0), -1, -1)
        freq_latent = freq_latent.unsqueeze(0).unsqueeze(2).expand(points.size(0), -1, len(self.pairs), -1)
        pair_output = self.decoder(torch.cat([pair_latent, freq_latent], dim=-1))
        full = torch.zeros(
            points.size(0),
            self.freq_grid.numel(),
            self.port_count,
            self.port_count,
            2,
            dtype=pair_output.dtype,
            device=pair_output.device,
        )
        for idx, (row, col) in enumerate(self.pairs):
            full[:, :, row, col] = pair_output[:, :, idx]
            full[:, :, col, row] = pair_output[:, :, idx]
        return full.view(points.size(0), self.freq_grid.numel(), self.port_count * self.port_count * 2)
def create_model(*, freq_grid: Sequence[float] | torch.Tensor, port_count: int, model_kind: str = "structured_pair_spectral_head", model_config: dict[str, Any] | None = None) -> nn.Module:
    config = model_config or {}
    hidden_dim = int(config.get("hidden_dim", 128))
    dropout = float(config.get("dropout", 0.1))
    if model_kind == "legacy_global_head":
        return LegacySpectrumPredictor(freq_bins=len(freq_grid), port_count=port_count, hidden_dim=hidden_dim)
    if model_kind == "symmetric_freq_decoder":
        return SpectrumPredictor(
            freq_grid=freq_grid,
            port_count=port_count,
            hidden_dim=hidden_dim,
            dropout=dropout,
            freq_bands=int(config.get("freq_bands", 8)),
        )
    if model_kind == "structured_token_decoder":
        return StructuredAntennaPredictor(
            freq_grid=freq_grid,
            port_count=port_count,
            hidden_dim=hidden_dim,
            dropout=dropout,
            freq_bands=int(config.get("freq_bands", 8)),
        )
    if model_kind == "structured_pair_spectral_head":
        return StructuredSpectralPredictor(
            freq_grid=freq_grid,
            port_count=port_count,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if model_kind == "structured_pair_split_decoder":
        return StructuredSpectralPredictor(
            freq_grid=freq_grid,
            port_count=port_count,
            hidden_dim=hidden_dim,
            dropout=dropout,
            split_decoder=True,
        )
    if model_kind == "structured_pair_pole_residue_head":
        return StructuredPoleResiduePredictor(
            freq_grid=freq_grid,
            port_count=port_count,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_poles=int(config.get("num_poles", 12)),
        )
    raise ValueError(f"Unsupported model kind: {model_kind}")
