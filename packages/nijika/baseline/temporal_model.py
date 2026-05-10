from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from baseline.temporal_encoder import TimeDomainEncoder
from baseline.transolver_encoder import TransolverEncoder


def _upper_triangle_pairs(port_count: int) -> list[tuple[int, int]]:
    return [(row, col) for row in range(port_count) for col in range(row, port_count)]


class TemporalSpectralPredictor(nn.Module):
    """Predict S-parameters from early FDTD time-domain signals + geometry.

    Combines three information sources:
    - Time-domain probe signals (first K steps of 9 port-pair waveforms)
    - Point cloud geometry via Transolver encoder
    - Port coordinates (normalized positions)
    """

    def __init__(
        self,
        freq_grid: Sequence[float] | torch.Tensor,
        port_count: int,
        hidden_dim: int = 160,
        dropout: float = 0.1,
        num_slices: int = 32,
        num_encoder_layers: int = 3,
    ):
        super().__init__()
        self.port_count = port_count
        self.freq_bins = len(freq_grid)
        self.pairs = _upper_triangle_pairs(port_count)

        self.temporal_encoder = TimeDomainEncoder(hidden_dim=hidden_dim)

        self.geometry_encoder = TransolverEncoder(
            hidden_dim=hidden_dim,
            num_slices=num_slices,
            num_layers=num_encoder_layers,
            dropout=dropout,
        )

        self.port_encoder = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.port_mixer = nn.TransformerEncoder(layer, num_layers=2)

        self.global_context = nn.Sequential(
            nn.Linear(hidden_dim + 6 + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.temporal_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.port_refiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.spectral_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, self.freq_bins * 2),
        )

    def _port_features(self, ports: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        start = ports[..., :3]
        end = ports[..., 3:]
        center = (start + end) * 0.5
        delta = end - start
        scale = geom[:, None, 3:].clamp_min(1e-4)
        origin = geom[:, None, :3]
        scale_feat = scale.expand(-1, ports.size(1), -1)
        return torch.cat(
            [
                (start - origin) / scale,
                (end - origin) / scale,
                (center - origin) / scale,
                delta / scale,
                torch.linalg.vector_norm(delta / scale, dim=-1, keepdim=True),
                scale_feat,
            ],
            dim=-1,
        )

    def forward(
        self,
        points: torch.Tensor,
        ports: torch.Tensor,
        geom: torch.Tensor,
        frame: torch.Tensor | None = None,
        cuts: torch.Tensor | None = None,
        nibs: torch.Tensor | None = None,
        temporal: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if temporal is None:
            raise ValueError("TemporalSpectralPredictor requires temporal signals")

        scale = geom[:, None, 3:].clamp_min(1e-4)
        normalized_points = (points - geom[:, None, :3]) / scale

        temporal_feats = self.temporal_encoder(temporal)
        temporal_feats = self.temporal_adapter(temporal_feats)

        geometry_latent = self.geometry_encoder(normalized_points)

        port_feats = self._port_features(ports, geom)
        port_tokens = self.port_encoder(port_feats)
        port_tokens = self.port_mixer(port_tokens)

        global_latent = self.global_context(
            torch.cat([geometry_latent, frame, geom[:, 3:]], dim=-1)
        )

        port_tokens = self.port_refiner(
            torch.cat(
                [port_tokens, global_latent.unsqueeze(1).expand_as(port_tokens)],
                dim=-1,
            )
        )

        pair_tokens = []
        for row, col in self.pairs:
            rt = port_tokens[:, row]
            ct = port_tokens[:, col]
            tt = temporal_feats[:, row * self.port_count + col]
            pair_tokens.append(
                torch.cat(
                    [rt, ct, torch.abs(rt - ct), rt * ct, tt, global_latent], dim=1
                )
            )
        pair_latent = self.pair_mlp(torch.stack(pair_tokens, dim=1))

        pair_output = self.spectral_decoder(pair_latent).view(
            pair_latent.size(0), len(self.pairs), self.freq_bins, 2
        )
        pair_output = pair_output.permute(0, 2, 1, 3)

        full = torch.zeros(
            pair_latent.size(0),
            self.freq_bins,
            self.port_count,
            self.port_count,
            2,
            dtype=pair_output.dtype,
            device=pair_output.device,
        )
        for idx, (row, col) in enumerate(self.pairs):
            full[:, :, row, col] = pair_output[:, :, idx]
            full[:, :, col, row] = pair_output[:, :, idx]

        return full.view(
            pair_latent.size(0),
            self.freq_bins,
            self.port_count * self.port_count * 2,
        )
