from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from baseline.antenna_features import MAX_CUTS, MAX_NIBS
from baseline.transolver_encoder import TransolverEncoder


def _upper_triangle_pairs(port_count: int) -> list[tuple[int, int]]:
    return [(row, col) for row in range(port_count) for col in range(row, port_count)]


class TransolverSpectralPredictor(nn.Module):
    """S-parameter predictor with Transolver point-cloud geometry encoder.

    Replaces the tokenized frame/cuts/nibs geometry path with a
    TransolverEncoder that operates directly on sampled surface points.
    Port interaction and pair-wise spectral decoding are preserved from
    the structured predictor family.
    """

    def __init__(
        self,
        freq_grid: Sequence[float] | torch.Tensor,
        port_count: int,
        hidden_dim: int = 160,
        dropout: float = 0.1,
        max_cuts: int = MAX_CUTS,
        max_nibs: int = MAX_NIBS,
        num_slices: int = 32,
        num_encoder_layers: int = 3,
    ):
        super().__init__()
        self.port_count = port_count
        self.freq_bins = len(freq_grid)
        self.pairs = _upper_triangle_pairs(port_count)

        self.geometry_encoder = TransolverEncoder(
            hidden_dim=hidden_dim,
            num_slices=num_slices,
            num_layers=num_encoder_layers,
            dropout=dropout,
        )

        self.port_encoder = nn.Sequential(
            nn.Linear(24, hidden_dim),
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
        self.port_mixer = nn.TransformerEncoder(layer, num_layers=3)

        self.global_context = nn.Sequential(
            nn.Linear(hidden_dim + 6 + 3, hidden_dim),
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
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
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
        start_local = (start - origin) / scale
        end_local = (end - origin) / scale
        center_local = (center - origin) / scale
        delta_local = delta / scale
        length = torch.linalg.vector_norm(delta_local, dim=-1, keepdim=True)
        scale_feat = scale.expand(-1, ports.size(1), -1)
        return torch.cat(
            [start_local, end_local, center_local, delta_local, length, scale_feat],
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
        **kwargs,
    ) -> torch.Tensor:
        scale = geom[:, None, 3:].clamp_min(1e-4)
        origin = geom[:, None, :3]
        normalized_points = (points - origin) / scale

        geometry_latent = self.geometry_encoder(normalized_points)

        port_feats = self._port_features(ports, geom)
        if nibs is not None:
            port_feats = torch.cat([port_feats, nibs[:, : ports.size(1)]], dim=-1)
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
            pair_tokens.append(
                torch.cat(
                    [rt, ct, torch.abs(rt - ct), rt * ct, global_latent], dim=1
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
