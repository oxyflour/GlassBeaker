from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from baseline.antenna_features import MAX_CUTS, MAX_NIBS


def _upper_triangle_pairs(port_count: int) -> list[tuple[int, int]]:
    return [(row, col) for row in range(port_count) for col in range(row, port_count)]


class StructuredAntennaPredictor(nn.Module):
    def __init__(
        self,
        freq_grid: Sequence[float] | torch.Tensor,
        port_count: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        freq_bands: int = 8,
        max_cuts: int = MAX_CUTS,
        max_nibs: int = MAX_NIBS,
    ):
        super().__init__()
        self.port_count = port_count
        self.max_cuts = max_cuts
        self.max_nibs = max_nibs
        self.pairs = _upper_triangle_pairs(port_count)
        self.frame_encoder = nn.Sequential(nn.Linear(6, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.cut_encoder = nn.Sequential(nn.Linear(7, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.nib_encoder = nn.Sequential(nn.Linear(8, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.port_encoder = nn.Sequential(nn.Linear(24, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.token_type = nn.Embedding(4, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.token_mixer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.port_refiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.global_context = nn.Sequential(
            nn.Linear(hidden_dim + 6 + 3, hidden_dim),
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
        self.register_buffer("pair_index", torch.tensor(self.pairs, dtype=torch.long), persistent=False)
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

    def _masked_mean(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = (~mask).unsqueeze(-1)
        summed = (tokens * valid).sum(dim=1)
        counts = valid.sum(dim=1).clamp_min(1)
        return summed / counts

    def _type_features(self, batch_size: int, device: torch.device) -> torch.Tensor:
        type_ids = torch.tensor(
            [0] + [1] * self.max_cuts + [2] * self.max_nibs + [3] * self.port_count,
            dtype=torch.long,
            device=device,
        )
        return self.token_type(type_ids).unsqueeze(0).expand(batch_size, -1, -1)

    def forward(
        self,
        points: torch.Tensor,
        ports: torch.Tensor,
        geom: torch.Tensor,
        frame: torch.Tensor | None = None,
        cuts: torch.Tensor | None = None,
        nibs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del points
        if frame is None or cuts is None or nibs is None:
            raise ValueError("StructuredAntennaPredictor requires frame/cuts/nibs features")
        port_nibs = nibs[:, : ports.size(1)]
        frame_token = self.frame_encoder(frame).unsqueeze(1)
        cut_tokens = self.cut_encoder(cuts)
        nib_tokens = self.nib_encoder(nibs)
        port_tokens = self.port_encoder(torch.cat([self._port_features(ports, geom), port_nibs], dim=-1))
        tokens = torch.cat([frame_token, cut_tokens, nib_tokens, port_tokens], dim=1)
        tokens = tokens + self._type_features(tokens.size(0), tokens.device)
        geom_mask = torch.cat(
            [
                torch.zeros(frame.size(0), 1, dtype=torch.bool, device=frame.device),
                cuts[..., 0] < 0.5,
                nibs[..., 0] < 0.5,
            ],
            dim=1,
        )
        port_mask = torch.zeros(frame.size(0), ports.size(1), dtype=torch.bool, device=frame.device)
        token_mask = torch.cat([geom_mask, port_mask], dim=1)
        tokens = self.token_mixer(tokens, src_key_padding_mask=token_mask)
        geometry_tokens = tokens[:, : 1 + self.max_cuts + self.max_nibs]
        geometry_latent = self._masked_mean(geometry_tokens, geom_mask)
        global_latent = self.global_context(torch.cat([geometry_latent, frame, geom[:, 3:]], dim=1))
        port_tokens = tokens[:, -ports.size(1) :]
        port_tokens = self.port_refiner(torch.cat([port_tokens, global_latent.unsqueeze(1).expand_as(port_tokens)], dim=-1))
        pair_tokens = []
        for row, col in self.pairs:
            row_token = port_tokens[:, row]
            col_token = port_tokens[:, col]
            pair_tokens.append(
                torch.cat(
                    [row_token, col_token, torch.abs(row_token - col_token), row_token * col_token, global_latent],
                    dim=1,
                )
            )
        pair_latent = self.pair_mlp(torch.stack(pair_tokens, dim=1))
        freq_latent = self.freq_encoder(self._freq_features())
        pair_latent = pair_latent.unsqueeze(1).expand(-1, freq_latent.size(0), -1, -1)
        freq_latent = freq_latent.unsqueeze(0).unsqueeze(2).expand(frame.size(0), -1, len(self.pairs), -1)
        pair_output = self.decoder(torch.cat([pair_latent, freq_latent], dim=-1))
        full = torch.zeros(
            frame.size(0),
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
        return full.view(frame.size(0), self.freq_grid.numel(), self.port_count * self.port_count * 2)
