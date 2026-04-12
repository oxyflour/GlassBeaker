from __future__ import annotations

import torch
from torch import nn


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
        point_features = self.mlp(points)
        return point_features.max(dim=1).values


class SpectrumPredictor(nn.Module):
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

    def forward(self, points: torch.Tensor, ports: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        point_latent = self.point_encoder(points)
        features = torch.cat([point_latent, ports.flatten(start_dim=1), geom], dim=1)
        latent = self.trunk(features)
        output = self.head(latent)
        return output.view(-1, self.freq_bins, self.port_count * self.port_count * 2)

