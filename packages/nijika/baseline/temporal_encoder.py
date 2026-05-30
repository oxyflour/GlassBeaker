from __future__ import annotations

import torch
from torch import nn


class TimeDomainEncoder(nn.Module):
    """1D ConvNet encoding FDTD time-domain probe signals into compact features.

    Shared across all port-pair signals. Compresses T time steps into a
    single feature vector per port pair via strided convolutions.
    """

    def __init__(self, hidden_dim: int = 160):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(128, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        B, P, T = signals.shape
        x = signals.view(B * P, 1, T)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return x.view(B, P, -1)
