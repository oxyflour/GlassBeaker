from __future__ import annotations

import torch
from torch import nn


class PhysicsAttention(nn.Module):
    """Transolver-style learnable-slice bottleneck attention.

    K learnable slice tokens compress an N-point geometry into a compact
    physics-aware latent via three steps: points→slices cross-attention,
    slice self-attention, and slices→points cross-attention.
    """

    def __init__(
        self,
        hidden_dim: int = 160,
        num_slices: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_slices = num_slices
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.slice_tokens = nn.Parameter(torch.randn(1, num_slices, hidden_dim) * 0.02)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.slice_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.slice_out = nn.Linear(hidden_dim, hidden_dim)

        self.slice_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.point_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm_slice_attn = nn.LayerNorm(hidden_dim)
        self.norm_slice_ffn = nn.LayerNorm(hidden_dim)
        self.norm_slice_self = nn.LayerNorm(hidden_dim)
        self.norm_point_attn = nn.LayerNorm(hidden_dim)

    def _cross_attn(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, L, D = q.shape
        S = kv.size(1)
        qp = self.q_proj(q).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        kp = self.k_proj(kv).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        vp = self.v_proj(kv).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (qp @ kp.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ vp).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, D = points.shape
        slices = self.slice_tokens.expand(B, -1, -1)

        slices = slices + self._cross_attn(self.norm_slice_attn(slices), points)
        slices = slices + self.slice_ffn(self.norm_slice_ffn(slices))

        qkv = self.slice_qkv(self.norm_slice_self(slices))
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        slices = slices + self.slice_out(
            (attn @ v).transpose(1, 2).contiguous().view(B, -1, D)
        )

        points = points + self._cross_attn(self.norm_point_attn(points), slices)
        points = points + self.point_ffn(points)

        return points, slices


class TransolverEncoder(nn.Module):
    """Geometry encoder: point cloud → Transolver blocks → global latent."""

    def __init__(
        self,
        hidden_dim: int = 160,
        num_slices: int = 32,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.point_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                PhysicsAttention(hidden_dim, num_slices, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        feats = self.point_embed(points)
        slices = feats.new_zeros(feats.size(0), 0, feats.size(-1))
        for block in self.blocks:
            feats, slices = block(feats)
        return self.global_pool(
            torch.cat([feats.max(dim=1).values, slices.mean(dim=1)], dim=-1)
        )
