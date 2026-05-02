from __future__ import annotations

import torch
from torch import nn


class GraphMessageBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, relation_count: int = 3):
        super().__init__()
        self.rel_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(relation_count)
            ]
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_state: torch.Tensor,
        node_mask: torch.Tensor,
        adj: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        batch, node_count, hidden_dim = node_state.shape
        src = node_state.unsqueeze(1).expand(batch, node_count, node_count, hidden_dim)
        dst = node_state.unsqueeze(2).expand(batch, node_count, node_count, hidden_dim)
        messages = torch.zeros_like(node_state)
        for rel_idx, rel_mlp in enumerate(self.rel_mlps):
            rel_pair = torch.cat([src, dst, edge_attr[:, rel_idx]], dim=-1)
            rel_msg = rel_mlp(rel_pair) * adj[:, rel_idx].unsqueeze(-1)
            messages = messages + rel_msg.sum(dim=2)
        updated = self.update(torch.cat([node_state, messages], dim=-1))
        updated = self.norm(node_state + updated)
        return updated * node_mask.unsqueeze(-1)
