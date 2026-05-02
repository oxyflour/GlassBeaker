from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from baseline.antenna_features import MAX_CUTS
from baseline.models.graph.layers import GraphMessageBlock
from baseline.models.graph.topology import GraphTopologyBuilder, MAX_SEGMENTS


def _upper_triangle_pairs(port_count: int) -> list[tuple[int, int]]:
    return [(row, col) for row in range(port_count) for col in range(row, port_count)]


class GraphTopologySpectralPredictor(nn.Module):
    def __init__(
        self,
        freq_grid: Sequence[float] | torch.Tensor,
        port_count: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        max_cuts: int = MAX_CUTS,
    ):
        super().__init__()
        del max_cuts
        self.uses_graph_features = True
        self.port_count = port_count
        self.max_segments = MAX_SEGMENTS
        self.freq_bins = len(freq_grid)
        self.pairs = _upper_triangle_pairs(port_count)
        self.topology = GraphTopologyBuilder(port_count=port_count, max_segments=self.max_segments)
        self.inner_encoder = nn.Sequential(nn.Linear(9, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.segment_encoder = nn.Sequential(nn.Linear(10, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.port_encoder = nn.Sequential(nn.Linear(28, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.type_embedding = nn.Embedding(3, hidden_dim)
        self.blocks = nn.ModuleList([GraphMessageBlock(hidden_dim=hidden_dim, edge_dim=6) for _ in range(3)])
        self.global_context = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pair_topology_encoder = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.GELU(),
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

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        active = mask.unsqueeze(-1)
        return (values * active).sum(dim=1) / active.sum(dim=1).clamp_min(1)

    def _type_ids(self, batch_size: int, device: torch.device) -> torch.Tensor:
        ids = [0] + [1] * self.max_segments + [2] * self.port_count
        return self.type_embedding(torch.tensor(ids, dtype=torch.long, device=device)).unsqueeze(0).expand(batch_size, -1, -1)

    def forward(
        self,
        points: torch.Tensor,
        ports: torch.Tensor,
        geom: torch.Tensor,
        frame: torch.Tensor | None = None,
        cuts: torch.Tensor | None = None,
        nibs: torch.Tensor | None = None,
        graph_inner: torch.Tensor | None = None,
        graph_segment: torch.Tensor | None = None,
        graph_port: torch.Tensor | None = None,
        graph_mask: torch.Tensor | None = None,
        graph_adj: torch.Tensor | None = None,
        graph_edge_attr: torch.Tensor | None = None,
        pair_topology: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del points
        if frame is None or cuts is None or nibs is None:
            raise ValueError("GraphTopologySpectralPredictor requires frame/cuts/nibs features")
        if graph_inner is None:
            topo = self.topology.build(frame=frame, cuts=cuts, nibs=nibs, ports=ports, geom=geom)
        else:
            topo = {
                "inner_raw": graph_inner,
                "segment_raw": graph_segment,
                "port_raw": graph_port,
                "node_mask": graph_mask,
                "adj": graph_adj,
                "edge_attr": graph_edge_attr,
                "pair_topology": pair_topology,
            }
        nodes = torch.cat(
            [
                self.inner_encoder(topo["inner_raw"]),
                self.segment_encoder(topo["segment_raw"]),
                self.port_encoder(topo["port_raw"]),
            ],
            dim=1,
        )
        nodes = nodes + self._type_ids(nodes.size(0), nodes.device)
        node_mask = topo["node_mask"]
        for block in self.blocks:
            nodes = block(nodes, node_mask, topo["adj"], topo["edge_attr"])
        global_latent = self.global_context(torch.cat([self._masked_mean(nodes, node_mask.to(nodes.dtype)), geom[:, 3:]], dim=1))
        port_nodes = nodes[:, 1 + self.max_segments : 1 + self.max_segments + self.port_count]
        pair_topology = self.pair_topology_encoder(topo["pair_topology"])
        pair_tokens = []
        for pair_idx, (row, col) in enumerate(self.pairs):
            row_node = port_nodes[:, row]
            col_node = port_nodes[:, col]
            pair_tokens.append(
                torch.cat(
                    [row_node, col_node, torch.abs(row_node - col_node), row_node * col_node, pair_topology[:, pair_idx], global_latent],
                    dim=1,
                )
            )
        pair_latent = self.pair_mlp(torch.stack(pair_tokens, dim=1))
        pair_output = self.spectral_decoder(pair_latent).view(frame.size(0), len(self.pairs), self.freq_bins, 2)
        pair_output = pair_output.permute(0, 2, 1, 3)
        full = torch.zeros(
            frame.size(0),
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
        return full.view(frame.size(0), self.freq_bins, self.port_count * self.port_count * 2)
