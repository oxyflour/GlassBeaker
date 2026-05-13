from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from baseline.antenna_features import MAX_CUTS, MAX_NIBS


def _upper_triangle_pairs(port_count: int) -> list[tuple[int, int]]:
    return [(row, col) for row in range(port_count) for col in range(row, port_count)]


class StructuredSpectralPredictor(nn.Module):
    def __init__(
        self,
        freq_grid: Sequence[float] | torch.Tensor,
        port_count: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        max_cuts: int = MAX_CUTS,
        max_nibs: int = MAX_NIBS,
        split_decoder: bool = False,
        use_pair_topology: bool = False,
        ffs_coeff_dim: int = 0,
    ):
        super().__init__()
        self.port_count = port_count
        self.max_cuts = max_cuts
        self.max_nibs = max_nibs
        self.freq_bins = len(freq_grid)
        self.ffs_coeff_dim = ffs_coeff_dim
        self.pairs = _upper_triangle_pairs(port_count)
        self.split_decoder = split_decoder
        self.use_pair_topology = use_pair_topology
        self.uses_graph_features = use_pair_topology
        if use_pair_topology:
            self.graph_feature_keys = ("pair_topology",)
        self.frame_encoder = nn.Sequential(nn.Linear(6, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.cut_encoder = nn.Sequential(nn.Linear(7, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.nib_encoder = nn.Sequential(nn.Linear(8, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.port_encoder = nn.Sequential(nn.Linear(24, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.token_type = nn.Embedding(4, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.token_mixer = nn.TransformerEncoder(layer, num_layers=3)
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
        if use_pair_topology:
            self.pair_topology_encoder = nn.Sequential(
                nn.Linear(8, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.pair_topology_adapter = nn.Linear(hidden_dim, hidden_dim)
            nn.init.zeros_(self.pair_topology_adapter.weight)
            nn.init.zeros_(self.pair_topology_adapter.bias)
            self.register_buffer(
                "pair_coupling_mask",
                torch.tensor([0.0 if row == col else 1.0 for row, col in self.pairs], dtype=torch.float32).view(1, -1, 1),
                persistent=False,
            )
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        if split_decoder:
            self.diag_indices = [i for i, (r, c) in enumerate(self.pairs) if r == c]
            self.offdiag_indices = [i for i, (r, c) in enumerate(self.pairs) if r != c]
            self.diag_decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, self.freq_bins * 2),
            )
            self.coupling_decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, self.freq_bins * 2),
            )
        else:
            self.spectral_decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, self.freq_bins * 2),
            )
        self.ffs_decoder = None
        if ffs_coeff_dim > 0:
            self.ffs_decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, ffs_coeff_dim),
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
        return torch.cat([start_local, end_local, center_local, delta_local, length, scale_feat], dim=-1)

    def _masked_mean(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = (~mask).unsqueeze(-1)
        return (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    def _type_features(self, batch_size: int, device: torch.device) -> torch.Tensor:
        type_ids = torch.tensor(
            [0] + [1] * self.max_cuts + [2] * self.max_nibs + [3] * self.port_count,
            dtype=torch.long,
            device=device,
        )
        return self.token_type(type_ids).unsqueeze(0).expand(batch_size, -1, -1)

    def _predict_outputs(
        self,
        points: torch.Tensor,
        ports: torch.Tensor,
        geom: torch.Tensor,
        frame: torch.Tensor | None = None,
        cuts: torch.Tensor | None = None,
        nibs: torch.Tensor | None = None,
        pair_topology: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del points
        if frame is None or cuts is None or nibs is None:
            raise ValueError("StructuredSpectralPredictor requires frame/cuts/nibs features")
        if self.use_pair_topology and pair_topology is None:
            raise ValueError("StructuredSpectralPredictor requires pair_topology features")
        frame_token = self.frame_encoder(frame).unsqueeze(1)
        cut_tokens = self.cut_encoder(cuts)
        nib_tokens = self.nib_encoder(nibs)
        port_tokens = self.port_encoder(torch.cat([self._port_features(ports, geom), nibs[:, : ports.size(1)]], dim=-1))
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
        token_mask = torch.cat(
            [geom_mask, torch.zeros(frame.size(0), ports.size(1), dtype=torch.bool, device=frame.device)],
            dim=1,
        )
        tokens = self.token_mixer(tokens, src_key_padding_mask=token_mask)
        geometry_latent = self._masked_mean(tokens[:, : 1 + self.max_cuts + self.max_nibs], geom_mask)
        global_latent = self.global_context(torch.cat([geometry_latent, frame, geom[:, 3:]], dim=1))
        port_tokens = tokens[:, -ports.size(1) :]
        port_tokens = self.port_refiner(torch.cat([port_tokens, global_latent.unsqueeze(1).expand_as(port_tokens)], dim=-1))
        pair_topology_latent = self.pair_topology_encoder(pair_topology) if self.use_pair_topology else None
        pair_tokens = []
        for row, col in self.pairs:
            row_token = port_tokens[:, row]
            col_token = port_tokens[:, col]
            pair_tokens.append(torch.cat([row_token, col_token, torch.abs(row_token - col_token), row_token * col_token, global_latent], dim=1))
        pair_latent = self.pair_mlp(torch.stack(pair_tokens, dim=1))
        if pair_topology_latent is not None:
            pair_delta = self.pair_topology_adapter(pair_topology_latent) * self.pair_coupling_mask
            pair_latent = pair_latent + pair_delta
        if self.split_decoder:
            pair_output = pair_latent.new_zeros(frame.size(0), len(self.pairs), self.freq_bins, 2)
            diag_latent = pair_latent[:, self.diag_indices]
            pair_output[:, self.diag_indices] = self.diag_decoder(diag_latent).view(
                frame.size(0), len(self.diag_indices), self.freq_bins, 2
            )
            offdiag_latent = pair_latent[:, self.offdiag_indices]
            pair_output[:, self.offdiag_indices] = self.coupling_decoder(offdiag_latent).view(
                frame.size(0), len(self.offdiag_indices), self.freq_bins, 2
            )
        else:
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
        outputs = {"s_pred": full.view(frame.size(0), self.freq_bins, self.port_count * self.port_count * 2)}
        if self.ffs_decoder is not None:
            outputs["ffs_coeff_pred"] = self.ffs_decoder(global_latent)
        return outputs

    def forward(
        self,
        points: torch.Tensor,
        ports: torch.Tensor,
        geom: torch.Tensor,
        frame: torch.Tensor | None = None,
        cuts: torch.Tensor | None = None,
        nibs: torch.Tensor | None = None,
        pair_topology: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._predict_outputs(points, ports, geom, frame, cuts, nibs, pair_topology)["s_pred"]

    def forward_with_aux(
        self,
        points: torch.Tensor,
        ports: torch.Tensor,
        geom: torch.Tensor,
        frame: torch.Tensor | None = None,
        cuts: torch.Tensor | None = None,
        nibs: torch.Tensor | None = None,
        pair_topology: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self._predict_outputs(points, ports, geom, frame, cuts, nibs, pair_topology)
