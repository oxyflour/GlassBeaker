from __future__ import annotations

import numpy as np
import torch

from baseline.models.graph.topology import GraphTopologyBuilder


def build_graph_features_np(
    *,
    frame: np.ndarray,
    cuts: np.ndarray,
    nibs: np.ndarray,
    ports: np.ndarray,
    geom: np.ndarray,
    port_count: int,
) -> dict[str, np.ndarray]:
    builder = GraphTopologyBuilder(port_count=port_count)
    topo = builder.build(
        frame=torch.tensor(frame, dtype=torch.float32).unsqueeze(0),
        cuts=torch.tensor(cuts, dtype=torch.float32).unsqueeze(0),
        nibs=torch.tensor(nibs, dtype=torch.float32).unsqueeze(0),
        ports=torch.tensor(ports, dtype=torch.float32).unsqueeze(0),
        geom=torch.tensor(geom, dtype=torch.float32).unsqueeze(0),
    )
    return {
        "graph_inner": topo["inner_raw"].squeeze(0).numpy(),
        "graph_segment": topo["segment_raw"].squeeze(0).numpy(),
        "graph_port": topo["port_raw"].squeeze(0).numpy(),
        "graph_mask": topo["node_mask"].squeeze(0).numpy(),
        "graph_adj": topo["adj"].squeeze(0).numpy(),
        "graph_edge_attr": topo["edge_attr"].squeeze(0).numpy(),
        "pair_topology": topo["pair_topology"].squeeze(0).numpy(),
    }
