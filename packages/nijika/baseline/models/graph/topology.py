from __future__ import annotations

import torch
from baseline.antenna_features import MAX_CUTS
MAX_SEGMENTS = max(MAX_CUTS, 1)

def _port_features(ports: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
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

def _edge_coord(side: torch.Tensor, distance: torch.Tensor, width: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    top = distance + width * 0.5
    right = width + height * 0.5 - distance
    bottom = width + height + width * 0.5 - distance
    left = width * 2 + height + distance + height * 0.5
    return torch.where(side == 2, top, torch.where(side == 1, right, torch.where(side == 3, bottom, left)))

def _side_from_coord(coord: torch.Tensor, width: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    return torch.where(
        coord < width,
        torch.full_like(coord, 2, dtype=torch.long),
        torch.where(
            coord < width + height,
            torch.full_like(coord, 1, dtype=torch.long),
            torch.where(coord < width * 2 + height, torch.full_like(coord, 3, dtype=torch.long), torch.zeros_like(coord, dtype=torch.long)),
        ),
    )


class GraphTopologyBuilder:
    def __init__(self, port_count: int, max_segments: int = MAX_SEGMENTS):
        self.port_count = port_count
        self.max_segments = max_segments
        self.pairs = [(row, col) for row in range(port_count) for col in range(row, port_count)]

    def build(
        self,
        frame: torch.Tensor,
        cuts: torch.Tensor,
        nibs: torch.Tensor,
        ports: torch.Tensor,
        geom: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del frame
        batch = geom.size(0)
        device = geom.device
        dtype = geom.dtype
        node_count = 1 + self.max_segments + self.port_count
        inner = torch.cat([geom[:, 3:], cuts.new_zeros(batch, 6)], dim=1).unsqueeze(1)
        segments = cuts.new_zeros(batch, self.max_segments, 10)
        ports_raw = cuts.new_zeros(batch, self.port_count, 28)
        node_mask = torch.zeros(batch, node_count, dtype=torch.bool, device=device)
        node_mask[:, 0] = True
        port_mask = nibs[:, : self.port_count, 0] > 0.5
        node_mask[:, 1 + self.max_segments : 1 + self.max_segments + self.port_count] = port_mask
        adj = torch.zeros(batch, 3, node_count, node_count, dtype=dtype, device=device)
        edge_attr = torch.zeros(batch, 3, node_count, node_count, 6, dtype=dtype, device=device)
        pair_topology = torch.zeros(batch, len(self.pairs), 8, dtype=dtype, device=device)
        port_geom = _port_features(ports, geom)
        for batch_idx in range(batch):
            width = geom[batch_idx, 3].clamp_min(1e-4)
            height = geom[batch_idx, 4].clamp_min(1e-4)
            perimeter = (width + height) * 2.0
            active_cuts = torch.nonzero(cuts[batch_idx, :, 0] > 0.5, as_tuple=False).flatten()
            cut_count = int(active_cuts.numel())
            seg_count = max(1, cut_count)
            seg_starts = torch.zeros(self.max_segments, dtype=dtype, device=device)
            seg_lens = torch.zeros(self.max_segments, dtype=dtype, device=device)
            cut_widths = torch.zeros(max(cut_count, 1), dtype=dtype, device=device)
            cut_centers = torch.zeros(max(cut_count, 1), dtype=dtype, device=device)
            if cut_count > 0:
                cut_rows = cuts[batch_idx, active_cuts]
                side = cut_rows[:, 1:5].argmax(dim=1)
                cross = torch.where(side < 2, height, width)
                distance = cut_rows[:, 5] * 0.5 * cross
                width_abs = cut_rows[:, 6] * cross
                center = _edge_coord(side, distance, width, height)
                order = torch.argsort(center)
                cut_centers = center[order]
                cut_widths = width_abs[order]
                cut_starts = torch.remainder(cut_centers - cut_widths * 0.5, perimeter)
                cut_ends = torch.remainder(cut_centers + cut_widths * 0.5, perimeter)
                for seg_idx in range(seg_count):
                    seg_starts[seg_idx] = cut_ends[seg_idx]
                    seg_end = cut_starts[(seg_idx + 1) % cut_count]
                    seg_lens[seg_idx] = torch.remainder(seg_end - seg_starts[seg_idx], perimeter)
                    center_t = torch.remainder(seg_starts[seg_idx] + seg_lens[seg_idx] * 0.5, perimeter)
                    side_t = _side_from_coord(center_t, width, height)
                    segments[batch_idx, seg_idx] = torch.cat(
                        [
                            seg_starts[seg_idx : seg_idx + 1] / perimeter,
                            (seg_starts[seg_idx : seg_idx + 1] + seg_lens[seg_idx : seg_idx + 1]) / perimeter,
                            seg_lens[seg_idx : seg_idx + 1] / perimeter,
                            center_t.unsqueeze(0) / perimeter,
                            torch.nn.functional.one_hot(side_t, num_classes=4).to(dtype=dtype),
                            cut_widths[seg_idx : seg_idx + 1] / perimeter,
                            cut_widths[(seg_idx + 1) % cut_count : (seg_idx + 1) % cut_count + 1] / perimeter,
                        ],
                        dim=0,
                    )
                    prev_seg = (seg_idx - 1) % seg_count
                    seg_node = 1 + seg_idx
                    prev_node = 1 + prev_seg
                    gap_feat = torch.tensor(
                        [cut_widths[seg_idx] / perimeter, seg_lens[prev_seg] / perimeter, seg_lens[seg_idx] / perimeter, 0.0, 0.0, 0.0],
                        dtype=dtype,
                        device=device,
                    )
                    adj[batch_idx, 0, seg_node, prev_node] = 1.0
                    edge_attr[batch_idx, 0, seg_node, prev_node] = gap_feat
            else:
                segments[batch_idx, 0, 2] = 1.0
                node_mask[batch_idx, 1] = True
            node_mask[batch_idx, 1 : 1 + seg_count] = True
            port_attach = torch.zeros(self.port_count, dtype=torch.long, device=device)
            port_pos = torch.zeros(self.port_count, dtype=dtype, device=device)
            port_gap = torch.linalg.vector_norm(ports[batch_idx, :, 3:] - ports[batch_idx, :, :3], dim=-1) / perimeter
            active_ports = int(port_mask[batch_idx].sum().item())
            if active_ports > 0:
                nib_rows = nibs[batch_idx, : self.port_count]
                nib_side = nib_rows[:, 1:5].argmax(dim=1)
                nib_cross = torch.where(nib_side < 2, height, width)
                nib_distance = nib_rows[:, 5] * 0.5 * nib_cross
                nib_coord = _edge_coord(nib_side, nib_distance, width, height)
                for port_idx in range(self.port_count):
                    if not port_mask[batch_idx, port_idx]:
                        continue
                    seg_idx = 0
                    if cut_count > 0:
                        rel = torch.remainder(nib_coord[port_idx] - seg_starts[:seg_count], perimeter)
                        seg_idx = int((rel <= seg_lens[:seg_count] + 1e-6).to(torch.int64).argmax().item())
                    port_attach[port_idx] = seg_idx
                    port_pos[port_idx] = nib_coord[port_idx] / perimeter
                    rel_pos = torch.remainder(nib_coord[port_idx] - seg_starts[seg_idx], perimeter) / seg_lens[seg_idx].clamp_min(1e-4)
                    ports_raw[batch_idx, port_idx] = torch.cat(
                        [
                            port_geom[batch_idx, port_idx],
                            nib_rows[port_idx],
                            torch.tensor([port_pos[port_idx], rel_pos, seg_lens[seg_idx] / perimeter, port_gap[port_idx]], dtype=dtype, device=device),
                        ],
                        dim=0,
                    )
                    port_node = 1 + self.max_segments + port_idx
                    seg_node = 1 + seg_idx
                    adj[batch_idx, 1, port_node, 0] = 1.0
                    edge_attr[batch_idx, 1, port_node, 0] = torch.tensor(
                        [nib_rows[port_idx, 6], nib_rows[port_idx, 7], port_gap[port_idx], port_pos[port_idx], 0.0, 0.0],
                        dtype=dtype,
                        device=device,
                    )
                    adj[batch_idx, 2, port_node, seg_node] = 1.0
                    edge_attr[batch_idx, 2, port_node, seg_node] = torch.tensor(
                        [seg_lens[seg_idx] / perimeter, rel_pos, port_gap[port_idx], 0.0, 0.0, 0.0],
                        dtype=dtype,
                        device=device,
                    )
                for pair_idx, (row, col) in enumerate(self.pairs):
                    delta = torch.remainder(port_pos[col] - port_pos[row], 1.0)
                    cw_cuts = 0.0
                    if cut_count > 0:
                        inside = (torch.remainder(cut_centers / perimeter - port_pos[row], 1.0) < delta).to(dtype=dtype)
                        cw_cuts = inside.sum() / cut_count
                    pair_topology[batch_idx, pair_idx] = torch.tensor(
                        [
                            float(port_attach[row] == port_attach[col]),
                            delta,
                            1.0 - delta,
                            cw_cuts,
                            1.0 - cw_cuts if cut_count > 0 else 0.0,
                            port_gap[row],
                            port_gap[col],
                            torch.abs(port_pos[row] - port_pos[col]),
                        ],
                        dtype=dtype,
                        device=device,
                    )
        return {
            "inner_raw": inner,
            "segment_raw": segments,
            "port_raw": ports_raw,
            "node_mask": node_mask,
            "adj": adj,
            "edge_attr": edge_attr,
            "pair_topology": pair_topology,
        }
