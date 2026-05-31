# Skill Update: Wide Copper Shape Generation for Power Nets

## Important Design Change

The intermediate connection layers are copper areas used for power routing.

Therefore, the algorithm must not generate thin routing lines as the final result.

Thin lines may only be used as temporary connection skeletons.

The final output should be wide copper shapes, preferably large copper islands or planes, while still keeping different groups electrically isolated.

The optimization goal is no longer shortest path.

The goal is:

```text
maximize usable copper area
maximize minimum neck width
minimize voltage drop / resistance proxy
maintain connectivity within each group
maintain clearance between different groups
```

---

## Correct Mental Model

Treat each group as a power net.

Each group should become one or more wide copper islands distributed over the intermediate layers.

The algorithm should work like:

```text
pins
  -> assign candidate layers
  -> reserve clearance around other nets
  -> generate an initial connected seed
  -> expand seed into available space
  -> smooth / simplify copper boundary
  -> validate connectivity and minimum width
```

The final geometry should look like copper pours, not traces.

---

## Objective Function

For each group, optimize the following objective:

```text
score =
    + area_weight * copper_area
    + neck_weight * minimum_neck_width
    + pin_weight * pin_coverage_quality
    - clearance_violation_penalty
    - fragmentation_penalty
    - boundary_complexity_penalty
```

A practical cost function can be:

```python
score = (
    w_area * area(shape)
    + w_neck * min_neck_width(shape)
    - w_perimeter * perimeter(shape)
    - w_conflict * clearance_violation(shape)
)
```

The routing algorithm should prefer wide, compact copper regions.

---

## Recommended Algorithm

Use the following pipeline:

```text
1. Build available region for each intermediate layer.
2. Assign each group to one or more candidate layers.
3. Generate a minimum connected seed for each group.
4. Expand the seed into a wide copper area.
5. Resolve conflicts between groups.
6. Add vertical pin connections and via pads.
7. Validate minimum width, clearance, and connectivity.
```

---

## Layer Available Region

For each intermediate layer, define the area where copper may exist.

```python
available_region[layer] = board_outline
```

Then subtract forbidden regions:

```python
available_region[layer] =
    board_outline
  - keepout_regions
  - already_assigned_other_group_regions.buffer(clearance)
```

For each group, the legal region on a layer is:

```python
legal_region[group, layer] =
    available_region[layer]
  - obstacles_from_other_groups[layer].buffer(clearance)
```

---

## Initial Connected Seed

Although final copper must be wide, it is useful to first generate a thin temporary skeleton to guarantee connectivity.

For each group:

```text
1. Project all group pins to a candidate layer.
2. Generate a temporary MST or Steiner skeleton.
3. Buffer the skeleton with a minimum required width.
4. Use this as the initial copper seed.
```

This seed is not the final shape.

It only guarantees that all pins are connected.

```python
seed = buffer(mst_skeleton, min_width / 2)
```

Then clip it to the legal region:

```python
seed = seed.intersection(legal_region)
```

If the clipped seed is no longer connected, the layer assignment or routing must be changed.

---

## Copper Expansion

After obtaining a connected seed, expand it as much as possible inside the legal region.

The simplest expansion is:

```python
expanded = legal_region_component_containing(seed)
```

That means:

```text
Take the connected component of the legal region that contains the seed.
Assign the whole component to this group.
```

This produces very wide copper areas.

However, if several groups compete for the same layer, directly taking the whole component may starve other groups.

Therefore use one of the following policies.

---

## Policy A: One Group Per Layer

If the number of layers is large enough, the simplest and most robust strategy is:

```text
assign each power group to its own intermediate layer
fill the largest legal region on that layer
connect pins vertically to that layer
```

For group A on layer k:

```python
shape_A_k = board_outline - keepouts
```

Then optionally remove clearance to vias or shapes from other groups.

This gives the widest copper and lowest resistance.

Use this when:

```text
L >= number_of_power_groups
```

This is the preferred power-plane strategy.

---

## Policy B: Partition One Layer Among Multiple Groups

If multiple groups must share a layer, partition the layer into large copper regions.

Recommended method:

```text
1. Use group pins as seeds.
2. Compute a weighted Voronoi partition on the layer.
3. Clip each partition cell by board outline and keepouts.
4. Intersect each group region with its connectivity seed.
5. Smooth and simplify the resulting polygons.
```

The result is that each group owns a large nearby region.

Pseudo process:

```python
for layer in layers:
    seeds = collect_group_seed_points_on_layer(groups)

    cells = weighted_voronoi(
        seeds=seeds,
        domain=board_outline,
        weights=group_priority_or_current_demand,
    )

    for group in groups_on_layer:
        candidate_region = union(cells[group])
        candidate_region = candidate_region.buffer(-clearance / 2)
        candidate_region = candidate_region.buffer(clearance / 2)

        if candidate_region.contains_or_connects(group_seed[group]):
            shape[group, layer] = candidate_region
```

This creates wide copper regions instead of narrow lines.

---

## Policy C: Region Growing From Connected Seeds

Another practical method is grid-based region growing.

For each layer:

```text
1. Rasterize the layer into grid cells.
2. Mark blocked cells.
3. Mark each group seed cells.
4. Grow all groups simultaneously.
5. Stop growth when groups meet clearance boundaries.
6. Convert occupied cells back to polygons.
```

This is similar to multi-source flood fill.

Each group grows from its connected seed.

At each iteration, assign neighboring free cells to the group with the best score:

```python
score =
    - distance_to_group_pins
    + group_priority
    - boundary_penalty
    + width_bonus
```

This naturally generates wide copper pours.

Pseudo code:

```python
def grow_copper_regions(layer, group_seeds, blocked):
    owner = initialize_grid(blocked)

    frontier = PriorityQueue()

    for group_id, seed_cells in group_seeds.items():
        for cell in seed_cells:
            owner[cell] = group_id
            frontier.push(cell, priority=0)

    while frontier:
        cell = frontier.pop()
        group_id = owner[cell]

        for nb in neighbors(cell):
            if owner[nb] is not None:
                continue

            if violates_clearance(nb, group_id, owner):
                continue

            priority = growth_cost(nb, group_id)
            owner[nb] = group_id
            frontier.push(nb, priority)

    return grid_to_polygons(owner)
```

This is often easier than exact computational geometry when the design is complex.

---

## Minimum Width Constraint

Because the copper is for power, every connected path should satisfy a minimum neck width.

After generating a copper shape, check:

```text
minimum local width >= min_power_width
```

Approximate checks:

```python
eroded = shape.buffer(-min_power_width / 2)
restored = eroded.buffer(min_power_width / 2)
```

If erosion disconnects the shape, then the original shape contains a neck narrower than `min_power_width`.

Validation:

```python
def has_narrow_neck(shape, min_width):
    eroded = shape.buffer(-min_width / 2)
    return not eroded.is_empty and not is_connected(eroded)
```

A stricter version:

```python
def validate_min_width(shape, min_width):
    eroded = shape.buffer(-min_width / 2)

    if eroded.is_empty:
        return False

    return is_connected(eroded)
```

If the shape fails this test:

```text
1. Try another layer.
2. Assign more area to this group.
3. Add additional layer shapes and vias.
4. Reject very narrow corridors.
```

---

## Avoid Thin Final Traces

The final copper shape must not be only:

```text
MST edge buffered by small width
```

This is forbidden for power nets unless no alternative exists.

Allowed use of MST:

```text
MST / Steiner tree may be used only as a connectivity seed.
```

Required final step:

```text
seed -> expand to large copper region
```

Final power copper should be generated by:

```text
connected legal region extraction
Voronoi partition
grid region growing
morphological expansion
polygon union and smoothing
```

---

## Suggested Practical MVP

The recommended MVP for power copper is:

```text
1. Assign one group to one layer if possible.
2. For each group:
   a. Project all pins to that layer.
   b. Build MST only as seed.
   c. Buffer seed by min_power_width / 2.
   d. Expand seed to the largest connected legal region.
   e. Use the expanded region as the final copper shape.
3. Add vertical pin connections.
4. Validate clearance and minimum neck width.
```

Pseudo code:

```python
def generate_power_copper_shapes(groups, layers, board_outline, clearance, min_width):
    results = {}
    obstacles = {layer.id: None for layer in layers}

    layer_assignment = assign_power_layers(groups, layers)

    for group_id, pins in groups.items():
        layer = layer_assignment[group_id]

        legal = compute_legal_region(
            layer=layer,
            board_outline=board_outline,
            obstacles=obstacles,
            clearance=clearance,
        )

        projected = [
            project_pin_to_layer(pin, layer)
            for pin in pins
        ]

        skeleton = build_mst_skeleton(projected)

        seed = skeleton.buffer(min_width / 2)
        seed = seed.intersection(legal)

        if not is_connected(seed):
            raise RuntimeError(f"Seed for group {group_id} is not connected")

        final_shape = connected_component_containing(
            region=legal,
            geometry=seed,
        )

        final_shape = cleanup_power_shape(final_shape, min_width)

        if not validate_min_width(final_shape, min_width):
            raise RuntimeError(f"Power shape for group {group_id} has narrow neck")

        results[group_id] = {
            "layer_shapes": {
                layer.id: final_shape
            },
            "vertical_connections": [
                make_vertical_connection(pin, layer)
                for pin in pins
            ]
        }

        blocked = final_shape.buffer(clearance)

        if obstacles[layer.id] is None:
            obstacles[layer.id] = blocked
        else:
            obstacles[layer.id] = unary_union([
                obstacles[layer.id],
                blocked,
            ])

    return results
```

---

## Cleanup for Power Copper

After expansion, clean the shape using morphological operations.

Recommended cleanup:

```python
def cleanup_power_shape(shape, min_width):
    # Remove tiny spikes and narrow artifacts
    cleaned = shape.buffer(-min_width * 0.25)
    cleaned = cleaned.buffer(min_width * 0.25)

    # Optionally simplify boundary
    cleaned = cleaned.simplify(min_width * 0.1, preserve_topology=True)

    return cleaned
```

This avoids sharp slivers and useless copper fragments.

---

## Via Strategy for Power Nets

For power nets, use multiple vias instead of a single via if possible.

Each pin should connect vertically to the copper layer with:

```text
one via for small pin
via array for large pin or high-current pin
```

Recommended output:

```python
VerticalConnection {
    group_id
    x
    y
    z1
    z2
    via_radius
    via_count
    via_pattern
}
```

For high-current nets, generate via arrays around the pin:

```python
def make_power_via_array(pin, target_layer, via_radius, pitch, max_count):
    points = generate_grid_around_pin(
        center=(pin.x, pin.y),
        pitch=pitch,
        max_count=max_count,
        inside_pin_or_pad=True,
    )

    return [
        make_vertical_connection_at(point, pin.z, target_layer.z, via_radius)
        for point in points
    ]
```

---

## Validation Rules for Power Copper

The implementation must validate:

```text
1. All pins in the same group are electrically connected.
2. Final copper shape is not a thin trace.
3. Minimum neck width is greater than or equal to min_power_width.
4. Different groups do not overlap.
5. Different groups satisfy clearance.
6. Vertical connections land inside the group copper shape.
7. If multiple layers are used by the same group, they are connected by vias.
8. Floating copper islands are removed unless explicitly allowed.
```

Connectivity check:

```python
def validate_pin_connection(group_shape, projected_pins):
    for p in projected_pins:
        if not group_shape.contains(Point(p.x, p.y)):
            return False
    return True
```

Floating island removal:

```python
def remove_floating_islands(shape, required_points):
    components = split_into_connected_components(shape)

    kept = []

    for comp in components:
        if any(comp.contains(Point(p.x, p.y)) for p in required_points):
            kept.append(comp)

    return unary_union(kept)
```

---

## Preferred Strategy Summary

For power copper, prefer this order:

```text
Best:
  one power group per layer
  large copper pour on that layer

Good:
  multiple groups per layer using Voronoi / region partition

Fallback:
  MST seed + maximum legal expansion

Avoid:
  final shape as narrow buffered MST trace
```

The final design should be judged by:

```text
large area
wide current paths
few narrow necks
short vertical access
sufficient clearance
simple polygon boundary
```

