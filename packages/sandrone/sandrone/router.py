from collections import defaultdict, deque
from heapq import heappop, heappush
from itertools import combinations

from shapely.geometry import Point as GeometryPoint, Polygon
from shapely.ops import unary_union

from .polygons import MIN_COPPER_GAP, build_copper_polygons
from .types import Layout, Point, Scenario, ValidationReport


DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
CELL_GROWTH_CLEARANCE = 0


def generate_power_copper_shapes(scenario: Scenario) -> Layout:
    layer_maps: dict[str, dict[Point, str]] = {}
    group_cells: dict[str, set[Point]] = defaultdict(set)

    for layer in scenario.layers:
        blocked = set(scenario.keepouts.get(layer, frozenset()))
        owner: dict[Point, str] = {}
        anchors: dict[str, Point] = {}
        heap: list[tuple[int, str, Point]] = []

        for group in (item for item in scenario.groups if item.layer == layer):
            seed = _connected_seed([pin.point for pin in group.pins], scenario, blocked, group.group_id)
            anchors[group.group_id] = _centroid(seed)
            for cell in seed:
                if cell in blocked:
                    raise ValueError(f"Seed for {group.group_id} lands in keepout at {cell}")
                if _touches_foreign_via(scenario, cell, group.group_id):
                    raise ValueError(f"Seed for {group.group_id} touches a foreign via at {cell}")
                previous = owner.get(cell)
                if previous not in (None, group.group_id):
                    raise ValueError(f"Seed conflict at {cell}: {previous} vs {group.group_id}")
                owner[cell] = group.group_id
                heappush(heap, (0, group.group_id, cell))

        while heap:
            _, group_id, cell = heappop(heap)
            for dx, dy in DIRS:
                nxt = (cell[0] + dx, cell[1] + dy)
                if not _inside(nxt, scenario) or nxt in blocked or nxt in owner:
                    continue
                if _touches_foreign_via(scenario, nxt, group_id):
                    continue
                if _touches_other(owner, nxt, group_id, CELL_GROWTH_CLEARANCE):
                    continue
                owner[nxt] = group_id
                anchor = anchors[group_id]
                cost = abs(nxt[0] - anchor[0]) + abs(nxt[1] - anchor[1])
                heappush(heap, (cost, group_id, nxt))

        layer_maps[layer] = owner
        for cell, group_id in owner.items():
            group_cells[group_id].add(cell)

    final_cells = {group.group_id: frozenset(group_cells[group.group_id]) for group in scenario.groups}
    return Layout(
        layers=layer_maps,
        group_cells=final_cells,
        group_polygons={
            group.group_id: build_copper_polygons(
                group.group_id,
                group.layer,
                final_cells[group.group_id],
                scenario,
            )
            for group in scenario.groups
        },
    )


def validate_layout(scenario: Scenario, layout: Layout) -> ValidationReport:
    errors: list[str] = []
    for layer, owner in layout.layers.items():
        keepout = scenario.keepouts.get(layer, frozenset())
        for cell, group_id in owner.items():
            if cell in keepout:
                errors.append(f"{group_id} overlaps keepout on {layer} at {cell}")
            if _touches_other(owner, cell, group_id, CELL_GROWTH_CLEARANCE):
                errors.append(f"{group_id} violates clearance on {layer} at {cell}")

    for group in scenario.groups:
        cells = layout.group_cells.get(group.group_id, frozenset())
        if not cells:
            errors.append(f"{group.group_id} has no copper cells")
            continue
        if not _is_connected(cells):
            errors.append(f"{group.group_id} copper is fragmented")
        for pin in group.pins:
            if pin.point not in cells:
                errors.append(f"{group.group_id} does not cover {pin.pin_id} at {pin.point}")
        for cell in cells:
            if _touches_foreign_via(scenario, cell, group.group_id):
                errors.append(f"{group.group_id} touches a foreign via column at {cell}")
        polygons = layout.group_polygons.get(group.group_id, tuple())
        if not polygons:
            errors.append(f"{group.group_id} has no output polygon")
            continue
        if len(polygons) > 1:
            errors.append(f"{group.group_id} polygon output is fragmented")
        shape = unary_union([Polygon(polygon.exterior, polygon.holes) for polygon in polygons])
        for pin in group.pins:
            if not shape.covers(GeometryPoint(pin.x, pin.y)):
                errors.append(f"{group.group_id} polygon misses {pin.pin_id} at {pin.point}")

    layer_polygons: dict[str, list[tuple[str, Polygon]]] = defaultdict(list)
    for group_id, polygons in layout.group_polygons.items():
        for polygon in polygons:
            layer_polygons[polygon.layer].append((group_id, Polygon(polygon.exterior, polygon.holes)))
    for layer, polygons in layer_polygons.items():
        for (left_id, left), (right_id, right) in combinations(polygons, 2):
            if left_id == right_id:
                continue
            distance = left.distance(right)
            if distance < MIN_COPPER_GAP:
                errors.append(
                    f"{left_id} and {right_id} are {distance:.3f} apart on {layer}; "
                    f"minimum is {MIN_COPPER_GAP:.3f}"
                )

    return ValidationReport(errors)


def _connected_seed(points: list[Point], scenario: Scenario, blocked: set[Point], group_id: str) -> set[Point]:
    connected = [points[0]]
    remaining = set(points[1:])
    seed = {points[0]}
    while remaining:
        _, start, end = min(
            (abs(a[0] - b[0]) + abs(a[1] - b[1]), a, b)
            for a in connected
            for b in remaining
        )
        path = _path(start, end, scenario, blocked, group_id)
        seed.update(path)
        connected.append(end)
        remaining.remove(end)
    return seed


def _path(start: Point, end: Point, scenario: Scenario, blocked: set[Point], group_id: str) -> list[Point]:
    first = _l_path(start, end, horizontal_first=True)
    if all(_is_legal_seed_cell(cell, scenario, blocked, group_id) for cell in first):
        return first
    second = _l_path(start, end, horizontal_first=False)
    if all(_is_legal_seed_cell(cell, scenario, blocked, group_id) for cell in second):
        return second
    return _bfs_path(start, end, scenario, blocked, group_id)


def _l_path(start: Point, end: Point, horizontal_first: bool) -> list[Point]:
    mid = (end[0], start[1]) if horizontal_first else (start[0], end[1])
    return _straight(start, mid) + _straight(mid, end)[1:]


def _straight(start: Point, end: Point) -> list[Point]:
    if start[0] == end[0]:
        step = 1 if end[1] >= start[1] else -1
        return [(start[0], y) for y in range(start[1], end[1] + step, step)]
    step = 1 if end[0] >= start[0] else -1
    return [(x, start[1]) for x in range(start[0], end[0] + step, step)]


def _bfs_path(start: Point, end: Point, scenario: Scenario, blocked: set[Point], group_id: str) -> list[Point]:
    queue: deque[Point] = deque([start])
    parent: dict[Point, Point | None] = {start: None}
    while queue:
        cell = queue.popleft()
        if cell == end:
            break
        for dx, dy in DIRS:
            nxt = (cell[0] + dx, cell[1] + dy)
            if nxt in parent or not _is_legal_seed_cell(nxt, scenario, blocked, group_id):
                continue
            parent[nxt] = cell
            queue.append(nxt)
    if end not in parent:
        raise ValueError(f"No seed path from {start} to {end}")
    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return list(reversed(path))


def _inside(cell: Point, scenario: Scenario) -> bool:
    return 0 <= cell[0] < scenario.width and 0 <= cell[1] < scenario.height


def _is_legal_seed_cell(cell: Point, scenario: Scenario, blocked: set[Point], group_id: str) -> bool:
    return _inside(cell, scenario) and cell not in blocked and not _touches_foreign_via(scenario, cell, group_id)


def _touches_foreign_via(scenario: Scenario, cell: Point, group_id: str) -> bool:
    for pin in scenario.pins:
        if pin.group_id == group_id:
            continue
        if abs(cell[0] - pin.x) <= scenario.clearance and abs(cell[1] - pin.y) <= scenario.clearance:
            return True
    return False


def _touches_other(owner: dict[Point, str], cell: Point, group_id: str, clearance: int) -> bool:
    for dx in range(-clearance, clearance + 1):
        for dy in range(-clearance, clearance + 1):
            other = owner.get((cell[0] + dx, cell[1] + dy))
            if other is not None and other != group_id:
                return True
    return False


def _centroid(cells: set[Point]) -> Point:
    return (
        round(sum(cell[0] for cell in cells) / len(cells)),
        round(sum(cell[1] for cell in cells) / len(cells)),
    )


def _is_connected(cells: frozenset[Point] | set[Point]) -> bool:
    queue = deque([next(iter(cells))])
    seen = {queue[0]}
    while queue:
        cell = queue.popleft()
        for dx, dy in DIRS:
            nxt = (cell[0] + dx, cell[1] + dy)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen) == len(cells)
