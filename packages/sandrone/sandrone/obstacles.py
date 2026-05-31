from random import Random

from shapely.geometry import LineString, Polygon, box

from .types import KeepoutShape, Point, PowerGroup


def build_random_obstacles(
    layers: tuple[str, ...],
    width: int,
    height: int,
    groups: tuple[PowerGroup, ...],
    seed: int,
    base_keepout: frozenset[Point] = frozenset(),
) -> dict[str, tuple[KeepoutShape, ...]]:
    rng = Random(seed)
    pin_keepout = _pin_keepout(groups, radius=3)
    obstacles: dict[str, tuple[KeepoutShape, ...]] = {}

    for layer in layers:
        used = set(base_keepout) | pin_keepout
        shapes: list[KeepoutShape] = []
        for kind in ("polygon", "polygon", "wire", "wire"):
            shape = _make_shape(kind, layer, rng, width, height, used)
            if shape is None:
                continue
            used.update(shape.cells)
            shapes.append(shape)
        obstacles[layer] = tuple(shapes)

    return obstacles


def merge_keepouts(
    layers: tuple[str, ...],
    base_keepout: frozenset[Point],
    obstacles: dict[str, tuple[KeepoutShape, ...]],
) -> dict[str, frozenset[Point]]:
    keepouts: dict[str, frozenset[Point]] = {}
    for layer in layers:
        cells = set(base_keepout)
        for shape in obstacles.get(layer, tuple()):
            cells.update(shape.cells)
        keepouts[layer] = frozenset(cells)
    return keepouts


def rectangle_obstacle(
    layer: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> KeepoutShape:
    geom = box(x1 - 0.5, y1 - 0.5, x2 - 0.5, y2 - 0.5)
    return KeepoutShape(layer, "polygon", _exterior(geom), _cells_for_geom(geom))


def _make_shape(
    kind: str,
    layer: str,
    rng: Random,
    width: int,
    height: int,
    used: set[Point],
) -> KeepoutShape | None:
    for _ in range(400):
        if kind == "polygon":
            geom = _random_polygon(rng, width, height)
        else:
            geom = _random_wire(rng, width, height)
        cells = _cells_for_geom(geom)
        if len(cells) < 8 or cells & used:
            continue
        return KeepoutShape(layer, kind, _exterior(geom), cells)
    return None


def _random_polygon(rng: Random, width: int, height: int):
    x1 = rng.randint(12, width - 28)
    y1 = rng.randint(10, height - 20)
    x2 = x1 + rng.randint(8, 18)
    y2 = y1 + rng.randint(6, 16)
    if rng.random() < 0.55:
        return box(x1, y1, x2, y2)

    chamfer = min(rng.randint(2, 5), (x2 - x1) // 2, (y2 - y1) // 2)
    corner = rng.choice(("top-left", "top-right", "bottom-right", "bottom-left"))
    if corner == "top-left":
        points = [(x1 + chamfer, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1 + chamfer)]
    elif corner == "top-right":
        points = [(x1, y1), (x2 - chamfer, y1), (x2, y1 + chamfer), (x2, y2), (x1, y2)]
    elif corner == "bottom-right":
        points = [(x1, y1), (x2, y1), (x2, y2 - chamfer), (x2 - chamfer, y2), (x1, y2)]
    else:
        points = [(x1, y1), (x2, y1), (x2, y2), (x1 + chamfer, y2), (x1, y2 - chamfer)]
    return Polygon(points)


def _random_wire(rng: Random, width: int, height: int):
    x1 = rng.randint(8, width - 34)
    y1 = rng.randint(8, height - 8)
    direction = rng.choice(((1, 0), (0, 1), (1, 1), (1, -1)))
    length = rng.randint(18, 42)
    x2 = x1 + direction[0] * length
    y2 = y1 + direction[1] * length
    if not (8 <= x2 <= width - 8 and 8 <= y2 <= height - 8):
        x2 = min(width - 8, x1 + length)
        y2 = y1
    return LineString(((x1, y1), (x2, y2))).buffer(
        rng.uniform(0.8, 1.5),
        cap_style=2,
        join_style=2,
    )


def _cells_for_geom(geom) -> frozenset[Point]:
    minx, miny, maxx, maxy = geom.bounds
    cells: set[Point] = set()
    for x in range(int(minx) - 1, int(maxx) + 2):
        for y in range(int(miny) - 1, int(maxy) + 2):
            if geom.intersects(box(x - 0.5, y - 0.5, x + 0.5, y + 0.5)):
                cells.add((x, y))
    return frozenset(cells)


def _pin_keepout(groups: tuple[PowerGroup, ...], radius: int) -> set[Point]:
    cells: set[Point] = set()
    for group in groups:
        for pin in group.pins:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    cells.add((pin.x + dx, pin.y + dy))
    return cells


def _exterior(geom) -> tuple[tuple[float, float], ...]:
    return tuple((round(float(x), 2), round(float(y), 2)) for x, y in geom.exterior.coords)
