from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from .types import CopperPolygon, Point, Scenario


SMOOTH_RADIUS = 0.15
COPPER_INSET = 0.48
MIN_COPPER_GAP = 0.48
SIMPLIFY_TOLERANCE = 0.55


def build_copper_polygons(
    group_id: str,
    layer: str,
    cells: frozenset[Point],
    scenario: Scenario,
) -> tuple[CopperPolygon, ...]:
    if not cells:
        return tuple()

    raw = unary_union([box(x - 0.5, y - 0.5, x + 0.5, y + 0.5) for x, y in cells])
    keepouts = _layer_keepouts(layer, scenario)
    foreign_vias = _foreign_via_keepouts(group_id, scenario)

    clean = raw.buffer(-SMOOTH_RADIUS, quad_segs=8, join_style=1)
    clean = clean.buffer(SMOOTH_RADIUS, quad_segs=8, join_style=1)
    clean = clean.buffer(-COPPER_INSET, quad_segs=8, join_style=1).buffer(0)
    clean = clean.difference(keepouts).buffer(0)
    clean = clean.difference(foreign_vias).buffer(0)
    clean = clean.simplify(SIMPLIFY_TOLERANCE, preserve_topology=True).buffer(0)
    clean = clean.difference(keepouts).buffer(0)
    clean = clean.difference(foreign_vias).buffer(0)
    geoms = clean.geoms if isinstance(clean, MultiPolygon) else (clean,)

    polygons: list[CopperPolygon] = []
    for geom in geoms:
        if geom.is_empty or not isinstance(geom, Polygon):
            continue
        polygons.append(
            CopperPolygon(
                group_id=group_id,
                layer=layer,
                exterior=_ring_coords(geom.exterior.coords),
                holes=tuple(_ring_coords(ring.coords) for ring in geom.interiors),
                area=float(geom.area),
            )
        )

    return tuple(sorted(polygons, key=lambda item: item.area, reverse=True))


def _ring_coords(coords) -> tuple[tuple[float, float], ...]:
    return tuple((round(float(x), 3), round(float(y), 3)) for x, y in coords)


def _foreign_via_keepouts(group_id: str, scenario: Scenario):
    radius = scenario.clearance + 0.6
    return unary_union(
        [
            box(pin.x - radius, pin.y - radius, pin.x + radius, pin.y + radius)
            .buffer(0.35, quad_segs=8, join_style=1)
            for pin in scenario.pins
            if pin.group_id != group_id
        ]
    )


def _layer_keepouts(layer: str, scenario: Scenario):
    obstacles = scenario.obstacles or {}
    shapes = [Polygon(shape.exterior) for shape in obstacles.get(layer, tuple())]
    if shapes:
        return unary_union(shapes)
    return unary_union(
        [
            box(x - 0.5, y - 0.5, x + 0.5, y + 0.5)
            for x, y in scenario.keepouts.get(layer, frozenset())
        ]
    )
