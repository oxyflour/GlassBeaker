from random import Random

from .obstacles import build_random_obstacles, merge_keepouts, rectangle_obstacle
from .types import Pin, Point, PowerGroup, Scenario


MARKERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123"


def build_balanced_case(name: str = "balanced-30g-100p") -> Scenario:
    width, height = 160, 90
    layers = tuple(f"L{i:02d}" for i in range(1, 4))
    groups = _build_groups(width, height, layers, seed=23017, mode="balanced", forbidden=frozenset())
    obstacles = build_random_obstacles(layers, width, height, groups, seed=41017)
    return Scenario(
        name=name,
        width=width,
        height=height,
        layers=layers,
        groups=groups,
        keepouts=merge_keepouts(layers, frozenset(), obstacles),
        obstacles=obstacles,
    )


def build_slot_keepout_case() -> Scenario:
    width, height = 160, 90
    layers = tuple(f"L{i:02d}" for i in range(1, 4))
    keepout = frozenset((x, y) for x in range(136, 142) for y in range(16, 74))
    groups = _build_groups(width, height, layers, seed=23029, mode="balanced", forbidden=keepout)
    obstacles = build_random_obstacles(layers, width, height, groups, seed=41029, base_keepout=keepout)
    obstacles = {
        layer: (rectangle_obstacle(layer, 136, 16, 142, 74), *obstacles[layer])
        for layer in layers
    }
    return Scenario(
        name="slot-keepout-30g-100p",
        width=width,
        height=height,
        layers=layers,
        groups=groups,
        keepouts=merge_keepouts(layers, keepout, obstacles),
        obstacles=obstacles,
    )


def build_edge_loaded_case() -> Scenario:
    width, height = 160, 90
    layers = tuple(f"L{i:02d}" for i in range(1, 4))
    groups = _build_groups(width, height, layers, seed=23041, mode="edge", forbidden=frozenset())
    obstacles = build_random_obstacles(layers, width, height, groups, seed=41041)
    return Scenario(
        name="edge-loaded-30g-100p",
        width=width,
        height=height,
        layers=layers,
        groups=groups,
        keepouts=merge_keepouts(layers, frozenset(), obstacles),
        obstacles=obstacles,
    )


def all_cases() -> tuple[Scenario, ...]:
    return (
        build_balanced_case(),
        build_slot_keepout_case(),
        build_edge_loaded_case(),
    )


def _build_groups(
    width: int,
    height: int,
    layers: tuple[str, ...],
    seed: int,
    mode: str,
    forbidden: frozenset[Point],
) -> tuple[PowerGroup, ...]:
    rng = Random(seed)
    used: set[Point] = set()
    anchors: list[Point] = []
    groups: list[PowerGroup] = []
    pin_index = 1

    for index in range(30):
        layer_index = index % len(layers)
        slot = index // len(layers)
        layer = layers[layer_index]
        anchor = _random_anchor(rng, width, height, anchors, slot, mode, forbidden)
        anchors.append(anchor)
        pin_count = 4 if index < 10 else 3
        if pin_count == 4:
            sides = ("top", "top", "bottom", "bottom")
        elif (index - 10) % 2 == 0:
            sides = ("top", "top", "bottom")
        else:
            sides = ("top", "bottom", "bottom")

        points = _points_near(rng, anchor, pin_count, used, set(used), width, height, forbidden)
        group_id = f"G{index + 1:02d}"
        pins: list[Pin] = []
        for side, point in zip(sides, points):
            pins.append(Pin(f"P{pin_index:03d}", group_id, side, point[0], point[1]))
            pin_index += 1
        groups.append(PowerGroup(group_id, MARKERS[index], layer, tuple(pins)))

    return tuple(groups)


def _random_anchor(
    rng: Random,
    width: int,
    height: int,
    existing: list[Point],
    slot: int,
    mode: str,
    forbidden: frozenset[Point],
) -> Point:
    for _ in range(2_000):
        if mode == "edge" and slot == 0:
            x = rng.randint(4, 17)
        elif mode == "edge" and slot == 1:
            x = rng.randint(width - 18, width - 5)
        else:
            x = rng.randint(5, width - 6)
        y = rng.randint(4, height - 5)
        point = (x, y)
        if point in forbidden or any(_distance(point, other) < 7 for other in existing):
            continue
        return point
    raise ValueError("Could not place a random group anchor")


def _points_near(
    rng: Random,
    anchor: Point,
    count: int,
    used: set[Point],
    foreign: set[Point],
    width: int,
    height: int,
    forbidden: frozenset[Point],
) -> tuple[Point, ...]:
    offsets = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4)]
    rng.shuffle(offsets)
    points: list[Point] = []
    for dx, dy in offsets:
        point = (anchor[0] + dx, anchor[1] + dy)
        if point in used or point in forbidden:
            continue
        if any(abs(point[0] - other[0]) <= 3 and abs(point[1] - other[1]) <= 3 for other in foreign):
            continue
        if not (1 <= point[0] < width - 1 and 1 <= point[1] < height - 1):
            continue
        used.add(point)
        points.append(point)
        if len(points) == count:
            return tuple(points)
    raise ValueError(f"Could not place {count} pins near {anchor}")


def _distance(left: Point, right: Point) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])
