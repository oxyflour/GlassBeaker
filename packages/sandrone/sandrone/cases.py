from random import Random

from .obstacles import build_random_obstacles, merge_keepouts, rectangle_obstacle
from .types import Pin, Point, PowerGroup, Scenario


MARKERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123"


def build_balanced_case(name: str | None = None, pin_count: int = 200) -> Scenario:
    width, height = 160, 90
    layers = tuple(f"L{i:02d}" for i in range(1, 11))
    groups = _build_groups(width, height, layers, pin_count, seed=23017, mode="balanced", forbidden=frozenset())
    obstacles = build_random_obstacles(layers, width, height, groups, seed=41017)
    return Scenario(
        name=name or f"balanced-30g-{pin_count}p",
        width=width,
        height=height,
        layers=layers,
        groups=groups,
        keepouts=merge_keepouts(layers, frozenset(), obstacles),
        obstacles=obstacles,
    )


def build_slot_keepout_case() -> Scenario:
    width, height = 160, 90
    layers = tuple(f"L{i:02d}" for i in range(1, 11))
    keepout = frozenset((x, y) for x in range(136, 142) for y in range(16, 74))
    groups = _build_groups(width, height, layers, 200, seed=23029, mode="balanced", forbidden=keepout)
    obstacles = build_random_obstacles(layers, width, height, groups, seed=41029, base_keepout=keepout)
    obstacles = {
        layer: (rectangle_obstacle(layer, 136, 16, 142, 74), *obstacles[layer])
        for layer in layers
    }
    return Scenario(
        name="slot-keepout-30g-200p",
        width=width,
        height=height,
        layers=layers,
        groups=groups,
        keepouts=merge_keepouts(layers, keepout, obstacles),
        obstacles=obstacles,
    )


def build_edge_loaded_case() -> Scenario:
    width, height = 160, 90
    layers = tuple(f"L{i:02d}" for i in range(1, 11))
    groups = _build_groups(width, height, layers, 200, seed=23041, mode="edge", forbidden=frozenset())
    obstacles = build_random_obstacles(layers, width, height, groups, seed=41041)
    return Scenario(
        name="edge-loaded-30g-200p",
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
    pin_count: int,
    seed: int,
    mode: str,
    forbidden: frozenset[Point],
) -> tuple[PowerGroup, ...]:
    rng = Random(seed)
    used: set[Point] = set()
    anchors: list[Point] = []
    groups: list[PowerGroup] = []
    pin_index = 1
    group_count = 30
    if pin_count < group_count * 2 or pin_count % 2 != 0:
        raise ValueError("pin_count must be even and at least 60")

    pin_counts = [4] * group_count
    remaining = pin_count - sum(pin_counts)
    if remaining < 0:
        pin_counts = [2] * group_count
        remaining = pin_count - sum(pin_counts)
    while remaining:
        choices = [index for index, count in enumerate(pin_counts) if count < 10]
        if not choices:
            raise ValueError("Could not allocate random pin counts")
        pin_counts[rng.choice(choices)] += 1
        remaining -= 1

    extra_side_count = pin_count // 2 - group_count
    extra_sides = ["top"] * extra_side_count + ["bottom"] * extra_side_count
    rng.shuffle(extra_sides)
    side_groups: list[list[str]] = []
    offset = 0
    for count in pin_counts:
        sides = ["top", "bottom"]
        sides.extend(extra_sides[offset : offset + count - 2])
        rng.shuffle(sides)
        side_groups.append(sides)
        offset += count - 2

    for index in range(group_count):
        slot = index // len(layers)
        anchor = _random_anchor(rng, width, height, anchors, slot, mode, forbidden)
        anchors.append(anchor)
        pin_count = pin_counts[index]
        sides = side_groups[index]

        points = _points_near(rng, anchor, pin_count, used, set(used), width, height, forbidden)
        group_id = f"G{index + 1:02d}"
        pins: list[Pin] = []
        for side, point in zip(sides, points):
            pins.append(Pin(f"P{pin_index:03d}", group_id, side, point[0], point[1]))
            pin_index += 1
        groups.append(PowerGroup(group_id, MARKERS[index], tuple(pins)))

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
