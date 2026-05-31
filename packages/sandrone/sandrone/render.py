from collections import defaultdict

from .types import Layout, Pin, Point, Scenario


def render_layer(scenario: Scenario, layout: Layout, layer: str, all_pins: bool = True) -> str:
    owner = layout.layers[layer]
    pins = scenario.pins if all_pins else tuple(
        pin for group in scenario.groups if group.layer == layer for pin in group.pins
    )
    pin_points = {pin.point for pin in pins}
    markers = {group.group_id: group.marker for group in scenario.groups}
    groups = [group for group in scenario.groups if group.layer == layer]
    legend = " ".join(f"{group.marker}={group.group_id}" for group in groups)
    lines = [f"Layer {layer}  {legend}"]

    for y in range(scenario.height):
        row: list[str] = []
        for x in range(scenario.width):
            point = (x, y)
            if point in pin_points:
                row.append("x")
            elif point in scenario.keepouts.get(layer, frozenset()):
                row.append("#")
            else:
                row.append(markers.get(owner.get(point, ""), "."))
        lines.append("".join(row))
    return "\n".join(lines)


def render_scenario(scenario: Scenario, layout: Layout) -> str:
    report_lines = [
        f"Scenario: {scenario.name}",
        f"Board: {scenario.width} x {scenario.height}",
        f"Groups: {len(scenario.groups)}",
        f"Pins: {len(scenario.pins)}",
        f"Intermediate layers: {len(scenario.layers)}",
        "",
        "Top pins:",
        _pin_catalog([pin for pin in scenario.pins if pin.side == "top"]),
        "",
        "Bottom pins:",
        _pin_catalog([pin for pin in scenario.pins if pin.side == "bottom"]),
        "",
    ]
    for layer in scenario.layers:
        report_lines.append(render_layer(scenario, layout, layer))
        report_lines.append("")
    return "\n".join(report_lines)


def _pin_catalog(pins: list[Pin]) -> str:
    by_group: dict[str, list[Point]] = defaultdict(list)
    for pin in pins:
        by_group[pin.group_id].append(pin.point)
    return "\n".join(
        f"  {group_id}: " + ", ".join(f"({x},{y})" for x, y in points)
        for group_id, points in sorted(by_group.items())
    )
