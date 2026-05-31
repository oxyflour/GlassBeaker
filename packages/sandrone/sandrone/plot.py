from colorsys import hsv_to_rgb
from math import ceil
from pathlib import Path

from matplotlib.colors import to_hex
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as PlotPath
import matplotlib.pyplot as plt

from .types import CopperPolygon, KeepoutShape, Layout, Scenario


def group_color_map(scenario: Scenario) -> dict[str, str]:
    return {
        group.group_id: _stable_color(index)
        for index, group in enumerate(scenario.groups)
    }


def save_layer_contact_sheet(scenario: Scenario, layout: Layout, path: Path) -> None:
    colors = group_color_map(scenario)
    cols = min(3, len(scenario.layers))
    rows = ceil(len(scenario.layers) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 4.6 * rows), constrained_layout=True)
    axes_flat = [axes] if len(scenario.layers) == 1 else list(axes.flat)

    for ax, layer in zip(axes_flat, scenario.layers):
        ax.set_facecolor("#f7f7f7")
        for group in scenario.groups:
            if group.layer != layer:
                continue
            for polygon in layout.group_polygons.get(group.group_id, tuple()):
                ax.add_patch(_polygon_patch(polygon, colors[group.group_id]))
        _draw_keepouts(ax, scenario, layer)
        _draw_pins(ax, scenario, colors)
        ax.set_title(_layer_title(scenario, layer), fontsize=10)
        ax.set_xlim(-0.5, scenario.width - 0.5)
        ax.set_ylim(scenario.height - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_flat[len(scenario.layers):]:
        ax.axis("off")

    fig.suptitle(
        f"{scenario.name}: 30 groups, 100 pins, {len(scenario.layers)} intermediate copper layers",
        fontsize=15,
    )
    fig.legend(
        handles=[
            Patch(facecolor=colors[group.group_id], label=f"{group.marker} {group.group_id}")
            for group in scenario.groups
        ],
        loc="outside lower center",
        ncol=15,
        fontsize=8,
    )
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _polygon_patch(polygon: CopperPolygon, color: str) -> PathPatch:
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    for ring in (polygon.exterior, *polygon.holes):
        if not ring:
            continue
        vertices.extend(ring)
        codes.extend([PlotPath.MOVETO] + [PlotPath.LINETO] * (len(ring) - 2) + [PlotPath.CLOSEPOLY])
    return PathPatch(
        PlotPath(vertices, codes),
        facecolor=color,
        edgecolor="none",
        linewidth=0,
        antialiased=True,
    )


def _draw_pins(ax, scenario: Scenario, colors: dict[str, str]) -> None:
    for group in scenario.groups:
        xs = [pin.x for pin in group.pins]
        ys = [pin.y for pin in group.pins]
        ax.scatter(xs, ys, marker="x", c=colors[group.group_id], s=18, linewidths=1.0)


def _draw_keepouts(ax, scenario: Scenario, layer: str) -> None:
    obstacles = (scenario.obstacles or {}).get(layer, tuple())
    if not obstacles and not scenario.keepouts.get(layer, frozenset()):
        return
    for shape in obstacles:
        ax.add_patch(_keepout_patch(shape))

    obstacle_cells = set()
    for shape in obstacles:
        obstacle_cells.update(shape.cells)
    remaining = scenario.keepouts.get(layer, frozenset()) - obstacle_cells
    if remaining:
        ax.scatter(
            [point[0] for point in remaining],
            [point[1] for point in remaining],
            marker="s",
            c="#767676",
            s=5,
            linewidths=0,
        )


def _keepout_patch(shape: KeepoutShape) -> PathPatch:
    return PathPatch(
        _ring_path(shape.exterior),
        facecolor="#777777",
        edgecolor="#4f4f4f",
        linewidth=0.8,
        alpha=0.78,
        antialiased=True,
    )


def _ring_path(ring: tuple[tuple[float, float], ...]) -> PlotPath:
    codes = [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(ring) - 2) + [PlotPath.CLOSEPOLY]
    return PlotPath(ring, codes)


def _layer_title(scenario: Scenario, layer: str) -> str:
    groups = [group for group in scenario.groups if group.layer == layer]
    return f"{layer}: " + ", ".join(f"{group.marker}={group.group_id}" for group in groups)


def _stable_color(index: int) -> str:
    hue = (index * 0.61803398875) % 1.0
    saturation = 0.58 + 0.16 * (index % 2)
    value = 0.78 - 0.08 * (index % 3 == 0)
    return to_hex(hsv_to_rgb(hue, saturation, value))
