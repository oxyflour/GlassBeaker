import sys
import unittest
from pathlib import Path

from shapely.geometry import Point, Polygon


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sandrone.cases import build_balanced_case
from sandrone.plot import group_color_map
from sandrone.render import render_layer
from sandrone.router import generate_power_copper_shapes, validate_layout


class PowerCopperTests(unittest.TestCase):
    def test_balanced_case_defines_requested_board_stack(self) -> None:
        scenario = build_balanced_case()

        self.assertEqual(30, len(scenario.groups))
        self.assertEqual(100, len(scenario.pins))
        self.assertEqual(10, len(scenario.layers))
        self.assertEqual(50, sum(1 for pin in scenario.pins if pin.side == "top"))
        self.assertEqual(50, sum(1 for pin in scenario.pins if pin.side == "bottom"))

    def test_balanced_case_uses_seeded_random_pin_distribution(self) -> None:
        scenario = build_balanced_case()
        rerun = build_balanced_case()

        self.assertEqual([pin.point for pin in scenario.pins], [pin.point for pin in rerun.pins])
        self.assertGreater(len({pin.x for pin in scenario.pins}), 35)
        self.assertGreater(len({pin.y for pin in scenario.pins}), 18)
        self.assertLess(max(_row_counts(scenario.pins).values()), 12)

    def test_router_generates_connected_isolated_shapes_covering_pins(self) -> None:
        scenario = build_balanced_case()
        layout = generate_power_copper_shapes(scenario)
        report = validate_layout(scenario, layout)

        self.assertEqual([], report.errors)
        self.assertEqual(set(scenario.layers), set(layout.layers))
        for group in scenario.groups:
            cells = layout.group_cells[group.group_id]
            self.assertGreater(len(cells), len(group.pins) * 8)
            for pin in group.pins:
                self.assertIn(pin.point, cells)

    def test_router_keeps_foreign_via_columns_clear_on_every_layer(self) -> None:
        scenario = build_balanced_case()
        layout = generate_power_copper_shapes(scenario)

        for group in scenario.groups:
            cells = layout.group_cells[group.group_id]
            foreign_pins = [pin for pin in scenario.pins if pin.group_id != group.group_id]
            for cell in cells:
                self.assertTrue(
                    all(
                        abs(cell[0] - pin.x) > scenario.clearance
                        or abs(cell[1] - pin.y) > scenario.clearance
                        for pin in foreign_pins
                    ),
                    f"{group.group_id} copper at {cell} touches a foreign via column",
                )

    def test_output_polygons_keep_foreign_via_columns_clear(self) -> None:
        scenario = build_balanced_case()
        layout = generate_power_copper_shapes(scenario)

        for group in scenario.groups:
            foreign_pins = [pin for pin in scenario.pins if pin.group_id != group.group_id]
            for copper in layout.group_polygons[group.group_id]:
                polygon = Polygon(copper.exterior, copper.holes)
                for pin in foreign_pins:
                    keepout = Point(pin.point).buffer(scenario.clearance + 0.5, cap_style=3)
                    self.assertTrue(
                        polygon.disjoint(keepout),
                        f"{group.group_id} final polygon touches {pin.group_id} via at {pin.point}",
                    )

    def test_router_outputs_smoothed_polygons_for_final_copper(self) -> None:
        scenario = build_balanced_case()
        layout = generate_power_copper_shapes(scenario)

        self.assertEqual({group.group_id for group in scenario.groups}, set(layout.group_polygons))
        for polygons in layout.group_polygons.values():
            self.assertGreaterEqual(len(polygons), 1)
            for polygon in polygons:
                self.assertGreater(polygon.area, 10)
                self.assertTrue(
                    any(
                        abs(x - round(x)) > 0.001 or abs(y - round(y)) > 0.001
                        for x, y in polygon.exterior
                    )
                )

    def test_plot_colors_are_unique_for_groups_sharing_a_layer(self) -> None:
        scenario = build_balanced_case()
        colors = group_color_map(scenario)

        for layer in scenario.layers:
            layer_colors = [
                colors[group.group_id]
                for group in scenario.groups
                if group.layer == layer
            ]
            self.assertEqual(len(layer_colors), len(set(layer_colors)))

    def test_layer_render_marks_pin_positions_with_x(self) -> None:
        scenario = build_balanced_case()
        layout = generate_power_copper_shapes(scenario)
        drawing = render_layer(scenario, layout, "L01")

        self.assertIn("Layer L01", drawing)
        self.assertGreaterEqual(drawing.count("x"), 8)
        self.assertTrue(any(char.isalpha() for char in drawing))


def _row_counts(pins):
    counts = {}
    for pin in pins:
        counts[pin.y] = counts.get(pin.y, 0) + 1
    return counts


if __name__ == "__main__":
    unittest.main()
