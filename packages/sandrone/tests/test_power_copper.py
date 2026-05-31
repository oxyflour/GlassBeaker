import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sandrone.cases import build_balanced_case
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

    def test_layer_render_marks_pin_positions_with_x(self) -> None:
        scenario = build_balanced_case()
        layout = generate_power_copper_shapes(scenario)
        drawing = render_layer(scenario, layout, "L01")

        self.assertIn("Layer L01", drawing)
        self.assertGreaterEqual(drawing.count("x"), 8)
        self.assertTrue(any(char.isalpha() for char in drawing))


if __name__ == "__main__":
    unittest.main()
