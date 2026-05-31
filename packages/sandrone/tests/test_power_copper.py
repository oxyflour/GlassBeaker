import sys
import unittest
from pathlib import Path

from shapely.geometry import Point, Polygon


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sandrone.agent import generate_power_copper_shapes_with_agent
from sandrone.cases import build_balanced_case
from sandrone.llm_agent import LLMLayerPlanningAgent
from sandrone.plot import group_color_map
from sandrone.render import render_layer
from sandrone.router import generate_power_copper_shapes, validate_layout
from sandrone.types import LayerPlan, Pin, PowerGroup, Scenario


class PowerCopperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.balanced_scenario = build_balanced_case()
        cls.balanced_layout = generate_power_copper_shapes(cls.balanced_scenario)
        cls.balanced_report = validate_layout(cls.balanced_scenario, cls.balanced_layout)

    def test_balanced_case_defines_requested_board_stack(self) -> None:
        scenario = self.balanced_scenario

        self.assertEqual(30, len(scenario.groups))
        self.assertEqual(200, len(scenario.pins))
        self.assertEqual(10, len(scenario.layers))
        self.assertEqual(100, sum(1 for pin in scenario.pins if pin.side == "top"))
        self.assertEqual(100, sum(1 for pin in scenario.pins if pin.side == "bottom"))

    def test_balanced_case_uses_seeded_random_pin_distribution(self) -> None:
        scenario = self.balanced_scenario
        rerun = build_balanced_case()
        pin_counts = [len(group.pins) for group in scenario.groups]

        self.assertEqual([pin.point for pin in scenario.pins], [pin.point for pin in rerun.pins])
        self.assertEqual(pin_counts, [len(group.pins) for group in rerun.groups])
        self.assertGreater(len(set(pin_counts)), 2)
        self.assertGreater(len({pin.x for pin in scenario.pins}), 35)
        self.assertGreater(len({pin.y for pin in scenario.pins}), 18)
        self.assertLess(max(_row_counts(scenario.pins).values()), 12)

    def test_balanced_case_accepts_requested_pin_count(self) -> None:
        scenario = build_balanced_case(pin_count=160)
        rerun = build_balanced_case(pin_count=160)
        pin_counts = [len(group.pins) for group in scenario.groups]

        self.assertEqual(160, len(scenario.pins))
        self.assertEqual(80, sum(1 for pin in scenario.pins if pin.side == "top"))
        self.assertEqual(80, sum(1 for pin in scenario.pins if pin.side == "bottom"))
        self.assertEqual(pin_counts, [len(group.pins) for group in rerun.groups])
        self.assertGreater(len(set(pin_counts)), 2)

    def test_keepout_shapes_use_regular_trace_angles(self) -> None:
        scenario = self.balanced_scenario

        for shapes in (scenario.obstacles or {}).values():
            for shape in shapes:
                for start, end in zip(shape.exterior, shape.exterior[1:]):
                    self.assertTrue(
                        _is_horizontal_vertical_or_45(start, end),
                        f"{shape.kind} edge {start}->{end} is not horizontal, vertical, or 45 degrees",
                    )

    def test_router_generates_connected_isolated_shapes_covering_pins(self) -> None:
        scenario = self.balanced_scenario
        layout = self.balanced_layout
        report = self.balanced_report

        self.assertEqual([], report.errors)
        self.assertEqual(set(scenario.layers), set(layout.layers))
        for group in scenario.groups:
            cells_by_layer = layout.group_cells[group.group_id]
            cells = set().union(*cells_by_layer.values())
            self.assertGreater(len(cells), len(group.pins) * 8)
            for pin in group.pins:
                self.assertIn(pin.point, cells)

    def test_router_keeps_foreign_via_columns_clear_on_every_layer(self) -> None:
        scenario = self.balanced_scenario
        layout = self.balanced_layout

        for group in scenario.groups:
            foreign_pins = [pin for pin in scenario.pins if pin.group_id != group.group_id]
            for layer, cells in layout.group_cells[group.group_id].items():
                for cell in cells:
                    self.assertTrue(
                        all(
                            abs(cell[0] - pin.x) > scenario.clearance
                            or abs(cell[1] - pin.y) > scenario.clearance
                            for pin in foreign_pins
                        ),
                        f"{group.group_id} copper at {cell} touches a foreign via column on {layer}",
                    )

    def test_output_polygons_keep_foreign_via_columns_clear(self) -> None:
        scenario = self.balanced_scenario
        layout = self.balanced_layout

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
        scenario = self.balanced_scenario
        layout = self.balanced_layout

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
        scenario = self.balanced_scenario
        layout = self.balanced_layout
        colors = group_color_map(scenario)

        for layer in scenario.layers:
            layer_colors = [
                colors[group_id]
                for group_id in _group_ids_on_layer(layout, layer)
            ]
            self.assertEqual(len(layer_colors), len(set(layer_colors)))

    def test_router_accepts_layer_plan_with_one_group_on_multiple_layers(self) -> None:
        scenario = _two_layer_single_group_scenario()
        plan = LayerPlan({"G01": ("L01", "L02")})

        layout = generate_power_copper_shapes(scenario, plan)
        report = validate_layout(scenario, layout)

        self.assertEqual([], report.errors)
        self.assertEqual({"L01", "L02"}, set(layout.group_cells["G01"]))
        self.assertGreater(len(layout.group_cells["G01"]["L01"]), 0)
        self.assertGreater(len(layout.group_cells["G01"]["L02"]), 0)
        self.assertEqual(
            {"L01", "L02"},
            {polygon.layer for polygon in layout.group_polygons["G01"]},
        )

    def test_agent_flow_selects_best_multilayer_plan(self) -> None:
        scenario = _two_layer_single_group_scenario()
        agent = _SequenceAgent(
            [
                [
                    LayerPlan({"G01": ("L01",)}),
                    LayerPlan({"G01": ("L01", "L02")}),
                ]
            ]
        )

        layout = generate_power_copper_shapes_with_agent(
            scenario,
            agent,
            max_rounds=1,
            candidates_per_round=2,
        )

        self.assertEqual({"L01", "L02"}, set(layout.group_cells["G01"]))

    def test_agent_flow_falls_back_to_default_plan_when_candidates_are_invalid(self) -> None:
        scenario = _two_group_two_layer_scenario()
        agent = _SequenceAgent([[LayerPlan({"G01": ("L01",)})]])

        layout = generate_power_copper_shapes_with_agent(
            scenario,
            agent,
            max_rounds=1,
            candidates_per_round=1,
        )
        report = validate_layout(scenario, layout)

        self.assertEqual([], report.errors)
        self.assertEqual({"L01"}, set(layout.group_cells["G01"]))
        self.assertEqual({"L02"}, set(layout.group_cells["G02"]))

    def test_llm_agent_parses_structured_layer_plan_response(self) -> None:
        scenario = _two_layer_single_group_scenario()
        agent = LLMLayerPlanningAgent(
            invoke=lambda prompt: '{"plans":[{"group_layers":{"G01":["L01","L02"]}}]}'
        )

        plans = agent.propose_layer_plans(scenario, tuple(), 1)

        self.assertEqual([LayerPlan({"G01": ("L01", "L02")})], plans)

    def test_layer_render_marks_pin_positions_with_x(self) -> None:
        scenario = self.balanced_scenario
        layout = self.balanced_layout
        drawing = render_layer(scenario, layout, "L01")

        self.assertIn("Layer L01", drawing)
        self.assertGreaterEqual(drawing.count("x"), 8)
        self.assertTrue(any(char.isalpha() for char in drawing))


def _row_counts(pins):
    counts = {}
    for pin in pins:
        counts[pin.y] = counts.get(pin.y, 0) + 1
    return counts


def _is_horizontal_vertical_or_45(start, end) -> bool:
    dx = round(end[0] - start[0], 3)
    dy = round(end[1] - start[1], 3)
    if abs(dx) < 0.001 or abs(dy) < 0.001:
        return True
    return abs(abs(dx) - abs(dy)) < 0.001


def _group_ids_on_layer(layout, layer: str) -> set[str]:
    return {
        polygon.group_id
        for polygons in layout.group_polygons.values()
        for polygon in polygons
        if polygon.layer == layer
    }


def _two_layer_single_group_scenario() -> Scenario:
    group = PowerGroup(
        "G01",
        "A",
        (
            Pin("P001", "G01", "top", 3, 3),
            Pin("P002", "G01", "bottom", 8, 8),
        ),
    )
    return Scenario(
        name="two-layer-single-group",
        width=12,
        height=12,
        layers=("L01", "L02"),
        groups=(group,),
        keepouts={"L01": frozenset(), "L02": frozenset()},
    )


def _two_group_two_layer_scenario() -> Scenario:
    first = PowerGroup(
        "G01",
        "A",
        (
            Pin("P001", "G01", "top", 3, 3),
            Pin("P002", "G01", "bottom", 4, 4),
        ),
    )
    second = PowerGroup(
        "G02",
        "B",
        (
            Pin("P003", "G02", "top", 8, 3),
            Pin("P004", "G02", "bottom", 9, 4),
        ),
    )
    return Scenario(
        name="two-group-two-layer",
        width=12,
        height=12,
        layers=("L01", "L02"),
        groups=(first, second),
        keepouts={"L01": frozenset(), "L02": frozenset()},
    )


class _SequenceAgent:
    def __init__(self, batches):
        self._batches = list(batches)

    def propose_layer_plans(self, scenario, history, count):
        return self._batches.pop(0)[:count]


if __name__ == "__main__":
    unittest.main()
