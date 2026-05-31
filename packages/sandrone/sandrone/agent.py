from dataclasses import dataclass
from typing import Protocol

from .router import generate_power_copper_shapes, validate_layout
from .types import LayerPlan, Layout, Scenario


@dataclass(frozen=True)
class PlanEvaluation:
    plan: LayerPlan
    valid: bool
    score: float
    errors: tuple[str, ...]
    metrics: dict[str, float]


class LayerPlanningAgent(Protocol):
    def propose_layer_plans(
        self,
        scenario: Scenario,
        history: tuple[PlanEvaluation, ...],
        count: int,
    ) -> list[LayerPlan]:
        ...


@dataclass(frozen=True)
class HeuristicLayerPlanningAgent:
    max_layers_per_group: int = 2

    def propose_layer_plans(
        self,
        scenario: Scenario,
        history: tuple[PlanEvaluation, ...],
        count: int,
    ) -> list[LayerPlan]:
        plans: list[LayerPlan] = []
        width = max(1, min(self.max_layers_per_group, len(scenario.layers)))
        for offset in range(max(1, count)):
            group_layers = {}
            for index, group in enumerate(scenario.groups):
                base = (index + offset) % len(scenario.layers)
                layer_count = 1 if offset == 0 else width
                group_layers[group.group_id] = tuple(
                    scenario.layers[(base + step) % len(scenario.layers)]
                    for step in range(layer_count)
                )
            plan = LayerPlan(group_layers)
            if plan not in plans:
                plans.append(plan)
            if len(plans) == count:
                break
        return plans


def generate_power_copper_shapes_with_agent(
    scenario: Scenario,
    agent: LayerPlanningAgent | None = None,
    max_rounds: int = 1,
    candidates_per_round: int = 3,
) -> Layout:
    planner = agent or HeuristicLayerPlanningAgent()
    history: list[PlanEvaluation] = []
    best: tuple[PlanEvaluation, Layout] | None = None

    for _ in range(max_rounds):
        plans = planner.propose_layer_plans(
            scenario,
            tuple(history),
            candidates_per_round,
        )
        if not plans:
            break

        for plan in plans[:candidates_per_round]:
            evaluation, layout = evaluate_layer_plan(scenario, plan)
            history.append(evaluation)
            if layout is None or not evaluation.valid:
                continue
            if best is None or evaluation.score > best[0].score:
                best = (evaluation, layout)

    if best is None:
        fallback_layout = generate_power_copper_shapes(scenario)
        fallback_report = validate_layout(scenario, fallback_layout)
        if not fallback_report.errors:
            return fallback_layout
        errors = "; ".join(error for item in history for error in item.errors)
        if fallback_report.errors:
            errors = (
                "; ".join((errors, *fallback_report.errors))
                if errors
                else "; ".join(fallback_report.errors)
            )
        raise ValueError(f"No valid layer plan found: {errors}")
    return best[1]


def evaluate_layer_plan(
    scenario: Scenario,
    plan: LayerPlan,
) -> tuple[PlanEvaluation, Layout | None]:
    try:
        layout = generate_power_copper_shapes(scenario, plan)
        report = validate_layout(scenario, layout)
        metrics = measure_layout(layout)
        score = score_layout(tuple(report.errors), metrics)
        return (
            PlanEvaluation(
                plan=plan,
                valid=not report.errors,
                score=score,
                errors=tuple(report.errors),
                metrics=metrics,
            ),
            layout,
        )
    except ValueError as exc:
        evaluation = PlanEvaluation(
            plan=plan,
            valid=False,
            score=float("-inf"),
            errors=(str(exc),),
            metrics={},
        )
        return evaluation, None


def measure_layout(layout: Layout) -> dict[str, float]:
    group_areas = [
        sum(polygon.area for polygon in polygons)
        for polygons in layout.group_polygons.values()
    ]
    layer_loads = [
        len(set(owner.values()))
        for owner in layout.layers.values()
    ]
    return {
        "total_area": sum(group_areas),
        "min_group_area": min(group_areas) if group_areas else 0.0,
        "layer_load_imbalance": _spread(layer_loads),
        "multi_layer_groups": float(
            sum(1 for cells in layout.group_cells.values() if len(cells) > 1)
        ),
    }


def score_layout(errors: tuple[str, ...], metrics: dict[str, float]) -> float:
    if errors:
        return -1_000_000.0 - (1_000.0 * len(errors))
    return (
        metrics["total_area"]
        + (0.25 * metrics["min_group_area"])
        - (5.0 * metrics["layer_load_imbalance"])
    )


def _spread(values: list[int]) -> float:
    if not values:
        return 0.0
    return float(max(values) - min(values))
