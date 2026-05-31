import json
import os
from dataclasses import dataclass
from typing import Callable

from .agent import HeuristicLayerPlanningAgent, PlanEvaluation
from .types import LayerPlan, Scenario


@dataclass(frozen=True)
class LLMLayerPlanningAgent:
    max_layers_per_group: int = 3
    invoke: Callable[[str], str] | None = None

    def propose_layer_plans(
        self,
        scenario: Scenario,
        history: tuple[PlanEvaluation, ...],
        count: int,
    ) -> list[LayerPlan]:
        prompt = _build_prompt(scenario, history, count, self.max_layers_per_group)
        text = self._invoke(prompt)
        plans = _parse_plans(scenario, text, count, self.max_layers_per_group)
        if plans:
            return plans
        return HeuristicLayerPlanningAgent(self.max_layers_per_group).propose_layer_plans(
            scenario,
            history,
            count,
        )

    def _invoke(self, prompt: str) -> str:
        if self.invoke is not None:
            return self.invoke(prompt)

        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=os.environ.get("COPILOTKIT_MODEL", "gpt-4o"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
        )
        response = model.invoke(prompt)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            return "".join(str(item) for item in content)
        return str(content)


def _build_prompt(
    scenario: Scenario,
    history: tuple[PlanEvaluation, ...],
    count: int,
    max_layers_per_group: int,
) -> str:
    summary = {
        "layers": scenario.layers,
        "groups": [
            {
                "group_id": group.group_id,
                "pin_count": len(group.pins),
                "bbox": _bbox([pin.point for pin in group.pins]),
                "centroid": _centroid([pin.point for pin in group.pins]),
            }
            for group in scenario.groups
        ],
        "recent_evaluations": [
            {
                "valid": item.valid,
                "score": item.score,
                "errors": item.errors[:5],
                "metrics": item.metrics,
            }
            for item in history[-4:]
        ],
    }
    return "\n".join(
        [
            "You are selecting intermediate copper layers for power nets.",
            "Return JSON only, with this schema:",
            '{"plans":[{"group_layers":{"G01":["L01","L02"]}}]}',
            f"Produce at most {count} plans.",
            f"Every group must have 1 to {max_layers_per_group} existing layers.",
            "Prefer more layers only for groups that benefit from extra copper area.",
            "Do not include markdown fences or prose.",
            json.dumps(summary, separators=(",", ":")),
        ]
    )


def _parse_plans(
    scenario: Scenario,
    text: str,
    count: int,
    max_layers_per_group: int,
) -> list[LayerPlan]:
    try:
        payload = json.loads(_json_text(text))
    except json.JSONDecodeError:
        return []

    raw_plans = payload.get("plans", payload if isinstance(payload, list) else [])
    if not isinstance(raw_plans, list):
        return []

    plans: list[LayerPlan] = []
    for raw_plan in raw_plans:
        group_layers = raw_plan.get("group_layers") if isinstance(raw_plan, dict) else None
        plan = _coerce_plan(scenario, group_layers, max_layers_per_group)
        if plan is not None and plan not in plans:
            plans.append(plan)
        if len(plans) == count:
            break
    return plans


def _coerce_plan(
    scenario: Scenario,
    raw_group_layers,
    max_layers_per_group: int,
) -> LayerPlan | None:
    if not isinstance(raw_group_layers, dict):
        return None

    scenario_groups = {group.group_id for group in scenario.groups}
    scenario_layers = set(scenario.layers)
    if set(raw_group_layers) != scenario_groups:
        return None

    group_layers: dict[str, tuple[str, ...]] = {}
    for group_id, raw_layers in raw_group_layers.items():
        if not isinstance(raw_layers, list):
            return None
        layers = tuple(layer for layer in raw_layers if isinstance(layer, str))
        if not 1 <= len(layers) <= max_layers_per_group:
            return None
        if len(layers) != len(set(layers)) or set(layers) - scenario_layers:
            return None
        group_layers[group_id] = layers
    return LayerPlan(group_layers)


def _json_text(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start : end + 1]


def _bbox(points) -> tuple[int, int, int, int]:
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def _centroid(points) -> tuple[float, float]:
    return (
        round(sum(point[0] for point in points) / len(points), 2),
        round(sum(point[1] for point in points) / len(points), 2),
    )
