import argparse
import json
import os
from pathlib import Path

from sandrone.agent import (
    HeuristicLayerPlanningAgent,
    evaluate_layer_plan,
)
from sandrone.cases import build_balanced_case
from sandrone.llm_agent import LLMLayerPlanningAgent
from sandrone.plot import save_layer_contact_sheet
from sandrone.router import generate_power_copper_shapes, validate_layout
from sandrone.types import Scenario


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[2]
    _load_env(root / args.env)

    scenario = _layer_limited_scenario(args.layers, args.pins)
    agent = _agent(args.planner, args.max_layers_per_group)
    history = []
    best = None

    print(f"model={os.environ.get('COPILOTKIT_MODEL', '')}")
    print(f"scenario={scenario.name}")
    print(f"groups={len(scenario.groups)}")
    print(f"layers={','.join(scenario.layers)}")

    for round_index in range(1, args.rounds + 1):
        plans = agent.propose_layer_plans(scenario, tuple(history), args.candidates)
        print(f"round_{round_index}_candidate_count={len(plans)}")
        for candidate_index, plan in enumerate(plans, start=1):
            evaluation, layout = evaluate_layer_plan(scenario, plan)
            history.append(evaluation)
            _print_candidate(round_index, candidate_index, plan, evaluation)
            if evaluation.valid and layout is not None:
                if best is None or evaluation.score > best[0].score:
                    best = (evaluation, layout, f"round-{round_index}/candidate-{candidate_index}")

    if best is None:
        final_layout = generate_power_copper_shapes(scenario)
        final_source = "fallback-default"
    else:
        final_layout = best[1]
        final_source = best[2]

    report = validate_layout(scenario, final_layout)
    image_path = root / args.out
    save_layer_contact_sheet(scenario, final_layout, image_path)

    print(f"history_count={len(history)}")
    print(f"valid_candidates={sum(1 for item in history if item.valid)}")
    print(f"final_source={final_source}")
    print(f"final_errors={len(report.errors)}")
    print(f"final_multilayer_groups={_multilayer_group_count(final_layout)}")
    print(f"image_path={image_path}")
    for error in report.errors[: args.error_limit]:
        print(f"final_error={error}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", choices=("llm", "heuristic"), default="llm")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--pins", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--max-layers-per-group", type=int, default=2)
    parser.add_argument("--error-limit", type=int, default=5)
    parser.add_argument("--env", default="apps/desktop/.env")
    parser.add_argument("--out", default="packages/sandrone/out/agent-3layer.png")
    return parser.parse_args()


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def _layer_limited_scenario(layer_count: int, pin_count: int) -> Scenario:
    base = build_balanced_case(pin_count=pin_count)
    layers = base.layers[:layer_count]
    obstacles = base.obstacles or {}
    return Scenario(
        name=f"{base.name}-{layer_count}layer",
        width=base.width,
        height=base.height,
        layers=layers,
        groups=base.groups,
        keepouts={layer: base.keepouts[layer] for layer in layers},
        obstacles={layer: obstacles[layer] for layer in layers},
        clearance=base.clearance,
        min_width=base.min_width,
    )


def _agent(planner: str, max_layers_per_group: int):
    if planner == "heuristic":
        return HeuristicLayerPlanningAgent(max_layers_per_group)
    return LLMLayerPlanningAgent(max_layers_per_group)


def _print_candidate(round_index, candidate_index, plan, evaluation) -> None:
    payload = {group_id: list(layers) for group_id, layers in sorted(plan.group_layers.items())}
    print(f"round_{round_index}_candidate_{candidate_index}_valid={str(evaluation.valid).lower()}")
    print(f"round_{round_index}_candidate_{candidate_index}_score={round(evaluation.score, 3)}")
    print(f"round_{round_index}_candidate_{candidate_index}_errors={len(evaluation.errors)}")
    print(f"round_{round_index}_candidate_{candidate_index}_multilayer_groups={_plan_multilayer_count(plan)}")
    print(
        f"round_{round_index}_candidate_{candidate_index}_plan="
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )
    for error in evaluation.errors[:5]:
        print(f"round_{round_index}_candidate_{candidate_index}_error={error}")


def _plan_multilayer_count(plan) -> int:
    return sum(1 for layers in plan.group_layers.values() if len(layers) > 1)


def _multilayer_group_count(layout) -> int:
    return sum(1 for cells in layout.group_cells.values() if len(cells) > 1)


if __name__ == "__main__":
    main()
