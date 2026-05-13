from __future__ import annotations

from utils.zapdos.manipulation.types import GroundedPick, SceneObject


def ground_pick_target(
    catalog: list[SceneObject],
    *,
    target_query: str,
    support_query: str | None = None,
) -> GroundedPick:
    query = _normalize(target_query)
    if not query:
        raise ValueError("target_query is required")
    body_to_object = {item["body"]: item for item in catalog}
    candidates = catalog
    if support_query:
        support_bodies = {
            item["body"]
            for item in catalog
            if _score(item, _normalize(support_query)) > 0
        }
        candidates = [item for item in catalog if item.get("support_body") in support_bodies]
    ranked = []
    for item in candidates:
        score = _score(item, query)
        if score > 0:
            ranked.append((score, item))
    if not ranked:
        raise LookupError(f"No pick target matches: {target_query}")
    ranked.sort(key=lambda pair: (pair[0], pair[1]["body"]), reverse=True)
    target = ranked[0][1]
    support = body_to_object.get(target.get("support_body") or "")
    return {"target": target, "support": support}


def _score(item: SceneObject, query: str) -> int:
    label = _normalize(item["label"])
    asset_id = _normalize(item["asset_id"] or "")
    tags = [_normalize(value) for value in item["tags"]]
    score = 0
    if label == query:
        score += 100
    elif query and query in label:
        score += 20
    if asset_id == query:
        score += 90
    elif query and query in asset_id:
        score += 15
    if any(tag == query for tag in tags):
        score += 70
    score += sum(1 for tag in tags if query and query in tag)
    return score


def _normalize(value: str) -> str:
    return " ".join(value.replace("_", " ").replace("-", " ").split()).strip().lower()


__all__ = ["ground_pick_target"]
