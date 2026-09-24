"""Score aggregation shared by every verifier step.

A verifier records per-criterion result rows; ``aggregate_score`` folds them into the
single ``score`` written to ``context.metadata["verifications"][verifier_id]``.
"""

from __future__ import annotations

from enum import Enum


class ScoreAggregator(Enum):
    ALL_PASS = "all_pass"
    ANY_PASS = "any_pass"
    WEIGHTED_AVERAGE = "weighted_average"


def aggregate_score(results: list[dict], aggregator: ScoreAggregator) -> float:
    if not results:
        return 0.0
    if aggregator == ScoreAggregator.ALL_PASS:
        return 1.0 if all(r.get("result") is True for r in results) else 0.0
    elif aggregator == ScoreAggregator.ANY_PASS:
        return 1.0 if any(r.get("result") is True for r in results) else 0.0
    elif aggregator == ScoreAggregator.WEIGHTED_AVERAGE:
        pos_num = 0.0
        pos_den = 0.0
        neg_penalty = 0.0
        for r in results:
            score = r.get("score", 1.0 if r.get("result") else 0.0)
            weight = float(r.get("weight", 1.0))
            if weight > 0:
                pos_num += score * weight
                pos_den += weight
            elif weight < 0:
                neg_penalty += (1.0 - score) * abs(weight)
        if pos_den == 0:
            return 0.0
        return max(0.0, min(1.0, (pos_num - neg_penalty) / pos_den))
    raise ValueError(f"Unknown aggregator: {aggregator}")
