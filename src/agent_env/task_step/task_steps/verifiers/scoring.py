"""Score aggregation shared by every verifier step.

A verifier records per-criterion result rows; ``aggregate_score`` folds them into the
single ``score`` written to ``context.metadata["verifications"][verifier_id]``.

A row passes when its ``result`` is true: the pass rule of the judge output format that built it
(``rubric_binary``: a score of 1.0; ``rubric_partial``: a score at or above the criterion's
``pass_threshold``).
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

# rubric_satisfaction's default weights: a mandatory row counts three times an optional one.
DEFAULT_MANDATORY_WEIGHT = 3.0
DEFAULT_OPTIONAL_WEIGHT = 1.0


class ScoreAggregator(Enum):
    ALL_PASS = "all_pass"
    ANY_PASS = "any_pass"
    WEIGHTED_AVERAGE = "weighted_average"
    # 1.0 when every row marked ``mandatory: true`` passed, whatever the optional rows did; 0.0 otherwise, and 0.0
    # when no row is mandatory (a pass needs something mandatory to have passed).
    MANDATORY_ALL_PASS = "mandatory_all_pass"


def is_mandatory(row: dict) -> bool:
    """Whether a criterion or result row is marked ``mandatory: true``; one without the field is optional."""
    return row.get("mandatory") is True


def aggregate_score(results: list[dict], aggregator: ScoreAggregator) -> float:
    if not results:
        return 0.0
    if aggregator == ScoreAggregator.ALL_PASS:
        return 1.0 if all(r.get("result") is True for r in results) else 0.0
    elif aggregator == ScoreAggregator.ANY_PASS:
        return 1.0 if any(r.get("result") is True for r in results) else 0.0
    elif aggregator == ScoreAggregator.MANDATORY_ALL_PASS:
        mandatory = [r for r in results if is_mandatory(r)]
        return 1.0 if mandatory and all(r.get("result") is True for r in mandatory) else 0.0
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


def mandatory_verdict(results: list[dict], *, mandatory_weight: float = DEFAULT_MANDATORY_WEIGHT,
                      optional_weight: float = DEFAULT_OPTIONAL_WEIGHT) -> dict[str, Any]:
    """What a ``mandatory_all_pass`` entry records beside its score, from the rows it scored:

    - ``passed``: every mandatory row passed (and there is one), the same rule as the score.
    - ``mandatory_failed``: the ids of the mandatory rows that didn't pass, in row order.
    - ``rubric_satisfaction``: the weighted share of rows that passed, a mandatory row weighing ``mandatory_weight``
      and an optional one ``optional_weight`` (a row's own ``weight`` doesn't count). Reported for analysis only: it
      never decides the pass. None when the rows weigh nothing.
    """
    mandatory_failed = [str(r.get("id")) for r in results if is_mandatory(r) and r.get("result") is not True]
    passed = any(is_mandatory(r) for r in results) and not mandatory_failed
    total = 0.0
    met = 0.0
    for r in results:
        weight = mandatory_weight if is_mandatory(r) else optional_weight
        total += weight
        if r.get("result") is True:
            met += weight
    return {
        "passed": passed,
        "mandatory_failed": mandatory_failed,
        "rubric_satisfaction": met / total if total > 0 else None,
    }


def rubric_satisfaction_weight_problems(mandatory_weight: Any, optional_weight: Any) -> list[str]:
    """Why the weights ``mandatory_verdict`` reads are unusable; empty when fine. None is the default weight. Each must
    be a finite number, at least 0; the mandatory one above 0, so a passing run always weighs something."""
    problems = []
    for name, value, floor in (("mandatory_weight", mandatory_weight, "above 0"),
                               ("optional_weight", optional_weight, "0 or more")):
        if value is None:
            continue
        number = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        if not number or value < 0 or (name == "mandatory_weight" and value == 0):
            problems.append(f"{name} must be a number {floor}, got {value!r}")
    return problems
