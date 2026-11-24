"""A judge's samples: ``judge_samples`` grades each criterion that many times and keeps the median, the lower of the two
middle scores for an even count, so it is always a score a sample gave. A score of 1 is always the agent's pass (a
negative criterion's is inverted), so a tie fails: two samples pass only together, four need three. For a 0-or-1
criterion an odd count is the majority vote. The row kept is that sample's (its justification, and its outcome on a
criterion with fixed outcomes), with every sample's score and justification beside it."""

from __future__ import annotations

import statistics
from typing import Any, Optional


def validated(value: Any) -> Optional[int]:
    """``value`` as a judge_samples count: None, or a positive integer; anything else is a ValueError."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"judge_samples must be a positive integer, got {value!r}")
    return value


def median_rows(sampled: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Each criterion's row across ``sampled``, one list of rows per sample, each with every criterion: the first
    sample's row that scored the low median, with ``samples``, each sample's score and justification in order."""
    by_sample = [{row["id"]: row for row in rows} for rows in sampled]
    merged = []
    for row in sampled[0]:
        rows = [sample[row["id"]] for sample in by_sample]
        median = statistics.median_low(each["score"] for each in rows)
        chosen = next(each for each in rows if each["score"] == median)
        merged.append({**chosen, "samples": [{"score": each["score"], "justification": each.get("justification")}
                                             for each in rows]})
    return merged
