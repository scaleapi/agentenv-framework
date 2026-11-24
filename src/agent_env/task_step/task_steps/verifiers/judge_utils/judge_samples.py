"""A judge's samples: ``judge_samples`` grades each criterion that many times and takes its median, which for a 0-or-1
criterion is the majority vote. The count is odd, so the median is a score a sample gave, and the row kept is that
sample's (its justification, and its outcome on a criterion with fixed outcomes), with every sample's score and
justification beside it."""

from __future__ import annotations

import statistics
from typing import Any, Optional


def validated(value: Any) -> Optional[int]:
    """``value`` as a judge_samples count: None, or a positive odd integer; anything else is a ValueError, an even
    count naming the odd ones beside it."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"judge_samples must be a positive odd integer, got {value!r}")
    if value % 2 == 0:
        raise ValueError(f"judge_samples must be odd, so each criterion's median is a score a sample gave: "
                         f"use {value - 1} or {value + 1}")
    return value


def median_rows(sampled: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Each criterion's row across ``sampled``, one list of rows per sample, each with every criterion: the first
    sample's row that scored the median, with ``samples``, each sample's score and justification in order."""
    by_sample = [{row["id"]: row for row in rows} for rows in sampled]
    merged = []
    for row in sampled[0]:
        rows = [sample[row["id"]] for sample in by_sample]
        median = statistics.median(each["score"] for each in rows)
        chosen = next(each for each in rows if each["score"] == median)
        merged.append({**chosen, "samples": [{"score": each["score"], "justification": each.get("justification")}
                                             for each in rows]})
    return merged
