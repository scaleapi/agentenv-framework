"""A judge's samples: ``judge_samples`` grades each criterion that many times and keeps the median, the lower of the two
middle scores for an even count, so it is always a score a sample gave. A score of 1 is always the agent's pass (a
negative criterion's is inverted), so a tie fails: two samples pass only together, four need three. For a 0-or-1
criterion an odd count is the majority vote. The row kept is that sample's (its justification, and its outcome on a
criterion with fixed outcomes), with every sample's score, verdict and justification beside it.

``min_judge_agreement`` is how many of the samples must give the kept row's verdict, as a share, or the grade fails
(``JudgeDisagreementError``). It is 0.5 by default, which the kept verdict always meets, so the median decides every
row unless a task or run asks for more: 1.0 fails any split. The share is kept to two places, so 0.67 is two samples
of three, and is 1.0 only when every sample agrees."""

from __future__ import annotations

import statistics
from typing import Any, Optional

DEFAULT_MIN_AGREEMENT = 0.5
# Agreement is kept to this many decimal places, so a bar of 0.67 is two samples of three.
AGREEMENT_PLACES = 2
# The most a split's agreement reads, one step under 1.0 at those places, so only unanimous samples read 1.0.
SPLIT_AGREEMENT_MAX = round(1 - 10 ** -AGREEMENT_PLACES, AGREEMENT_PLACES)


class JudgeDisagreementError(RuntimeError):
    """A sampled judge's samples split on a criterion more than ``min_judge_agreement`` allows."""


def validated(value: Any) -> Optional[int]:
    """``value`` as a judge_samples count: None, or a positive integer; anything else is a ValueError."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"judge_samples must be a positive integer, got {value!r}")
    return value


def validated_agreement(value: Any) -> Optional[float]:
    """``value`` as a min_judge_agreement: None, or a number from 0 to 1; anything else is a ValueError."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"min_judge_agreement must be a number from 0 to 1, got {value!r}")
    return float(value)


def _share(agreeing: int, total: int) -> float:
    """``agreeing`` of ``total`` to two places, 1.0 only when all agree, so no split reads as unanimous."""
    return 1.0 if agreeing == total else min(round(agreeing / total, AGREEMENT_PLACES), SPLIT_AGREEMENT_MAX)


def median_rows(sampled: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Each criterion's row across ``sampled``, one list of rows per sample, each with every criterion: the first
    sample's row that scored the low median, with its ``agreement``, the share of samples whose verdict (``result``)
    is that row's, and ``samples``, each sample's score, verdict and justification in order."""
    by_sample = [{row["id"]: row for row in rows} for rows in sampled]
    merged = []
    for row in sampled[0]:
        rows = [sample[row["id"]] for sample in by_sample]
        median = statistics.median_low(each["score"] for each in rows)
        chosen = next(each for each in rows if each["score"] == median)
        verdict = chosen.get("result") is True
        agreeing = sum(1 for each in rows if (each.get("result") is True) == verdict)
        merged.append({**chosen, "agreement": _share(agreeing, len(rows)),
                       "samples": [{"score": each["score"], "passed": each.get("result") is True,
                                    "justification": each.get("justification")} for each in rows]})
    return merged


def overrides(samples: Optional[int], bar: Optional[float]) -> dict[str, Any]:
    """The ``user_overrides`` a run's sample count and agreement bar make: each one given, a bar of 0 included."""
    return {**({"judge_samples": samples} if samples is not None else {}),
            **({"min_judge_agreement": bar} if bar is not None else {})}


def _sampled(row: dict[str, Any]) -> bool:
    """Whether ``median_rows`` made ``row``, so it carries its ``samples`` and ``agreement``."""
    return "samples" in row


def _tally(row: dict[str, Any]) -> str:
    """How a sampled row's samples split, as ``<passes> of <samples>``."""
    return f"{sum(1 for sample in row['samples'] if sample['passed'])} of {len(row['samples'])}"


def disagreeing(rows: list[dict[str, Any]], bar: float) -> list[dict[str, Any]]:
    """The sampled rows whose ``agreement`` is under ``bar``."""
    return [row for row in rows if _sampled(row) and row["agreement"] < bar]


def splits(rows: list[dict[str, Any]]) -> list[str]:
    """Each sampled row its samples didn't all agree on, as ``<id> (<passes> of <samples> pass)``."""
    return [f"{row.get('id')} ({_tally(row)} pass)" for row in rows if _sampled(row) and row["agreement"] < 1.0]


def disagreement(verifier_id: str, rows: list[dict[str, Any]], bar: float) -> str:
    """Why a grade whose samples split on ``rows`` failed, naming how each split."""
    named = ", ".join(f"{row.get('id')} ({_tally(row)} samples pass it)" for row in rows)
    which = "a criterion" if len(rows) == 1 else f"{len(rows)} criteria"
    return (f"Verifier '{verifier_id}': its judge samples split on {which}, under min_judge_agreement {bar:g}: "
            f"{named}. Each sample's verdict and justification are on the grade's rows; a lower "
            "min_judge_agreement keeps the median instead.")
