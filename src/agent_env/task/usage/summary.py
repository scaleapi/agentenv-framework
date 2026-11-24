"""A run's usage for people to read: its ledger (``ledger``) summed by role, then by model. An entry with per-model rows
(``models``) splits along them; one without counts under its own model. Usage the map couldn't price is unpriced, never
$0, and makes the total a lower bound, as a partial or a still running entry does; a cost priced on an assumed rate
(``cost_assumed``) makes it an upper bound."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from agent_env.config.model import ModelRole

# The order roles are shown in; any other comes after them.
ROLES = tuple(role.value for role in ModelRole)


@dataclass
class Share:
    """One role's usage on one model: its tokens, each None when nothing counted it, and the cost of the part that was
    priced, with ``unpriced`` when some part had no cost, ``assumed`` when some part's cost is an upper bound
    (``cost_assumed``), and ``partial`` when its counts are a lower bound (a partial or a still running entry)."""

    role: str
    model: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    unpriced: bool = False
    assumed: bool = False
    partial: bool = False

    def add(self, part: Mapping) -> None:
        self.input_tokens = _plus(self.input_tokens, part.get("input_tokens"))
        self.output_tokens = _plus(self.output_tokens, part.get("output_tokens"))
        cost = part.get("cost_usd")
        if _number(cost):
            self.cost_usd = (self.cost_usd or 0.0) + cost
            self.assumed = self.assumed or bool(part.get("cost_assumed"))
        else:
            self.unpriced = True


@dataclass(frozen=True)
class UsageSummary:
    """A ledger summed: its shares in role order, the cost of everything priced (None when nothing was), whether that
    cost is a lower bound, an upper bound or, both being true, an estimate, how many attempts the ledger holds, and
    where its costs came from (``cost_source``)."""

    shares: tuple[Share, ...]
    cost_usd: float | None
    lower_bound: bool
    upper_bound: bool
    attempts: int
    sources: tuple[str, ...]


def summarize(ledger: Iterable[Mapping]) -> UsageSummary | None:
    """``ledger``'s entries, a task instance's ``usage``, summed by role and model; None when it has none."""
    entries = [entry for entry in ledger if isinstance(entry, Mapping)]
    if not entries:
        return None
    shares: dict[tuple[str, str | None], Share] = {}
    sources: set[str] = set()
    for entry in entries:
        role = str(entry.get("role") or "unknown")
        for model, part in _parts(entry):
            share = shares.setdefault((role, model), Share(role, model))
            share.add(part)
            share.partial = share.partial or bool(entry.get("partial") or entry.get("running"))
            if _number(part.get("cost_usd")) and part.get("cost_source"):
                sources.add(part["cost_source"])
    ordered = sorted(shares.values(), key=lambda share: (_rank(share.role), share.role, share.model or ""))
    priced = [share.cost_usd for share in ordered if share.cost_usd is not None]
    lower_bound = any(share.unpriced or share.partial for share in ordered)
    return UsageSummary(shares=tuple(ordered), cost_usd=sum(priced) if priced else None, lower_bound=lower_bound,
                        upper_bound=any(share.assumed for share in ordered),
                        attempts=len({entry.get("attempt") for entry in entries}), sources=tuple(sorted(sources)))


def combine(summaries: Iterable[UsageSummary | None]) -> UsageSummary | None:
    """Several runs' summaries as one, as an eval's total: their shares summed by role and model."""
    present = [summary for summary in summaries if summary is not None]
    if not present:
        return None
    shares: dict[tuple[str, str | None], Share] = {}
    for summary in present:
        for share in summary.shares:
            total = shares.setdefault((share.role, share.model), Share(share.role, share.model))
            total.input_tokens = _plus(total.input_tokens, share.input_tokens)
            total.output_tokens = _plus(total.output_tokens, share.output_tokens)
            if share.cost_usd is not None:
                total.cost_usd = (total.cost_usd or 0.0) + share.cost_usd
            total.unpriced = total.unpriced or share.unpriced
            total.assumed = total.assumed or share.assumed
            total.partial = total.partial or share.partial
    ordered = sorted(shares.values(), key=lambda share: (_rank(share.role), share.role, share.model or ""))
    priced = [summary.cost_usd for summary in present if summary.cost_usd is not None]
    return UsageSummary(shares=tuple(ordered), cost_usd=sum(priced) if priced else None,
                        lower_bound=any(summary.lower_bound for summary in present),
                        upper_bound=any(summary.upper_bound for summary in present),
                        attempts=sum(summary.attempts for summary in present),
                        sources=tuple(sorted({source for summary in present for source in summary.sources})))


def _parts(entry: Mapping) -> list[tuple[str | None, Mapping]]:
    rows = entry.get("models")
    if isinstance(rows, list) and rows:
        return [(row.get("model") or entry.get("model"), row) for row in rows if isinstance(row, Mapping)]
    return [(entry.get("model"), entry)]


def _rank(role: str) -> int:
    return ROLES.index(role) if role in ROLES else len(ROLES)


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _plus(total: int | None, value: object) -> int | None:
    return total if not _number(value) else (total or 0) + int(value)
