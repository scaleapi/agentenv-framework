"""A run's usage for people to read: its ledger (``ledger``) summed by role, then by model. An entry with per-model rows
(``models``) splits along them; one without counts under its own model. Usage the map couldn't price is unpriced, never
$0, and makes the cost a lower bound, as a partial or a still running entry does; a cost priced on an assumed rate
(``cost_assumed``) makes it an upper bound."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent_env.config.model import ModelRole

if TYPE_CHECKING:
    from agent_env.task_step.usage import ModelEntry, UsageEntry

# The order roles are shown in; any other comes after them.
_ROLES = tuple(role.value for role in ModelRole)


@dataclass
class Share:
    """One role's usage on one model: its tokens, each None when nothing counted it, and the cost of the part that was
    priced, with ``unpriced`` when some part had no cost, ``assumed`` when some part's cost was priced on an assumed
    rate (``cost_assumed``), and ``partial`` when its counts fall short (a partial or a still running entry)."""

    role: str
    model: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    unpriced: bool = False
    assumed: bool = False
    partial: bool = False

    @property
    def lower_bound(self) -> bool:
        return self.unpriced or self.partial

    @property
    def upper_bound(self) -> bool:
        return self.assumed

    def add(self, part: UsageEntry | ModelEntry, partial: bool) -> None:
        self.input_tokens = _plus(self.input_tokens, part.get("input_tokens"))
        self.output_tokens = _plus(self.output_tokens, part.get("output_tokens"))
        if (cost := part.get("cost_usd")) is None:
            self.unpriced = True
        else:
            self.cost_usd = (self.cost_usd or 0.0) + cost
            self.assumed = self.assumed or bool(part.get("cost_assumed"))
        self.partial = self.partial or partial


@dataclass(frozen=True)
class UsageSummary:
    """A ledger summed: its shares in role order, how many attempts it holds, and where its costs came from
    (``cost_source``). Its cost and bounds are its shares'."""

    shares: tuple[Share, ...]
    attempts: int
    sources: tuple[str, ...]

    @property
    def cost_usd(self) -> float | None:
        """The cost of everything priced; None when nothing was."""
        priced = [share.cost_usd for share in self.shares if share.cost_usd is not None]
        return sum(priced) if priced else None

    @property
    def lower_bound(self) -> bool:
        return any(share.lower_bound for share in self.shares)

    @property
    def upper_bound(self) -> bool:
        return any(share.upper_bound for share in self.shares)


def summarize(ledger: Iterable[UsageEntry]) -> UsageSummary | None:
    """``ledger``'s entries summed by role and model; None when it has none. A task instance's ``usage`` gives a run's;
    several runs' together give their total, whose ``attempts`` counts attempt numbers, not the runs' attempts."""
    entries = list(ledger)
    if not entries:
        return None
    shares: dict[tuple[str, str | None], Share] = {}
    sources: set[str] = set()
    for entry in entries:
        partial = bool(entry.get("partial") or entry.get("running"))
        for model, part in _parts(entry):
            shares.setdefault((entry["role"], model), Share(entry["role"], model)).add(part, partial)
            if part.get("cost_usd") is not None and part.get("cost_source"):
                sources.add(part["cost_source"])
    ordered = sorted(shares.values(), key=lambda share: (_rank(share.role), share.role, share.model or ""))
    return UsageSummary(shares=tuple(ordered), attempts=len({entry.get("attempt") for entry in entries}),
                        sources=tuple(sorted(sources)))


def _parts(entry: UsageEntry) -> list[tuple[str | None, UsageEntry | ModelEntry]]:
    if rows := entry.get("models"):
        return [(row.get("model") or entry.get("model"), row) for row in rows]
    return [(entry.get("model"), entry)]


def _rank(role: str) -> int:
    return _ROLES.index(role) if role in _ROLES else len(_ROLES)


def _plus(total: int | None, value: int | None) -> int | None:
    return total if value is None else (total or 0) + value
