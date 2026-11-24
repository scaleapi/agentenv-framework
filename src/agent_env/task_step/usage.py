"""The usage a prompt step keeps for each role it ran turns of: the turns' reported usage summed into one entry per
role and priced, in the step's ``PromptResponse.usage`` and, as each turn ends, in ``metadata["step_usage"]``, so a
step that raises keeps its finished turns. An entry's counts have ``Usage``'s keys, which ``price`` reads."""

from __future__ import annotations

import operator
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from functools import reduce
from typing import TYPE_CHECKING, Any, NotRequired, Optional, TypedDict

from agentenv_protocol.a2a_agent import Usage

from agent_env.config.model import ModelRole
from agent_env.config.model_pricing import (
    Cost, CostStatus, PriceTable, ReportedBy, Tokens, litellm_prices, price,
)
from agent_env.task_step.context import ATTEMPT_EPOCH, DeployedAgent, TaskStepContext

if TYPE_CHECKING:
    from agent_env.a2a_agent.protocol import TerminalResponse
    from agent_env.config.model_call import ModelReply

STEP_USAGE_KEY = "step_usage"
_COUNTERS = ("tool_call_count", "input_tokens", "output_tokens", "total_tokens", "cache_read_tokens",
             "cache_write_tokens")
_TOKEN_COUNTS = _COUNTERS[1:]
_MODEL_COUNTS = ("calls", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
# How many turns' unreadable values an entry keeps; ``unparsed_turns`` counts them all.
_UNPARSED_TURNS_KEPT = 5


class CostFields(TypedDict):
    """A cost as an entry or a model's row keeps it (``Cost.fields``)."""

    cost_usd: Optional[float]
    cost_status: str
    cost_source: Optional[str]
    reported_cost_usd: Optional[float]
    uncached_input_tokens: NotRequired[int]
    unit_prices: NotRequired[dict[str, float]]
    cost_assumed: NotRequired[list[str]]
    unpriced_reason: NotRequired[str]


class ModelEntry(CostFields):
    """One model's share of a usage entry: its counts summed over the role's turns, and its cost at that model's own
    rate."""

    model: Optional[str]
    calls: Optional[int]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cache_read_tokens: Optional[int]
    cache_write_tokens: Optional[int]
    input_includes_cache: Optional[bool]


class UsageEntry(CostFields):
    """One role's usage in one step, as ``PromptResponse.usage`` and ``metadata["step_usage"]`` hold it. A count or
    a cost is None when nothing reported it, never 0. ``cost_usd`` is the cost ``cost_status`` and ``cost_source``
    describe: priced from LiteLLM's price map, the ``uncached_input_tokens`` and the cache and output counts each at
    its ``unit_prices`` rate, the agent's own figure, or None when unpriced, with the agent's figure always in
    ``reported_cost_usd``. When a turn's calls used several models (``Usage.models``), ``models`` holds each model's
    counts and cost at its own rate, ``cost_usd`` is their sum, and the entry has no ``unit_prices`` or
    ``uncached_input_tokens`` of its own."""

    role: str
    model: Optional[str]
    endpoint_host: Optional[str]
    attempt: Optional[int]
    turns: int
    tool_call_count: Optional[int]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    cache_read_tokens: Optional[int]
    cache_write_tokens: Optional[int]
    input_includes_cache: Optional[bool]
    models: NotRequired[list[ModelEntry]]
    partial: bool
    unreported_calls: NotRequired[int]
    unparsed: NotRequired[list[dict]]
    unparsed_turns: NotRequired[int]


@dataclass
class RoleUsage:
    """One role's turns in one step: the model it ran, the host of the endpoint it called, the attempt epoch the step
    ran under, and each turn's usage, None for a turn that reported none. ``call_costs`` are the costs of calls core
    made itself (``StepUsage.reply``), each costed when it was made; a role with them is costed from them alone."""

    role: str
    model: Optional[str] = None
    endpoint_host: Optional[str] = None
    attempt: Optional[int] = field(default_factory=ATTEMPT_EPOCH.get)
    turns: list[Optional[Usage]] = field(default_factory=list)
    unparsed: list[dict] = field(default_factory=list)
    call_costs: list[Cost] = field(default_factory=list)
    # Whose figure a turn's ``cost_usd`` is: an agent's own (the price map wins over it), or an endpoint's bill (it
    # wins over the map).
    reported_by: ReportedBy = ReportedBy.HARNESS

    def add(self, usage: Optional[Usage], unparsed: Optional[dict] = None) -> None:
        """One finished turn's usage, and what of it couldn't be read."""
        self.turns.append(usage)
        if unparsed:
            self.unparsed.append(unparsed)

    def entry(self, prices: PriceTable = PriceTable()) -> UsageEntry:
        """The turns summed and priced: each count the sum of the turns that reported it, None when none did, never
        0, and the sum's cost by ``price`` from ``prices``, the agent's own figure kept beside it; each
        turn's tokens are read by its own ``input_includes_cache`` before they're summed, so turns that disagree on it
        are still charged for their cached input once. The map prices the entry only when every turn that reported a
        token count gave input and output; otherwise its sums can't be charged in full, and it isn't priced short.
        ``input_includes_cache`` is the turns' flag when they agree, else None. ``partial`` says the sum is a lower
        bound: some turns reported a token count and others none, or a turn said some of its calls reported none
        (``unreported_calls``, which the entry sums, or a proxy's ``unreported``), or a turn was cut off
        (``StepUsage.cut_off``). When a turn says its counts by model (``Usage.models``), each model is priced at its own
        rate (``_by_model``)."""
        reported = [usage for usage in self.turns if usage is not None]
        counted = [usage for usage in reported
                   if usage.models or any(getattr(usage, name) is not None for name in _TOKEN_COUNTS)]
        unreported = [n for usage in reported if (n := _unreported_calls(usage))]
        partial = (0 < len(counted) < len(self.turns) or bool(unreported)
                   or any(usage.provider_details.get("unreported") for usage in reported))
        figures = [usage.cost_usd if usage is not None else None for usage in self.turns]
        models = (self._by_model(counted, prices)
                  if not self.call_costs and any(usage.models for usage in counted) else None)
        if self.call_costs:
            cost = _combined(self.call_costs)
        elif models is None:
            cost = _cost_of(_tokens(counted), figures, model=self.model, prices=prices, reported_by=self.reported_by)
        else:
            cost = _summed([cost for _, cost in models], _total(reported, "cost_usd"))
        entry = UsageEntry(role=self.role, model=self.model, endpoint_host=self.endpoint_host, attempt=self.attempt,
                           turns=len(self.turns), **{name: _total(reported, name) for name in _COUNTERS},
                           input_includes_cache=_agreed(counted), **cost.fields(), partial=partial)
        if models is not None:
            entry["models"] = [row for row, _ in models]
        if unreported:
            entry["unreported_calls"] = sum(unreported)
        if self.unparsed:
            entry["unparsed"] = self.unparsed[:_UNPARSED_TURNS_KEPT]
            entry["unparsed_turns"] = len(self.unparsed)
        return entry

    def _by_model(self, counted: list[Usage], prices: PriceTable) -> list[tuple[ModelEntry, Cost]]:
        """The turns' counts by model, each priced at its own model's rate: a turn's ``models``, and a turn without
        them counted as the role's model's."""
        shares: dict[Optional[str], list[Any]] = {}
        for usage in counted:
            if usage.models:
                for share in usage.models:
                    shares.setdefault(share.model, []).append(share)
            else:
                shares.setdefault(self.model, []).append(usage)
        rows = []
        for model, of in shares.items():
            cost = _cost_of(_tokens(of), [share.cost_usd for share in of], model=model, prices=prices,
                            reported_by=self.reported_by)
            row = ModelEntry(model=model, **{name: _total(of, name) for name in _MODEL_COUNTS},
                             input_includes_cache=_agreed(of), **cost.fields())
            rows.append((row, cost))
        return rows


def _total(records: Iterable[Any], name: str) -> Any:
    """The sum of ``name`` over the records that give it, None when none does, never 0."""
    values = [value for record in records if (value := getattr(record, name, None)) is not None]
    return sum(values) if values else None


def _tokens(records: list[Any]) -> Optional[Tokens]:
    """The records' tokens summed, each read by its own ``input_includes_cache``; None when one lacks its input or
    output, so a sum that can't be charged in full isn't priced short."""
    charged = [Tokens.of(record.to_dict()) for record in records]
    return reduce(operator.add, charged) if charged and None not in charged else None


def _agreed(records: list[Any]) -> Optional[bool]:
    """The records' ``input_includes_cache`` when they all agree on it, else None."""
    flags = {record.input_includes_cache for record in records}
    return flags.pop() if len(flags) == 1 else None


def _cost_of(tokens: Optional[Tokens], figures: list[Optional[float]], *, model: Optional[str], prices: PriceTable,
             reported_by: ReportedBy) -> Cost:
    """``price`` given the turns' (or a model's shares') own figures: they stand in for the map only when every one
    came, since a sum of some isn't the cost, so a model the map lacks is then unpriced; their sum is kept beside the
    cost as ``reported_usd`` either way."""
    given = [figure for figure in figures if figure is not None]
    summed = sum(given) if given else None
    cost = price(tokens, summed if len(given) == len(figures) else None, model=model, prices=prices,
                 reported_by=reported_by)
    return cost if cost.reported_usd == summed else replace(cost, reported_usd=summed)


def _summed(costs: list[Cost], reported_usd: Optional[float]) -> Cost:
    """An entry's cost from its models' costs: their sum, as sure as the least sure of them. Unpriced when one is, so
    a sum short of a model is never shown; reported, with its source, when one is the agent's figure; else priced."""
    unpriced = next((cost for cost in costs if cost.status is CostStatus.UNPRICED), None)
    if unpriced is not None:
        return Cost(None, CostStatus.UNPRICED, None, reported_usd, unpriced=unpriced.unpriced)
    least = next((cost for cost in costs if cost.status is not CostStatus.PRICED), costs[0])
    assumed = tuple(dict.fromkeys(assumed for cost in costs for assumed in cost.assumed))
    return Cost(sum(cost.usd for cost in costs), least.status, least.source, reported_usd, assumed=assumed)


def _combined(costs: list[Cost]) -> Cost:
    """Calls' costs summed: billed (``reported``) only when every costed call was billed, otherwise ``priced``, an
    estimate, with the source of the priced calls, the assumptions any made, and the bills kept in ``reported_usd``.
    Unpriced when none was costed; a call with no cost is left out of the sum, which ``partial`` marks."""
    costed = [c for c in costs if c.usd is not None]
    if not costed:
        return Cost(None, CostStatus.UNPRICED, None, None, unpriced=costs[-1].unpriced)
    status = CostStatus.REPORTED if all(c.status is CostStatus.REPORTED for c in costed) else CostStatus.PRICED
    sources = {c.source for c in costed if c.status is status}
    bills = [c.reported_usd for c in costs if c.reported_usd is not None]
    assumed = tuple(dict.fromkeys(a for c in costed for a in c.assumed))
    priced = [c for c in costed if c.status is CostStatus.PRICED]
    uncached = [c.uncached_input for c in priced]
    rates = {repr(c.unit_prices): c.unit_prices for c in priced}
    return Cost(sum(c.usd for c in costed), status, sources.pop() if len(sources) == 1 else None,
                sum(bills) if bills else None,
                uncached_input=sum(uncached) if priced and None not in uncached else None,
                unit_prices=next(iter(rates.values())) if len(rates) == 1 else None, assumed=assumed)


def _unreported() -> Usage:
    """A turn or call whose usage never came back."""
    return Usage(provider_details={"unreported": True})


def _unreported_calls(usage: Usage) -> int:
    """The calls a turn's usage says reported none, as the shipped agents put it in ``provider_details``."""
    value = usage.provider_details.get("unreported_calls")
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


class StepUsage:
    """A prompt step's usage, role by role: each role ``start``ed with its model and agent, each of its finished turns
    recorded with ``turn``, and the entries, priced from LiteLLM's price map, written to
    ``metadata["step_usage"][step_id]`` after every turn."""

    def __init__(self, context: TaskStepContext, step_id: str) -> None:
        self._context = context
        self._step_id = step_id
        self._roles: dict[ModelRole, RoleUsage] = {}
        self._prices = litellm_prices()

    def start(self, role: ModelRole, model: Optional[str], agent: Optional[DeployedAgent]) -> None:
        """Count ``role``'s turns on ``model``, recording the model and the host of the endpoint ``agent`` was given in
        ``metadata.model_endpoints``, under the role's own name."""
        host = agent.model_endpoint_host if agent else None
        self._context.record_model_endpoint(role.value, self._step_id, model, host)
        self._roles[role] = RoleUsage(role.value, model, host)

    def turn(self, role: ModelRole, reply: TerminalResponse) -> None:
        """One finished turn of ``role``, written at once, so a step that raises later keeps it. A role that wasn't
        started, a human peer's, is skipped."""
        usage = self._roles.get(role)
        if usage is not None:
            usage.add(reply.usage, reply.usage_unparsed)
            self.entries()

    def call(self, role: ModelRole, usage: Optional[Usage], *, model: Optional[str], endpoint_host: Optional[str],
             reported_by: ReportedBy = ReportedBy.HARNESS) -> None:
        """One model call of ``role`` that isn't an agent's turn, as a judge's, written at once. The role starts at its
        first call, with the model and the endpoint host that call reached; ``reported_by`` says whose figure a call's
        ``cost_usd`` is, and the role keeps its first call's."""
        self._add_call(role, usage, model, endpoint_host, reported_by)

    def reply(self, role: ModelRole, reply: ModelReply) -> None:
        """One model call the step made itself as ``role`` through ``model_call`` (a user-sim gate's judge), written
        at once. Its cost is kept as the call was costed (``_combined`` sums them); one with no cost leaves the entry a
        lower bound, as ``cut_off`` does."""
        usage = self._role_for_call(role, reply.model, reply.endpoint_host, ReportedBy.HARNESS)
        counts = Usage.from_model_response(reply.usage) if reply.usage else Usage()
        if reply.cost.usd is None:
            counts = replace(counts, provider_details={**counts.provider_details, "unreported": True})
        usage.call_costs.append(reply.cost)
        usage.add(counts)
        self.entries()

    def _role_for_call(self, role: ModelRole, model: Optional[str], endpoint_host: Optional[str],
                       reported_by: ReportedBy) -> RoleUsage:
        """The role a call is counted under, started at its first call."""
        if role not in self._roles:
            self._context.record_model_endpoint(role.value, self._step_id, model, endpoint_host)
            self._roles[role] = RoleUsage(role.value, model, endpoint_host, reported_by=reported_by)
        return self._roles[role]

    def cut_off(self, role: ModelRole) -> None:
        """A turn of ``role`` that was sent but never replied: cut off by a time limit, by a cancel, or by its agent no
        longer answering (``AgentUnreachableError``, a ``TimeoutError``). Its calls are billed but their usage is
        unknown, so it counts as a turn of unknown usage, marking the role's entry a lower bound. A role that wasn't
        started, a human peer's, is skipped."""
        usage = self._roles.get(role)
        if usage is not None:
            usage.add(_unreported())
            self.entries()

    def cut_off_call(self, role: ModelRole, *, model: Optional[str], endpoint_host: Optional[str],
                     reported_by: ReportedBy = ReportedBy.HARNESS) -> None:
        """A model call of ``role`` that was sent but never answered, cut off by a time limit or a cancel: a call of
        unknown usage, marking the role's entry a lower bound as ``cut_off`` marks a turn's, recorded as ``call``
        records one, so it starts the role when it's the first."""
        self._add_call(role, _unreported(), model, endpoint_host, reported_by)

    def _add_call(self, role: ModelRole, usage: Optional[Usage], model: Optional[str], endpoint_host: Optional[str],
                  reported_by: ReportedBy) -> None:
        self._role_for_call(role, model, endpoint_host, reported_by).add(usage)
        self.entries()

    def entries(self) -> Optional[list[UsageEntry]]:
        """The entries of the roles that ran turns, written to ``metadata["step_usage"]``; None, and nothing written,
        when none did."""
        entries = [usage.entry(self._prices) for usage in self._roles.values() if usage.turns]
        if not entries:
            return None
        self._context.metadata.setdefault(STEP_USAGE_KEY, {})[self._step_id] = entries
        return entries

