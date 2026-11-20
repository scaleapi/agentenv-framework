"""The usage a prompt step keeps for each role it ran turns of: the turns' reported usage summed into one entry per
role, in the step's ``PromptResponse.usage`` and, as each turn ends, in ``metadata["step_usage"]``, so a step that
raises keeps its finished turns. An entry's counts and cost have ``Usage``'s keys, so a price reads it as a Usage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NotRequired, Optional, TypedDict

from agentenv_protocol.a2a_agent import Usage

from agent_env.config.model import ModelRole
from agent_env.task_step.context import ATTEMPT_EPOCH, DeployedAgent, TaskStepContext

if TYPE_CHECKING:
    from agent_env.a2a_agent.protocol import TerminalResponse

STEP_USAGE_KEY = "step_usage"
_COUNTERS = ("tool_call_count", "input_tokens", "output_tokens", "total_tokens", "cache_read_tokens",
             "cache_write_tokens")
_TOKEN_COUNTS = _COUNTERS[1:]
# How many turns' unreadable values an entry keeps; ``unparsed_turns`` counts them all.
_UNPARSED_TURNS_KEPT = 5


class UsageEntry(TypedDict):
    """One role's usage in one step, as ``PromptResponse.usage`` and ``metadata["step_usage"]`` hold it. A count or
    the cost is None when no turn reported it, never 0."""

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
    cost_usd: Optional[float]
    partial: bool
    unreported_calls: NotRequired[int]
    unparsed: NotRequired[list[dict]]
    unparsed_turns: NotRequired[int]


@dataclass
class RoleUsage:
    """One role's turns in one step: the model it ran, the host of the endpoint it called, the attempt epoch the step
    ran under, and each turn's usage, None for a turn that reported none."""

    role: str
    model: Optional[str] = None
    endpoint_host: Optional[str] = None
    attempt: Optional[int] = field(default_factory=ATTEMPT_EPOCH.get)
    turns: list[Optional[Usage]] = field(default_factory=list)
    unparsed: list[dict] = field(default_factory=list)

    def add(self, usage: Optional[Usage], unparsed: Optional[dict] = None) -> None:
        """One finished turn's usage, and what of it couldn't be read."""
        self.turns.append(usage)
        if unparsed:
            self.unparsed.append(unparsed)

    def entry(self) -> UsageEntry:
        """The turns summed: each count and the cost the sum of the turns that reported it, None when none did, never
        0. ``input_includes_cache`` is the turns' flag when they agree, else None. ``partial`` says the sum is a lower
        bound: some turns reported a token count and others none, or a turn said some of its calls reported none
        (``unreported_calls``, which the entry sums, or a proxy's ``unreported``)."""
        reported = [usage for usage in self.turns if usage is not None]

        def total(name: str) -> Any:
            values = [getattr(usage, name) for usage in reported if getattr(usage, name) is not None]
            return sum(values) if values else None

        counted = [usage for usage in reported if any(getattr(usage, name) is not None for name in _TOKEN_COUNTS)]
        flags = {usage.input_includes_cache for usage in counted}
        unreported = [n for usage in reported if (n := _unreported_calls(usage))]
        partial = (0 < len(counted) < len(self.turns) or bool(unreported)
                   or any(usage.provider_details.get("unreported") for usage in reported))
        entry = UsageEntry(
            role=self.role, model=self.model, endpoint_host=self.endpoint_host, attempt=self.attempt,
            turns=len(self.turns), tool_call_count=total("tool_call_count"), input_tokens=total("input_tokens"),
            output_tokens=total("output_tokens"), total_tokens=total("total_tokens"),
            cache_read_tokens=total("cache_read_tokens"), cache_write_tokens=total("cache_write_tokens"),
            input_includes_cache=flags.pop() if len(flags) == 1 else None, cost_usd=total("cost_usd"),
            partial=partial,
        )
        if unreported:
            entry["unreported_calls"] = sum(unreported)
        if self.unparsed:
            entry["unparsed"] = self.unparsed[:_UNPARSED_TURNS_KEPT]
            entry["unparsed_turns"] = len(self.unparsed)
        return entry


def _unreported_calls(usage: Usage) -> int:
    """The calls a turn's usage says reported none, as the shipped agents put it in ``provider_details``."""
    value = usage.provider_details.get("unreported_calls")
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


class StepUsage:
    """A prompt step's usage, role by role: each role ``start``ed with its model and agent, each of its finished turns
    recorded with ``turn``, and the entries written to ``metadata["step_usage"][step_id]`` after every turn."""

    def __init__(self, context: TaskStepContext, step_id: str) -> None:
        self._context = context
        self._step_id = step_id
        self._roles: dict[ModelRole, RoleUsage] = {}

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

    def entries(self) -> Optional[list[UsageEntry]]:
        """The entries of the roles that ran turns, written to ``metadata["step_usage"]``; None, and nothing written,
        when none did."""
        entries = [usage.entry() for usage in self._roles.values() if usage.turns]
        if not entries:
            return None
        self._context.metadata.setdefault(STEP_USAGE_KEY, {})[self._step_id] = entries
        return entries
