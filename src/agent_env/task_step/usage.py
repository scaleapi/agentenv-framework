"""The usage a prompt step keeps for each role it ran turns of: the turns' reported usage summed into one entry per
role, in the step's ``PromptResponse.usage`` and, as each turn ends, in ``metadata["step_usage"]``, so a step that
raises keeps its finished turns. An entry's counts and cost have ``Usage``'s keys, so a price reads it as a Usage."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Optional

from agentenv_protocol.a2a_agent import Usage

from agent_env.task_step.context import ATTEMPT_EPOCH, TaskStepContext

STEP_USAGE_KEY = "step_usage"
_COUNTERS = ("tool_call_count", "input_tokens", "output_tokens", "total_tokens", "cache_read_tokens",
             "cache_write_tokens")
_TOKEN_COUNTS = _COUNTERS[1:]
# How many turns' unreadable values an entry keeps; ``unparsed_turns`` counts them all.
_UNPARSED_TURNS_KEPT = 5


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

    def entry(self) -> dict[str, Any]:
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
        entry: dict[str, Any] = {
            "role": self.role, "model": self.model, "endpoint_host": self.endpoint_host, "attempt": self.attempt,
            "turns": len(self.turns), **{name: total(name) for name in _COUNTERS},
            "input_includes_cache": flags.pop() if len(flags) == 1 else None, "cost_usd": total("cost_usd"),
            "partial": partial,
        }
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


def record_step_usage(context: TaskStepContext, step_id: str,
                      roles: Iterable[Optional[RoleUsage]]) -> Optional[list[dict]]:
    """The entries of ``roles`` that ran turns, written to ``metadata["step_usage"][step_id]`` and returned; None, and
    nothing written, when none did."""
    entries = [role.entry() for role in roles if role is not None and role.turns]
    if not entries:
        return None
    context.metadata.setdefault(STEP_USAGE_KEY, {})[step_id] = entries
    return entries
