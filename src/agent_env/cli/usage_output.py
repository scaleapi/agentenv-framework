"""How the run summaries print a run's usage (``task.usage.summary``): one cost line per run and one closing line
naming where the costs came from."""

from __future__ import annotations

from collections.abc import Iterable

from agent_env.task.usage.summary import Share, UsageSummary

_ROLE_NAMES = {"user_simulator": "user simulator"}
# The cost sources that aren't the price map, as ``cost_source`` names them, in the order they're named.
_REPORTED = (("provider", "the endpoint's bills"), ("harness", "agents' own figures"))


def cost_line(summary: UsageSummary | None, label: str = "cost", attempts: bool = True) -> str | None:
    """``cost: $0.0532 (solver gpt-4.1-mini $0.0412, 115k in, 1.2k out; judge ...)``, ``at least`` when the cost is a
    lower bound and, with ``attempts``, the attempts named when there were more than one; None for no usage."""
    if summary is None:
        return None
    total = "unpriced" if summary.cost_usd is None else _dollars(summary.cost_usd)
    if summary.lower_bound and summary.cost_usd is not None:
        total = f"at least {total}"
    parts = [f"{summary.attempts} attempts"] if attempts and summary.attempts > 1 else []
    return f"{label}: {total} ({'; '.join([*parts, *map(_share, summary.shares)])})"


def sources_line(summaries: Iterable[UsageSummary | None]) -> str | None:
    """Where the summaries' costs came from, as ``Costs priced from LiteLLM's price map, litellm@1.80.0.``; None when
    none was priced."""
    sources = {source for summary in summaries if summary is not None for source in summary.sources}
    if not sources:
        return None
    named = [f"LiteLLM's price map, {source}" for source in sorted(sources) if source.startswith("litellm@")]
    named += [name for source, name in _REPORTED if source in sources]
    named += sorted(source for source in sources if not source.startswith("litellm@") and source not in dict(_REPORTED))
    return f"Costs priced from {'; '.join(named)}."


def _share(share: Share) -> str:
    cost = "unpriced" if share.cost_usd is None else _dollars(share.cost_usd)
    if share.unpriced and share.cost_usd is not None:
        cost = f"at least {cost}"
    words = [f"{_ROLE_NAMES.get(share.role, share.role)} {share.model or 'an unnamed model'} {cost}"]
    if share.input_tokens is not None:
        words.append(f"{_count(share.input_tokens)} in")
    if share.output_tokens is not None:
        words.append(f"{_count(share.output_tokens)} out")
    if share.partial:
        words.append("partial")
    return ", ".join(words)


def _dollars(usd: float) -> str:
    return "$0" if usd == 0 else f"${usd:.4f}" if usd >= 0.0001 else "under $0.0001"


def _count(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 10_000:
        return f"{n / 1000:.1f}k"
    if n < 1_000_000:
        return f"{round(n / 1000)}k"
    return f"{n / 1_000_000:.1f}M"
