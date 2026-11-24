"""How the run summaries print a run's usage (``task.usage.summary``): one cost line per run and one closing line
naming where the costs came from."""

from __future__ import annotations

from agent_env.config.model import ModelRole
from agent_env.config.model_pricing import ReportedBy
from agent_env.task.usage.summary import Share, UsageSummary

_ROLE_NAMES = {role.value: role.record_key for role in ModelRole}
# The cost sources that aren't the price map, in the order they're named.
_REPORTED = {ReportedBy.PROVIDER: "the endpoint's bills", ReportedBy.HARNESS: "agents' own figures"}


def cost_line(summary: UsageSummary | None, label: str = "cost", attempts: bool = True) -> str | None:
    """``cost: $0.0532 (solver gpt-4.1-mini $0.0412, 115k in, 1.2k out; judge ...)``, each cost as ``_cost`` gives it
    and, with ``attempts``, the attempts named when there were more than one; None for no usage."""
    if summary is None:
        return None
    parts = [f"{summary.attempts} attempts"] if attempts and summary.attempts > 1 else []
    return f"{label}: {_cost(summary)} ({'; '.join([*parts, *map(_share, summary.shares)])})"


def sources_line(summary: UsageSummary | None) -> str | None:
    """Where ``summary``'s costs came from, as ``Costs priced from LiteLLM's price map, litellm@1.80.0.``; None when
    nothing was priced."""
    sources = summary.sources if summary is not None else ()
    if not sources:
        return None
    named = [f"LiteLLM's price map, {source}" for source in sources if source.startswith("litellm@")]
    named += [name for source, name in _REPORTED.items() if source in sources]
    named += [source for source in sources if not source.startswith("litellm@") and source not in _REPORTED]
    return f"Costs priced from {'; '.join(named)}."


def _share(share: Share) -> str:
    words = [f"{_ROLE_NAMES.get(share.role, share.role)} {share.model or 'an unnamed model'} {_cost(share)}"]
    if share.input_tokens is not None:
        words.append(f"{_count(share.input_tokens)} in")
    if share.output_tokens is not None:
        words.append(f"{_count(share.output_tokens)} out")
    if share.partial:
        words.append("partial")
    return ", ".join(words)


def _cost(of: UsageSummary | Share) -> str:
    """``of``'s cost with what it is: ``at least`` when some usage went unpriced or uncounted, ``at most`` when some
    was priced on an assumed rate, ``estimated`` when both, and ``unpriced`` when nothing was priced."""
    if of.cost_usd is None:
        return "unpriced"
    lower, upper = of.lower_bound, of.upper_bound
    word = "estimated " if lower and upper else "at least " if lower else "at most " if upper else ""
    return f"{word}{_dollars(of.cost_usd)}"


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
