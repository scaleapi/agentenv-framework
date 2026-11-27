"""How the run summaries print a run's usage (``task.usage.summary``): one cost line per run and one closing line
naming where the costs came from."""

from __future__ import annotations

from agent_env.config.model import ModelRole
from agent_env.config.model_pricing import ReportedBy
from agent_env.task.usage.summary import Bound, Share, UsageSummary

_ROLE_NAMES = {role.value: role.record_key for role in ModelRole}
# The cost sources that aren't the price map, in the order they're named.
_REPORTED = {ReportedBy.PROVIDER: "the endpoint's bills", ReportedBy.HARNESS: "agents' own figures"}
_DOLLAR_PLACES = 4
_SMALLEST_DOLLARS = 0.0001  # the least a cost shows at _DOLLAR_PLACES; a smaller one reads "under" it
_THOUSAND, _MILLION = 1_000, 1_000_000
_TENTHS_OF_THOUSANDS_BELOW = 10 * _THOUSAND  # a count under it shows one decimal: 1.2k, not 1k


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
    """``of``'s cost after the word its ``bound`` gives, none when exact; ``unpriced`` when nothing was priced."""
    if of.bound is Bound.UNPRICED:
        return "unpriced"
    word = "" if of.bound is Bound.EXACT else f"{of.bound.value.replace('_', ' ')} "
    return f"{word}{_dollars(of.cost_usd)}"


def _dollars(usd: float) -> str:
    if usd == 0:
        return "$0"
    if usd < _SMALLEST_DOLLARS:
        return f"under ${_SMALLEST_DOLLARS:.{_DOLLAR_PLACES}f}"
    return f"${usd:.{_DOLLAR_PLACES}f}"


def _count(n: int) -> str:
    if n < _THOUSAND:
        return str(n)
    if n < _TENTHS_OF_THOUSANDS_BELOW:
        return f"{n / _THOUSAND:.1f}k"
    if n < _MILLION:
        return f"{round(n / _THOUSAND)}k"
    return f"{n / _MILLION:.1f}M"
