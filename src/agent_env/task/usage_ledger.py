"""A run's usage ledger, on its task instance beside the context: every attempt's usage entries, kept by attempt, step
and role, and their total. A retry's rollback, a write a later attempt refused, a failure, a cancel and a resume all
replace or drop the context's usage, never the ledger's, so the ledger is what a run spent. The store's writes merge
into it; nothing else changes it."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any, Optional

LEDGER_KEY = "usage"
TOTAL_KEY = "usage_total"
_COUNTS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")


def entries_of(step_usage: Optional[Mapping], step_ids: Optional[Collection[str]] = None) -> list[dict]:
    """The ledger entries of a context's ``metadata.step_usage``: each step's usage entries with their ``step_id``,
    only those of ``step_ids`` when given."""
    if not isinstance(step_usage, Mapping):
        return []
    return [{**entry, "step_id": step_id} for step_id, entries in step_usage.items()
            if step_ids is None or step_id in step_ids
            for entry in entries or [] if isinstance(entry, dict)]


def _key(entry: dict) -> tuple:
    return entry.get("attempt"), entry.get("step_id"), entry.get("role")


def merged(ledger: list[dict], entries: list[dict]) -> list[dict]:
    """``ledger`` with ``entries`` in it. An entry replaces the one of its attempt, step and role: a step's entry is its
    sum so far, so a write sent again, or a later one of the same attempt, never counts twice. A retry or a resume
    runs under a new attempt epoch, so its entries are new ones."""
    out = list(ledger)
    at = {_key(entry): i for i, entry in enumerate(out)}
    for entry in entries:
        key = _key(entry)
        if key in at:
            out[at[key]] = entry
        else:
            at[key] = len(out)
            out.append(entry)
    return out


def total(ledger: list[dict]) -> dict[str, Any]:
    """The ledger summed, as a budget reads it: the cost and the token counts, each None when no entry has one, never
    0; ``partial`` when an entry's counts are a lower bound; ``unpriced_entries`` the entries with no cost, which
    make the cost a lower bound too."""
    def sum_of(name: str) -> Any:
        values = [v for e in ledger if isinstance(v := e.get(name), (int, float)) and not isinstance(v, bool)]
        return sum(values) if values else None

    return {"entries": len(ledger), "cost_usd": sum_of("cost_usd"), **{name: sum_of(name) for name in _COUNTS},
            "partial": any(bool(e.get("partial")) for e in ledger),
            "unpriced_entries": sum(1 for e in ledger if e.get("cost_usd") is None)}


def ledger_fields(doc: dict, entries: list[dict], *, attempt: Optional[int] = None,
                  gathered: bool = False) -> dict[str, Any]:
    """The instance fields that merge ``entries`` into ``doc``'s ledger, to set in the same write; none when there are
    no entries. ``attempt`` is the epoch the write was made under: a completion's, the epoch its step was dispatched
    under; a failure's, cancel's, undo's or teardown's, the attempt it ends. An entry with no attempt, from a runner
    that doesn't set the attempt epoch, takes it (the instance's current epoch when the write has none), so two
    attempts of a step never share a key. A ``gathered`` write, one that takes a whole context's entries, leaves out
    such an entry when the ledger already holds it, its counts equal: an earlier write recorded it under its own
    attempt."""
    if not entries:
        return {}
    ledger = doc.get(LEDGER_KEY) or []
    epoch = attempt if attempt is not None else doc.get("attempt_epoch", 0)
    placed = [e for e in (_placed(e, ledger, epoch, gathered) for e in entries) if e is not None]
    if not placed:
        return {}
    ledger = merged(ledger, placed)
    return {LEDGER_KEY: ledger, TOTAL_KEY: total(ledger)}


def _placed(entry: dict, ledger: list[dict], epoch: int, gathered: bool) -> Optional[dict]:
    if entry.get("attempt") is not None:
        return entry
    if gathered and any(_without_attempt(e) == _without_attempt(entry) for e in ledger):
        return None
    return {**entry, "attempt": epoch}


def _without_attempt(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if k != "attempt"}
