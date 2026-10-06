"""The step journal's replay rules: how one step's diff is applied, in what order, and what
that reconstructs. Pure — the store owns the collection, the sequence, and the CAS."""

from __future__ import annotations

import dataclasses
from typing import Iterable

from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.context_ops import ContextUpdateOps

_SEED_STEP_ID = "__seed__"
# The scheduler's own record (retry audit), written by ``undo_steps_sync`` in the same CAS as
# the undo and replayed right after the seed, so a retry's bookkeeping survives a later undo.
_SCHEDULER_STEP_ID = "__scheduler__"
_RESERVED_STEP_IDS = frozenset({_SEED_STEP_ID, _SCHEDULER_STEP_ID})

def _get_path(doc: dict, dotted: str):
    """Read a dotted path (e.g. 'context.metadata.x') from a doc, or None."""
    cur: object = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _union(existing: list, items: list, *, move_to_end: bool = False, path: str = "") -> list:
    """Union by value (deployed envs by instance), order-stable. ``move_to_end`` is the re-record case: that step's
    completion moves to the end of the replay order, so its items must move with it."""
    key = _item_key(path)
    fresh, fresh_keys = [], []
    for item in items:
        item_key = key(item)
        if item_key not in fresh_keys:
            fresh.append(item)
            fresh_keys.append(item_key)
    existing_keys = [key(item) for item in existing]
    if move_to_end:
        return [item for item, item_key in zip(existing, existing_keys) if item_key not in fresh_keys] + fresh
    return list(existing) + [x for x, k in zip(fresh, fresh_keys) if k not in existing_keys]


def _set_path(doc: dict, dotted: str, value) -> None:
    cur = doc
    parts = dotted.split(".")
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _unset_path(doc: dict, dotted: str) -> None:
    cur = doc
    parts = dotted.split(".")
    for part in parts[:-1]:
        cur = cur.get(part)
        if not isinstance(cur, dict):
            return
    cur.pop(parts[-1], None)


def _apply_forward(doc: dict, ops: ContextUpdateOps, *, move_to_end: bool = False) -> None:
    """Apply one journaled diff in place with the completion write's semantics."""
    for path, value in ops.sets.items():
        _set_path(doc, path, value)
    for path in ops.unsets:
        _unset_path(doc, path)
    for path, items in ops.add_to_sets.items():
        _set_path(doc, path, _union(_get_path(doc, path) or [], items, move_to_end=move_to_end, path=path))


def commit_ordered(entries: list[dict], completed_steps: list[dict]) -> list[dict]:
    """Entries in commit order: seed first, then the scheduler's record, then by position in
    ``completed_steps``, which the CAS appends and so outranks ``seq``. A step with no
    completion never landed; drop it."""
    pos = {c.get("step_id"): i for i, c in enumerate(completed_steps)}
    pos[_SCHEDULER_STEP_ID] = -1
    pos[_SEED_STEP_ID] = -2
    kept = [e for e in entries if e["step_id"] in pos]
    return sorted(kept, key=lambda e: pos[e["step_id"]])


def replay_context(entries: list[dict], rerecorded_steps: Iterable[str]) -> dict:
    """The context ``entries`` produce over an empty one; order them with ``commit_ordered``.
    ``rerecorded_steps`` is the instance's own list, the only record of which steps re-recorded:
    a flag on the entry would be a pre-CAS read and can disagree with the write that landed."""
    flagged = set(rerecorded_steps)
    doc = {"context": dataclasses.asdict(TaskStepContext())}
    for e in entries:
        ops = ContextUpdateOps.from_journal_dict(e["ops"])
        _apply_forward(doc, ops, move_to_end=e["step_id"] in flagged)
    return doc["context"]


def _item_key(path: str):
    """How a context list's items are told apart: a deployed env by its instance, so two kernels' copies stay one entry."""
    if path == "context.deployed_envs":
        return lambda item: (item.get("instance_id") if isinstance(item, dict) else None) or item
    return lambda item: item
