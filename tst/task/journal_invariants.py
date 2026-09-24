"""Invariants the step journal must hold after a run, shared by the integration
tests so a change to the mechanism is one edit rather than one per call site."""

from __future__ import annotations

import json
from typing import Iterable

from agent_env.store.document_store import Filter
from agent_env.task.step_journal import commit_ordered, replay_context
from agent_env.task.store import (
    TASK_INSTANCES_COLLECTION,
    TASK_STEP_JOURNAL_COLLECTION,
    get_task_instance_store,
)
from agent_env.task_step.context import TaskStepContext

SEED = "__seed__"


def instance_doc(instance_id: str) -> dict | None:
    """The raw task-instance document from whichever backend is configured."""
    store = get_task_instance_store()
    return store._doc_store.find_one(TASK_INSTANCES_COLLECTION, Filter.of(instance_id=instance_id))


def all_journal_rows(instance_id: str) -> list[dict]:
    """Every journal row, including ones a re-register retired (``journal_entries_sync`` hides those)."""
    store = get_task_instance_store()
    return store._doc_store.query(TASK_STEP_JOURNAL_COLLECTION, Filter.of(instance_id=instance_id))


def _canonical(value, sort_lists: bool):
    if isinstance(value, dict):
        return {k: _canonical(v, sort_lists) for k, v in sorted(value.items())}
    if isinstance(value, list):
        items = [_canonical(v, sort_lists) for v in value]
        if sort_lists:
            items.sort(key=lambda v: json.dumps(v, sort_keys=True, default=str))
        return items
    return value


def _comparable(value, sort_lists: bool):
    """Both sides through one serializer, so a tuple or a datetime can't fake a difference."""
    return _canonical(json.loads(json.dumps(value, default=str)), sort_lists)


def assert_journal_replays(
    instance_id: str,
    *,
    expect_steps: Iterable[str] | None = None,
    live_context: TaskStepContext | None = None,
    seeded: bool = True,
    exact_list_order: bool = False,
) -> dict:
    """Assert the journal replays to the stored context; return the instance doc.

    ``live_context`` adds an independent oracle: nothing writes ``context`` wholesale on the
    success path, so only this catches a diff that under-reports a real write."""
    store = get_task_instance_store()
    doc = instance_doc(instance_id)
    assert doc is not None, f"no task instance {instance_id!r}"

    entries = store.journal_entries_sync(instance_id)
    step_ids = [e["step_id"] for e in entries]
    assert len(set(step_ids)) == len(step_ids), f"duplicate journal entries: {step_ids}"
    seqs = [e["seq"] for e in entries]
    assert len(set(seqs)) == len(seqs), f"journal seqs must be unique: {seqs}"
    missing = sorted(c["step_id"] for c in doc["completed_steps"]
                     if c.get("status") != "failure" and c["step_id"] not in step_ids)
    assert missing == [], f"steps completed with no journal entry: {missing}"
    if seeded:
        assert SEED in step_ids, f"the run's seed was never journaled: {sorted(step_ids)}"
    if expect_steps is not None:
        expected = {*expect_steps} | ({SEED} if seeded else set())
        assert set(step_ids) == expected, f"journal holds {sorted(step_ids)}, expected {sorted(expected)}"

    replayed = replay_context(commit_ordered(entries, doc["completed_steps"]), doc["rerecorded_steps"])
    assert replayed == doc["context"], "replaying the journal did not reproduce the stored context"

    if live_context is not None:
        sort_lists = not exact_list_order
        assert _comparable(doc["context"], sort_lists) == _comparable(live_context.to_safe_dict(), sort_lists), (
            "the stored context diverges from the context the steps actually built in memory"
        )
    return doc
