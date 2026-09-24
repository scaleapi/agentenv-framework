"""Write-path tests for record_task_failure and append_step_attempt_failure.

The retry tests fake the doc store and make it fail a controllable number of times.
The ledger test drives a real LocalSqliteDocumentStore so it exercises actual filter +
update semantics (a mock would bypass those and mask the absent-array bug). No external
services. `_FAILURE_WRITE_BACKOFF_SECONDS` is zeroed so the retries don't sleep.
"""

from __future__ import annotations

import pytest

import agent_env.task.store as store_mod
from agent_env.store import LocalSqliteDocumentStore
from agent_env.task.store import (
    StepAttemptFailure,
    TaskInstanceStore,
    append_step_attempt_failure,
    record_task_failure,
    set_task_instance_store,
)
from agent_env.task_step.context import TaskStepContext
from agent_env.config import set_document_store


class _FlakyDocStore:
    """Raises on the first ``fail_times`` ``update`` calls, then succeeds."""
    def ensure_index(self, *a, **k):   # a DocumentStore has one; this double stands in for it
        pass


    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0

    def update(self, collection, filt, spec):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient mongo error")


def _store_with(doc_store) -> TaskInstanceStore:
    store = TaskInstanceStore()
    set_document_store(doc_store)
    return store


def _sqlite_store(tmp_path) -> TaskInstanceStore:
    """A store over a real LocalSqliteDocumentStore, so appends actually run the filter +
    update (a mock would bypass those and mask the absent-array bug)."""
    doc = LocalSqliteDocumentStore(str(tmp_path / "instances.db"))
    doc.ensure_index(store_mod.TASK_INSTANCES_COLLECTION, ["instance_id"], unique=True)
    return _store_with(doc)


async def _record(instance_id="inst-1"):
    await record_task_failure(
        instance_id=instance_id,
        failing_step_id="prompt_agent",
        context=TaskStepContext(),
        error=RuntimeError("the model refused this task"),
        completed_at_utc="2026-07-20 21:30 UTC",
    )


@pytest.mark.asyncio
async def test_record_task_failure_retries_until_write_succeeds(monkeypatch):
    monkeypatch.setattr(store_mod, "_FAILURE_WRITE_BACKOFF_SECONDS", 0)
    flaky = _FlakyDocStore(fail_times=2)
    set_task_instance_store(_store_with(flaky))
    try:
        await _record()
    finally:
        set_task_instance_store(None)
    assert flaky.calls == 3


@pytest.mark.asyncio
async def test_record_task_failure_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(store_mod, "_FAILURE_WRITE_BACKOFF_SECONDS", 0)
    always = _FlakyDocStore(fail_times=99)
    set_task_instance_store(_store_with(always))
    try:
        await _record()
    finally:
        set_task_instance_store(None)
    assert always.calls == store_mod._FAILURE_WRITE_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_append_step_attempt_failure_lands_on_absent_array_and_is_idempotent(tmp_path, monkeypatch):
    """First append lands on the absent array, distinct attempts accumulate, an identical
    re-write is deduped, and the terminal status/error are untouched. (A prior ``Ne`` +
    ``$exists`` guard silently dropped this first append.)"""
    monkeypatch.setattr(store_mod, "_FAILURE_WRITE_BACKOFF_SECONDS", 0)
    store = _sqlite_store(tmp_path)
    store.upsert_instance(instance_id="inst-1", task_id="t", task_version=1, total_steps=5)
    set_task_instance_store(store)
    try:
        # First append: `step_attempt_failures` does not exist yet on the fresh instance doc.
        await append_step_attempt_failure(
            "inst-1",
            StepAttemptFailure(
                attempt=1,
                step_type="prompt_agent",
                error_class="httpx.ReadTimeout",
                error_message="stream timed out at turn 389",
            ),
        )
        led = store.get("inst-1").step_attempt_failures
        assert len(led) == 1, "first append was dropped (absent-array guard regression)"
        assert led[0]["error_class"] == "httpx.ReadTimeout"
        assert led[0]["error_message"] == "stream timed out at turn 389"

        # A distinct attempt accumulates (append-only, ordered by insertion).
        await append_step_attempt_failure("inst-1", StepAttemptFailure(attempt=2, error_class="RuntimeError"))
        assert [e["attempt"] for e in store.get("inst-1").step_attempt_failures] == [1, 2]

        # An identical re-write (e.g. the best-effort retry wrapper re-firing) is deduped.
        await append_step_attempt_failure("inst-1", StepAttemptFailure(attempt=2, error_class="RuntimeError"))
        assert [e["attempt"] for e in store.get("inst-1").step_attempt_failures] == [1, 2]

        # The ledger never clobbers the terminal fields another writer owns.
        inst = store.get("inst-1")
        assert inst.status == "running"
        assert inst.error is None
    finally:
        set_task_instance_store(None)


@pytest.mark.asyncio
async def test_append_step_attempt_failure_retries_transient_write(monkeypatch):
    monkeypatch.setattr(store_mod, "_FAILURE_WRITE_BACKOFF_SECONDS", 0)
    flaky = _FlakyDocStore(fail_times=2)
    set_task_instance_store(_store_with(flaky))
    try:
        await append_step_attempt_failure("inst-1", StepAttemptFailure(attempt=2))
    finally:
        set_task_instance_store(None)
    assert flaky.calls == 3


@pytest.mark.asyncio
async def test_append_and_record_are_independent_on_the_same_instance(tmp_path, monkeypatch):
    """On the final attempt the worker calls both; they touch disjoint fields, so the terminal
    failure is recorded and the ledger entry survives regardless of order."""
    monkeypatch.setattr(store_mod, "_FAILURE_WRITE_BACKOFF_SECONDS", 0)
    store = _sqlite_store(tmp_path)
    store.upsert_instance(instance_id="inst-1", task_id="t", task_version=1, total_steps=5)
    set_task_instance_store(store)
    try:
        await append_step_attempt_failure(
            "inst-1", StepAttemptFailure(attempt=3, error_class="httpx.ReadTimeout")
        )
        await record_task_failure(
            instance_id="inst-1",
            failing_step_id="prompt_agent",
            context=TaskStepContext(),
            error=RuntimeError("A2A task failed (...): agent-config 404"),
            completed_at_utc="2026-07-24 08:46 UTC",
        )
        inst = store.get("inst-1")
        # terminal failure recorded (the must-not-lose write) ...
        assert inst.status == "failed"
        assert "agent-config 404" in (inst.error or "")
        assert any(s.step_id == "prompt_agent" for s in inst.completed_steps)
        # ... and the ledger append (a disjoint field) was not clobbered by it.
        assert [e["attempt"] for e in inst.step_attempt_failures] == [3]
        assert inst.step_attempt_failures[0]["error_class"] == "httpx.ReadTimeout"
    finally:
        set_task_instance_store(None)
