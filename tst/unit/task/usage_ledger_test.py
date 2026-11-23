"""A run's usage ledger through ``Task.run``: each step's usage reaches the instance as the step completes, and a
retried span, a straggler's completion held past the rollback, a fatal failure, a cancel and a resume all keep what
every attempt spent. Real LocalSqliteDocumentStore; the steps record usage as a prompt step does."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from agentenv_protocol.a2a_agent import Usage

from agent_env.a2a_agent.protocol import TerminalResponse
from agent_env.config import set_document_store
from agent_env.config.model import ModelRole
import agent_env.task.task as task_mod
from agent_env.store import LocalSqliteDocumentStore
from agent_env.task.store import SupersededAttemptError, TaskInstanceStore, get_task_instance_store, \
    set_task_instance_store
from agent_env.task.task import Task
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import RetryConfig, TaskStep
from agent_env.task_step.usage import StepUsage


@pytest.fixture
def store(tmp_path):
    set_document_store(LocalSqliteDocumentStore(str(tmp_path / "instances.db")))
    s = TaskInstanceStore()
    set_task_instance_store(s)
    try:
        yield s
    finally:
        set_task_instance_store(None)


class _Spends(TaskStep):
    """Records one solver turn of ``input_tokens`` on gpt-4.1-mini, then sets ``spent``, waits for ``hold`` and fails
    on the attempts ``fail_on`` names."""

    type = "spends"

    def __init__(self, id, *, input_tokens=100, fail_on=(), hold=None, **kwargs):
        super().__init__(id, 1, **kwargs)
        self.input_tokens, self.fail_on, self.hold, self.attempts = input_tokens, set(fail_on), hold, 0
        self.spent = asyncio.Event()

    async def execute(self, context):
        self.attempts += 1
        usage = StepUsage(context, self.id)
        usage.start(ModelRole.SOLVER, "gpt-4.1-mini", None)
        usage.turn(ModelRole.SOLVER, TerminalResponse(response_text="", usage=Usage(
            input_tokens=self.input_tokens, output_tokens=10, input_includes_cache=True)))
        self.spent.set()
        if self.hold is not None:
            await self.hold.wait()
        if self.attempts in self.fail_on:
            raise RuntimeError(f"{self.id} fails on attempt {self.attempts}")
        return context


class _ReadsTheLedger(TaskStep):
    type = "reads_the_ledger"

    async def execute(self, context):
        self.seen = get_task_instance_store().get(context.instance_id).usage_total
        return context


def _ledger(store, instance_id):
    return [(e["attempt"], e["step_id"], e["input_tokens"]) for e in store.get(instance_id).usage]


@pytest.mark.asyncio
async def test_the_first_steps_priced_usage_is_in_the_store_before_the_second_ends(store):
    reads = _ReadsTheLedger("reads", 1)
    await Task(id="t", version=1, steps=[_Spends("ask"), reads]).run(context=TaskStepContext())
    assert (reads.seen["entries"], reads.seen["input_tokens"]) == (1, 100)
    assert reads.seen["cost_usd"] > 0 and reads.seen["unpriced_entries"] == 0


@pytest.mark.asyncio
async def test_a_retried_span_keeps_the_rolled_back_attempts_usage(store):
    step = _Spends("ask", fail_on={1}, retry_config=RetryConfig(retry_from_step_id="ask"))
    context = await Task(id="t", version=1, steps=[step]).run(context=TaskStepContext())
    assert _ledger(store, context.instance_id) == [(0, "ask", 100), (1, "ask", 100)]
    assert store.get(context.instance_id).usage_total["input_tokens"] == 200
    assert [e["attempt"] for e in context.metadata["step_usage"]["ask"]] == [1]


@pytest.mark.asyncio
async def test_a_siblings_completion_held_past_the_rollback_lands_its_usage_not_its_context(store, monkeypatch):
    loop = asyncio.get_running_loop()
    deploy_writing = asyncio.Event()
    undo_landed, deploy_write_done = threading.Event(), threading.Event()
    real_complete, real_undo = store.record_step_complete_sync, store.undo_steps_sync
    held: list = []

    def complete_after_the_undo(instance_id, entry, *args):
        if entry["step_id"] != "deploy" or held:
            return real_complete(instance_id, entry, *args)
        held.append(entry)
        loop.call_soon_threadsafe(deploy_writing.set)
        undo_landed.wait(10)
        try:
            return real_complete(instance_id, entry, *args)
        except SupersededAttemptError as refused:
            held.append(refused)
            raise
        finally:
            deploy_write_done.set()

    def undo_then_release(*args, **kwargs):
        try:
            return real_undo(*args, **kwargs)
        finally:
            undo_landed.set()

    monkeypatch.setattr(store, "record_step_complete_sync", complete_after_the_undo)
    monkeypatch.setattr(store, "undo_steps_sync", undo_then_release)
    flaky = _Spends("flaky", input_tokens=7, fail_on={1}, hold=deploy_writing, depends_on=["a"],
                    retry_config=RetryConfig(retry_from_step_id="flaky"))
    steps = [_Spends("a", input_tokens=5), flaky, _Spends("deploy", input_tokens=40, depends_on=["a"])]

    context = await Task(id="t", version=1, steps=steps).run(context=TaskStepContext())
    assert deploy_write_done.wait(10)

    assert isinstance(held[-1], SupersededAttemptError)
    assert sorted(_ledger(store, context.instance_id)) == [
        (0, "a", 5), (0, "deploy", 40), (0, "flaky", 7), (1, "deploy", 40), (1, "flaky", 7)]
    stored = store.get(context.instance_id).context["metadata"]["step_usage"]
    assert {step_id: [e["attempt"] for e in entries] for step_id, entries in stored.items()} == {
        "a": [0], "flaky": [1], "deploy": [1]}


@pytest.mark.asyncio
async def test_a_fatal_failure_keeps_the_usage_of_every_step_that_ran(store):
    context = TaskStepContext()
    with pytest.raises(RuntimeError, match="b fails"):
        await Task(id="t", version=1, steps=[_Spends("a", input_tokens=5), _Spends("b", input_tokens=9, fail_on={1})]
                   ).run(context=context)
    instance = store.get(context.instance_id)
    assert instance.status == "failed"
    assert _ledger(store, context.instance_id) == [(0, "a", 5), (0, "b", 9)]


@pytest.mark.asyncio
async def test_a_cancelled_run_keeps_the_usage_of_every_step_that_ran(store):
    context = TaskStepContext()
    waits = _Spends("b", input_tokens=9, hold=asyncio.Event())
    run = asyncio.create_task(Task(id="t", version=1, steps=[_Spends("a", input_tokens=5), waits]).run(context=context))
    await asyncio.wait_for(waits.spent.wait(), 10)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert store.get(context.instance_id).status == "cancelled"
    assert _ledger(store, context.instance_id) == [(0, "a", 5), (0, "b", 9)]


@pytest.mark.asyncio
async def test_a_resumed_run_keeps_the_usage_from_before_it(store):
    context = TaskStepContext()
    with pytest.raises(RuntimeError):
        await Task(id="t", version=1, steps=[_Spends("a", input_tokens=5), _Spends("b", input_tokens=9, fail_on={1})]
                   ).run(context=context)
    persisted = TaskStepContext.from_dict(store.get(context.instance_id).context)
    resumed = await Task(id="t", version=1, steps=[_Spends("a", input_tokens=5), _Spends("b", input_tokens=9)]).run(
        context=persisted, start_step=1, instance_id=context.instance_id)
    assert _ledger(store, resumed.instance_id) == [(0, "a", 5), (0, "b", 9), (1, "b", 9)]
    assert store.get(resumed.instance_id).usage_total["input_tokens"] == 23



@pytest.mark.asyncio
async def test_a_runner_that_sets_no_attempt_epoch_gets_its_runs_usage_without_retries(store, monkeypatch):
    """As the hub's worker runs steps today: the entries carry no attempt, and a run without retries is exact."""
    monkeypatch.setattr(task_mod, "ATTEMPT_EPOCH", SimpleNamespace(set=lambda epoch: None))
    context = await Task(id="t", version=1, steps=[_Spends("ask"), _Spends("next", input_tokens=5)]).run(
        context=TaskStepContext())
    assert _ledger(store, context.instance_id) == [(None, "ask", 100), (None, "next", 5)]
    assert store.get(context.instance_id).usage_total["input_tokens"] == 105
