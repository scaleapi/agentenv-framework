"""A running step's usage reaches its run's ledger while the step runs: a budget reading the ledger sees a long step's
turns before the step ends, a worker that dies mid-step keeps them, and the step's completion replaces them with its
final entry. Real LocalSqliteDocumentStore; the steps record usage as a prompt step does."""

from __future__ import annotations

import asyncio

import pytest
from agentenv_protocol.a2a_agent import Usage

from agent_env.a2a_agent.protocol import TerminalResponse
from agent_env.config import set_document_store
from agent_env.config.model import ModelRole
from agent_env.store import LocalSqliteDocumentStore
from agent_env.task.usage import running
from agent_env.task.store import TaskInstanceStore, get_task_instance_store, set_task_instance_store
from agent_env.task.task import Task
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.usage import STEP_USAGE_KEY, StepUsage


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(running, "INTERVAL_S", 0.02)
    set_document_store(LocalSqliteDocumentStore(str(tmp_path / "instances.db")))
    s = TaskInstanceStore()
    set_task_instance_store(s)
    try:
        yield s
    finally:
        set_task_instance_store(None)


def _ledger(instance_id):
    return [(e["attempt"], e["step_id"], e["turns"], e["input_tokens"], bool(e.get("running")))
            for e in get_task_instance_store().get(instance_id).usage]


def _partial(instance_id):
    return get_task_instance_store().get(instance_id).usage_total["partial"]


async def _until(instance_id, want):
    for _ in range(250):
        if _ledger(instance_id) == want:
            return want
        await asyncio.sleep(0.02)
    return _ledger(instance_id)


class _Turns(TaskStep):
    """Records a solver turn on gpt-4.1-mini of each of ``inputs`` input tokens, after each waiting until the ledger
    holds the step's turns so far, marked running, and noting whether the run's total said it was partial, then holds
    on ``hold`` when given."""

    type = "turns"

    def __init__(self, id, inputs, hold=None, **kwargs):
        super().__init__(id, 1, **kwargs)
        self.inputs, self.hold, self.seen, self.partial = inputs, hold, [], []

    async def execute(self, context):
        usage = StepUsage(context, self.id)
        usage.start(ModelRole.SOLVER, "gpt-4.1-mini", None)
        for turns, tokens in enumerate(self.inputs, start=1):
            usage.turn(ModelRole.SOLVER, TerminalResponse(response_text="", usage=Usage(
                input_tokens=tokens, output_tokens=10, input_includes_cache=True)))
            self.seen.append(await _until(context.instance_id, [(0, self.id, turns, sum(self.inputs[:turns]), True)]))
            self.partial.append(_partial(context.instance_id))
        if self.hold is not None:
            await self.hold.wait()
        return context


@pytest.mark.asyncio
async def test_a_long_steps_turns_reach_the_ledger_before_it_ends_and_its_completion_replaces_them(store):
    step = _Turns("long", [100, 200])
    context = await Task(id="t", version=1, steps=[step]).run(context=TaskStepContext())

    assert step.seen == [[(0, "long", 1, 100, True)], [(0, "long", 2, 300, True)]] and step.partial == [True, True]
    assert _ledger(context.instance_id) == [(0, "long", 2, 300, False)]
    assert get_task_instance_store().get(context.instance_id).usage_total["input_tokens"] == 300
    assert not _partial(context.instance_id)


@pytest.mark.asyncio
async def test_a_worker_that_dies_mid_step_keeps_the_turns_written_before(store, monkeypatch):
    """The run's end writes never come, as when its worker dies: the step's turn is in the ledger all the same."""
    for write in ("record_step_complete_sync", "record_task_failure_sync", "record_task_cancelled_sync",
                  "record_teardown_sync"):
        monkeypatch.setattr(store, write, lambda *args, **kwargs: None)
    context, step = TaskStepContext(), _Turns("dies", [100], hold=asyncio.Event())
    run = asyncio.create_task(Task(id="t", version=1, steps=[step]).run(context=context))
    while not step.seen:
        await asyncio.sleep(0.02)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert _ledger(context.instance_id) == [(0, "dies", 1, 100, True)] and _partial(context.instance_id)


@pytest.mark.asyncio
async def test_usage_is_written_only_when_it_changed_and_stop_waits_for_a_write_in_flight(monkeypatch):
    monkeypatch.setattr(running, "INTERVAL_S", 0.01)
    writes, landed = asyncio.Event(), []

    async def slow(instance_id, usage):
        writes.set()
        await asyncio.sleep(0.1)
        landed.append(usage)
        return True

    monkeypatch.setattr(running.store, "record_running_usage", slow)
    context = TaskStepContext(metadata={STEP_USAGE_KEY: {"s": [{"role": "solver", "attempt": 0, "turns": 1}]}})
    async with running.running_usage("instance", "s", context):
        await writes.wait()
    assert landed == [[{"role": "solver", "attempt": 0, "turns": 1, "step_id": "s"}]]

    async with running.running_usage("instance", "s", context):
        await asyncio.sleep(0.25)
    assert len(landed) == 2


@pytest.mark.asyncio
async def test_a_snapshot_that_lands_late_never_takes_the_ledger_back(store):
    """As a write a cancel left running in its thread, landing after the step's newer entry: a role's turns only grow,
    so the ledger keeps the entry with more of them."""
    context = await Task(id="t", version=1, steps=[_Turns("long", [100, 200])]).run(context=TaskStepContext())
    late = {"attempt": 0, "step_id": "long", "role": "solver", "turns": 1, "input_tokens": 100}

    store.record_running_usage_sync(context.instance_id, [late])
    assert _ledger(context.instance_id) == [(0, "long", 2, 300, False)]

    store.record_running_usage_sync(context.instance_id, [{**late, "turns": 3, "input_tokens": 450}])
    assert _ledger(context.instance_id) == [(0, "long", 3, 450, True)]


@pytest.mark.asyncio
async def test_a_write_that_failed_is_tried_again_though_nothing_changed(monkeypatch):
    monkeypatch.setattr(running, "INTERVAL_S", 0.01)
    tries = []

    async def flaky(instance_id, usage):
        tries.append(usage)
        return len(tries) > 1

    monkeypatch.setattr(running.store, "record_running_usage", flaky)
    context = TaskStepContext(metadata={STEP_USAGE_KEY: {"s": [{"role": "solver", "attempt": 0, "turns": 1}]}})
    async with running.running_usage("instance", "s", context):
        await asyncio.sleep(0.2)

    assert len(tries) == 2 and tries[0] == tries[1]
