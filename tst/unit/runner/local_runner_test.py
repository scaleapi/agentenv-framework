"""LocalRunner (in-process): submit runs Task.run() in this process, bounded by a worker
semaphore. Task execution is Task.run()'s job and covered elsewhere; what these tests pin
is the dispatch layer — a submit that returns before the work finishes, terminal-state
bookkeeping, bounded concurrency, real cancellation, and orphan reconciliation on start().
"""

import asyncio

import pytest

from agent_env.config import configure, reset_config, set_document_store
from agent_env.runner import store as run_store
from agent_env.runner.local_runner import LocalRunner
from agent_env.runner.runner import RunRecord, RunStatus
from agent_env.store.document_store.sqlite_document_store import LocalSqliteDocumentStore


@pytest.fixture
def docs(tmp_path):
    configure()
    store = LocalSqliteDocumentStore(str(tmp_path / "documents.db"))
    set_document_store(store)
    run_store.ensure_indexes()
    yield store
    reset_config()


class _FakeTask:
    """Stands in for a stored Task; records how Task.run() was invoked."""

    def __init__(self, on_run=None, failed_steps=None):
        self.on_run = on_run
        self.failed_steps = failed_steps
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        if self.on_run:
            await self.on_run(kwargs)
        ctx = kwargs.get("context")
        if self.failed_steps and ctx is not None:
            ctx.metadata["failed_steps"] = self.failed_steps
        return ctx


def _install_task(monkeypatch, task):
    from agent_env.task import Task
    monkeypatch.setattr(Task, "get", classmethod(lambda cls, tid, ver=None: task))


async def _drain(runner, run_id, timeout=5.0):
    """Wait for a run to reach a terminal state."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        rec = await runner.status(run_id)
        if rec and rec.status.is_terminal:
            return rec
        await asyncio.sleep(0.02)
    raise AssertionError(f"run {run_id} never reached a terminal state")


@pytest.mark.asyncio
async def test_submit_runs_the_task_and_completes(docs, monkeypatch):
    task = _FakeTask()
    _install_task(monkeypatch, task)

    runner = LocalRunner(workers=1)
    await runner.start()
    handle = await runner.submit("t1", 1, agent_model="claude-x")
    assert handle.run_id.startswith("local-")
    assert handle.instance_id                       # minted at submit, so a caller can link now
    try:
        rec = await _drain(runner, handle.run_id)
    finally:
        await runner.stop()

    assert rec.status is RunStatus.COMPLETED
    kw = task.calls[0]                              # the overrides + run identity reached Task.run()
    assert kw["agent_model"] == "claude-x"
    assert kw["instance_id"] == handle.instance_id
    assert kw["context"].metadata["workflow_id"] == handle.run_id
    assert kw["context"].metadata["runner"] == "local"


@pytest.mark.asyncio
async def test_failed_steps_mark_the_run_failed(docs, monkeypatch):
    # One well-formed entry and one malformed (non-dict) one: the malformed entry must
    # count as a failure, not crash _finish into a misreported outcome.
    _install_task(monkeypatch, _FakeTask(
        failed_steps=[{"step_id": "deploy_env", "error": "boom"}, "legacy-string-entry"],
    ))
    runner = LocalRunner(workers=1)
    await runner.start()
    handle = await runner.submit("t1", 1)
    try:
        rec = await _drain(runner, handle.run_id)
    finally:
        await runner.stop()
    # a step can fail without raising (fail_task_on_error=False); failed_steps is the
    # authoritative signal, whichever runner executed the task
    assert rec.status is RunStatus.FAILED
    assert "deploy_env" in rec.error and "AttributeError" not in rec.error


@pytest.mark.asyncio
async def test_retried_failed_steps_do_not_fail_the_run(docs, monkeypatch):
    # A failure the scheduler recovered via retry_config stays in failed_steps as audit,
    # flagged `retried`; it must not turn a recovered run into FAILED.
    _install_task(monkeypatch, _FakeTask(
        failed_steps=[{"step_id": "prompt", "error": "flaky", "retried": True}],
    ))
    runner = LocalRunner(workers=1)
    await runner.start()
    handle = await runner.submit("t1", 1)
    try:
        rec = await _drain(runner, handle.run_id)
    finally:
        await runner.stop()
    assert rec.status is RunStatus.COMPLETED


@pytest.mark.asyncio
async def test_raising_task_marks_the_run_failed(docs, monkeypatch):
    async def boom(_kw):
        raise RuntimeError("sandbox exploded")
    _install_task(monkeypatch, _FakeTask(on_run=boom))

    runner = LocalRunner(workers=1)
    await runner.start()
    handle = await runner.submit("t1", 1)
    try:
        rec = await _drain(runner, handle.run_id)
    finally:
        await runner.stop()
    assert rec.status is RunStatus.FAILED
    assert "sandbox exploded" in rec.error


@pytest.mark.asyncio
async def test_missing_task_fails_the_run_not_the_runner(docs, monkeypatch):
    from agent_env.task import Task
    monkeypatch.setattr(Task, "get", classmethod(lambda cls, tid, ver=None: None))

    runner = LocalRunner(workers=1)
    await runner.start()
    handle = await runner.submit("nope", 1)
    try:
        rec = await _drain(runner, handle.run_id)
        second = await runner.submit("nope", 1)     # the runner must still be usable
        rec2 = await _drain(runner, second.run_id)
    finally:
        await runner.stop()
    assert rec.status is RunStatus.FAILED and "not found" in rec.error
    assert rec2.status is RunStatus.FAILED


@pytest.mark.asyncio
async def test_workers_bound_concurrency(docs, monkeypatch):
    """workers=N means at most N runs execute Task.run() at once — the semaphore is the queue."""
    gauge = {"cur": 0, "max": 0}

    async def track(_kw):
        gauge["cur"] += 1
        gauge["max"] = max(gauge["max"], gauge["cur"])
        await asyncio.sleep(0.15)
        gauge["cur"] -= 1

    _install_task(monkeypatch, _FakeTask(on_run=track))
    runner = LocalRunner(workers=2)
    await runner.start()
    handles = [await runner.submit("t1", 1) for _ in range(4)]
    try:
        for h in handles:
            await _drain(runner, h.run_id)
    finally:
        await runner.stop()
    assert gauge["max"] == 2, f"ran {gauge['max']} at once, expected 2"


@pytest.mark.asyncio
async def test_cancel_preempts_a_running_task(docs, monkeypatch):
    """Cancel actually stops the work — Task.run() is preempted at its current await, so a
    later step never runs (not just a relabel to CANCELED)."""
    started = asyncio.Event()
    ran_to_end = []

    async def slow(_kw):
        started.set()
        await asyncio.sleep(2)
        ran_to_end.append(1)                        # must NOT happen

    _install_task(monkeypatch, _FakeTask(on_run=slow))
    runner = LocalRunner(workers=1)
    await runner.start()
    handle = await runner.submit("t1", 1)
    await asyncio.wait_for(started.wait(), 2)
    assert await runner.cancel(handle.run_id) is True
    rec = await _drain(runner, handle.run_id)
    await runner.stop()
    assert rec.status is RunStatus.CANCELED
    assert ran_to_end == [], "cancel did not preempt the in-flight task"


@pytest.mark.asyncio
async def test_cancel_a_queued_run_before_it_starts(docs, monkeypatch):
    started = asyncio.Event()

    async def block(_kw):
        started.set()
        await asyncio.sleep(2)

    task = _FakeTask(on_run=block)
    _install_task(monkeypatch, task)
    runner = LocalRunner(workers=1)
    await runner.start()
    blocker = await runner.submit("t1", 1)
    await asyncio.wait_for(started.wait(), 2)        # blocker holds the one slot
    waiting = await runner.submit("t1", 1)           # no free slot → QUEUED

    await asyncio.sleep(0.05)
    assert (await runner.status(waiting.run_id)).status is RunStatus.QUEUED
    assert await runner.cancel(waiting.run_id) is True
    rec = await _drain(runner, waiting.run_id)            # _run records CANCELED once its wait is interrupted
    assert rec.status is RunStatus.CANCELED
    assert await runner.cancel(waiting.run_id) is False   # already terminal
    await runner.stop()
    assert len(task.calls) == 1, "the queued run must never have executed"


@pytest.mark.asyncio
async def test_cancel_racing_completion_reports_completed(docs, monkeypatch):
    """A cancel that lands as the run finishes must report COMPLETED, not CANCELED — the work
    (and its side effects) are done, so `_run` honors the completed Task.run() over the cancel.

    The window is exact: Task.run() has finished but `_run` hasn't persisted COMPLETED. We hit
    it deterministically by scheduling the cancel from inside the work itself, so it fires after
    run_task is done (can no longer be cancelled) but before `_run` resumes."""
    ran = []
    holder = {}

    async def work(_kw):
        ran.append(1)
        asyncio.get_running_loop().call_soon(holder["cancel"])

    _install_task(monkeypatch, _FakeTask(on_run=work))
    runner = LocalRunner(workers=1)
    await runner.start()
    handle = await runner.submit("t1", 1)
    holder["cancel"] = lambda: runner._inflight[handle.run_id].cancel()
    rec = await _drain(runner, handle.run_id)
    await runner.stop()
    assert rec.status is RunStatus.COMPLETED
    assert ran == [1]                                     # the work actually finished


@pytest.mark.asyncio
async def test_start_fails_a_run_left_running_by_a_previous_process(docs):
    # A run the prior process left mid-flight: a fresh process can't be executing it.
    run_store.insert_run(RunRecord(run_id="orphan", runner="local", task_id="t",
                                   task_version=1, instance_id="i", status=RunStatus.RUNNING))
    runner = LocalRunner(workers=1)
    await runner.start()
    await runner.stop()
    rec = run_store.get_run("orphan")
    assert rec.status is RunStatus.FAILED
    assert "restart" in rec.error


@pytest.mark.asyncio
async def test_start_reconciles_an_orphan_behind_many_terminal_runs(docs):
    # Reconcile is by status, not a window of the newest records: an orphan older than a
    # large backlog of finished runs must still be failed.
    run_store.insert_run(RunRecord(run_id="old-orphan", runner="local", task_id="t",
                                   task_version=1, instance_id="i", status=RunStatus.RUNNING))
    for i in range(600):
        run_store.insert_run(RunRecord(run_id=f"done-{i}", runner="local", task_id="t",
                                       task_version=1, instance_id="i", status=RunStatus.COMPLETED))
    runner = LocalRunner(workers=1)
    await runner.start()
    await runner.stop()
    assert run_store.get_run("old-orphan").status is RunStatus.FAILED


async def _settled(runner, run_id, timeout=5.0):
    """Wait for a run's task to finish, teardown included: its terminal state is recorded first."""
    deadline = asyncio.get_event_loop().time() + timeout
    while run_id in runner._inflight:
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError(f"run {run_id} never finished")
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["completes", "raises", "is cancelled"])
async def test_what_a_run_deployed_is_torn_down_however_it_ends(docs, monkeypatch, ending):
    from agent_env.runner import local_runner
    from agent_env.task.teardown import TeardownReport

    torn_down = []

    async def teardown_run(context):
        torn_down.append(context.metadata["workflow_id"])
        return TeardownReport()

    monkeypatch.setattr(local_runner, "teardown_run", teardown_run)
    started = asyncio.Event()

    async def on_run(_):
        started.set()
        if ending == "raises":
            raise RuntimeError("boom")
        if ending == "is cancelled":
            await asyncio.sleep(30)

    _install_task(monkeypatch, _FakeTask(on_run=on_run))
    runner = LocalRunner(workers=1)
    await runner.start()
    handle = await runner.submit("t1", 1)
    try:
        if ending == "is cancelled":
            await asyncio.wait_for(started.wait(), 5)
            await runner.cancel(handle.run_id)
        await _drain(runner, handle.run_id)
        await _settled(runner, handle.run_id)
    finally:
        await runner.stop()

    assert torn_down == [handle.run_id]


@pytest.mark.asyncio
async def test_stop_lets_a_teardown_under_way_finish(docs, monkeypatch):
    from agent_env.runner import local_runner
    from agent_env.task.teardown import TeardownReport

    tearing, finished = asyncio.Event(), []

    async def teardown_run(context):
        tearing.set()
        await asyncio.sleep(0.3)
        finished.append(context.metadata["workflow_id"])
        return TeardownReport()

    monkeypatch.setattr(local_runner, "teardown_run", teardown_run)
    _install_task(monkeypatch, _FakeTask())
    runner = LocalRunner(workers=1)
    await runner.start()
    handle = await runner.submit("t1", 1)
    await asyncio.wait_for(tearing.wait(), 5)

    await runner.stop()

    assert finished == [handle.run_id]


@pytest.mark.asyncio
async def test_a_teardown_that_never_ends_is_given_up_on_so_stop_ends(docs, monkeypatch):
    from agent_env.runner import local_runner

    tearing = asyncio.Event()

    async def teardown_run(context):
        tearing.set()
        await asyncio.sleep(3600)

    monkeypatch.setattr(local_runner, "teardown_run", teardown_run)
    monkeypatch.setattr(LocalRunner, "TEARDOWN_WAIT_SECONDS", 0.2)
    _install_task(monkeypatch, _FakeTask())
    runner = LocalRunner(workers=1)
    await runner.start()
    await runner.submit("t1", 1)
    await asyncio.wait_for(tearing.wait(), 5)

    await asyncio.wait_for(runner.stop(), 5)

    assert not runner._inflight



@pytest.mark.asyncio
async def test_a_teardown_gets_its_whole_wait_however_long_the_run_took_to_stop(docs, monkeypatch):
    from agent_env.runner import local_runner
    from agent_env.task.teardown import TeardownReport

    finished = []

    async def teardown_run(context):
        await asyncio.sleep(0.3)
        finished.append(context.metadata["workflow_id"])
        return TeardownReport()

    async def slow_to_stop(_):
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await asyncio.sleep(0.3)  # a step that takes a while to unwind
            raise

    monkeypatch.setattr(local_runner, "teardown_run", teardown_run)
    monkeypatch.setattr(LocalRunner, "TEARDOWN_WAIT_SECONDS", 0.5)
    _install_task(monkeypatch, _FakeTask(on_run=slow_to_stop))
    runner = LocalRunner(workers=1)
    await runner.start()
    handle = await runner.submit("t1", 1)
    await asyncio.sleep(0.1)

    await asyncio.wait_for(runner.stop(), 5)

    assert finished == [handle.run_id]
