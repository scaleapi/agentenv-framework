"""Pins for task-level retry_config (scheduler-driven rollback-and-re-dispatch through the
step journal).

When a step that declares ``retry_config`` fails and has retries left, ``Task._drive_dag``
drains in-flight work, rolls the failed span back through the store's ``undo_steps`` (the
surviving steps' journaled diffs are replayed over the run's seed in one CAS, together with
the scheduler's retry record and an ``attempt_epoch`` bump), rebuilds the live context from
the result, and re-dispatches the span as ordinary steps on a freshly redeployed world.
Sandboxes are not torn down here; the ``retry_resets`` marker records the ones the undo
stripped, for the executor's cleanup path. These tests pin that contract: everything the
span wrote goes (typed entries, custom metadata, an overwritten scalar), the span re-runs on
the guard-free deploy path, the journal still replays to the stored context afterwards,
``max_retries`` bounds the retries, a stale-epoch write is fenced without punching a hole in
the journal, and a refused, unpersisted or cancelled rollback leaves a true record.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

import agent_env.task.task as task_module
import agent_env.task.store as store_mod
from agent_env.env.env import DeployedEnv, DeployedGatewayEnv
from agent_env.store import LocalSqliteDocumentStore
from agent_env.store.document_store import Filter
from agent_env.task.step_journal import _SCHEDULER_STEP_ID, _SEED_STEP_ID, commit_ordered, replay_context
from agent_env.task.store import (
    TASK_INSTANCES_COLLECTION,
    TASK_STEP_JOURNAL_COLLECTION,
    TaskInstanceStore,
    set_task_instance_store,
)
from agent_env.task.task import Task, _rolled_back_sandboxes
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.context_ops import ContextUpdateOps, build_context_update_ops
from agent_env.task_step.task_step import RetryConfig
from agent_env.config import set_document_store
from tst.task import journal_invariants as journal


# --------------------------------------------------------------------------- #
# Fakes                                                                         #
# --------------------------------------------------------------------------- #
class _Dep:
    def __init__(self, task_step_id: str):
        self.task_step_id = task_step_id


class _FakeDeployEnv:
    """Deploys an env into the context; raises on a duplicate (the guard-free deploy path).
    A retry must roll the env out first, so a clean re-run never trips the guard. Nothing is
    stamped: the journal, not the entry, says who wrote it. Each run mints a new sandbox id
    so the marker can be checked for the rolled-back one."""

    type = "deploy_env"

    def __init__(self, id: str, depends_on=None):
        self.id = id
        self.depends_on = [_Dep(d) for d in depends_on] if depends_on is not None else None
        self.retry_config = None
        self.fail_task_on_error = True
        self.deploy_calls: list[str] = []

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        if any(e.env_id == self.id for e in context.deployed_envs):
            raise RuntimeError(f"Env '{self.id}' is already deployed")
        self.deploy_calls.append(self.id)
        context.deployed_envs.append(
            DeployedGatewayEnv(
                env_id=self.id, env_version=1, gateway_url="g", mcp_url="m", db_web_url=None,
                sandbox_id=f"sb-{self.id}-{len(self.deploy_calls)}", sandbox_type="fake",
            )
        )
        return context


class _FlakyPrompt:
    """Fails its first ``fail_times`` attempts, then succeeds. Each attempt (even a failed
    one) writes a prompt_response, a metadata dict keyed by itself that no allowlist knows
    about (the custom-step case), and a metadata list entry, before it may raise, so the
    rollback has real contributions to strip."""

    type = "prompt_agent"

    def __init__(
        self, id: str, retry_config=None, fail_times: int = 1, depends_on=None,
        fail_task_on_error: bool = True,
    ):
        self.id = id
        self.depends_on = [_Dep(d) for d in depends_on] if depends_on is not None else None
        self.retry_config = retry_config
        self.fail_task_on_error = fail_task_on_error
        self._fail_times = fail_times
        self.attempts = 0

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        self.attempts += 1
        context.prompt_responses.append(
            PromptResponse(prompt_id=self.id, response=f"attempt-{self.attempts}", step_id=self.id)
        )
        context.metadata.setdefault("my_custom_step_metadata", {})[self.id] = self.attempts
        context.metadata.setdefault("loaded_urls", []).append({"by": self.id, "attempt": self.attempts})
        if self.attempts <= self._fail_times:
            raise RuntimeError("flaky boom")
        return context


# --------------------------------------------------------------------------- #
# Store fixtures                                                                #
# --------------------------------------------------------------------------- #
def _sqlite_store(tmp_path) -> TaskInstanceStore:
    doc = LocalSqliteDocumentStore(str(tmp_path / "instances.db"))
    doc.ensure_index(TASK_INSTANCES_COLLECTION, ["instance_id"], unique=True)
    doc.ensure_index(TASK_STEP_JOURNAL_COLLECTION, ["instance_id", "step_id"], unique=True)
    doc.ensure_index(TASK_STEP_JOURNAL_COLLECTION, ["instance_id", "seq"])
    store = TaskInstanceStore()
    set_document_store(doc)
    return store


@pytest.fixture
def store(tmp_path):
    s = _sqlite_store(tmp_path)
    set_task_instance_store(s)
    try:
        yield s
    finally:
        set_task_instance_store(None)


def _doc(store, instance_id):
    return store._doc_store.find_one(TASK_INSTANCES_COLLECTION, Filter.of(instance_id=instance_id))


def _rows(store, instance_id):
    return store.journal_entries_sync(instance_id)


# --------------------------------------------------------------------------- #
# Scheduler-level: rollback + re-dispatch                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_retry_rolls_back_span_and_redispatches_on_fresh_world(store):
    deploy = _FakeDeployEnv("s0")
    flaky = _FlakyPrompt("s1", retry_config=RetryConfig(retry_from_step_id="s0"), fail_times=1)
    task = Task(id="t", version=1, steps=[deploy, flaky])

    ctx = TaskStepContext()
    result = await task.run(context=ctx)

    # The span re-ran on a clean world: deploy_env ran twice (guard-free), the
    # flaky step ran twice (fail then succeed).
    assert deploy.deploy_calls == ["s0", "s0"]
    assert flaky.attempts == 2

    # Final context reflects only the successful attempt's contributions.
    assert [e.sandbox_id for e in result.deployed_envs] == ["sb-s0-2"]
    assert [p.response for p in result.prompt_responses] == ["attempt-2"]

    # The reset was recorded for the run history; the failure is kept as audit, flagged.
    resets = result.metadata["retry_resets"]
    assert len(resets) == 1
    assert resets[0]["attempt"] == 2
    assert resets[0]["step_id"] == "s1"
    assert resets[0]["retry_from_step_id"] == "s0"
    assert resets[0]["replayed"] == ["s0", "s1"]
    # The undo stripped attempt 1's env; the marker names its sandbox so cleanup can find it.
    assert resets[0]["rolled_back_sandboxes"] == {
        "deployed_envs": [{"sandbox_id": "sb-s0-1", "sandbox_type": "fake", "sandbox_ids": {}}],
        "deployed_agents": [],
        "deployed_sandboxes": [],
    }
    assert [f.get("retried") for f in result.metadata["failed_steps"]] == [True]

    # Persisted instance: completed once each, run completed, epoch bumped, and the
    # journal (with the scheduler's record) still replays to the stored context, which
    # matches what the steps built in memory.
    inst = store.get(ctx.instance_id)
    assert inst.status == "completed"
    assert sorted(s.step_id for s in inst.completed_steps) == ["s0", "s1"]
    assert inst.attempt_epoch == 1
    doc = journal.assert_journal_replays(ctx.instance_id, live_context=ctx)
    assert len(doc["context"]["deployed_envs"]) == 1
    assert {e["step_id"] for e in _rows(store, ctx.instance_id)} == {_SEED_STEP_ID, _SCHEDULER_STEP_ID, "s0", "s1"}


@pytest.mark.asyncio
async def test_custom_metadata_and_lists_written_by_the_span_are_rolled_back(store):
    # No allowlist, no stamp: the journal knows what the span wrote. The dict keyed by the
    # step and the list entry from the failed attempt are gone; only attempt 2's remain.
    deploy = _FakeDeployEnv("s0")
    flaky = _FlakyPrompt("s1", retry_config=RetryConfig(retry_from_step_id="s0"), fail_times=1)
    result = await Task(id="t", version=1, steps=[deploy, flaky]).run(context=TaskStepContext())

    assert result.metadata["my_custom_step_metadata"] == {"s1": 2}
    assert result.metadata["loaded_urls"] == [{"by": "s1", "attempt": 2}]
    # failed_steps is an audit trail: the failed attempt is still recorded.
    assert [f["step_id"] for f in result.metadata["failed_steps"]] == ["s1"]


@pytest.mark.asyncio
async def test_overwritten_scalar_is_restored_by_the_rollback(store):
    # Stamps could only remove; replay restores. s0 sets a scalar, the flaky step
    # overwrites it and fails; on its re-run it sees s0's value again.
    class _SetModel:
        type = "install_agent"

        def __init__(self, id: str):
            self.id = id
            self.depends_on = None
            self.retry_config = None
            self.fail_task_on_error = True

        async def execute(self, context):
            context.default_agent_model = "base-model"
            return context

    class _OverwriteThenFail:
        type = "prompt_agent"

        def __init__(self, id: str):
            self.id = id
            self.depends_on = None
            self.retry_config = RetryConfig(retry_from_step_id=id)  # self-retry
            self.fail_task_on_error = True
            self.seen: list[str | None] = []

        async def execute(self, context):
            self.seen.append(context.default_agent_model)
            context.default_agent_model = f"changed-{len(self.seen)}"
            if len(self.seen) == 1:
                raise RuntimeError("boom")
            return context

    s0, s1 = _SetModel("s0"), _OverwriteThenFail("s1")
    result = await Task(id="t", version=1, steps=[s0, s1]).run(context=TaskStepContext())
    assert s1.seen == ["base-model", "base-model"]  # attempt 2 saw the restored value
    assert result.default_agent_model == "changed-2"


@pytest.mark.asyncio
async def test_self_retry_reruns_only_the_failed_step(store):
    deploy = _FakeDeployEnv("s0")
    # retry_from = the step itself -> span is just {s1}, deploy is not rolled back.
    flaky = _FlakyPrompt("s1", retry_config=RetryConfig(retry_from_step_id="s1"), fail_times=1)
    task = Task(id="t", version=1, steps=[deploy, flaky])

    ctx = TaskStepContext()
    result = await task.run(context=ctx)

    assert deploy.deploy_calls == ["s0"]  # deploy NOT re-run
    assert flaky.attempts == 2
    assert result.metadata["retry_resets"][0]["replayed"] == ["s1"]
    assert result.metadata["retry_resets"][0]["rolled_back_sandboxes"]["deployed_envs"] == []
    assert store.get(ctx.instance_id).status == "completed"


@pytest.mark.asyncio
async def test_max_retries_bounds_the_retries(store):
    deploy = _FakeDeployEnv("s0")
    # Fails more times than allowed -> terminal failure.
    flaky = _FlakyPrompt(
        "s1", retry_config=RetryConfig(retry_from_step_id="s0", max_retries=1), fail_times=5
    )
    task = Task(id="t", version=1, steps=[deploy, flaky])

    ctx = TaskStepContext()
    with pytest.raises(RuntimeError, match="flaky boom"):
        await task.run(context=ctx)

    # One retry allowed: two attempts total.
    assert flaky.attempts == 2
    inst = store.get(ctx.instance_id)
    assert inst.status == "failed"


@pytest.mark.asyncio
async def test_retry_aborts_and_fails_when_the_undo_cannot_persist(store, monkeypatch):
    # The store's undo fails (None). The retry must abort and the task fail safely, not
    # re-dispatch onto a stored record that still describes the failed attempt.
    async def _fail_undo(*a, **k):
        return None

    monkeypatch.setattr(store_mod, "undo_steps", _fail_undo)

    deploy = _FakeDeployEnv("s0")
    flaky = _FlakyPrompt("s1", retry_config=RetryConfig(retry_from_step_id="s0"), fail_times=1)
    task = Task(id="t", version=1, steps=[deploy, flaky])

    ctx = TaskStepContext()
    with pytest.raises(RuntimeError, match="flaky boom"):
        await task.run(context=ctx)

    # No re-dispatch: the flaky step ran once, deploy was not re-run.
    assert flaky.attempts == 1
    assert deploy.deploy_calls == ["s0"]
    # The live context was never touched, so the failure record keeps s0's env evidence.
    inst = store.get(ctx.instance_id)
    assert inst.status == "failed"
    assert [e["env_id"] for e in inst.context["deployed_envs"]] == ["s0"]
    # No retry happened, so the record must not claim one: the failure is not flagged
    # `retried` and no retry_resets marker exists, live or persisted.
    for md in (ctx.metadata, inst.context["metadata"]):
        assert [f.get("retried") for f in md["failed_steps"]] == [None]
        assert "retry_resets" not in md


@pytest.mark.asyncio
async def test_retry_is_refused_on_a_run_the_journal_cannot_rebuild(store, caplog):
    # A run whose seed was never journaled (a pre-journal instance) cannot be rebuilt by
    # replay; the store refuses and the scheduler fails with the ORIGINAL error rather than
    # re-dispatching on a world it cannot clean.
    class _DropSeed:
        type = "load_artifact"

        def __init__(self, id: str):
            self.id = id
            self.depends_on = None
            self.retry_config = None
            self.fail_task_on_error = True

        async def execute(self, context):
            store._doc_store.delete(
                TASK_STEP_JOURNAL_COLLECTION,
                Filter.of(instance_id=context.instance_id, step_id=_SEED_STEP_ID),
            )
            return context

    deploy = _FakeDeployEnv("s0")
    flaky = _FlakyPrompt("s2", retry_config=RetryConfig(retry_from_step_id="s0"), fail_times=1)
    ctx = TaskStepContext()
    with caplog.at_level(logging.ERROR, logger="agent_env.task.task"):
        with pytest.raises(RuntimeError, match="flaky boom"):
            await Task(id="t", version=1, steps=[deploy, _DropSeed("s1"), flaky]).run(context=ctx)

    assert flaky.attempts == 1 and deploy.deploy_calls == ["s0"]
    assert any("no journal seed" in m for m in caplog.messages)
    assert [e.env_id for e in ctx.deployed_envs] == ["s0"]  # live context intact
    assert store.get(ctx.instance_id).status == "failed"


@pytest.mark.asyncio
async def test_retry_is_refused_without_a_task_instance_record(store, monkeypatch, caplog):
    # No instance (the store could not register one): there is nothing to undo through, so
    # the failure is terminal with the original error instead of a partial in-memory retry.
    monkeypatch.setattr(store_mod, "register_task_instance", lambda *a, **k: None)
    deploy = _FakeDeployEnv("s0")
    flaky = _FlakyPrompt("s1", retry_config=RetryConfig(retry_from_step_id="s0"), fail_times=1)
    with caplog.at_level(logging.ERROR, logger="agent_env.task.task"):
        with pytest.raises(RuntimeError, match="flaky boom"):
            await Task(id="t", version=1, steps=[deploy, flaky]).run(context=TaskStepContext())
    assert flaky.attempts == 1
    assert any("no task-instance record" in m for m in caplog.messages)


@pytest.mark.asyncio
async def test_a_survivor_that_ran_concurrently_with_the_span_blocks_the_retry(store, caplog):
    # A and B are independent siblings. A completes while B is mid-execute, so A's journaled
    # diff was taken against a context that may already hold B's partial writes; replaying A
    # after undoing B could rebuild B's world dirty. Until per-step isolation makes diffs
    # exact, the scheduler refuses the retry and fails with the original error.
    b_started = asyncio.Event()
    a_committed = asyncio.Event()

    class _A:
        type = "prompt_agent"

        def __init__(self):
            self.id = "A"
            self.depends_on = [_Dep("s0")]
            self.retry_config = None
            self.fail_task_on_error = True

        async def execute(self, context):
            await b_started.wait()  # B is now executing; A's post snapshot may see B's writes
            context.metadata["a"] = True
            return context

    class _B:
        type = "prompt_agent"

        def __init__(self):
            self.id = "B"
            self.depends_on = [_Dep("s0")]
            self.retry_config = RetryConfig(retry_from_step_id="B")
            self.fail_task_on_error = True
            self.attempts = 0

        async def execute(self, context):
            self.attempts += 1
            b_started.set()
            context.metadata["b_partial"] = True
            await a_committed.wait()  # A completed and journaled while B was running
            raise RuntimeError("B boom")

    def _on_complete(idx, total, step, context, duration):
        if step.id == "A":
            a_committed.set()

    b = _B()
    with caplog.at_level(logging.ERROR, logger="agent_env.task.task"):
        with pytest.raises(RuntimeError, match="B boom"):
            await Task(id="t", version=1, steps=[_FakeDeployEnv("s0"), _A(), b]).run(
                context=TaskStepContext(), on_step_complete=_on_complete,
            )
    assert b.attempts == 1  # refused, not retried
    assert any("completed while the span was executing" in m and "['A']" in m for m in caplog.messages)


@pytest.mark.asyncio
async def test_a_survivor_that_finished_before_the_span_started_does_not_block(store):
    # Control for the gate: the sibling completed before the retrying step was dispatched,
    # so its diff cannot carry the span's writes and the retry proceeds.
    deploy = _FakeDeployEnv("s0")
    early = _FlakyPrompt("early", fail_times=0, depends_on=["s0"])
    flaky = _FlakyPrompt("late", retry_config=RetryConfig(retry_from_step_id="late"),
                         fail_times=1, depends_on=["early"])
    result = await Task(id="t", version=1, steps=[deploy, early, flaky]).run(context=TaskStepContext())
    assert early.attempts == 1 and flaky.attempts == 2
    assert result.metadata["my_custom_step_metadata"] == {"early": 1, "late": 2}


@pytest.mark.asyncio
async def test_concurrent_retryable_failures_fail_terminally(store):
    # Two retryable steps failing in the SAME scheduler batch must fail the task,
    # not retry one and silently re-dispatch the other un-rolled-back. A barrier
    # releases both raises in the same event-loop cycle so they land in one asyncio.wait.
    barrier = asyncio.Barrier(2)

    class _FailTogether:
        type = "prompt_agent"

        def __init__(self, id: str):
            self.id = id
            self.depends_on = [_Dep("s0")]
            self.retry_config = RetryConfig(retry_from_step_id=id)  # self-retry, budget 1
            self.fail_task_on_error = True
            self.attempts = 0

        async def execute(self, context):
            self.attempts += 1
            await barrier.wait()  # both arrive, then both raise in the same batch
            raise RuntimeError(f"{self.id} boom")

    s0 = _FakeDeployEnv("s0")
    a = _FailTogether("A")
    b = _FailTogether("B")
    task = Task(id="t", version=1, steps=[s0, a, b])

    ctx = TaskStepContext()
    with pytest.raises(RuntimeError, match="boom"):
        await task.run(context=ctx)

    # Neither was retried (each ran exactly once); the batch was terminal.
    assert a.attempts == 1
    assert b.attempts == 1
    assert store.get(ctx.instance_id).status == "failed"


@pytest.mark.asyncio
async def test_retry_reruns_independent_descendant_of_span(store):
    # Diamond: s0 <- {branch, mid}, mid <- fail(retry_from=s0). `branch` depends
    # on s0 but is NOT an ancestor of the failed step. Rolling back to s0 tears
    # down s0, so `branch` MUST re-run too, or it is left completed against a
    # torn-down world.
    s0 = _FakeDeployEnv("s0")
    branch = _FlakyPrompt("branch", fail_times=0, depends_on=["s0"])
    mid = _FakeDeployEnv("mid", depends_on=["s0"])
    fail = _FlakyPrompt(
        "fail", retry_config=RetryConfig(retry_from_step_id="s0"),
        fail_times=1, depends_on=["mid"],
    )
    task = Task(id="t", version=1, steps=[s0, branch, mid, fail])

    ctx = TaskStepContext()
    result = await task.run(context=ctx)

    # Everything downstream of s0 re-ran, including the independent branch.
    assert s0.deploy_calls == ["s0", "s0"]
    assert mid.deploy_calls == ["mid", "mid"]
    assert branch.attempts == 2
    assert fail.attempts == 2
    assert result.metadata["retry_resets"][0]["replayed"] == ["s0", "branch", "mid", "fail"]
    # No stale duplicates: each contribution appears once.
    assert sorted(e.env_id for e in result.deployed_envs) == ["mid", "s0"]
    assert sorted(p.step_id for p in result.prompt_responses) == ["branch", "fail"]
    journal.assert_journal_replays(ctx.instance_id, live_context=ctx)


@pytest.mark.asyncio
async def test_rollback_of_a_branching_dag_spans_the_dependents_closure(store):
    # Reviewer's graph:  L -> A -> {P -> {Q, S}, X -> {Y, Z}}.
    # Q fails with retry_from_step_id=A, so the span is the dependents-closure of
    # A = {A, P, X, Q, S, Y, Z}; L (A's ancestor) is untouched and the span
    # re-runs starting from A.
    steps = [
        _FlakyPrompt("L", fail_times=0),
        _FlakyPrompt("A", fail_times=0, depends_on=["L"]),
        _FlakyPrompt("P", fail_times=0, depends_on=["A"]),
        _FlakyPrompt("X", fail_times=0, depends_on=["A"]),
        _FlakyPrompt("Q", retry_config=RetryConfig(retry_from_step_id="A"),
                     fail_times=1, depends_on=["P"]),
        _FlakyPrompt("S", fail_times=0, depends_on=["P"]),
        _FlakyPrompt("Y", fail_times=0, depends_on=["X"]),
        _FlakyPrompt("Z", fail_times=0, depends_on=["X"]),
    ]
    by_id = {s.id: s for s in steps}
    ctx = TaskStepContext()
    result = await Task(id="t", version=1, steps=steps).run(context=ctx)

    reset = result.metadata["retry_resets"][0]
    assert reset["retry_from_step_id"] == "A"
    # The span is exactly A's dependents-closure; L is not in it.
    assert set(reset["replayed"]) == {"A", "P", "X", "Q", "S", "Y", "Z"}
    assert "L" not in reset["replayed"]
    # Deterministic regardless of concurrency timing: L (ancestor) ran once; Q
    # failed then succeeded; the run completed.
    assert by_id["L"].attempts == 1
    assert by_id["Q"].attempts == 2
    assert store.get(ctx.instance_id).status == "completed"
    assert result.metadata["my_custom_step_metadata"]["L"] == 1  # L's write survived the undo


@pytest.mark.asyncio
async def test_cancelled_sibling_is_rolled_back_and_redispatched(store):
    # A self-retrying step fails while an independent sibling is still in-flight.
    # The drain cancels the sibling mid-execute (after it appended a DeployedEnv),
    # so the sibling must be rolled back too, or its clean re-run trips the
    # guard-free duplicate check.
    release = asyncio.Event()

    class _Blocking:
        type = "deploy_env"

        def __init__(self, id: str, depends_on):
            self.id = id
            self.depends_on = [_Dep(d) for d in depends_on]
            self.retry_config = None
            self.fail_task_on_error = True
            self.runs = 0

        async def execute(self, context):
            if any(e.env_id == self.id for e in context.deployed_envs):
                raise RuntimeError(f"Env '{self.id}' is already deployed")
            self.runs += 1
            context.deployed_envs.append(
                DeployedGatewayEnv(env_id=self.id, env_version=1, gateway_url="g", mcp_url="m",
                            db_web_url=None, sandbox_id="")
            )
            await release.wait()
            return context

    class _FailThenRelease:
        type = "prompt_agent"

        def __init__(self, id: str, depends_on):
            self.id = id
            self.depends_on = [_Dep(d) for d in depends_on]
            self.retry_config = RetryConfig(retry_from_step_id=id)  # self-retry
            self.fail_task_on_error = True
            self.attempts = 0

        async def execute(self, context):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("fail once")
            release.set()  # let the re-dispatched sibling finish
            return context

    s0 = _FakeDeployEnv("s0")
    blk = _Blocking("blk", depends_on=["s0"])
    fail = _FailThenRelease("fail", depends_on=["s0"])
    task = Task(id="t", version=1, steps=[s0, blk, fail])

    result = await task.run(context=TaskStepContext())

    assert fail.attempts == 2
    assert blk.runs == 2  # cancelled once, re-ran cleanly (no "already deployed")
    assert [e.env_id for e in result.deployed_envs] == ["s0", "blk"]


@pytest.mark.asyncio
async def test_two_rollback_cycles_keep_the_audit_trail(store):
    # max_retries=2 with two consecutive failures: each cycle appends its own marker and
    # flags its own failure, and the second undo must not strip the first cycle's audit
    # (the scheduler's record is replayed after the seed and is never undone).
    deploy = _FakeDeployEnv("s0")
    flaky = _FlakyPrompt("s1", retry_config=RetryConfig(retry_from_step_id="s0", max_retries=2), fail_times=2)
    ctx = TaskStepContext()
    result = await Task(id="t", version=1, steps=[deploy, flaky]).run(context=ctx)

    assert deploy.deploy_calls == ["s0"] * 3 and flaky.attempts == 3
    assert [r["attempt"] for r in result.metadata["retry_resets"]] == [2, 3]
    assert [r["rolled_back_sandboxes"]["deployed_envs"][0]["sandbox_id"] for r in result.metadata["retry_resets"]] == ["sb-s0-1", "sb-s0-2"]
    assert [f.get("retried") for f in result.metadata["failed_steps"]] == [True, True]
    inst = store.get(ctx.instance_id)
    assert inst.status == "completed" and inst.attempt_epoch == 2
    doc = journal.assert_journal_replays(ctx.instance_id, live_context=ctx)
    assert len(doc["context"]["metadata"]["retry_resets"]) == 2


@pytest.mark.asyncio
async def test_tolerant_step_with_retry_config_is_retried_then_tolerated(store):
    # retry_config on a fail_task_on_error=False step: the budget is spent first, and
    # only then does tolerance apply, so a flaky-but-optional verifier gets its retry
    # instead of a silent single run.
    deploy = _FakeDeployEnv("s0")
    flaky = _FlakyPrompt(
        "s1", retry_config=RetryConfig(retry_from_step_id="s1", max_retries=1),
        fail_times=5, fail_task_on_error=False,
    )
    after = _FlakyPrompt("s2", fail_times=0, depends_on=["s1"])
    ctx = TaskStepContext()
    result = await Task(id="t", version=1, steps=[deploy, flaky, after]).run(context=ctx)  # no raise

    assert flaky.attempts == 2 and after.attempts == 1 and deploy.deploy_calls == ["s0"]
    # Two failures recorded: the retried one is flagged, the tolerated one is not.
    assert [f.get("retried") for f in result.metadata["failed_steps"]] == [True, None]
    assert store.get(ctx.instance_id).status == "completed"
    journal.assert_journal_replays(ctx.instance_id, live_context=ctx)


@pytest.mark.asyncio
async def test_every_fatal_failure_in_a_batch_is_logged(store, caplog):
    # Two independent fatal steps failing in the same scheduler batch: only the first
    # becomes the task's error, but both must be logged.
    barrier = asyncio.Barrier(2)

    class _FailTogether:
        type = "prompt_agent"

        def __init__(self, id: str):
            self.id = id
            self.depends_on = [_Dep("s0")]
            self.retry_config = None
            self.fail_task_on_error = True

        async def execute(self, context):
            await barrier.wait()
            raise RuntimeError(f"{self.id} boom")

    task = Task(id="t", version=1, steps=[_FakeDeployEnv("s0"), _FailTogether("A"), _FailTogether("B")])
    with caplog.at_level(logging.ERROR, logger="agent_env.task.task"):
        with pytest.raises(RuntimeError, match="boom"):
            await task.run(context=TaskStepContext())
    fatal_logged = {m.split()[1] for m in caplog.messages if "failed (fatal)" in m}
    assert fatal_logged == {"A", "B"}


@pytest.mark.asyncio
async def test_legacy_failed_steps_entry_does_not_break_the_retry(store):
    # A caller-seeded context can carry a legacy non-dict failed_steps entry (the
    # runner tolerates one). The rollback must skip it, not raise and abort the
    # retry with an AttributeError that masks the real failure.
    deploy = _FakeDeployEnv("s0")
    flaky = _FlakyPrompt("s1", retry_config=RetryConfig(retry_from_step_id="s0"), fail_times=1)
    ctx = TaskStepContext(metadata={"failed_steps": ["legacy-string-entry"]})
    result = await Task(id="t", version=1, steps=[deploy, flaky]).run(context=ctx)

    assert flaky.attempts == 2
    assert result.metadata["failed_steps"][0] == "legacy-string-entry"  # untouched
    assert result.metadata["failed_steps"][1]["retried"] is True
    assert store.get(ctx.instance_id).status == "completed"


@pytest.mark.asyncio
async def test_secrets_survive_the_rebuild_from_the_stored_context(store):
    # The stored document never holds redacted keys; rebuilding the live context from it
    # must graft them back, or the re-dispatched span loses its credentials.
    deploy = _FakeDeployEnv("s0")
    flaky = _FlakyPrompt("s1", retry_config=RetryConfig(retry_from_step_id="s0"), fail_times=1)
    ctx = TaskStepContext(metadata={"user_overrides": {"litellm_api_key": "SECRET", "agent_effort": "high"}})
    result = await Task(id="t", version=1, steps=[deploy, flaky]).run(context=ctx)
    assert result.metadata["user_overrides"] == {"litellm_api_key": "SECRET", "agent_effort": "high"}
    assert "SECRET" not in str(_doc(store, ctx.instance_id)) and "SECRET" not in str(_rows(store, ctx.instance_id))


@pytest.mark.asyncio
async def test_cancel_during_the_undo_leaves_a_consistent_record(store, monkeypatch):
    # The undo write lands in a thread and cannot be interrupted. If the run is cancelled
    # while it is in flight, the live context must still be rebuilt to match the stored
    # completed_steps, or the failure handler persists the un-rolled-back context over a
    # record whose completed_steps were just cleared.
    run_task: asyncio.Task | None = None
    real_undo = store.undo_steps_sync

    async def _undo_then_cancel(instance_id, step_ids, **kw):
        doc = real_undo(instance_id, set(step_ids), **kw)  # lands
        run_task.cancel()
        return doc

    monkeypatch.setattr(store_mod, "undo_steps", _undo_then_cancel)

    deploy = _FakeDeployEnv("s0")
    flaky = _FlakyPrompt("s1", retry_config=RetryConfig(retry_from_step_id="s0"), fail_times=1)
    ctx = TaskStepContext()
    run_task = asyncio.ensure_future(Task(id="t", version=1, steps=[deploy, flaky]).run(context=ctx))
    with pytest.raises(asyncio.CancelledError):
        await run_task

    inst = store.get(ctx.instance_id)
    assert inst.status == "failed"
    # Stored record and live context agree: the span is gone from both.
    assert inst.completed_steps == []
    assert inst.context["deployed_envs"] == [] and ctx.deployed_envs == []
    assert inst.attempt_epoch == 1
    assert flaky.attempts == 1  # cancelled before any re-dispatch


@pytest.mark.asyncio
async def test_retry_reaches_a_resume_point_before_start_step(store):
    # A resumed run (start_step=1) seeds s0 as completed without running it. If s2 then
    # fails with retry_from_step_id=s0, the span reaches back past start_step: s0 must be
    # rolled back and RE-DISPATCHED so the world is genuinely rebuilt, not left half-built
    # with the seeded step skipped. (The seed itself, the caller's world, is never undone.)
    deploy = _FakeDeployEnv("s0")
    mid = _FlakyPrompt("s1", fail_times=0, depends_on=["s0"])
    flaky = _FlakyPrompt("s2", retry_config=RetryConfig(retry_from_step_id="s0"),
                         fail_times=1, depends_on=["s1"])
    task = Task(id="t", version=1, steps=[deploy, mid, flaky])

    result = await task.run(context=TaskStepContext(), start_step=1)

    # s0 never ran on the initial (resumed) pass; the retry re-dispatched it.
    assert deploy.deploy_calls == ["s0"]
    assert mid.attempts == 2 and flaky.attempts == 2
    assert set(result.metadata["retry_resets"][0]["replayed"]) == {"s0", "s1", "s2"}


@pytest.mark.asyncio
async def test_recovered_failure_is_marked_retried(store):
    # A failure the scheduler recovers via retry stays in failed_steps (audit) but
    # is flagged `retried`, so a consumer keying off failed_steps sees the run as
    # recovered rather than failed.
    deploy = _FakeDeployEnv("s0")
    flaky = _FlakyPrompt("s1", retry_config=RetryConfig(retry_from_step_id="s0"), fail_times=1)
    result = await Task(id="t", version=1, steps=[deploy, flaky]).run(context=TaskStepContext())

    fs = result.metadata["failed_steps"]
    assert [f["step_id"] for f in fs] == ["s1"]
    assert all(f.get("retried") for f in fs)


def test_rolled_back_sandboxes_is_the_difference_between_stored_and_replayed():
    before = {
        "deployed_envs": [
            {"env_id": "e", "sandbox_id": "sb-e", "sandbox_type": "beta", "sandbox_ids": {"vm": "sb-vm"}},
            {"env_id": "keep", "sandbox_id": "sb-k", "sandbox_type": "beta", "sandbox_ids": {}},
        ],
        "deployed_agents": [
            {"agent_name": "a", "sandbox_id": "sb-a", "sandbox_type": "local"},
            {"agent_name": "hosted", "sandbox_id": None},  # no sandbox: nothing to reclaim
        ],
        "deployed_sandboxes": [{"sandbox_name": "x", "sandbox_id": "sb-x", "sandbox_type": None}],
    }
    after = {
        "deployed_envs": [before["deployed_envs"][1]],
        "deployed_agents": [],
        "deployed_sandboxes": [],
    }
    assert _rolled_back_sandboxes(before, after) == {
        "deployed_envs": [{"sandbox_id": "sb-e", "sandbox_type": "beta", "sandbox_ids": {"vm": "sb-vm"}}],
        "deployed_agents": [{"sandbox_id": "sb-a", "sandbox_type": "local"}],
        "deployed_sandboxes": [{"sandbox_id": "sb-x", "sandbox_type": None}],
    }


def test_retry_config_from_dict_validates():
    with pytest.raises(ValueError):
        RetryConfig.from_dict({})  # missing resume point
    with pytest.raises(ValueError):
        RetryConfig.from_dict({"retry_from_step_id": "s0", "max_retries": "3"})  # str
    with pytest.raises(ValueError):
        RetryConfig.from_dict({"retry_from_step_id": "s0", "max_retries": True})  # bool
    with pytest.raises(ValueError):
        RetryConfig.from_dict({"retry_from_step_id": "s0", "max_retries": -1})  # negative
    rc = RetryConfig.from_dict({"go_to_step_id": "s0"})  # wire alias, default budget
    assert rc.retry_from_step_id == "s0" and rc.max_retries == 1


def test_retry_config_round_trips_on_non_prompt_step():
    # deploy_env's __init__ does not accept retry_config; attach_retry_config must
    # still hydrate it on the generic parse path, and it must round-trip out.
    doc = {
        "id": "t", "type": "task", "version": 1,
        "steps": [{
            "id": "s0", "type": "deploy_env", "env_id": "e",
            "retry_config": {"retry_from_step_id": "s0"},
        }],
    }
    task = Task.from_dict(doc)
    assert task.steps[0].retry_config.retry_from_step_id == "s0"
    assert task.to_dict()["steps"][0]["retry_config"]["retry_from_step_id"] == "s0"


# --------------------------------------------------------------------------- #
# Store-level: the scheduler's record, the attempt-epoch fence                  #
# --------------------------------------------------------------------------- #
def _ops(pre_md: dict, post_md: dict) -> ContextUpdateOps:
    return build_context_update_ops(TaskStepContext(metadata=dict(pre_md)), TaskStepContext(metadata=dict(post_md)))


def _record(store, iid, step_id, pre, post, epoch=0, total=3):
    store.record_step_complete_sync(iid, {"step_id": step_id, "status": "success"}, _ops(pre, post), total, "now", epoch)


def _seeded(store, iid="i"):
    store.upsert_instance(iid, "t", 1, total_steps=3)
    store.seed_context(iid, build_context_update_ops(None, TaskStepContext()))
    return iid


def test_undo_applies_the_callers_record_in_the_same_cas_and_journals_it(store):
    iid = _seeded(store)
    _record(store, iid, "s0", {}, {"s0": 1})
    _record(store, iid, "s1", {"s0": 1}, {"s0": 1, "s1": 1})

    seen = {}

    def _extra(before, after):
        seen["before"], seen["after"] = before["metadata"], after["metadata"]
        return ContextUpdateOps(sets={"context.metadata.audit": ["undid s1"]})

    doc = store.undo_steps_sync(iid, {"s1"}, extra_ops=_extra, bump_epoch=True)
    assert seen["before"] == {"s0": 1, "s1": 1} and seen["after"] == {"s0": 1}
    assert doc["context"]["metadata"] == {"s0": 1, "audit": ["undid s1"]}
    assert doc["attempt_epoch"] == 1
    rows = {e["step_id"]: e for e in _rows(store, iid)}
    assert rows[_SCHEDULER_STEP_ID]["ops"]["sets"] == [{"path": "context.metadata.audit", "value": ["undid s1"]}]
    # Replay reproduces the doc, the record replays right after the seed, and a second undo
    # (of a different step, no record) keeps it: the scheduler's row is never undone.
    d = _doc(store, iid)
    ordered = commit_ordered(_rows(store, iid), d["completed_steps"])
    assert [e["step_id"] for e in ordered] == [_SEED_STEP_ID, _SCHEDULER_STEP_ID, "s0"]
    assert replay_context(ordered, d["rerecorded_steps"]) == d["context"]
    doc2 = store.undo_steps_sync(iid, {"s0", _SCHEDULER_STEP_ID})
    assert doc2["context"]["metadata"] == {"audit": ["undid s1"]}
    assert _SCHEDULER_STEP_ID in {e["step_id"] for e in _rows(store, iid)}


def test_stale_epoch_write_is_fenced_and_leaves_no_journal_hole(store):
    # A completion still in flight from a rolled-back attempt lands after the span was
    # undone and re-run. The epoch fence must drop it, and dropping it must not remove the
    # re-run's journal entry for the same step: the journal-first write would otherwise
    # displace the live row and then delete it, leaving a committed step with no entry.
    iid = _seeded(store)
    _record(store, iid, "s0", {}, {"s0": 1}, epoch=0)
    _record(store, iid, "s1", {"s0": 1}, {"s0": 1, "s1": "attempt1"}, epoch=0)
    doc = store.undo_steps_sync(iid, {"s1"}, bump_epoch=True)
    assert doc["attempt_epoch"] == 1 and "s1" not in doc["context"]["metadata"]

    _record(store, iid, "s1", {"s0": 1}, {"s0": 1, "s1": "attempt2"}, epoch=1)   # the re-run
    _record(store, iid, "s1", {"s0": 1}, {"s0": 1, "s1": "attempt1"}, epoch=0)   # the zombie

    d = _doc(store, iid)
    assert d["context"]["metadata"]["s1"] == "attempt2"                      # fenced
    assert [c["step_id"] for c in d["completed_steps"]] == ["s0", "s1"]
    rows = _rows(store, iid)
    assert {e["step_id"] for e in rows} >= {"s0", "s1"}                       # no hole
    assert replay_context(commit_ordered(rows, d["completed_steps"]), d["rerecorded_steps"]) == d["context"]


def test_a_writer_that_passed_the_fence_before_the_undo_cannot_displace_the_reruns_row(store, monkeypatch):
    # The fence check and the row write are two operations on two collections. A stale
    # writer can pass the check, pause, and resume after the undo and the re-run have both
    # landed. Its seq is older than the re-run's, so the row write must refuse to go
    # backwards; otherwise it would replace the re-run's row and then delete it when its
    # own CAS is declined.
    iid = _seeded(store)
    _record(store, iid, "s0", {}, {"s0": 1}, epoch=0)
    _record(store, iid, "s1", {"s0": 1}, {"s0": 1, "s1": "attempt1"}, epoch=0)
    stale_view = store._bump_journal_seq(store._doc_store, iid)   # the stale writer's read: epoch 0, seq N
    store.undo_steps_sync(iid, {"s1"}, bump_epoch=True)
    _record(store, iid, "s1", {"s0": 1}, {"s0": 1, "s1": "attempt2"}, epoch=1)   # the re-run, seq N+1

    real_bump = store._bump_journal_seq
    monkeypatch.setattr(store, "_bump_journal_seq", lambda _s, _iid: dict(stale_view))  # resumes on its stale read
    _record(store, iid, "s1", {"s0": 1}, {"s0": 1, "s1": "attempt1"}, epoch=0)
    monkeypatch.setattr(store, "_bump_journal_seq", real_bump)

    d = _doc(store, iid)
    rows = {e["step_id"]: e for e in _rows(store, iid)}
    assert d["context"]["metadata"]["s1"] == "attempt2"
    assert "attempt2" in str(rows["s1"]["ops"]) and rows["s1"]["seq"] > stale_view["journal_seq"]
    assert replay_context(commit_ordered(_rows(store, iid), d["completed_steps"]), d["rerecorded_steps"]) == d["context"]


def test_a_higher_seq_writer_that_loses_the_first_insert_race_still_wins_the_row(store, monkeypatch):
    # Two first-time writers for one step both miss the conditional replace (no row yet). If
    # the lower-seq one inserts before the other's read, the higher-seq writer must compare
    # seqs and replace, not give up: giving up would leave the stale row as the step's entry.
    iid = _seeded(store)
    real_replace = store._doc_store.replace
    seen = {"n": 0}

    def racy_replace(coll, filt, doc, upsert=False):
        seen["n"] += 1
        if seen["n"] == 1 and coll == TASK_STEP_JOURNAL_COLLECTION:
            stale = {**doc, "seq": doc["seq"] - 1,
                     "ops": {"sets": [{"path": "context.metadata.s0", "value": "stale"}], "unsets": [], "add_to_sets": []}}
            store._doc_store.insert(TASK_STEP_JOURNAL_COLLECTION, stale)   # the lower-seq writer lands first
            return 0                                                        # our replace saw no row
        return real_replace(coll, filt, doc, upsert)

    monkeypatch.setattr(store._doc_store, "replace", racy_replace)
    _record(store, iid, "s0", {}, {"s0": "fresh"}, epoch=0)
    row = {e["step_id"]: e for e in _rows(store, iid)}["s0"]
    assert "fresh" in str(row["ops"]) and "stale" not in str(row["ops"])
    d = _doc(store, iid)
    assert replay_context(commit_ordered(_rows(store, iid), d["completed_steps"]), d["rerecorded_steps"]) == d["context"]


def test_current_epoch_write_lands_after_an_undo(store):
    iid = _seeded(store)
    _record(store, iid, "s0", {}, {"s0": 1}, epoch=0)
    store.undo_steps_sync(iid, {"s0"}, bump_epoch=True)
    _record(store, iid, "s0", {}, {"s0": 2}, epoch=1)
    d = _doc(store, iid)
    assert d["context"]["metadata"] == {"s0": 2} and d["attempt_epoch"] == 1


# --------------------------------------------------------------------------- #
# _validate_dag: retry_config resume point                                      #
# --------------------------------------------------------------------------- #
def test_validate_dag_rejects_non_ancestor_resume_point():
    a = _FakeDeployEnv("a", depends_on=[])
    b = _FakeDeployEnv("b", depends_on=[])  # independent of a
    c = _FlakyPrompt("c", retry_config=RetryConfig(retry_from_step_id="b"), depends_on=["a"])
    with pytest.raises(ValueError, match="not in the step's dependency ancestry"):
        Task(id="t", version=1, steps=[a, b, c])


def test_validate_dag_rejects_unknown_resume_point():
    step = _FlakyPrompt("step", retry_config=RetryConfig(retry_from_step_id="missing"))

    with pytest.raises(ValueError, match="not in the step's dependency ancestry"):
        Task(id="t", version=1, steps=[step])


def test_validate_dag_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="Duplicate step id 'same' at position 1"):
        Task(id="t", version=1, steps=[_FakeDeployEnv("same"), _FakeDeployEnv("same")])


@pytest.mark.parametrize("self_retry", [False, True])
def test_implicit_validation_skips_prefix_slices_and_self_retry_ancestry(monkeypatch, self_retry):
    class _SliceCountingSteps(list):
        slice_count = 0

        def __getitem__(self, key):
            if isinstance(key, slice):
                self.slice_count += 1
            return super().__getitem__(key)

    class _Step:
        def __init__(self, index):
            self.id = f"s{index}"
            self.depends_on = None
            self.retry_config = (
                RetryConfig(retry_from_step_id=self.id) if self_retry else None
            )

    ancestry_calls = 0
    original_ancestor_ids = task_module._ancestor_ids

    def count_ancestry(*args):
        nonlocal ancestry_calls
        ancestry_calls += 1
        return original_ancestor_ids(*args)

    monkeypatch.setattr(task_module, "_ancestor_ids", count_ancestry)
    steps = _SliceCountingSteps(_Step(i) for i in range(100))

    Task(id="t", version=1, steps=steps)

    assert steps.slice_count == 0
    assert ancestry_calls == 0


def test_validate_dag_accepts_ancestor_resume_point():
    a = _FakeDeployEnv("a", depends_on=[])
    c = _FlakyPrompt("c", retry_config=RetryConfig(retry_from_step_id="a"), depends_on=["a"])
    # Should not raise.
    Task(id="t", version=1, steps=[a, c])
