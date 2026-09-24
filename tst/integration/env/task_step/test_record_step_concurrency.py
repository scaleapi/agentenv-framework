"""Integration tests proving record_step_complete is safe under parallel writers
to disjoint metadata paths — no CAS retry loop needed — and, end to end on real Mongo,
that the step journal replays to the stored context through a run's lifecycle."""

import asyncio
import uuid

import pytest

from agent_env.env.env import DeployedEnv
from agent_env.store.document_store import Filter
from agent_env.task.store import (
    TASK_INSTANCES_COLLECTION,
    get_task_instance_store,
    record_step_complete,
    seed_task_instance_context,
    undo_steps,
)
from agent_env.task_step.context import TaskStepContext, regraft_redacted_keys
from agent_env.task_step.context_ops import build_context_update_ops
from tst.task import journal_invariants as journal


def _fresh_instance(total_steps: int) -> str:
    store = get_task_instance_store()
    inst = store.create_instance(
        task_id=f"concurrency-test-{uuid.uuid4().hex[:6]}",
        task_version=1,
        total_steps=total_steps,
    )
    return inst.instance_id


def _assert_journal_matches(instance_id: str, n: int) -> None:
    """Every concurrent completion journaled its diff with a unique seq, and
    replaying the journal in commit order reproduces the stored context on real Mongo.
    These instances come from ``create_instance`` with no seed of their own."""
    journal.assert_journal_replays(
        instance_id, expect_steps={f"s{i}" for i in range(n)}, seeded=False,
    )


def _env(instance_id: str) -> DeployedEnv:
    return DeployedEnv(
        env_id="e", env_version=1, gateway_url="g", mcp_url="m",
        db_web_url=None, sandbox_id="s", instance_id=instance_id,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_disjoint_metadata_paths_all_land():
    """10 workers each write a distinct metadata key. All 10 keys land, no
    merge logic, no CAS retry. This is the headline correctness claim."""
    n = 10
    instance_id = _fresh_instance(total_steps=n)

    async def write_step(i: int) -> None:
        post = TaskStepContext(metadata={f"step_{i}": {"result": i}})
        ops = build_context_update_ops(TaskStepContext(), post)
        await record_step_complete(
            instance_id, step_id=f"s{i}", ops=ops,
            total_steps=n, completed_at_utc="2026-04-24 12:00 UTC",
        )

    await asyncio.gather(*(write_step(i) for i in range(n)))

    inst = get_task_instance_store().get(instance_id)
    assert len(inst.completed_steps) == n
    assert inst.current_step == n
    assert inst.status == "completed"
    assert inst.completed_at_utc == "2026-04-24 12:00 UTC"
    assert inst.context is not None
    metadata = inst.context.get("metadata") or {}
    for i in range(n):
        assert metadata.get(f"step_{i}") == {"result": i}, (
            f"step_{i} missing from persisted metadata"
        )
    _assert_journal_matches(instance_id, n)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_deployed_envs_all_dedup_append():
    """10 workers each append a distinct DeployedEnv. All land via $addToSet."""
    n = 10
    instance_id = _fresh_instance(total_steps=n)

    async def write_step(i: int) -> None:
        post = TaskStepContext(deployed_envs=[_env(instance_id=f"env_{i}")])
        ops = build_context_update_ops(TaskStepContext(), post)
        await record_step_complete(
            instance_id, step_id=f"s{i}", ops=ops,
            total_steps=n, completed_at_utc="2026-04-24 12:00 UTC",
        )

    await asyncio.gather(*(write_step(i) for i in range(n)))

    inst = get_task_instance_store().get(instance_id)
    envs = (inst.context or {}).get("deployed_envs") or []
    persisted_ids = {e.get("instance_id") for e in envs}
    assert persisted_ids == {f"env_{i}" for i in range(n)}
    _assert_journal_matches(instance_id, n)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_nested_sibling_leaves_coexist():
    """Two workers write to sibling leaves under the same nested parent. Both
    survive — this is the bug the path-level refactor fixes."""
    instance_id = _fresh_instance(total_steps=2)

    async def write(step_id: str, key: str, value: dict) -> None:
        post = TaskStepContext(metadata={"verifications": {key: value}})
        ops = build_context_update_ops(TaskStepContext(), post)
        await record_step_complete(
            instance_id, step_id=step_id, ops=ops,
            total_steps=2, completed_at_utc="2026-04-24 12:00 UTC",
        )

    await asyncio.gather(
        write("sA", "A", {"score": 1.0}),
        write("sB", "B", {"score": 0.5}),
    )

    inst = get_task_instance_store().get(instance_id)
    verifications = (inst.context or {}).get("metadata", {}).get("verifications", {})
    assert verifications == {"A": {"score": 1.0}, "B": {"score": 0.5}}
    journal.assert_journal_replays(instance_id, expect_steps={"sA", "sB"}, seeded=False)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_journal_lifecycle():
    """The journal on the configured store: register -> seed (with a secret) -> 8 concurrent
    completions that all touch one shared list -> same-status re-record -> undo a subset through
    the async API -> re-record -> re-register. At every stage the journal replays to the stored
    context. (Mongo-only invariants — BSON dates, no TTL — are asserted in the sdk tier.)"""
    store = get_task_instance_store()
    n = 8
    iid = f"journal-lifecycle-{uuid.uuid4().hex[:8]}"
    all_steps = {f"s{i}" for i in range(n)}

    def doc() -> dict:
        return store._doc_store.find_one(TASK_INSTANCES_COLLECTION, Filter.of(instance_id=iid))

    def assert_replay_matches(expected_steps: set[str]) -> tuple[list[dict], dict]:
        d = journal.assert_journal_replays(iid, expect_steps=expected_steps)
        return store.journal_entries_sync(iid), d

    async def complete(i: int, value: int) -> None:
        post = TaskStepContext(metadata={f"step_{i}": {"result": value}, "log": [f"s{i}"]})
        await record_step_complete(
            iid, step_id=f"s{i}", ops=build_context_update_ops(TaskStepContext(), post),
            total_steps=n, completed_at_utc="2026-04-24 12:00 UTC",
        )

    try:
        store.upsert_instance(instance_id=iid, task_id="journal-lifecycle", task_version=1, total_steps=n)
        live_md = {"base": 1, "user_overrides": {"litellm_api_key": "sk-lifecycle-secret"}}
        seed_task_instance_context(iid, TaskStepContext(metadata=dict(live_md)))

        await asyncio.gather(*(complete(i, i) for i in range(n)))      # the shared `log` forces CAS retries
        entries, d = assert_replay_matches(all_steps)
        assert d["status"] == "completed" and sorted(d["context"]["metadata"]["log"]) == sorted(all_steps)
        assert "sk-lifecycle-secret" not in repr(entries) and "sk-lifecycle-secret" not in repr(d["context"])

        await complete(3, 33)                                            # same status, no undo in between
        entries, d = assert_replay_matches(all_steps)
        assert d["completed_steps"][-1]["step_id"] == "s3" and d["context"]["metadata"]["step_3"] == {"result": 33}

        undone = {"s1", "s3", "s6"}
        assert await undo_steps(iid, undone) is not None
        entries, d = assert_replay_matches(all_steps - undone)
        assert {c["step_id"] for c in d["completed_steps"]} == all_steps - undone and d["current_step"] == n - 3
        assert d["status"] == "running" and d["completed_at_utc"] is None
        md = d["context"]["metadata"]
        assert not {"step_1", "step_3", "step_6"} & md.keys() and sorted(md["log"]) == sorted(all_steps - undone)
        regraft_redacted_keys(live_md, md)
        assert md["user_overrides"] == {"litellm_api_key": "sk-lifecycle-secret"}   # the secret-only parent comes back

        await asyncio.gather(*(complete(int(s[1:]), int(s[1:])) for s in undone))
        entries, d = assert_replay_matches(all_steps)
        assert d["status"] == "completed"

        seq_before = doc()["journal_seq"]
        store.upsert_instance(instance_id=iid, task_id="journal-lifecycle", task_version=1, total_steps=n)  # re-register
        assert store.journal_entries_sync(iid) == [] and doc()["completed_steps"] == []
        assert doc()["journal_seq"] == seq_before                         # the counter is monotonic across re-registers
    finally:
        store.clear_journal_sync(iid)
        store._doc_store.delete(TASK_INSTANCES_COLLECTION, Filter.of(instance_id=iid))
