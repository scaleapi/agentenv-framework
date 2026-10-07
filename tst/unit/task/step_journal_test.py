"""task_step_journal: the seed and every completion journal their forward diff
(journal first), a re-recorded step replaces its entry under a fresh seq, replaying the
journal reproduces the stored context, and undo_steps_sync rebuilds the instance by replaying
the survivors inside the CAS. Real LocalSqliteDocumentStore throughout."""

from __future__ import annotations

import asyncio

import pytest

import agent_env.task.step_journal as journal_mod
import agent_env.task.store as store_mod
from agent_env.store import Filter, LocalSqliteDocumentStore, Sort
from agent_env.task.step_journal import _union, commit_ordered, replay_context
from agent_env.task.store import (
    TASK_STEP_JOURNAL_COLLECTION, TaskInstanceStore, TaskStepResult, TaskStepStatus,
    seed_task_instance_context, set_task_instance_store,
    undo_steps,
)
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.context_ops import ContextUpdateOps, build_context_update_ops
from agent_env.config import set_document_store

INST = store_mod.TASK_INSTANCES_COLLECTION


@pytest.fixture
def store(tmp_path) -> TaskInstanceStore:
    doc = LocalSqliteDocumentStore(str(tmp_path / "j.db"))
    s = TaskInstanceStore()
    set_document_store(doc)
    doc.ensure_index(INST, ["instance_id"], unique=True)
    doc.ensure_index(TASK_STEP_JOURNAL_COLLECTION, ["instance_id", "step_id"], unique=True)
    set_task_instance_store(s)
    try:
        yield s
    finally:
        set_task_instance_store(None)


def _new(store, total=9, seed_md=None, **kw):
    inst = store.create_instance("t", 1, total_steps=total, **kw)
    seed_task_instance_context(inst.instance_id, TaskStepContext(metadata=dict(seed_md or {})))
    return inst.instance_id


def _ops(pre_md, post_md, **scalars):
    return build_context_update_ops(TaskStepContext(metadata=dict(pre_md)),
                                    TaskStepContext(metadata=dict(post_md), **scalars))


def _record(store, iid, step_id, ops, total=9, status="success"):
    store.record_step_complete_sync(iid, {"step_id": step_id, "status": status}, ops, total, "now")


def _journal(store, iid):
    return store._doc_store.query(TASK_STEP_JOURNAL_COLLECTION, Filter.of(instance_id=iid), sort=Sort.by("seq", descending=False))


def _replay(store, iid):
    d = _doc(store, iid)
    return replay_context(commit_ordered(_journal(store, iid), d["completed_steps"]), d["rerecorded_steps"])


def _doc(store, iid):
    return store._doc_store.find_one(INST, Filter.of(instance_id=iid))


# ── journal write ────────────────────────────────────────────────────────────

def test_seed_and_completions_journal_forward_diffs_with_a_monotonic_seq(store):
    iid = _new(store, seed_md={"base": 1})
    _record(store, iid, "s0", _ops({"base": 1}, {"base": 1, "a": 1}))
    _record(store, iid, "s1", _ops({"base": 1, "a": 1}, {"base": 1, "a": 2, "l": [1]}))
    j = _journal(store, iid)
    assert [(e["step_id"], e["seq"], e["status"]) for e in j] == [("__seed__", 1, "seed"), ("s0", 2, "success"), ("s1", 3, "success")]
    ops1 = ContextUpdateOps.from_journal_dict(j[2]["ops"])
    assert ops1.sets == {"context.metadata.a": 2} and ops1.add_to_sets == {"context.metadata.l": [1]}
    assert _doc(store, iid)["journal_seq"] == 3


def test_an_empty_seed_is_still_journaled_as_the_replay_base(store):
    iid = _new(store)
    assert [e["step_id"] for e in _journal(store, iid)] == ["__seed__"]


def test_rerecording_a_step_replaces_its_entry_under_a_fresh_seq(store):
    iid = _new(store)
    _record(store, iid, "s0", _ops({}, {"a": 1}))
    _record(store, iid, "s1", _ops({"a": 1}, {"a": 2}))
    _record(store, iid, "s0", _ops({}, {"a": 7}))  # s0 runs again after a rollback
    j = _journal(store, iid)
    assert [(e["step_id"], e["seq"]) for e in j] == [("__seed__", 1), ("s1", 3), ("s0", 4)]


def test_rerecording_a_committed_step_moves_its_completion_to_the_end(store):
    # Same status, no undo in between (e.g. the public record_step_complete API re-issued):
    # the completion must sit where its latest write landed, or replay diverges from the doc.
    iid = _new(store)
    _record(store, iid, "s0", _ops({}, {"a": 1, "l": ["s0"]}))
    _record(store, iid, "s1", _ops({"a": 1, "l": ["s0"]}, {"a": 2, "l": ["s0", "s1"]}))
    _record(store, iid, "s0", _ops({"a": 2, "l": ["s1"]}, {"a": 7, "l": ["s1", "s0"]}))
    d = _doc(store, iid)
    assert [c["step_id"] for c in d["completed_steps"]] == ["s1", "s0"] and d["current_step"] == 2
    assert d["context"]["metadata"] == {"a": 7, "l": ["s1", "s0"]}   # its re-added list item moved to the end too
    assert _replay(store, iid) == d["context"]
    _record(store, iid, "s1", _ops({}, {}), status="failure")   # a differing status replaces, never duplicates
    assert [(c["step_id"], c["status"]) for c in _doc(store, iid)["completed_steps"]] == [("s0", "success"), ("s1", "failure")]


def test_a_first_record_leaves_an_item_another_step_already_committed_in_place(store):
    # Plain $addToSet: only a re-record moves items. s1 re-emitting s0's item must not
    # reorder the list under it, and the doc still matches what replay produces.
    iid = _new(store)
    _record(store, iid, "s0", _ops({}, {"l": ["x"]}))
    _record(store, iid, "s1", _ops({"l": ["x"]}, {"l": ["x", "y"]}))
    _record(store, iid, "s2", _ops({"l": ["y"]}, {"l": ["y", "x"]}))   # re-emits x, first record for s2
    d = _doc(store, iid)
    assert d["context"]["metadata"]["l"] == ["x", "y"]                 # x kept its place
    assert [c["step_id"] for c in d["completed_steps"]] == ["s0", "s1", "s2"]
    assert _replay(store, iid) == d["context"]


def test_a_rerecord_that_re_emits_an_item_moves_it_and_replay_follows(store):
    # The one case that does move an item: s0's completion moves to the end, so its items
    # move with it — and the journal entry says so, which is how replay stays in step.
    iid = _new(store)
    _record(store, iid, "s0", _ops({}, {"l": ["x"]}))
    _record(store, iid, "s1", _ops({"l": ["x"]}, {"l": ["x", "y"]}))
    _record(store, iid, "s0", _ops({"l": ["y"]}, {"l": ["y", "x"]}))   # re-record of s0
    d = _doc(store, iid)
    assert d["context"]["metadata"]["l"] == ["y", "x"]
    assert _doc(store, iid)["rerecorded_steps"] == ["s0"]   # the instance is the only record
    assert _replay(store, iid) == d["context"]


def test_journal_is_written_before_the_completion(store, monkeypatch):
    iid = _new(store)
    monkeypatch.setattr(store_mod, "compare_and_swap", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cas down")))
    with pytest.raises(RuntimeError, match="cas down"):
        _record(store, iid, "s0", _ops({}, {"a": 1}))
    assert [e["step_id"] for e in _journal(store, iid)] == ["__seed__", "s0"]
    assert _doc(store, iid)["completed_steps"] == []


def test_journal_failure_fails_open_leaving_a_completion_with_no_entry(store, monkeypatch):
    iid = _new(store)
    monkeypatch.setattr(store, "_journal_step", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("journal down")))
    _record(store, iid, "s0", _ops({}, {"a": 1}))            # must NOT raise
    d = _doc(store, iid)
    assert [c["step_id"] for c in d["completed_steps"]] == ["s0"]   # the completion landed
    assert d["context"]["metadata"] == {"a": 1}
    assert [e["step_id"] for e in _journal(store, iid)] == ["__seed__"]   # the only trace of the failure
    with pytest.raises(ValueError, match=r"steps \['s0'\] completed without a journal entry"):
        store.undo_steps_sync(iid, {"never-ran"})


def test_seed_lands_even_when_its_journal_write_fails(store, monkeypatch):
    inst = store.create_instance("t", 1, total_steps=2)
    real = store._journal_step
    monkeypatch.setattr(store, "_journal_step", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("journal down")))
    seed_task_instance_context(inst.instance_id, TaskStepContext(metadata={"base": 1}))   # logs, must not raise
    assert _doc(store, inst.instance_id)["context"]["metadata"] == {"base": 1}
    monkeypatch.setattr(store, "_journal_step", real)
    _record(store, inst.instance_id, "s0", _ops({"base": 1}, {"base": 1, "a": 1}))
    with pytest.raises(ValueError, match="no journal seed"):   # refuses rather than replaying from an empty base
        store.undo_steps_sync(inst.instance_id, {"s0"})


def test_a_successful_rerecord_restores_the_missing_entry(store, monkeypatch):
    iid = _new(store)
    real = store._journal_step
    monkeypatch.setattr(store, "_journal_step", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    _record(store, iid, "s0", _ops({}, {"a": 1}))
    monkeypatch.setattr(store, "_journal_step", real)
    _record(store, iid, "s0", _ops({}, {"a": 1}))            # e.g. heartbeat recovery re-records it
    assert [e["step_id"] for e in _journal(store, iid)] == ["__seed__", "s0"]
    assert store.undo_steps_sync(iid, {"s0"}) is not None   # no longer unreplayable


def test_undo_refuses_when_a_surviving_step_has_no_journal_entry(store, monkeypatch):
    iid = _new(store)
    _record(store, iid, "A", _ops({}, {"a": 1}))
    monkeypatch.setattr(store, "_journal_step", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    _record(store, iid, "S", _ops({"a": 1}, {"a": 1, "s": 1}))   # S's writes are in the doc, not the journal
    with pytest.raises(ValueError, match=r"steps \['S'\] completed without a journal entry"):
        store.undo_steps_sync(iid, {"A"})
    # undoing the unjournaled step itself is fine: replay simply omits it
    after = store.undo_steps_sync(iid, {"S"})
    assert after["context"]["metadata"] == {"a": 1} and [c["step_id"] for c in after["completed_steps"]] == ["A"]


def test_completion_for_a_missing_instance_leaves_no_journal_orphan(store):
    _record(store, "ghost", "s0", _ops({}, {"a": 1}))
    assert _journal(store, "ghost") == []


def test_completion_that_does_not_land_deletes_its_journal_entry(store, monkeypatch):
    iid = _new(store)
    monkeypatch.setattr(store_mod, "compare_and_swap", lambda *a, **k: None)   # instance deleted / write declined
    _record(store, iid, "s0", _ops({}, {"a": 1}))
    assert [e["step_id"] for e in _journal(store, iid)] == ["__seed__"]


# ── replay reproduces the store ───────────────────────────────────────────────

def test_replaying_every_entry_reproduces_the_stored_context(store):
    iid = _new(store, seed_md={"base": 1, "user_overrides": {"litellm_api_key": "sk", "agent_effort": "high"}})
    _record(store, iid, "A", _ops({"base": 1}, {"base": 1, "a": {"k": 1}, "shared": "A", "l": [1]}))
    _record(store, iid, "B", _ops({"shared": "A", "l": [1]}, {"shared": "B", "l": [1, 2], "gone": None}, agent_model="m"))
    _record(store, iid, "C", _ops({"l": [1, 2], "gone": None}, {"l": [3, 1]}))  # reorder -> wholesale set; key removed
    d = _doc(store, iid)
    assert _replay(store, iid) == d["context"]


def test_undo_rebuilds_context_and_completed_steps_and_deletes_the_entries(store):
    iid = _new(store, total=4, seed_md={"base": 1})
    _record(store, iid, "L", _ops({"base": 1}, {"base": 1, "loaded": True}), total=4)
    _record(store, iid, "A", _ops({"loaded": True}, {"loaded": True, "a": {"k": 1}, "shared": "from-A"}), total=4)
    _record(store, iid, "J", _ops({"shared": "from-A"}, {"shared": "from-J", "judge": True}), total=4)   # independent branch
    _record(store, iid, "P", _ops({}, {"p": [1]}, default_agent_model="gpt-P"), total=4)
    before = _doc(store, iid)
    assert before["status"] == "completed" and before["current_step"] == 4

    after = store.undo_steps_sync(iid, {"A", "P"})
    md = after["context"]["metadata"]
    assert md == {"base": 1, "loaded": True, "shared": "from-J", "judge": True}   # survivor J's write stands; a/p gone
    assert after["context"]["default_agent_model"] is None                         # scalar restored
    assert [c["step_id"] for c in after["completed_steps"]] == ["L", "J"] and after["current_step"] == 2
    assert [e["step_id"] for e in _journal(store, iid)] == ["__seed__", "L", "J"]
    assert after["rev"] == before["rev"] + 1


def test_no_survivor_restores_the_base_not_a_siblings_value(store):
    # Concurrent dispatch: A snapshotted P=0 and set 1; B snapshotted P=1 (A's in-flight write)
    # and set 2; B COMPLETED first. Undoing both must restore the base (0), not B's prior (1).
    iid = _new(store, seed_md={"p": 0})
    _record(store, iid, "B", ContextUpdateOps(sets={"context.metadata.p": 2}))
    _record(store, iid, "A", ContextUpdateOps(sets={"context.metadata.p": 1}))
    assert store.undo_steps_sync(iid, {"A", "B"})["context"]["metadata"]["p"] == 0


def test_second_undo_on_the_same_instance_is_still_correct(store):
    iid = _new(store)
    _record(store, iid, "A", ContextUpdateOps(sets={"context.metadata.p": 1}))
    _record(store, iid, "S", ContextUpdateOps(sets={"context.metadata.p": 2}))
    assert store.undo_steps_sync(iid, {"A"})["context"]["metadata"]["p"] == 2   # S wrote last
    assert "p" not in store.undo_steps_sync(iid, {"S"})["context"]["metadata"]   # back to the base: absent


def test_undone_wholesale_list_set_keeps_a_survivors_append(store):
    iid = _new(store, seed_md={"l": ["x", "y"]})
    _record(store, iid, "A", ContextUpdateOps(sets={"context.metadata.l": ["y", "x"]}))      # reorder
    _record(store, iid, "B", ContextUpdateOps(add_to_sets={"context.metadata.l": ["z"]}))
    assert store.undo_steps_sync(iid, {"A"})["context"]["metadata"]["l"] == ["x", "y", "z"]


def test_undone_append_does_not_pull_from_a_survivors_authoritative_set(store):
    iid = _new(store)
    _record(store, iid, "A", ContextUpdateOps(add_to_sets={"context.metadata.l": ["a"]}))
    _record(store, iid, "B", ContextUpdateOps(sets={"context.metadata.l": ["a", "x"]}))
    assert store.undo_steps_sync(iid, {"A"})["context"]["metadata"]["l"] == ["a", "x"]


def test_survivors_wholesale_subtree_stands_over_an_undone_leaf(store):
    iid = _new(store)
    _record(store, iid, "A", ContextUpdateOps(sets={"context.metadata.x.y": 1}))
    _record(store, iid, "B", ContextUpdateOps(sets={"context.metadata.x": {"y": 5, "z": 2, "k.dotted": 3}}))
    assert store.undo_steps_sync(iid, {"A"})["context"]["metadata"]["x"] == {"y": 5, "z": 2, "k.dotted": 3}


def test_list_created_by_undone_and_extended_by_survivor_keeps_the_survivors_items(store):
    iid = _new(store)
    _record(store, iid, "A", ContextUpdateOps(add_to_sets={"context.metadata.l": [1]}))
    _record(store, iid, "B", ContextUpdateOps(add_to_sets={"context.metadata.l": [2]}))
    assert store.undo_steps_sync(iid, {"A"})["context"]["metadata"]["l"] == [2]


def test_uncommitted_journal_entry_is_not_replayed(store, monkeypatch):
    iid = _new(store)
    _record(store, iid, "A", ContextUpdateOps(sets={"context.metadata.a": 1}))
    real = store_mod.compare_and_swap
    monkeypatch.setattr(store_mod, "compare_and_swap", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cas down")))
    with pytest.raises(RuntimeError):
        _record(store, iid, "S", ContextUpdateOps(sets={"context.metadata.s": 1}))  # journaled, never completed
    monkeypatch.setattr(store_mod, "compare_and_swap", real)
    after = store.undo_steps_sync(iid, {"A"})
    assert "s" not in after["context"]["metadata"]


def test_journal_is_read_inside_the_cas(store, monkeypatch):
    # A completion lands between the undo's first read and its write; the retry must see it.
    iid = _new(store)
    _record(store, iid, "A", ContextUpdateOps(sets={"context.metadata.a": 1}))
    calls = {"n": 0}
    real_query = store._journal_entries

    def racing_query(docs, instance_id, generation=None):
        calls["n"] += 1
        if calls["n"] == 1:
            _record(store, iid, "S", ContextUpdateOps(sets={"context.metadata.s": 1}))  # bumps rev -> CAS retries
        return real_query(docs, instance_id, generation)
    monkeypatch.setattr(store, "_journal_entries", racing_query)
    after = store.undo_steps_sync(iid, {"A"})
    assert calls["n"] >= 2 and after["context"]["metadata"] == {"s": 1}
    assert [c["step_id"] for c in after["completed_steps"]] == ["S"]


def test_a_recompletion_after_the_cas_keeps_its_fresh_journal_entry(store, monkeypatch):
    # The undo's trim must target the exact seq it read: if the undone step completes again
    # between the CAS and the trim, that new entry is the live one and must survive.
    iid = _new(store)
    _record(store, iid, "A", ContextUpdateOps(sets={"context.metadata.a": 1}))
    real_cas = store_mod.compare_and_swap

    def racing_cas(*a, **k):
        out = real_cas(*a, **k)
        monkeypatch.setattr(store_mod, "compare_and_swap", real_cas)                # fire once
        _record(store, iid, "A", ContextUpdateOps(sets={"context.metadata.a": 7}))  # re-run lands post-CAS
        return out
    stale_seq = {e["step_id"]: e["seq"] for e in _journal(store, iid)}["A"]
    monkeypatch.setattr(store_mod, "compare_and_swap", racing_cas)
    store.undo_steps_sync(iid, {"A"})

    entries = {e["step_id"]: e for e in _journal(store, iid)}
    assert "A" in entries, "the trim deleted the entry the re-completion wrote"
    assert entries["A"]["seq"] > stale_seq       # the live entry, not the one the undo read
    assert _doc(store, iid)["context"]["metadata"]["a"] == 7   # and the re-run's write stands


def test_an_undone_step_with_no_entry_left_to_trim_is_not_an_error(store):
    # start_step-seeded completions have no journal entry at all; the trim has nothing to do.
    seeded = [TaskStepResult(step_id="s0", status=TaskStepStatus.SUCCESS)]
    iid = _new(store, total=3, start_step=1, completed_steps=seeded)
    _record(store, iid, "s1", _ops({}, {"a": 1}), total=3)
    after = store.undo_steps_sync(iid, {"s0"})
    assert [c["step_id"] for c in after["completed_steps"]] == ["s1"]
    assert [e["step_id"] for e in _journal(store, iid)] == ["__seed__", "s1"]


def test_pre_journal_instance_refuses_undo(store):
    inst = store.create_instance("t", 1, total_steps=2)   # no seed_task_instance_context: no journal at all
    with pytest.raises(ValueError, match="no journal seed"):   # even with nothing completed: no base to replay over
        store.undo_steps_sync(inst.instance_id, {"s0"})
    store._doc_store.update(INST, Filter.of(instance_id=inst.instance_id),
                            store_mod.UpdateSpec(set={"completed_steps": [{"step_id": "s0", "status": "success"}]}))
    with pytest.raises(ValueError, match="no journal seed"):
        store.undo_steps_sync(inst.instance_id, {"s0"})


def test_a_missing_seed_refuses_undo_even_when_a_newer_entry_survives(store):
    # Whatever removed the seed, replaying the survivors from an empty base would invent a
    # context nobody wrote, so a missing seed is a hard refusal rather than a best effort.
    iid = _new(store, seed_md={"base": 1})
    _record(store, iid, "A", _ops({"base": 1}, {"base": 1, "a": 1}))
    store._doc_store.delete(TASK_STEP_JOURNAL_COLLECTION, Filter.of(instance_id=iid, step_id="__seed__"))
    assert [e["step_id"] for e in _journal(store, iid)] == ["A"]
    with pytest.raises(ValueError, match="no journal seed"):
        store.undo_steps_sync(iid, {"A"})


def test_journal_entries_record_when_they_were_written(store):
    # A real datetime rather than an ISO string: nothing reads it today, but a retention
    # index would have to key on it, and Mongo only expires a BSON date.
    from datetime import datetime
    written = []
    real_replace = store._doc_store.replace
    def spy(collection, filt, doc, upsert=False):
        if collection == TASK_STEP_JOURNAL_COLLECTION:
            written.append(doc)
        return real_replace(collection, filt, doc, upsert=upsert)
    store._doc_store.replace = spy
    iid = _new(store)
    _record(store, iid, "s0", _ops({}, {"a": 1}))
    assert written and all(isinstance(d["recorded_at"], datetime) and d["recorded_at"].tzinfo is not None for d in written)
    assert "recorded_at_utc" not in written[-1]


def test_undo_reopens_a_finished_run_and_a_rerecord_closes_it_again(store):
    iid = _new(store, total=2)
    _record(store, iid, "A", _ops({}, {"a": 1}), total=2)
    _record(store, iid, "B", _ops({"a": 1}, {"a": 1, "b": 1}), total=2)
    assert _doc(store, iid)["status"] == "completed"
    assert store.undo_steps_sync(iid, {"never-ran"})["status"] == "completed"   # nothing came off: not reopened
    after = store.undo_steps_sync(iid, {"B"})
    assert after["status"] == "running" and after["completed_at_utc"] is None and after["error"] is None
    _record(store, iid, "B", _ops({"a": 1}, {"a": 1, "b": 2}), total=2)
    d = _doc(store, iid)
    assert d["status"] == "completed" and d["completed_at_utc"] == "now" and d["context"]["metadata"] == {"a": 1, "b": 2}


def test_undo_of_the_failed_step_clears_the_failure_and_its_unjournaled_partial_writes(store):
    iid = _new(store, total=2)
    _record(store, iid, "A", _ops({}, {"a": 1}), total=2)
    store.record_task_failure_sync(iid, "B", TaskStepContext(metadata={"a": 1, "partial": True}), RuntimeError("boom"), "now")
    d = _doc(store, iid)
    assert d["status"] == "failed" and d["error"] == "boom" and [c["step_id"] for c in d["completed_steps"]] == ["A", "B"]
    after = store.undo_steps_sync(iid, {"B"})
    assert after["status"] == "running" and after["error"] is None and after["completed_at_utc"] is None
    assert after["context"]["metadata"] == {"a": 1}   # the failure's wholesale context write is gone; replay is the truth
    assert [c["step_id"] for c in after["completed_steps"]] == ["A"]


def test_undoing_a_healthy_step_leaves_a_failed_run_failed(store):
    # The failing completion is still on the doc, so the run has not become healthy:
    # reopening it would hide the error and report a failed run as running.
    iid = _new(store, total=2)
    _record(store, iid, "A", _ops({}, {"a": 1}), total=2)
    store.record_task_failure_sync(iid, "B", TaskStepContext(metadata={"a": 1}), RuntimeError("boom"), "now")
    after = store.undo_steps_sync(iid, {"A"})
    assert after["status"] == "failed" and after["error"] == "boom"
    assert [c["step_id"] for c in after["completed_steps"]] == ["B"]


def test_undo_refuses_an_instance_predating_the_rerecord_field(store):
    # Absent is not empty, and it is not recoverable either: the entries' own flags are
    # pre-CAS reads, so a losing writer can have left one stale where the winning CAS did
    # move the item. Replaying off them would silently write the wrong list order.
    iid = _new(store, seed_md={"l": ["a", "x"]})
    add_a = ContextUpdateOps(add_to_sets={"context.metadata.l": ["a"]})
    _record(store, iid, "s0", add_a)
    _record(store, iid, "s1", ContextUpdateOps(sets={"context.metadata.other": 1}))
    _record(store, iid, "s0", add_a)                      # re-record: "a" moves to the end
    before = _doc(store, iid)
    assert before["context"]["metadata"]["l"] == ["x", "a"]
    store._doc_store.update(INST, Filter.of(instance_id=iid),
                            store_mod.UpdateSpec(unset={"rerecorded_steps"}))   # a pre-field instance

    with pytest.raises(ValueError, match="predates rerecorded_steps"):
        store.undo_steps_sync(iid, {"s1"})
    # Refused, not half-applied: context, completions and the journal are untouched.
    after = _doc(store, iid)
    assert after["context"] == before["context"]
    assert [c["step_id"] for c in after["completed_steps"]] == [c["step_id"] for c in before["completed_steps"]]
    assert [e["step_id"] for e in _journal(store, iid)] == ["__seed__", "s1", "s0"]


def test_a_fresh_instance_carries_the_rerecord_field(store):
    # create_instance is what makes "absent" mean pre-journal above; born without it, a
    # brand-new run would take the refusal on its first undo.
    assert _doc(store, _new(store))["rerecorded_steps"] == []


def test_the_seed_is_never_undone(store):
    iid = _new(store, seed_md={"base": 1})
    after = store.undo_steps_sync(iid, {"__seed__"})
    assert after["context"]["metadata"] == {"base": 1} and [e["step_id"] for e in _journal(store, iid)] == ["__seed__"]


def test_undo_of_start_step_seeded_completions_drops_them_without_touching_context(store):
    seeded = [TaskStepResult(step_id=s, status=TaskStepStatus.SUCCESS) for s in ("s0", "s1")]
    iid = _new(store, total=4, start_step=2, completed_steps=seeded)
    _record(store, iid, "s2", _ops({}, {"a": 1}), total=4)
    before = _doc(store, iid)["context"]
    after = store.undo_steps_sync(iid, {"s0"})
    assert [c["step_id"] for c in after["completed_steps"]] == ["s1", "s2"]
    assert after["current_step"] == 2 and after["context"] == before


def test_reregister_retires_the_previous_runs_journal_without_deleting_it(store):
    # No delete on the re-register path: one could race a completion still landing, and a
    # failed one left the old seed replayable. The generation retires the entries instead.
    store.upsert_instance(instance_id="i", task_id="t", task_version=1, total_steps=3)
    seed_task_instance_context("i", TaskStepContext(metadata={"base": 1}))
    _record(store, "i", "s0", _ops({"base": 1}, {"base": 1, "a": 1}))
    assert [e["step_id"] for e in store.journal_entries_sync("i")] == ["__seed__", "s0"]
    store.upsert_instance(instance_id="i", task_id="t", task_version=1, total_steps=3)  # reset the run
    assert [e["step_id"] for e in _journal(store, "i")] == ["__seed__", "s0"]  # rows are still there
    assert store.journal_entries_sync("i") == []                              # and no longer this run's
    assert _doc(store, "i")["context"]["metadata"] == {}                      # the run state was reset


def test_a_retired_entry_is_never_replayed_into_the_new_run(store):
    # The sharp edge a delete-on-reset was there to prevent: the old run's seed must not
    # become the replay base for the new one.
    store.upsert_instance(instance_id="i", task_id="t", task_version=1, total_steps=3)
    seed_task_instance_context("i", TaskStepContext(metadata={"old": True}))
    store.upsert_instance(instance_id="i", task_id="t", task_version=1, total_steps=3)
    with pytest.raises(ValueError, match="no journal seed"):
        store.undo_steps_sync("i", {"s0"})          # the retired seed is not a base
    seed_task_instance_context("i", TaskStepContext(metadata={"new": True}))
    _record(store, "i", "s0", _ops({"new": True}, {"new": True, "a": 1}))
    after = store.undo_steps_sync("i", {"s0"})
    assert after["context"]["metadata"] == {"new": True}   # the new run's seed, not {"old": True}


def test_a_step_journaling_across_a_reregister_belongs_to_one_run_only(store):
    # The delete this replaced could remove an entry written just after the reset. Here the
    # entry survives; it is simply scoped to the generation it was written under.
    store.upsert_instance(instance_id="i", task_id="t", task_version=1, total_steps=3)
    seed_task_instance_context("i", TaskStepContext())
    store.upsert_instance(instance_id="i", task_id="t", task_version=1, total_steps=3)
    seed_task_instance_context("i", TaskStepContext())
    _record(store, "i", "s0", _ops({}, {"a": 1}))                 # lands under the new generation
    assert [e["step_id"] for e in store.journal_entries_sync("i")] == ["__seed__", "s0"]
    assert store.undo_steps_sync("i", {"s0"})["context"]["metadata"] == {}


def test_a_completion_journaled_before_a_reregister_never_joins_the_new_run(store, monkeypatch):
    # Without binding the CAS to the generation captured while journaling, a step from the
    # old run joins the fresh one while its row stays retired — unreplayable and unremovable.
    store.upsert_instance(instance_id="i", task_id="t", task_version=1, total_steps=3)
    seed_task_instance_context("i", TaskStepContext(metadata={"base": 1}))
    real = store._journal_step

    def journal_then_reregister(docs, instance_id, step_id, ops, status, **kw):
        out = real(docs, instance_id, step_id, ops, status, **kw)
        monkeypatch.setattr(store, "_journal_step", real)                               # fire once
        store.upsert_instance(instance_id="i", task_id="t", task_version=1, total_steps=3)
        return out
    monkeypatch.setattr(store, "_journal_step", journal_then_reregister)
    _record(store, "i", "s0", _ops({"base": 1}, {"base": 1, "a": 1}), total=3)   # must be declined

    d = _doc(store, "i")
    assert d["completed_steps"] == []                     # the superseded run's step did not land
    assert "a" not in d["context"]["metadata"]            # nor its context write
    assert [e["step_id"] for e in _journal(store, "i")] == ["__seed__"]   # orphan row cleaned up


def test_the_rerecord_flag_replay_uses_comes_from_the_winning_cas(store, monkeypatch):
    # Two writers both journal rerecorded=False, but the CAS landing second unions with
    # move_to_end=True — so the flag replay trusts lives on the instance, written in that CAS.
    iid = _new(store)
    _record(store, iid, "s1", _ops({}, {"l": ["x"]}))            # x committed by a *different* step
    real = store._journal_step

    def journal_then_race(docs, instance_id, step_id, ops, status, **kw):
        out = real(docs, instance_id, step_id, ops, status, **kw)
        monkeypatch.setattr(store, "_journal_step", real)                       # fire once
        _record(store, iid, "s0", _ops({"l": ["x"]}, {"l": ["x"]}))             # the other writer lands first
        return out
    monkeypatch.setattr(store, "_journal_step", journal_then_race)
    _record(store, iid, "s0", _ops({"l": ["x"]}, {"l": ["x"]}))                 # this CAS sees s0 completed

    d = _doc(store, iid)
    assert d["rerecorded_steps"] == ["s0"]                       # the winning attempt's verdict
    assert d["context"]["metadata"]["l"] == ["x"]
    entries, completed = _journal(store, iid), d["completed_steps"]
    # The instance flags reproduce the stored context; the entry's pre-CAS flag is stale here.
    assert replay_context(commit_ordered(entries, completed), d["rerecorded_steps"]) == d["context"]


def test_undo_drops_the_undone_steps_rerecord_flag(store):
    iid = _new(store)
    _record(store, iid, "s0", _ops({}, {"l": ["x"]}))
    _record(store, iid, "s0", _ops({"l": ["x"]}, {"l": ["x", "y"]}))   # re-record -> flagged
    assert _doc(store, iid)["rerecorded_steps"] == ["s0"]
    after = store.undo_steps_sync(iid, {"s0"})
    assert after["rerecorded_steps"] == [] and after["context"]["metadata"] == {}


def test_undo_of_unknown_instance_returns_none(store):
    assert store.undo_steps_sync("nope", {"x"}) is None


def test_the_async_wrapper_propagates_a_refusal_but_not_a_missing_instance(store):
    # A refusal and "nothing to undo" must not look the same to a scheduler: on the first it
    # has to abandon the retry, on the second it can carry on.
    iid = _new(store)
    _record(store, iid, "A", _ops({}, {"a": 1}))
    store._doc_store.delete(TASK_STEP_JOURNAL_COLLECTION, Filter.of(instance_id=iid, step_id="__seed__"))
    with pytest.raises(ValueError, match="no journal seed"):
        asyncio.run(undo_steps(iid, {"A"}))
    assert asyncio.run(undo_steps("nope", {"x"})) is None


def test_a_deployed_env_written_by_two_kernels_stays_one_entry():
    """The same deployment re-serialized by a newer kernel (a field added or derived since) matches by instance, not by value."""
    old = {"env_id": "e", "instance_id": "i1", "mcp_server_name": None}
    new = {"env_id": "e", "instance_id": "i1", "mcp_server_name": "env1234", "env_provider_type": "gateway"}
    other, unregistered = {"env_id": "e2", "instance_id": "i2"}, {"env_id": "e3"}
    assert _union([old, unregistered], [new, other, unregistered], path="context.deployed_envs") == [old, unregistered, other]
    assert _union([old, other], [new], move_to_end=True, path="context.deployed_envs") == [other, new]
    assert _union([old], [new], path="context.prompt_responses") == [old, new]  # every other list stays by value


def test_union_projects_each_item_key_once(monkeypatch):
    calls = 0

    def item_key(path):
        def project(item):
            nonlocal calls
            calls += 1
            return item

        return project

    monkeypatch.setattr(journal_mod, "_item_key", item_key)
    existing = [["old"], ["shared"]]
    items = [["shared"], ["new"], ["new"]]

    assert _union(existing, items) == [["old"], ["shared"], ["new"]]
    assert calls == len(existing) + len(items)

    calls = 0
    moved_items = [["old"]]
    assert _union(existing, moved_items, move_to_end=True) == [["shared"], ["old"]]
    assert calls == len(existing) + len(moved_items)


def test_a_completion_that_re_adds_a_deployed_env_keeps_one_entry(store):
    iid = _new(store)
    old = {"env_id": "e", "instance_id": "i1", "mcp_server_name": None}
    _record(store, iid, "s0", ContextUpdateOps(add_to_sets={"context.deployed_envs": [old]}))
    _record(store, iid, "s1", ContextUpdateOps(add_to_sets={"context.deployed_envs": [{**old, "mcp_server_name": "env1234"}]}))
    assert _doc(store, iid)["context"]["deployed_envs"] == [old]
    assert _replay(store, iid)["deployed_envs"] == [old]


def test_a_seed_that_carries_one_deployment_twice_keeps_one_entry(store):
    """A context persisted with a duplicate (e.g. from a mixed-version rollout) seeds one entry, which its replay agrees with."""
    old = {"env_id": "e", "instance_id": "i1", "mcp_server_name": None}
    inst = store.create_instance("t", 1, total_steps=9)
    store.seed_context(inst.instance_id, ContextUpdateOps(add_to_sets={"context.deployed_envs": [old, {**old, "mcp_server_name": "env1234"}]}))
    assert _doc(store, inst.instance_id)["context"]["deployed_envs"] == [old]
    assert _replay(store, inst.instance_id)["deployed_envs"] == [old]
