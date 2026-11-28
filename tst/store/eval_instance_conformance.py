"""Eval instance conformance: the ``eval_instances`` and ``eval_slots`` records on any configured document store.

Each case is ``case(root)``, run with the store under test installed as the configured document store: ``root`` is
the eval id its records are rooted in, bare or in a local namespace, so the cases also hold through namespace routing.
No record in the store may use ``root`` yet: give each case its own, as this machine's local store outlives a test.
Every document store must pass every case in ``CASES``.
"""

import uuid

import pytest

from agent_env.config import get_config
from agent_env.eval import instance_store
from agent_env.eval.instance import (
    Attempt,
    EvalInstance,
    EvalInstanceStatus,
    EvalSettings,
    EvalSource,
    Lineage,
    RunnerDeclarations,
    Slot,
    SlotState,
    SourceKind,
    new_eval_instance_id,
    planned_slots,
    slot_id,
)
from agent_env.store import DuplicateKeyError


def an_eval_instance_and_its_slots_round_trip(root):
    instance = _instance(root, status=EvalInstanceStatus.RUNNING, intent_lock={"task": {"id": "t", "version": 2}},
                         framework={"agentenv-framework": "0.9.1", "result_rules_version": 0},
                         summary={"slots_rev": 0}, driver={"host": "h", "pid": 1, "seen": "2026-11-27T00:00:00+00:00"},
                         lineage=Lineage(rerun_of=f"{root}-before", not_like_for_like=True), created_by="me",
                         invocation_id="inv-1")
    slots = planned_slots(instance.id, [(f"{root}/tasks/a", 2), (f"{root}/tasks/b", 5)], k=2)
    instance_store.create_eval_instance(instance, slots)

    assert instance_store.get_eval_instance(instance.id) == instance
    assert instance_store.get_slots(instance.id) == slots
    assert [slot.id for slot in slots] == [
        f"{root}/tasks/a@2#1", f"{root}/tasks/b@5#1", f"{root}/tasks/a@2#2", f"{root}/tasks/b@5#2"]
    assert instance_store.get_eval_instance(f"{root}-missing0") is None


def slot_ids_name_the_task_version_and_replica_counted_from_one(root):
    instance = created_instance(root, k=3)
    assert [slot.id for slot in instance_store.get_slots(instance.id)] == [
        f"{root}/tasks/a@2#1", f"{root}/tasks/a@2#2", f"{root}/tasks/a@2#3"]
    with pytest.raises(ValueError):
        slot_id(f"{root}/tasks/a", 2, 0)
    with pytest.raises(ValueError):
        planned_slots(instance.id, [(f"{root}/tasks/a", 2)], k=0)
    with pytest.raises(ValueError):
        planned_slots(instance.id, [(f"{root}/tasks/a", 2), (f"{root}/tasks/b", 1), (f"{root}/tasks/a", 2)], k=1)
    assert len(planned_slots(instance.id, [(f"{root}/tasks/a", 2), (f"{root}/tasks/a", 3)], k=2)) == 4


def an_attempt_is_added_after_the_earlier_ones_and_never_replaces_them(root):
    instance = created_instance(root)
    slot = f"{root}/tasks/a@2#1"
    first, second = _attempt("first"), _attempt("second")

    assert instance_store.append_attempt(instance.id, slot, first).attempts == [first]
    instance_store.record_slot(instance.id, slot, SlotState.INTERRUPTED, {"result": "ungraded", "cause": "interrupted"},
                               run_id=first.run_id)
    instance_store.append_attempt(instance.id, slot, second)
    graded = instance_store.record_slot(instance.id, slot, SlotState.GRADED, {"result": "passed"}, run_id=second.run_id)

    assert graded.attempts == [first, second]
    assert (graded.state, graded.result) == (SlotState.GRADED, {"result": "passed"})
    assert instance_store.get_slot(instance.id, slot) == graded


def an_attempt_sent_again_is_not_added_twice(root):
    """As when a write that landed is retried because its answer was lost."""
    instance = created_instance(root)
    slot = f"{root}/tasks/a@2#1"
    first = _attempt("first")
    instance_store.append_attempt(instance.id, slot, first)
    assert instance_store.append_attempt(instance.id, slot, first).attempts == [first]


def a_result_lands_only_for_the_slots_last_attempt(root):
    """So a late result from an attempt a retry replaced never overwrites the slot."""
    instance = created_instance(root, k=2)
    slot, untouched = f"{root}/tasks/a@2#1", f"{root}/tasks/a@2#2"
    first, second = _attempt("first"), _attempt("second")
    instance_store.append_attempt(instance.id, slot, first)
    instance_store.append_attempt(instance.id, slot, second)

    for stale in (first.run_id, None, "run-never"):
        with pytest.raises(instance_store.StaleAttemptError):
            instance_store.record_slot(instance.id, slot, SlotState.GRADED, {"result": "passed"}, run_id=stale)
    assert instance_store.get_slot(instance.id, slot).state is SlotState.RUNNING
    with pytest.raises(instance_store.StaleAttemptError):
        instance_store.record_slot(instance.id, untouched, SlotState.GRADED, run_id=first.run_id)
    assert instance_store.record_slot(instance.id, untouched, SlotState.NOT_RUN_BUDGET, run_id=None).state is (
        SlotState.NOT_RUN_BUDGET)


def a_slot_write_is_counted_before_and_after_it_lands(root):
    """A summary takes slots_rev before the slots, so one taken during a write, or before a driver stopped between the
    steps of one, is stamped older than slots_rev."""
    instance = created_instance(root)
    slot = f"{root}/tasks/a@2#1"
    assert instance_store.read_slots(instance.id)[0] == 0
    instance_store.append_attempt(instance.id, slot, _attempt("first"))
    assert instance_store.read_slots(instance.id)[0] == 2
    with pytest.raises(LookupError):
        instance_store.record_slot(instance.id, f"{root}/tasks/never@1#1", SlotState.GRADED, run_id=None)
    assert instance_store.get_eval_instance(instance.id).slots_rev == 3
    with pytest.raises(LookupError):
        instance_store.append_attempt(f"{root}-missing0", slot, _attempt("x"))
    with pytest.raises(LookupError):
        instance_store.read_slots(f"{root}-missing0")


def each_eval_instance_keeps_its_own_slots(root):
    first, second = created_instance(root), created_instance(root)
    slot = f"{root}/tasks/a@2#1"
    instance_store.record_slot(first.id, slot, SlotState.CANCELLED, run_id=None)

    assert instance_store.get_slot(first.id, slot).state is SlotState.CANCELLED
    assert instance_store.get_slot(second.id, slot).state is SlotState.PLANNED
    assert {s.eval_instance_id for s in instance_store.get_slots(second.id)} == {second.id}


def a_creation_stopped_partway_is_not_seen_and_running_it_again_finishes_it(root):
    """The slots are written before the instance, so slots left without one are read by nobody."""
    instance = _instance(root)
    slots = planned_slots(instance.id, [(f"{root}/tasks/a", 2), (f"{root}/tasks/b", 2)], k=2)
    instance_store.ensure_indexes()
    for slot in slots[:3]:
        get_config().get_document_store().insert(instance_store.EVAL_SLOTS_COLLECTION, slot.to_dict())

    assert instance_store.get_eval_instance(instance.id) is None
    assert instance.id not in [i.id for i in instance_store.list_eval_instances(eval_id=root)]
    with pytest.raises(LookupError):
        instance_store.get_slots(instance.id)
    with pytest.raises(LookupError):
        instance_store.get_slot(instance.id, slots[0].id)
    with pytest.raises(LookupError):
        instance_store.append_attempt(instance.id, slots[0].id, _attempt("early"))

    instance_store.create_eval_instance(instance, slots)
    instance_store.create_eval_instance(instance, slots)
    assert instance_store.get_eval_instance(instance.id) == instance
    assert instance_store.get_slots(instance.id) == slots


def a_different_record_under_a_recorded_id_is_refused_and_writes_nothing(root):
    instance = created_instance(root)
    planned = planned_slots(instance.id, [(f"{root}/tasks/a", 2)], k=1)
    slot = planned[0].id
    more = planned_slots(instance.id, [(f"{root}/tasks/a", 2), (f"{root}/tasks/b", 2)], k=2)
    for differing in ((_instance(root, id=instance.id, settings=EvalSettings(k=9)), more), (instance, more), (instance, [])):
        with pytest.raises(DuplicateKeyError):
            instance_store.create_eval_instance(*differing)
    assert instance_store.get_slots(instance.id) == planned
    assert instance_store.get_eval_instance(instance.id) == instance

    instance_store.append_attempt(instance.id, slot, _attempt("first"))
    with pytest.raises(DuplicateKeyError):
        instance_store.create_eval_instance(instance, planned)
    assert len(instance_store.get_slot(instance.id, slot).attempts) == 1


def a_record_is_stored_as_json_so_a_retry_compares_equal(root):
    """Tuples become lists and keys text on every backend alike; what JSON can't hold is refused before any write."""
    instance = _instance(root, intent_lock={"pins": ("a", 1)}, settings=EvalSettings(k=1, overrides={7: "seven"}))
    slots = planned_slots(instance.id, [(f"{root}/tasks/a", 2)], k=1)
    instance_store.create_eval_instance(instance, slots)
    instance_store.create_eval_instance(instance, slots)
    stored = instance_store.get_eval_instance(instance.id)
    assert (stored.intent_lock, stored.settings.overrides) == ({"pins": ["a", 1]}, {"7": "seven"})

    for unstorable in ({"seen": {"a"}}, {"score": float("nan")}):
        bad = _instance(root, intent_lock=unstorable)
        with pytest.raises((TypeError, ValueError)):
            instance_store.create_eval_instance(bad, planned_slots(bad.id, [(f"{root}/tasks/a", 2)], k=1))
        assert instance_store.get_eval_instance(bad.id) is None


def a_slot_must_name_a_task_in_its_eval_instances_namespace(root):
    """So an eval's records all live in one store, the local one for an @local or @hf eval."""
    elsewhere = "tasks/a" if root.startswith("@") else "@local/~/elsewhere/tasks/a"
    instance = _instance(root)
    with pytest.raises(ValueError):
        instance_store.create_eval_instance(instance, planned_slots(instance.id, [(elsewhere, 1)], k=1))
    assert instance_store.get_eval_instance(instance.id) is None


def per_run_secrets_are_not_stored(root):
    """As the run index strips them from a run's overrides; a bare env: or secret: reference names one and is kept."""
    overrides = {"litellm_api_key": "sk-not-stored", "judge_litellm_api_key": "env:JUDGE_KEY", "judge_model": "m",
                 "keys": ({"usersim_api_key": "sk-not-stored"},)}
    instance = _instance(root, settings=EvalSettings(k=1, overrides=overrides))
    instance_store.create_eval_instance(instance, planned_slots(instance.id, [(f"{root}/tasks/a", 2)], k=1))
    instance_store.create_eval_instance(instance, planned_slots(instance.id, [(f"{root}/tasks/a", 2)], k=1))
    graded = instance_store.record_slot(instance.id, f"{root}/tasks/a@2#1", SlotState.CANCELLED,
                                        {"overrides": {"usersim_api_key": "sk-not-stored"}}, run_id=None)

    stored = instance_store.get_eval_instance(instance.id).settings.overrides
    assert stored == {"judge_litellm_api_key": "env:JUDGE_KEY", "judge_model": "m", "keys": [{}]}
    assert graded.result == {"overrides": {}}


def eval_instances_list_newest_first_and_by_eval(root):
    other_root = f"{root}-other"
    times = ["2026-11-27T10:00:00+00:00", "2026-11-27T05:01:00.5-05:00", "2026-11-27T10:02:00Z"]
    made = [created_instance(root, created_at=when) for when in times]
    created_instance(other_root, created_at="2026-11-27T10:05:00+00:00")

    ours = instance_store.list_eval_instances(eval_id=root)
    assert [i.id for i in ours] == [i.id for i in reversed(made)]
    assert made[1].created_at == "2026-11-27T10:01:00.500000+00:00"
    assert [i.id for i in instance_store.list_eval_instances(eval_id=root, limit=1, offset=1)] == [made[1].id]
    every = instance_store.list_eval_instances()
    assert [i.created_at for i in every] == sorted((i.created_at for i in every), reverse=True)


def an_unknown_status_state_kind_or_time_is_refused_when_written(root):
    known = _instance(root)
    unknown = [
        (_instance(root, status="done"), None),
        (_instance(root, source=EvalSource(kind="folder", eval_id=root)), None),
        (_instance(root, created_at="2026-11-27T10:00:00"), None),
        (_instance(root, created_at="yesterday"), None),
        (known, Slot(known.id, f"{root}/tasks/a", 2, 1, state="passed")),
    ]
    for instance, slot in unknown:
        with pytest.raises(ValueError):
            instance_store.create_eval_instance(instance, [slot] if slot else [])
        assert instance_store.get_eval_instance(instance.id) is None
    instance = created_instance(root)
    with pytest.raises(ValueError):
        instance_store.record_slot(instance.id, f"{root}/tasks/a@2#1", "passed", run_id=None)
    assert instance_store.get_slot(instance.id, f"{root}/tasks/a@2#1").state is SlotState.PLANNED


def a_record_a_newer_version_wrote_still_loads(root):
    """Several versions share a store, so values this one doesn't know load as stored rather than failing a listing."""
    instance = created_instance(root)
    docs = get_config().get_document_store()
    newer = {**instance.to_dict(), "id": f"{instance.id}n", "status": "paused", "created_at": "2026-11-27 10:00:00",
             "source": {**instance.source.to_dict(), "kind": "workspace"}}
    docs.insert(instance_store.EVAL_INSTANCES_COLLECTION, newer)
    docs.insert(instance_store.EVAL_SLOTS_COLLECTION, {
        **planned_slots(newer["id"], [(f"{root}/tasks/a", 2)], k=1)[0].to_dict(), "id": "a newer scheme's id",
        "state": "queued"})

    loaded = instance_store.get_eval_instance(newer["id"])
    assert (loaded.status, loaded.source.kind, loaded.created_at) == ("paused", "workspace", "2026-11-27 10:00:00")
    assert newer["id"] in [i.id for i in instance_store.list_eval_instances(eval_id=root)]
    [slot] = instance_store.get_slots(newer["id"])
    assert (slot.id, slot.state) == ("a newer scheme's id", "queued")


CASES = [
    an_eval_instance_and_its_slots_round_trip,
    slot_ids_name_the_task_version_and_replica_counted_from_one,
    an_attempt_is_added_after_the_earlier_ones_and_never_replaces_them,
    an_attempt_sent_again_is_not_added_twice,
    a_result_lands_only_for_the_slots_last_attempt,
    a_slot_write_is_counted_before_and_after_it_lands,
    each_eval_instance_keeps_its_own_slots,
    a_creation_stopped_partway_is_not_seen_and_running_it_again_finishes_it,
    a_different_record_under_a_recorded_id_is_refused_and_writes_nothing,
    a_record_is_stored_as_json_so_a_retry_compares_equal,
    a_slot_must_name_a_task_in_its_eval_instances_namespace,
    per_run_secrets_are_not_stored,
    eval_instances_list_newest_first_and_by_eval,
    an_unknown_status_state_kind_or_time_is_refused_when_written,
    a_record_a_newer_version_wrote_still_loads,
]


def _instance(root, **fields) -> EvalInstance:
    return EvalInstance(**{
        "id": new_eval_instance_id(root),
        "source": EvalSource(kind=SourceKind.EVAL, eval_id=root, eval_version=3),
        "settings": EvalSettings(k=2, max_concurrency=4, overrides={"judge_model": "m"}),
        "runner": RunnerDeclarations(type="local"),
        "created_at": "2026-11-27T10:00:00+00:00",
        **fields,
    })


def created_instance(root, k=1, **fields) -> EvalInstance:
    instance = _instance(root, **fields)
    instance_store.create_eval_instance(instance, planned_slots(instance.id, [(f"{root}/tasks/a", 2)], k=k))
    return instance


def _attempt(name) -> Attempt:
    return Attempt(run_id=f"run-{name}-{uuid.uuid4().hex[:8]}", instance_id=f"task-{name}",
                   submitted_at="2026-11-27T10:00:00+00:00")
