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


def an_attempt_is_added_after_the_earlier_ones_and_never_replaces_them(root):
    instance = created_instance(root)
    slot = f"{root}/tasks/a@2#1"
    first, second = _attempt("first"), _attempt("second")

    assert instance_store.append_attempt(instance.id, slot, first).attempts == [first]
    instance_store.record_slot(instance.id, slot, SlotState.INTERRUPTED, {"result": "ungraded", "cause": "interrupted"})
    instance_store.append_attempt(instance.id, slot, second)
    graded = instance_store.record_slot(instance.id, slot, SlotState.GRADED, {"result": "passed"})

    assert graded.attempts == [first, second]
    assert (graded.state, graded.result) == (SlotState.GRADED, {"result": "passed"})
    assert instance_store.get_slot(instance.id, slot) == graded
    assert instance_store.get_eval_instance(instance.id).slots_rev == 4


def a_slot_write_counts_in_slots_rev_before_it_lands(root):
    """So a summary stamped with an older slots_rev reads as stale even when the write after the count never came."""
    instance = created_instance(root)
    with pytest.raises(LookupError):
        instance_store.record_slot(instance.id, f"{root}/tasks/never@1#1", SlotState.GRADED)
    assert instance_store.get_eval_instance(instance.id).slots_rev == 1
    with pytest.raises(LookupError):
        instance_store.append_attempt(f"{root}-missing0", f"{root}/tasks/a@2#1", _attempt("x"))


def each_eval_instance_keeps_its_own_slots(root):
    first, second = created_instance(root), created_instance(root)
    slot = f"{root}/tasks/a@2#1"
    instance_store.record_slot(first.id, slot, SlotState.GRADED, {"result": "passed"})

    assert instance_store.get_slot(first.id, slot).state is SlotState.GRADED
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


def a_different_record_under_a_recorded_id_is_refused(root):
    instance = created_instance(root)
    slot = f"{root}/tasks/a@2#1"
    with pytest.raises(DuplicateKeyError):
        instance_store.create_eval_instance(_instance(root, id=instance.id, settings=EvalSettings(k=9)), [])
    instance_store.append_attempt(instance.id, slot, _attempt("first"))
    with pytest.raises(DuplicateKeyError):
        instance_store.create_eval_instance(instance, planned_slots(instance.id, [(f"{root}/tasks/a", 2)], k=1))
    assert len(instance_store.get_slot(instance.id, slot).attempts) == 1


def per_run_secrets_are_not_stored(root):
    """As the run index strips them from a run's overrides; a bare env: or secret: reference names one and is kept."""
    overrides = {"litellm_api_key": "sk-not-stored", "judge_litellm_api_key": "env:JUDGE_KEY", "judge_model": "m"}
    instance = _instance(root, settings=EvalSettings(k=1, overrides=overrides))
    instance_store.create_eval_instance(instance, planned_slots(instance.id, [(f"{root}/tasks/a", 2)], k=1))
    instance_store.create_eval_instance(instance, planned_slots(instance.id, [(f"{root}/tasks/a", 2)], k=1))
    graded = instance_store.record_slot(instance.id, f"{root}/tasks/a@2#1", SlotState.GRADED,
                                        {"result": "passed", "overrides": {"usersim_api_key": "sk-not-stored"}})

    stored = instance_store.get_eval_instance(instance.id).settings.overrides
    assert stored == {"judge_litellm_api_key": "env:JUDGE_KEY", "judge_model": "m"}
    assert graded.result == {"result": "passed", "overrides": {}}


def eval_instances_list_newest_first_and_by_eval(root):
    other_root = f"{root}-other"
    made = [created_instance(root, created_at=f"2026-11-27T10:0{i}:00+00:00") for i in range(3)]
    created_instance(other_root, created_at="2026-11-27T10:05:00+00:00")

    ours = instance_store.list_eval_instances(eval_id=root)
    assert [i.id for i in ours] == [i.id for i in reversed(made)]
    assert [i.id for i in instance_store.list_eval_instances(eval_id=root, limit=1, offset=1)] == [made[1].id]
    every = instance_store.list_eval_instances()
    assert [i.created_at for i in every] == sorted((i.created_at for i in every), reverse=True)


def an_unknown_status_or_slot_state_is_refused(root):
    with pytest.raises(ValueError):
        _instance(root, status="done")
    with pytest.raises(ValueError):
        Slot(f"{root}-x", f"{root}/tasks/a", 2, 1, state="passed")
    with pytest.raises(ValueError):
        EvalSource(kind="folder")
    instance = created_instance(root)
    with pytest.raises(ValueError):
        instance_store.record_slot(instance.id, f"{root}/tasks/a@2#1", "passed")
    assert instance_store.get_slot(instance.id, f"{root}/tasks/a@2#1").state is SlotState.PLANNED
    with pytest.raises(ValueError):
        EvalInstance.from_dict({**instance.to_dict(), "status": "done"})


CASES = [
    an_eval_instance_and_its_slots_round_trip,
    slot_ids_name_the_task_version_and_replica_counted_from_one,
    an_attempt_is_added_after_the_earlier_ones_and_never_replaces_them,
    a_slot_write_counts_in_slots_rev_before_it_lands,
    each_eval_instance_keeps_its_own_slots,
    a_creation_stopped_partway_is_not_seen_and_running_it_again_finishes_it,
    a_different_record_under_a_recorded_id_is_refused,
    per_run_secrets_are_not_stored,
    eval_instances_list_newest_first_and_by_eval,
    an_unknown_status_or_slot_state_is_refused,
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
