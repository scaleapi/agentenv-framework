"""Eval instances and their slots in the configured document store: ``eval_instances`` holds one record per eval run,
``eval_slots`` one document per slot, keyed by its eval instance, so no document grows with the number of slots.

An eval instance's own document is what makes its slots count: creating one writes the slots first and the instance
last, so a creation stopped partway leaves only slots that nothing reads, and running it again finishes it. Records go
through JSON before they are written, so every backend stores, and a retry compares, the same values.

Only the process driving an eval instance writes it after that, but a retry may overlap it, so every slot write is
checked against the slot as read: an attempt already on the slot isn't added again, and a result is recorded only for
the slot's last attempt. Each slot write is counted in the instance's ``slots_rev`` both before and after it lands, and
``read_slots`` takes ``slots_rev`` before the slots: a summary stamped with what it took is current only while no slot
write has begun since, even when a driver stopped between the steps of one.

The per-run secrets the run index strips from a run's overrides (``task_step.context._REDACTED_KEYS``) are stripped
here too, so a resume reads them from its own environment and config again. An eval instance's slots must name tasks
in its own namespace, so an ``@local`` or ``@hf`` eval's records all stay in this machine's local store and a
registry eval's in the configured one.
"""

from __future__ import annotations

import json
from typing import Callable, Iterable, Optional

from agent_env.config import get_config
from agent_env.eval.instance import Attempt, EvalInstance, Slot, SlotState
from agent_env.store import DuplicateKeyError, Filter, Sort, UpdateSpec, compare_and_swap
from agent_env.store.ids import local_namespace
from agent_env.task_step.context import _strip_redacted_keys

EVAL_INSTANCES_COLLECTION = "eval_instances"
EVAL_SLOTS_COLLECTION = "eval_slots"

_indexed = None


class StaleAttemptError(RuntimeError):
    """A result recorded for an attempt that is no longer the slot's last."""


def _docs():
    """The configured document store, with the indexes behind each read made once per store."""
    global _indexed
    docs = get_config().get_document_store()
    if _indexed is not docs:
        docs.ensure_index(EVAL_INSTANCES_COLLECTION, ["id"], unique=True)
        docs.ensure_index(EVAL_INSTANCES_COLLECTION, ["eval_id", "created_at"])
        docs.ensure_index(EVAL_SLOTS_COLLECTION, ["eval_instance_id", "id"], unique=True)
        _indexed = docs
    return docs


def ensure_indexes() -> None:
    """Make the indexes now rather than on first use."""
    _docs()


def create_eval_instance(instance: EvalInstance, slots: Iterable[Slot]) -> None:
    """Record ``instance`` and its planned ``slots``: the slots first, then the instance, whose write is what makes
    them its. Running it again with the same record finishes a creation that stopped partway, or whose last write
    landed unanswered. Once the instance is recorded nothing is written again: a call whose instance or slots differ
    from what is stored, as after a slot moved on, is refused with DuplicateKeyError."""
    slots = list(slots)
    instance.validate()
    for slot in slots:
        slot.validate()
        if slot.eval_instance_id != instance.id:
            raise ValueError(f"slot {slot.id} is eval instance {slot.eval_instance_id}'s, not {instance.id}'s")
        if local_namespace(slot.task_id) != local_namespace(instance.id):
            raise ValueError(f"slot {slot.id} names a task outside eval instance {instance.id}'s namespace")
    docs = _docs()
    record = _as_stored(instance.to_dict())
    planned = {slot.id: _as_stored(slot.to_dict()) for slot in slots}
    if (stored := docs.find_one(EVAL_INSTANCES_COLLECTION, Filter.of(id=instance.id))) is not None:
        _refuse_unless_recorded_as(docs, stored, record, planned)
        return
    for slot_id, doc in planned.items():
        try:
            docs.insert(EVAL_SLOTS_COLLECTION, doc)
        except DuplicateKeyError:
            if docs.find_one(EVAL_SLOTS_COLLECTION, Filter.of(eval_instance_id=instance.id, id=slot_id)) != doc:
                raise
    try:
        docs.insert(EVAL_INSTANCES_COLLECTION, record)
    except DuplicateKeyError:
        _refuse_unless_recorded_as(
            docs, docs.find_one(EVAL_INSTANCES_COLLECTION, Filter.of(id=instance.id)), record, planned)


def _as_stored(doc: Optional[dict]) -> Optional[dict]:
    """``doc`` as every backend stores it: through JSON, tuples as lists and keys as text, with no per-run secret. A
    value JSON can't hold, NaN included, is refused before anything is written."""
    return _strip_redacted_keys(json.loads(json.dumps(doc, allow_nan=False)))


def _refuse_unless_recorded_as(docs, stored: dict, record: dict, planned: dict[str, dict]) -> None:
    """Accept a creation already done as asked; refuse one whose instance or slots are stored otherwise."""
    if stored != record:
        raise DuplicateKeyError(f"eval instance {record['id']} is recorded with other contents")
    slots = docs.query(EVAL_SLOTS_COLLECTION, Filter.of(eval_instance_id=record["id"]))
    if {doc["id"]: doc for doc in slots} != planned:
        raise DuplicateKeyError(f"eval instance {record['id']} is recorded with other slots")


def get_eval_instance(eval_instance_id: str) -> Optional[EvalInstance]:
    doc = _docs().find_one(EVAL_INSTANCES_COLLECTION, Filter.of(id=eval_instance_id))
    return EvalInstance.from_dict(doc) if doc else None


def list_eval_instances(eval_id: Optional[str] = None, limit: int = 50, offset: int = 0) -> list[EvalInstance]:
    """Eval instances newest first, those of ``eval_id`` alone when it's given."""
    docs = _docs().query(
        EVAL_INSTANCES_COLLECTION, Filter.of(eval_id=eval_id) if eval_id else Filter(),
        sort=Sort.by("created_at", descending=True), limit=limit, offset=offset,
    )
    return [EvalInstance.from_dict(doc) for doc in docs]


def get_slots(eval_instance_id: str) -> list[Slot]:
    """Every slot of the eval instance, replica 1 of each task first."""
    return read_slots(eval_instance_id)[1]


def read_slots(eval_instance_id: str) -> tuple[int, list[Slot]]:
    """The eval instance's ``slots_rev``, read before its slots, and its slots: a summary of them stamped with that
    ``slots_rev`` is current while it still equals the instance's."""
    instance = _docs().find_one(EVAL_INSTANCES_COLLECTION, Filter.of(id=eval_instance_id))
    if instance is None:
        raise LookupError(f"no eval instance {eval_instance_id}")
    docs = _docs().query(EVAL_SLOTS_COLLECTION, Filter.of(eval_instance_id=eval_instance_id))
    slots = sorted((Slot.from_dict(doc) for doc in docs), key=lambda slot: (slot.replica, slot.task_id, slot.task_version))
    return instance.get("slots_rev", 0), slots


def get_slot(eval_instance_id: str, slot_id: str) -> Optional[Slot]:
    _recorded(eval_instance_id)
    doc = _docs().find_one(EVAL_SLOTS_COLLECTION, Filter.of(eval_instance_id=eval_instance_id, id=slot_id))
    return Slot.from_dict(doc) if doc else None


def _recorded(eval_instance_id: str) -> None:
    """Refuse an eval instance that isn't recorded, so no reader takes up the slots a stopped creation left behind."""
    if _docs().find_one(EVAL_INSTANCES_COLLECTION, Filter.of(id=eval_instance_id)) is None:
        raise LookupError(f"no eval instance {eval_instance_id}")


def append_attempt(eval_instance_id: str, slot_id: str, attempt: Attempt) -> Slot:
    """Add ``attempt`` after the slot's earlier ones, which it never replaces, and mark the slot running. An attempt
    already on the slot, as when a write that landed is sent again, isn't added twice."""
    def add(doc: dict) -> Optional[UpdateSpec]:
        if any(a.get("run_id") == attempt.run_id for a in doc.get("attempts") or []):
            return None
        return UpdateSpec(push={"attempts": [attempt.to_dict()]}, set={"state": str(SlotState.RUNNING)})

    return _write_slot(eval_instance_id, slot_id, add)


def record_slot(eval_instance_id: str, slot_id: str, state: SlotState, result: Optional[dict] = None, *,
                run_id: Optional[str]) -> Slot:
    """Set where the slot stands and its result, which is the attempt ``run_id``'s, or none's (None) for a slot never
    submitted. Refused with StaleAttemptError unless that attempt is still the slot's last, so a late result from an
    attempt a retry replaced never lands."""
    state = SlotState(state)
    result = _as_stored(result)

    def settle(doc: dict) -> UpdateSpec:
        attempts = doc.get("attempts") or []
        last = attempts[-1].get("run_id") if attempts else None
        if last != run_id:
            raise StaleAttemptError(f"slot {slot_id}'s last attempt is {last}, not {run_id}")
        return UpdateSpec(set={"state": str(state), "result": result})

    return _write_slot(eval_instance_id, slot_id, settle)


def _write_slot(eval_instance_id: str, slot_id: str, change: Callable[[dict], Optional[UpdateSpec]]) -> Slot:
    """Apply ``change`` to the slot as read, under a compare-and-swap on its ``rev``, counted in the instance's
    ``slots_rev`` before and after. ``change`` returns None when the slot already holds what it asks."""
    docs = _docs()
    key = Filter.of(eval_instance_id=eval_instance_id, id=slot_id)
    count = UpdateSpec(inc={"slots_rev": 1})
    if not docs.update(EVAL_INSTANCES_COLLECTION, Filter.of(id=eval_instance_id), count):
        raise LookupError(f"no eval instance {eval_instance_id}")
    written = compare_and_swap(docs, EVAL_SLOTS_COLLECTION, key, change, counter_field="rev")
    if written is None and (written := docs.find_one(EVAL_SLOTS_COLLECTION, key)) is None:
        raise LookupError(f"eval instance {eval_instance_id} has no slot {slot_id}")
    docs.update(EVAL_INSTANCES_COLLECTION, Filter.of(id=eval_instance_id), count)
    return Slot.from_dict(written)
