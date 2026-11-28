"""Eval instances and their slots in the configured document store: ``eval_instances`` holds one record per eval run,
``eval_slots`` one document per slot, keyed by its eval instance, so no document grows with the number of slots.

An eval instance's own document is what makes its slots count: creating one writes the slots first and the instance
last, so a creation stopped partway leaves only slots that nothing reads, and running it again finishes it. Only the
process driving an eval instance writes it after that. A slot write first counts itself in the eval instance's
``slots_rev`` and then lands, so a summary stamped with an older ``slots_rev`` is known to be stale even when the
driver stopped between the two.

The per-run secrets the run index strips from a run's overrides (``task_step.context._REDACTED_KEYS``) are stripped
here too, so a resume reads them from its own environment and config again. An ``@local`` or ``@hf`` eval's records
stay in this machine's local store, since an eval instance's id and each slot's ``eval_instance_id`` are rooted in the
eval's id, which places them there.
"""

from __future__ import annotations

from typing import Iterable, Optional

from agent_env.config import get_config
from agent_env.eval.instance import Attempt, EvalInstance, Slot, SlotState
from agent_env.store import DuplicateKeyError, Filter, Sort, UpdateSpec, compare_and_swap
from agent_env.task_step.context import _strip_redacted_keys

EVAL_INSTANCES_COLLECTION = "eval_instances"
EVAL_SLOTS_COLLECTION = "eval_slots"

_indexed = None


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
    landed unanswered; an instance or slot already stored with other contents is refused with DuplicateKeyError."""
    slots = list(slots)
    if strays := sorted({slot.eval_instance_id for slot in slots} - {instance.id}):
        raise ValueError(f"slots of {', '.join(strays)} given for eval instance {instance.id}")
    docs = _docs()
    for slot in slots:
        _insert_once(docs, EVAL_SLOTS_COLLECTION, Filter.of(eval_instance_id=instance.id, id=slot.id), slot.to_dict())
    _insert_once(docs, EVAL_INSTANCES_COLLECTION, Filter.of(id=instance.id), instance.to_dict())


def _insert_once(docs, collection: str, key: Filter, doc: dict) -> None:
    doc = _strip_redacted_keys(doc)
    try:
        docs.insert(collection, doc)
    except DuplicateKeyError:
        if docs.find_one(collection, key) != doc:
            raise


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
    _recorded(eval_instance_id)
    docs = _docs().query(EVAL_SLOTS_COLLECTION, Filter.of(eval_instance_id=eval_instance_id))
    return sorted((Slot.from_dict(doc) for doc in docs), key=lambda slot: (slot.replica, slot.task_id, slot.task_version))


def get_slot(eval_instance_id: str, slot_id: str) -> Optional[Slot]:
    _recorded(eval_instance_id)
    doc = _docs().find_one(EVAL_SLOTS_COLLECTION, Filter.of(eval_instance_id=eval_instance_id, id=slot_id))
    return Slot.from_dict(doc) if doc else None


def _recorded(eval_instance_id: str) -> None:
    """Refuse an eval instance that isn't recorded, so no reader takes up the slots a stopped creation left behind."""
    if _docs().find_one(EVAL_INSTANCES_COLLECTION, Filter.of(id=eval_instance_id)) is None:
        raise LookupError(f"no eval instance {eval_instance_id}")


def append_attempt(eval_instance_id: str, slot_id: str, attempt: Attempt) -> Slot:
    """Add ``attempt`` after the slot's earlier ones, which it never replaces, and mark the slot running."""
    return _write_slot(eval_instance_id, slot_id, UpdateSpec(
        push={"attempts": [attempt.to_dict()]}, set={"state": str(SlotState.RUNNING)}))


def record_slot(eval_instance_id: str, slot_id: str, state: SlotState, result: Optional[dict] = None) -> Slot:
    """Set where the slot stands and its result, its last attempt's."""
    return _write_slot(eval_instance_id, slot_id, UpdateSpec(
        set={"state": str(SlotState(state)), "result": _strip_redacted_keys(result)}))


def _write_slot(eval_instance_id: str, slot_id: str, spec: UpdateSpec) -> Slot:
    docs = _docs()
    if not docs.update(EVAL_INSTANCES_COLLECTION, Filter.of(id=eval_instance_id), UpdateSpec(inc={"slots_rev": 1})):
        raise LookupError(f"no eval instance {eval_instance_id}")
    written = compare_and_swap(
        docs, EVAL_SLOTS_COLLECTION, Filter.of(eval_instance_id=eval_instance_id, id=slot_id),
        lambda _: UpdateSpec(set=dict(spec.set), push=dict(spec.push)), counter_field="rev",
    )
    if written is None:
        raise LookupError(f"eval instance {eval_instance_id} has no slot {slot_id}")
    return Slot.from_dict(written)
