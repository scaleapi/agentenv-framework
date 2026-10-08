"""Task store for MongoDB persistence."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import logging
import random
import string
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import TYPE_CHECKING, Callable, Optional, Self

from agent_env.store.base import NotFoundError
from agent_env.config import get_config
from agent_env.store.document_store import DocumentStore, DuplicateKeyError, Filter, Lte, Ne, Sort, SortKey, UpdateSpec, VersionedEntityStore, VersionedEntityStoreCache, compare_and_swap
from agent_env.store.query import QueryBuilder, to_document_query
from agent_env.task.step_journal import (
    _RESERVED_STEP_IDS,
    _SCHEDULER_STEP_ID,
    _SEED_STEP_ID,
    _apply_forward,
    _get_path,
    _union,
    commit_ordered,
    replay_context,
)
from agent_env.task_step.context import TaskStepContext

if TYPE_CHECKING:
    from agent_env.task.task import Task
    from agent_env.task_step.context_ops import ContextUpdateOps

logger = logging.getLogger(__name__)

TASKS_COLLECTION = "tasks"
TASK_INSTANCES_COLLECTION = "task_instances"
TASK_STEP_JOURNAL_COLLECTION = "task_step_journal"
_REV = "rev"
_JOURNAL_SEQ = "journal_seq"
# Steps whose latest completion was a re-record, decided inside the CAS so never a stale read.
_RERECORDED = "rerecorded_steps"
# Which run the journal entries belong to; a re-register bumps it, retiring them without a delete.
_GENERATION = "run_generation"
class TaskQuery(QueryBuilder["Task"]):
    def __init__(self, store: Optional["TaskStore"] = None) -> None:
        super().__init__()
        self._store = store

    def _clone(self) -> Self:
        clone = TaskQuery(self._store)
        clone._filters = self._filters.copy()
        clone._sort_field = self._sort_field
        clone._sort_desc = self._sort_desc
        clone._limit_value = self._limit_value
        clone._offset_value = self._offset_value
        return clone

    def type(self, task_type: str) -> Self:
        return self._add_filter("type", task_type)

    def execute(self) -> list[Task]:
        if self._store is None:
            self._store = get_task_store()
        return self._store.execute_query(self)

    def _execute_count(self) -> int:
        if self._store is None:
            self._store = get_task_store()
        return self._store.execute_count(self)


class TaskStore:
    def __init__(self) -> None:
        self._versioned_cache: VersionedEntityStoreCache[Task] = VersionedEntityStoreCache(
            TASKS_COLLECTION, self._serialize, self._deserialize, secondary_indexes=[["type"]],
        )

    @property
    def _doc_store(self):
        return get_config().get_document_store()

    @property
    def _versioned(self) -> VersionedEntityStore[Task]:
        return self._versioned_cache.for_store(self._doc_store)

    def _serialize(self, task: Task) -> dict:
        doc = task.to_dict()
        doc["created_at_utc"] = datetime.now(timezone.utc)
        return doc

    def get(self, id: str, version: Optional[int] = None) -> Task:
        task = self._versioned.get(id, version)
        if task is None:
            version_str = f" version={version}" if version is not None else ""
            raise NotFoundError(f"Task {id}{version_str} not found")
        return task

    def next_version(self, id: str) -> int:
        return self._versioned.next_version(id)

    def put_document(self, task: Task) -> Task:
        task.version = self._versioned.put(task)
        return task

    def _deserialize(self, doc: dict) -> Task:
        from agent_env.task.registry import get_task_registry

        doc.pop("_id", None)
        doc.pop("created_at_utc", None)
        task_type = doc["type"]
        registry = get_task_registry()
        cls = registry.get(task_type)
        if cls is None:
            raise ValueError(f"Unknown task type: {task_type}")
        return cls.from_dict(doc)

    def execute_query(self, query: TaskQuery) -> list[Task]:
        filt, sort = to_document_query(query)
        return self._versioned.query(filt, sort, query._limit_value, query._offset_value)

    def execute_count(self, query: TaskQuery) -> int:
        filt, _ = to_document_query(query)
        return self._versioned.count(filt)


_task_store: Optional[TaskStore] = None


def get_task_store() -> TaskStore:
    global _task_store
    if _task_store is None:
        _task_store = TaskStore()
    return _task_store


def set_task_store(store: TaskStore) -> None:
    global _task_store
    _task_store = store


def reset_task_store() -> None:
    global _task_store
    _task_store = None


# --- TaskInstance store (non-versioned, tracks task run instances) ---


class TaskStepStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass
class TaskStepResult:
    step_id: str
    status: TaskStepStatus

    @classmethod
    def from_dict(cls, data: dict) -> TaskStepResult:
        return cls(step_id=data["step_id"], status=TaskStepStatus(data["status"]))


@dataclass
class TaskInstance:
    instance_id: str
    task_id: str
    task_version: int
    status: str
    current_step: int
    total_steps: int
    context: dict | None = None
    error: str | None = None
    created_at_utc: str | None = None
    completed_at_utc: str | None = None
    rev: int = 0
    completed_steps: list[TaskStepResult] = field(default_factory=list)
    step_attempt_failures: list[dict] = field(default_factory=list)
    # Bumped by ``undo_steps_sync(bump_epoch=True)`` on each retry. A completion write
    # carries the epoch it was dispatched under and no-ops when older than the stored one,
    # fencing a writer from a rolled-back attempt. Missing (pre-field docs) reads as 0.
    attempt_epoch: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> TaskInstance:
        return cls(
            instance_id=data["instance_id"],
            task_id=data["task_id"],
            task_version=data["task_version"],
            status=data["status"],
            current_step=data["current_step"],
            total_steps=data["total_steps"],
            context=data.get("context"),
            error=data.get("error"),
            created_at_utc=data.get("created_at_utc"),
            completed_at_utc=data.get("completed_at_utc"),
            rev=data.get("rev", 0),
            completed_steps=[TaskStepResult.from_dict(e) for e in data.get("completed_steps", [])],
            step_attempt_failures=data.get("step_attempt_failures", []),
            attempt_epoch=data.get("attempt_epoch", 0),
        )


@dataclass
class StepAttemptFailure:
    """Recorded once per failed step attempt, so a later retry can't clobber the earlier,
    more useful failure messages."""

    attempt: int
    step_id: str | None = None
    step_type: str | None = None
    error_class: str | None = None
    error_message: str | None = None
    duration_s: float | None = None
    occurred_at_utc: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _fresh_run_state(start_step: int, completed_step_docs: list[dict]) -> dict:
    """The fields a run owns. A re-register resets exactly these, so both registration paths
    have to agree on them; ``rev`` and the generation are bumped there instead and live in
    ``_new_instance_doc``."""
    return {
        "context": dataclasses.asdict(TaskStepContext()),
        "completed_steps": completed_step_docs,
        "status": "running",
        "error": None,
        "completed_at_utc": None,
        "current_step": start_step,
        # Present from birth: undo reads this as the authoritative record, so *absent* has to
        # mean "predates the journal", not "no re-records yet".
        _RERECORDED: [],
        "attempt_epoch": 0,
    }


def _new_instance_doc(
    instance_id: str, task_id: str, task_version: int, total_steps: int,
    created_at_utc: str, run_state: dict,
) -> dict:
    """A brand-new instance document: identity, the counters a re-register bumps, run state."""
    return {
        "instance_id": instance_id,
        "task_id": task_id,
        "task_version": task_version,
        "total_steps": total_steps,
        "created_at_utc": created_at_utc,
        "rev": 0,
        _GENERATION: 0,
        **run_state,
    }


# A run records only the minute it started, so the instance id orders runs of one minute and pages never overlap.
# Both keys descend so Mongo reads the sort from the (task_id, created_at_utc, instance_id) index scanned backward;
# a mixed-direction sort matches no index and sorts every run's whole document in memory.
_NEWEST_FIRST = Sort((SortKey("created_at_utc", descending=True), SortKey("instance_id", descending=True)))


def _task_filter(task_id: str, task_version: int | None) -> Filter:
    return Filter.of(task_id=task_id) if task_version is None else Filter.of(task_id=task_id, task_version=task_version)


class TaskInstanceStore:
    def __init__(self) -> None:
        self._indexed = None

    @property
    def _doc_store(self):
        # Resolved per call: a cached store outlives reset_config(), so a process that
        # re-pointed would read the new config and write the old backend. An operation that
        # writes more than once resolves here once and passes ``store`` down, so a reset
        # between its writes cannot leave half of it in each backend.
        store = get_config().get_document_store()
        if self._indexed is not store:
            store.ensure_index(TASK_INSTANCES_COLLECTION, ["instance_id"], unique=True)
            store.ensure_index(TASK_INSTANCES_COLLECTION, ["task_id"])
            store.ensure_index(TASK_INSTANCES_COLLECTION, ["task_id", "created_at_utc", "instance_id"])
            store.ensure_index(TASK_INSTANCES_COLLECTION, ["task_id", "task_version", "created_at_utc", "instance_id"])
            store.ensure_index(TASK_STEP_JOURNAL_COLLECTION, ["instance_id", "step_id"], unique=True)
            store.ensure_index(TASK_STEP_JOURNAL_COLLECTION, ["instance_id", "seq"])
            self._indexed = store
        return store

    def create_instance(
        self,
        task_id: str,
        task_version: int,
        total_steps: int,
        start_step: int = 0,
        completed_steps: list[TaskStepResult] | None = None,
    ) -> TaskInstance:
        completed_step_docs = [
            dataclasses.asdict(e) for e in completed_steps or []
        ]
        suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        instance_id = f"{task_id}-{suffix}"
        created_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        doc = _new_instance_doc(
            instance_id, task_id, task_version, total_steps, created_at_utc,
            _fresh_run_state(start_step, completed_step_docs),
        )
        store = self._doc_store
        store.insert(TASK_INSTANCES_COLLECTION, doc)
        self._journal_seeded_completions(store, instance_id, completed_step_docs)
        return TaskInstance.from_dict(doc)

    def upsert_instance(
        self,
        instance_id: str,
        task_id: str,
        task_version: int,
        total_steps: int,
        start_step: int = 0,
        completed_steps: list[TaskStepResult] | None = None,
    ) -> TaskInstance | None:
        """Idempotent register on a caller-supplied ``instance_id``."""
        current_time_str = datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
        completed_step_docs = [
            dataclasses.asdict(e) for e in completed_steps or []
        ]
        run_state = _fresh_run_state(start_step, completed_step_docs)
        fresh_doc = _new_instance_doc(
            instance_id, task_id, task_version, total_steps, current_time_str, run_state,
        )
        store = self._doc_store
        try:
            store.insert(TASK_INSTANCES_COLLECTION, fresh_doc)
            self._journal_seeded_completions(store, instance_id, completed_step_docs)
            return TaskInstance.from_dict(fresh_doc)
        except DuplicateKeyError:
            # atomic inc (not read-then-set rev): stays monotonic under concurrent re-registers + racing step CAS.
            # The same write bumps the generation, retiring the old journal without a delete that
            # could race a completion still landing.
            updated = store.update_one_and_get(
                TASK_INSTANCES_COLLECTION,
                Filter.of(instance_id=instance_id),
                UpdateSpec(set=dict(run_state), inc={_REV: 1, _GENERATION: 1}),
                return_after=True,
            )
            if updated is None:
                raise RuntimeError(
                    f"register_task_instance: instance {instance_id!r} "
                    "disappeared during re-register"
                )
            self._journal_seeded_completions(store, instance_id, completed_step_docs)
            return TaskInstance.from_dict(updated)

    def file_artifact_universe_ids_for_batch(self, batch_id: str) -> list[str]:
        """Return FileArtifactUniverse ids produced by task instances in a batch.

        ``batch_id`` is a batch-runner concept stored on the task instance
        (``context.metadata.batch_id``). The ``collect_artifacts`` step writes
        a forward pointer (``context.metadata.file_artifact_universe.id``) on
        the instance, so aggregating those ids over every instance matching
        the batch gives the set of universes that batch produced.

        De-duplicates while preserving first-appearance order: instances by creation time (to the
        minute), then by instance id. The order is imposed here, since a document store returns a
        query's matches in no particular order.
        """
        docs = self._doc_store.query(
            TASK_INSTANCES_COLLECTION, Filter.of(**{"context.metadata.batch_id": batch_id})
        )
        seen: set[str] = set()
        ids: list[str] = []
        for doc in sorted(docs, key=lambda d: (d.get("created_at_utc") or "", d.get("instance_id") or "")):
            uid = _get_path(doc, "context.metadata.file_artifact_universe.id")
            if uid and uid not in seen:
                seen.add(uid)
                ids.append(uid)
        return ids

    def get(self, instance_id: str) -> TaskInstance:
        doc = self._doc_store.find_one(TASK_INSTANCES_COLLECTION, Filter.of(instance_id=instance_id))
        if not doc:
            raise NotFoundError(f"TaskInstance '{instance_id}' not found")
        return TaskInstance.from_dict(doc)

    def find(
        self, task_id: str, *, task_version: int | None = None, limit: int | None = None, offset: int = 0,
    ) -> list[TaskInstance]:
        docs = self._doc_store.query(
            TASK_INSTANCES_COLLECTION, _task_filter(task_id, task_version),
            sort=_NEWEST_FIRST, limit=limit, offset=offset,
        )
        return [TaskInstance.from_dict(doc) for doc in docs]

    def count(self, task_id: str, *, task_version: int | None = None) -> int:
        return self._doc_store.count(TASK_INSTANCES_COLLECTION, _task_filter(task_id, task_version))

    def seed_context(self, instance_id: str, ops: "ContextUpdateOps") -> None:
        """Apply the run's initial context, then journal it as the replay base (even when empty)."""
        store = self._doc_store
        # One entry per deployment, as completions and the replay of this seed write them.
        ops = dataclasses.replace(ops, add_to_sets={p: _union([], items, path=p) for p, items in ops.add_to_sets.items()})
        if not ops.is_empty():
            store.update(
                TASK_INSTANCES_COLLECTION, Filter.of(instance_id=instance_id), ops.to_update_spec()
            )
        # Doc first: a failed journal write (logged by the caller) must not cost the run its seed.
        self._journal_step(store, instance_id, _SEED_STEP_ID, ops, "seed")

    def _bump_journal_seq(self, store: DocumentStore, instance_id: str) -> dict | None:
        """Atomically bump the instance's journal sequence; the post-update doc, or None if the
        instance is missing."""
        return store.update_one_and_get(
            TASK_INSTANCES_COLLECTION, Filter.of(instance_id=instance_id),
            UpdateSpec(inc={_JOURNAL_SEQ: 1}),
        )

    def _journal_step(
        self, store: DocumentStore, instance_id: str, step_id: str, ops: "ContextUpdateOps", status: str,
        attempt_epoch: int | None = None,
    ) -> tuple[int, int] | None:
        """Record a step's diff, replacing any earlier entry for it under a fresh ``seq``.
        Returns ``(seq, generation)``, or None if the instance is missing, the write is
        already fenced by ``attempt_epoch``, or a later attempt already owns the row: a
        completion that will not land must not displace the entry the re-run committed."""
        doc = self._bump_journal_seq(store, instance_id)
        if doc is None:
            return None
        if attempt_epoch is not None and doc.get("attempt_epoch", 0) > attempt_epoch:
            return None
        seq = doc[_JOURNAL_SEQ]
        generation = doc.get(_GENERATION, 0)
        row = {
            "instance_id": instance_id,
            "step_id": step_id,
            "seq": seq,
            "generation": generation,
            "status": status,
            "recorded_at": datetime.now(timezone.utc),  # a real datetime, so a retention index could key on it later
            "ops": ops.to_journal_dict(),
        }
        if not self._write_journal_row(store, instance_id, step_id, row):
            return None
        return seq, generation

    def _write_journal_row(self, store: DocumentStore, instance_id: str, step_id: str, row: dict) -> bool:
        """Land ``row`` for the step unless a row with a higher ``seq`` already exists.

        The seq is taken before the epoch fence and the completion CAS, so a writer that
        passed the fence and then paused always carries a lower seq than any writer that
        started after the undo. Never going backwards in seq is what keeps that writer from
        displacing the re-run's entry between the fence and its own (declined) CAS; the fence
        and this write cannot be one atomic operation across two collections. Returns
        whether the row landed."""
        key = Filter.of(instance_id=instance_id, step_id=step_id)
        older = key.where("seq", Lte(row["seq"] - 1))
        for _ in range(5):
            if store.replace(TASK_STEP_JOURNAL_COLLECTION, older, row):
                return True
            existing = store.find_one(TASK_STEP_JOURNAL_COLLECTION, key)
            if existing is None:
                try:
                    store.insert(TASK_STEP_JOURNAL_COLLECTION, row)
                    return True
                except DuplicateKeyError:
                    continue  # lost the first-insert race; compare seqs on the next pass
            if existing.get("seq", 0) >= row["seq"]:
                return False  # a later attempt owns the row
            # An older row landed between the replace and the read; replace it on the next pass.
        raise RuntimeError(
            f"journal row for step {step_id!r} on instance {instance_id!r} kept changing under us"
        )

    def journal_step_sync(self, instance_id: str, step_id: str, ops: "ContextUpdateOps", status: str) -> int | None:
        """``_journal_step`` on the current backend, for callers that only need the seq."""
        journaled = self._journal_step(self._doc_store, instance_id, step_id, ops, status)
        return None if journaled is None else journaled[0]

    def _journal_seeded_completions(
        self, store: DocumentStore, instance_id: str, completed_step_docs: list[dict],
    ) -> None:
        """Journal an empty entry per completion carried in from a previous attempt, so a
        completion with no entry means exactly one thing: its journal write failed. Empty is
        the truthful diff — the seed already holds whatever those steps wrote."""
        from agent_env.task_step.context_ops import ContextUpdateOps

        for c in completed_step_docs:
            try:
                self._journal_step(store, instance_id, c["step_id"], ContextUpdateOps(), c["status"])
            except Exception:
                logger.warning("Failed to journal seeded step %s for instance_id=%s",
                               c["step_id"], instance_id, exc_info=True)

    def journal_entries_sync(self, instance_id: str, generation: int | None = None) -> list[dict]:
        """The instance's entries for one run generation, in seq order; a previous run's are
        retired and not returned. ``generation`` defaults to the instance's current one."""
        return self._journal_entries(self._doc_store, instance_id, generation)

    def _journal_entries(
        self, store: DocumentStore, instance_id: str, generation: int | None,
    ) -> list[dict]:
        if generation is None:
            instance = store.find_one(TASK_INSTANCES_COLLECTION, Filter.of(instance_id=instance_id))
            generation = 0 if instance is None else instance.get(_GENERATION, 0)
        entries = store.query(
            TASK_STEP_JOURNAL_COLLECTION, Filter.of(instance_id=instance_id),
            sort=Sort.by("seq", descending=False),
        )
        return [e for e in entries if e.get("generation", 0) == generation]

    def clear_journal_sync(self, instance_id: str) -> None:
        """Drop every journal entry for an instance, whatever its generation. Not on the
        re-register path — a delete racing a live completion is exactly what generations
        exist to avoid — this is for an explicit teardown."""
        store = self._doc_store
        entries = store.query(
            TASK_STEP_JOURNAL_COLLECTION, Filter.of(instance_id=instance_id),
        )
        for e in entries:
            store.delete(
                TASK_STEP_JOURNAL_COLLECTION, Filter.of(instance_id=instance_id, step_id=e["step_id"]),
            )

    def undo_steps_sync(
        self,
        instance_id: str,
        step_ids: set[str],
        *,
        extra_ops: "Callable[[dict, dict], ContextUpdateOps] | None" = None,
        bump_epoch: bool = False,
    ) -> dict | None:
        """Roll the named steps out by replaying the surviving committed entries over the seed,
        in one CAS on ``rev``, then delete those entries at the seq it read. None if the
        instance is missing. Refuses a missing seed or an unjournaled survivor; callers
        regraft secrets into the returned doc.

        ``extra_ops(before_context, after_context)`` lets the caller write its own record in
        the same CAS (the scheduler's retry audit): it sees the context as stored and as
        replayed, returns forward ops, and those are applied on top of the replay and then
        journaled under ``__scheduler__``, which replays right after the seed and is never
        undone. ``bump_epoch`` advances ``attempt_epoch`` in the same write, so a completion
        still in flight from the undone attempt is fenced (``record_step_complete_sync``).
        """
        step_ids = set(step_ids) - _RESERVED_STEP_IDS
        undone_seqs: dict[str, int] = {}
        applied_extra: list["ContextUpdateOps"] = []
        store = self._doc_store

        def _mutate(doc: dict) -> UpdateSpec:
            entries = self._journal_entries(store, instance_id, doc.get(_GENERATION, 0))
            # The seq this attempt read, so the trim can't delete a row a re-completion wrote after the CAS.
            undone_seqs.clear()
            undone_seqs.update({e["step_id"]: e["seq"] for e in entries if e["step_id"] in step_ids})
            before = doc.get("completed_steps", [])
            completed_ids = {c.get("step_id") for c in before}
            if not any(e["step_id"] == _SEED_STEP_ID for e in entries):
                raise ValueError(f"instance {instance_id!r} has no journal seed (pre-journal or unseeded); cannot undo")
            journaled = {e["step_id"] for e in entries}
            unreplayable = sorted(
                c["step_id"] for c in before
                if c.get("status") != TaskStepStatus.FAILURE
                and c["step_id"] not in journaled and c["step_id"] not in step_ids
            )
            if unreplayable:
                raise ValueError(
                    f"steps {unreplayable} completed without a journal entry; their writes cannot be "
                    f"replayed, refusing to undo {sorted(step_ids)} on instance {instance_id!r}"
                )
            completed = [c for c in before if c.get("step_id") not in step_ids]
            keep = commit_ordered([e for e in entries if e["step_id"] not in step_ids], completed)
            # The instance's own flags, not the entries' pre-CAS ones (see ``replay_context``).
            # Every instance this code creates carries the field, so absent means the run
            # predates the journal: its entries' flags are pre-CAS guesses that a losing
            # writer can have left stale, and the winning CAS's verdict is unrecoverable.
            flags = doc.get(_RERECORDED)
            if flags is None:
                raise ValueError(
                    f"instance {instance_id!r} predates {_RERECORDED} (pre-journal run); its "
                    "entries carry only pre-CAS re-record flags, so replay order cannot be "
                    f"reconstructed — refusing to undo {sorted(step_ids)}"
                )
            rerecords = [s for s in flags if s not in step_ids]
            replayed = {"context": replay_context(keep, rerecords)}
            applied_extra.clear()
            if extra_ops is not None:
                # Copies: the callback must not be able to edit the CAS's working state.
                ops = extra_ops(copy.deepcopy(doc.get("context") or {}), copy.deepcopy(replayed["context"]))
                _apply_forward(replayed, ops)
                applied_extra.append(ops)
            fields: dict = {
                "context": replayed["context"],
                "completed_steps": completed,
                "current_step": len(completed),
                _RERECORDED: rerecords,
            }
            if bump_epoch:
                fields["attempt_epoch"] = doc.get("attempt_epoch", 0) + 1
            if len(completed) < len(before):
                undone_the_failure = any(
                    c.get("status") == TaskStepStatus.FAILURE
                    for c in before if c.get("step_id") in step_ids
                )
                if doc.get("status") != "failed" or undone_the_failure:
                    # A completion came off a finished run: it is running again until re-recorded.
                    fields.update(status="running", error=None, completed_at_utc=None)
                # Otherwise the step that failed the run is still recorded — undoing a
                # *different* step does not make the run healthy, so its error stands.
            return UpdateSpec(set=fields)

        result = compare_and_swap(
            store, TASK_INSTANCES_COLLECTION, Filter.of(instance_id=instance_id),
            _mutate, counter_field=_REV,
        )
        if result is None:
            return None
        for sid, seq in undone_seqs.items():
            store.delete(
                TASK_STEP_JOURNAL_COLLECTION,
                Filter.of(instance_id=instance_id, step_id=sid, seq=seq),
            )
        if applied_extra:
            # Doc first, like the seed: the record is in the context already; a failed journal
            # write only means replay lacks it until the next undo rewrites this row.
            try:
                self._journal_step(store, instance_id, _SCHEDULER_STEP_ID, applied_extra[0], "scheduler")
            except Exception:
                logger.warning("Failed to journal the scheduler record for instance_id=%s", instance_id, exc_info=True)
        return result

    def record_step_complete_sync(
        self,
        instance_id: str,
        entry: dict,
        ops: "ContextUpdateOps",
        total_steps: int,
        completed_at_utc: str,
        attempt_epoch: int = 0,
    ) -> None:
        """Journal the step's diff, then upsert the step into ``completed_steps`` (moved to the
        end) and derive ``current_step``/``status`` under an optimistic compare-and-swap on
        ``rev`` (retried).

        Journal first but fail open: a failed journal write is logged and the completion still
        lands, leaving a completion with no entry, which is what undo refuses to replay over.
        The CAS is bound to the generation captured while journaling, so a completion from a run
        a re-register superseded is declined rather than joining the fresh one.

        ``attempt_epoch`` is the epoch the step was dispatched under. A write older than the
        stored epoch is from an attempt a retry already rolled back; it neither journals nor
        lands, so it cannot resurrect rolled-back state. Cancellation does not interrupt an
        in-flight ``to_thread`` call, so this fence, not the cancel, stops the drained
        step's write.

        Replaces the old aggregation-pipeline update: the dedupe and the
        ``$size``/``$cond`` derivations run in Python, so any backend can implement it
        with a plain conditional update.
        """
        step_id = entry["step_id"]
        store = self._doc_store
        try:
            journaled = self._journal_step(
                store, instance_id, step_id, ops, entry["status"], attempt_epoch=attempt_epoch
            )
        except Exception:
            logger.warning(
                "Failed to journal step %s for instance_id=%s; completing without an undo record",
                step_id, instance_id, exc_info=True,
            )
            journaled = None
        seq, journaled_generation = journaled if journaled is not None else (None, None)

        def _mutate(doc: dict) -> UpdateSpec | None:
            if journaled_generation is not None and doc.get(_GENERATION, 0) != journaled_generation:
                logger.warning(
                    "Declining completion of step %s for instance_id=%s: the run was "
                    "re-registered (generation %s -> %s) after the step journaled",
                    step_id, instance_id, journaled_generation, doc.get(_GENERATION, 0),
                )
                return None
            if doc.get("attempt_epoch", 0) > attempt_epoch:
                return None
            # Position is the replay order, so a re-record moves to the end, items included.
            rerecorded = any(c.get("step_id") == step_id for c in doc.get("completed_steps", []))
            set_fields: dict = dict(ops.sets)
            for path, items in ops.add_to_sets.items():
                set_fields[path] = _union(_get_path(doc, path) or [], items, move_to_end=rerecorded, path=path)
            completed = [c for c in doc.get("completed_steps", []) if c.get("step_id") != step_id] + [entry]
            set_fields["completed_steps"] = completed
            set_fields["current_step"] = len(completed)
            # Computed from the doc this attempt won on, and recorded with the write it
            # describes: the only record of a re-record, and the one replay reads.
            rerecords = [s for s in doc.get(_RERECORDED, []) if s != step_id]
            if rerecorded:
                rerecords.append(step_id)
            set_fields[_RERECORDED] = rerecords
            if len(completed) >= total_steps and doc.get("status") != "failed":
                set_fields["status"] = "completed"
                set_fields["completed_at_utc"] = completed_at_utc
            return UpdateSpec(set=set_fields, unset=set(ops.unsets))

        result = compare_and_swap(
            store,
            TASK_INSTANCES_COLLECTION,
            Filter.of(instance_id=instance_id),
            _mutate,
            counter_field=_REV,
        )
        if result is None and journaled is not None:
            # The completion never landed, so the entry describes nothing in the document.
            # Delete at the exact seq, so a re-record that replaced the row keeps its own.
            store.delete(
                TASK_STEP_JOURNAL_COLLECTION,
                Filter.of(instance_id=instance_id, step_id=step_id, seq=seq),
            )

    def record_task_failure_sync(
        self,
        instance_id: str,
        failing_step_id: str | None,
        context: "TaskStepContext",
        error: BaseException,
        completed_at_utc: str,
    ) -> None:
        """Terminal failure write — one atomic update guarded on ``status != completed``
        (so a sibling step that finished mid-cancel isn't overwritten)."""
        spec = UpdateSpec(
            set={
                "status": "failed",
                "error": str(error),
                "context": context.to_safe_dict(),
                "completed_at_utc": completed_at_utc,
            },
            inc={"rev": 1},
        )
        if failing_step_id is not None:
            entry = dataclasses.asdict(TaskStepResult(step_id=failing_step_id, status=TaskStepStatus.FAILURE))
            spec.add_to_set = {"completed_steps": [entry]}
        self._doc_store.update(
            TASK_INSTANCES_COLLECTION,
            Filter.of(instance_id=instance_id).where("status", Ne("completed")),
            spec,
        )

    def record_task_cancelled_sync(self, instance_id: str, reason: str, completed_at_utc: str) -> None:
        """Mark a run cancelled, over the ``failed`` its unwinding recorded, but never a completed one."""
        self._doc_store.update(
            TASK_INSTANCES_COLLECTION,
            Filter.of(instance_id=instance_id).where("status", Ne("completed")),
            UpdateSpec(
                set={"status": "cancelled", "error": reason, "completed_at_utc": completed_at_utc},
                inc={"rev": 1},
            ),
        )

    def append_step_attempt_failure_sync(self, instance_id: str, failure: "StepAttemptFailure") -> None:
        """Best-effort append of one attempt's failure, designed not to conflict with the
        ``record_task_failure`` write. ``add_to_set`` creates the array on the first append and
        dedupes an identical re-issued write, so it's idempotent."""
        self._doc_store.update(
            TASK_INSTANCES_COLLECTION,
            Filter.of(instance_id=instance_id).where("status", Ne("completed")),
            UpdateSpec(add_to_set={"step_attempt_failures": [failure.to_dict()]}),
        )


_task_instance_store: Optional[TaskInstanceStore] = None


def get_task_instance_store() -> TaskInstanceStore:
    global _task_instance_store
    if _task_instance_store is None:
        _task_instance_store = TaskInstanceStore()
    return _task_instance_store


def set_task_instance_store(store: TaskInstanceStore) -> None:
    global _task_instance_store
    _task_instance_store = store


def reset_task_instance_store() -> None:
    global _task_instance_store
    _task_instance_store = None


def task_instances(
    task_id: str, *, task_version: int | None = None, limit: int | None = None, offset: int = 0,
) -> list[TaskInstance]:
    """A task's recorded runs, newest first, from the configured document store. A run keeps only the minute it
    started, so runs of one minute come in descending instance-id order, which keeps pages from overlapping."""
    return get_task_instance_store().find(task_id, task_version=task_version, limit=limit, offset=offset)


def count_task_instances(task_id: str, *, task_version: int | None = None) -> int:
    return get_task_instance_store().count(task_id, task_version=task_version)


def find_task_instance(instance_id: str) -> TaskInstance | None:
    """One recorded run by its instance id, or None when there is no such run."""
    try:
        return get_task_instance_store().get(instance_id)
    except NotFoundError:
        return None


def register_task_instance(
    task_id: str,
    task_version: int,
    total_steps: int,
    start_step: int = 0,
    instance_id: str | None = None,
    completed_steps: list[TaskStepResult] | None = None,
) -> TaskInstance | None:
    """Register a task_instance row.

    With ``instance_id``: idempotent upsert. Without: random-id insert.
    """
    try:
        store = get_task_instance_store()
        if instance_id is None:
            return store.create_instance(
                task_id, task_version, total_steps, start_step, completed_steps,
            )
        return store.upsert_instance(
            instance_id,
            task_id,
            task_version,
            total_steps,
            start_step,
            completed_steps,
        )
    except Exception:
        logger.warning("Failed to create task instance record", exc_info=True)
        return None


def seed_task_instance_context(instance_id: str, context: "TaskStepContext") -> None:
    """Apply the caller's initial context to an instance row created by
    ``register_task_instance``, as path-level Mongo updates.

    Must be called before any step's ``record_step_complete`` fires (steps
    haven't started yet at this point in ``Task.run()``).
    """
    from agent_env.task_step.context_ops import build_context_update_ops

    ops = build_context_update_ops(None, context)
    try:
        get_task_instance_store().seed_context(instance_id, ops)
    except Exception:
        logger.warning("Failed to seed task instance context", exc_info=True)


async def record_step_complete(
    instance_id: str,
    step_id: str,
    ops: "ContextUpdateOps",
    total_steps: int,
    completed_at_utc: str,
    status: TaskStepStatus = TaskStepStatus.SUCCESS,
    attempt_epoch: int = 0,
) -> None:
    """Union the step into ``completed_steps`` and derive ``current_step`` /
    ``status`` / ``completed_at_utc`` under an optimistic compare-and-swap on
    ``rev`` (read -> compute in Python -> conditional write, retried on
    contention). Concurrent finishers converge without losing each other's
    entries; ``completed_steps`` keeps first-appearance order. ``attempt_epoch``
    fences a write from a rolled-back attempt (see ``record_step_complete_sync``).
    """
    entry = dataclasses.asdict(TaskStepResult(step_id=step_id, status=status))
    try:
        await asyncio.to_thread(
            get_task_instance_store().record_step_complete_sync,
            instance_id, entry, ops, total_steps, completed_at_utc, attempt_epoch,
        )
    except Exception:
        logger.error(
            "Failed to record step complete for instance_id=%s step_id=%s",
            instance_id, step_id, exc_info=True,
        )


async def undo_steps(
    instance_id: str,
    step_ids: set[str],
    *,
    extra_ops: "Callable[[dict, dict], ContextUpdateOps] | None" = None,
    bump_epoch: bool = False,
) -> dict | None:
    """Async ``TaskInstanceStore.undo_steps_sync``; None if the instance is missing or the
    write failed (logged). A refusal propagates: the store is saying it cannot rebuild that
    context, and a caller that re-dispatched anyway would run the span on a dirty one."""
    try:
        return await asyncio.to_thread(
            get_task_instance_store().undo_steps_sync, instance_id, set(step_ids),
            extra_ops=extra_ops, bump_epoch=bump_epoch,
        )
    except ValueError:
        raise
    except Exception:
        logger.error("Failed to undo steps %s for instance_id=%s", sorted(step_ids), instance_id, exc_info=True)
        return None


_FAILURE_WRITE_MAX_ATTEMPTS = 3
_FAILURE_WRITE_BACKOFF_SECONDS = 0.5


async def record_task_failure(
    instance_id: str,
    failing_step_id: str | None,
    context: "TaskStepContext",
    error: BaseException,
    completed_at_utc: str,
) -> None:
    """Write terminal failure state.

    Filter excludes already-completed instances: a sibling step's
    `record_step_complete` may land between the scheduler's cancel and this
    write (cancellation doesn't interrupt an in-flight `asyncio.to_thread`
    call), and its CAS could have stamped `status="completed"`. The filter
    makes this write a no-op in that case rather than flipping a successful run
    to "failed".
    """
    store = get_task_instance_store()
    last_exc: BaseException | None = None
    for attempt in range(_FAILURE_WRITE_MAX_ATTEMPTS):
        try:
            await asyncio.to_thread(
                store.record_task_failure_sync,
                instance_id, failing_step_id, context, error, completed_at_utc,
            )
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt + 1 < _FAILURE_WRITE_MAX_ATTEMPTS:
                await asyncio.sleep(_FAILURE_WRITE_BACKOFF_SECONDS * (2**attempt))
    logger.warning(
        "Failed to record task failure after %d attempts; the task instance record "
        "will not show it",
        _FAILURE_WRITE_MAX_ATTEMPTS, exc_info=last_exc,
    )




def record_task_cancelled(instance_id: str, reason: str, completed_at_utc: str) -> None:
    """Write a cancelled run's terminal state, retried like ``record_task_failure``. It logs a write that keeps
    failing rather than raising. Synchronous: ``run_bundle`` calls it once its event loop has closed."""
    store = get_task_instance_store()
    last_exc: BaseException | None = None
    for attempt in range(_FAILURE_WRITE_MAX_ATTEMPTS):
        try:
            store.record_task_cancelled_sync(instance_id, reason, completed_at_utc)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt + 1 < _FAILURE_WRITE_MAX_ATTEMPTS:
                time.sleep(_FAILURE_WRITE_BACKOFF_SECONDS * (2**attempt))
    logger.warning(
        "Failed to record the cancellation of instance_id=%s after %d attempts",
        instance_id, _FAILURE_WRITE_MAX_ATTEMPTS, exc_info=last_exc,
    )

# Best-effort, diagnostic-only wrapper around ``append_step_attempt_failure_sync``: retries a
# transient write (like ``record_task_failure``), then logs and swallows — never raises.
async def append_step_attempt_failure(instance_id: str, failure: "StepAttemptFailure") -> None:
    store = get_task_instance_store()
    last_exc: BaseException | None = None
    for attempt in range(_FAILURE_WRITE_MAX_ATTEMPTS):
        try:
            await asyncio.to_thread(store.append_step_attempt_failure_sync, instance_id, failure)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt + 1 < _FAILURE_WRITE_MAX_ATTEMPTS:
                await asyncio.sleep(_FAILURE_WRITE_BACKOFF_SECONDS * (2**attempt))
    logger.warning(
        "Failed to append attempt-failure ledger entry for instance_id=%s after %d attempts",
        instance_id, _FAILURE_WRITE_MAX_ATTEMPTS, exc_info=last_exc,
    )
