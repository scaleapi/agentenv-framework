"""The run index every ``[runner]`` writes: ``insert_run`` at submit, ``set_running`` when the run
starts, ``mark_terminal`` when it ends. The explorer lists, counts and groups runs from here and never
asks the runner, so a run missing here is invisible and one never marked terminal stays in flight.

It is an index, not a work queue: there is no lease or claim here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from agent_env.config import get_config
from agent_env.runner.runner import RunRecord, RunStatus
from agent_env.store import Eq, Filter, In, Sort, UpdateSpec
from agent_env.task_step.context import _strip_redacted_keys

RUNS_COLLECTION = "runs"

# Terminal states are never overwritten: mark_terminal/set_running only act on a run that
# is still QUEUED or RUNNING, so a completion racing a cancel can't resurrect a CANCELED run.
_NON_TERMINAL = [In([str(RunStatus.QUEUED), str(RunStatus.RUNNING)])]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_indexed = None


def _docs():
    """The configured document store, with the indexes backing the two reads (status-by-run-id and the per-task
    listing) made once per store: a caller that never ``start()``s a runner still gets them."""
    global _indexed
    docs = get_config().get_document_store()
    if _indexed is not docs:
        docs.ensure_index(RUNS_COLLECTION, ["run_id"], unique=True)
        docs.ensure_index(RUNS_COLLECTION, ["task_id", "created_at_utc"])
        _indexed = docs
    return docs


def ensure_indexes() -> None:
    """Make the run index's indexes now rather than on first use."""
    _docs()


def insert_run(record: RunRecord) -> RunRecord:
    record.created_at_utc = record.created_at_utc or _now()
    doc = record.to_dict()
    doc["overrides"] = _strip_redacted_keys(doc["overrides"])
    _docs().insert(RUNS_COLLECTION, doc)
    return record


def get_run(run_id: str) -> Optional[RunRecord]:
    doc = _docs().find_one(RUNS_COLLECTION, Filter.of(run_id=run_id))
    return RunRecord.from_dict(doc) if doc else None


def list_runs(
    task_id: Optional[str] = None, limit: int = 50, offset: int = 0
) -> list[RunRecord]:
    filt = Filter.of(task_id=task_id) if task_id else Filter()
    docs = _docs().query(
        RUNS_COLLECTION, filt, sort=Sort.by("created_at_utc", descending=True),
        limit=limit, offset=offset,
    )
    return [RunRecord.from_dict(d) for d in docs]


def count_runs(task_id: Optional[str] = None) -> int:
    return _docs().count(RUNS_COLLECTION, Filter.of(task_id=task_id) if task_id else Filter())


def active_runs(runner: str, limit: int = 100_000) -> list[RunRecord]:
    """Every non-terminal (QUEUED/RUNNING) run for ``runner`` — used at startup to reconcile
    orphans a dead process left behind, no matter how many terminal runs precede them."""
    docs = _docs().query(
        RUNS_COLLECTION,
        Filter(conditions={"runner": [Eq(runner)], "status": _NON_TERMINAL}),
        limit=limit,
    )
    return [RunRecord.from_dict(d) for d in docs]


def set_running(run_id: str) -> None:
    """QUEUED -> RUNNING once a worker slot frees. No-op if the run is no longer QUEUED
    (e.g. it was canceled while waiting for the slot)."""
    _docs().update(
        RUNS_COLLECTION,
        Filter(conditions={"run_id": [Eq(run_id)], "status": [Eq(str(RunStatus.QUEUED))]}),
        UpdateSpec(set={"status": str(RunStatus.RUNNING), "started_at_utc": _now()}),
    )


def see(run_id: str) -> None:
    """Mark a non-terminal run as still seen by its owner: what keeps an owner on another host counting as alive."""
    _docs().update(
        RUNS_COLLECTION,
        Filter(conditions={"run_id": [Eq(run_id)], "status": _NON_TERMINAL}),
        UpdateSpec(set={"owner.seen": _now()}),
    )


def mark_terminal(run_id: str, status: RunStatus, error: Optional[str] = None) -> None:
    """Move a run to a terminal status, but only from QUEUED/RUNNING — so a run that a
    concurrent cancel already terminated is left as-is (cancel wins, and it's idempotent)."""
    _docs().update(
        RUNS_COLLECTION,
        Filter(conditions={"run_id": [Eq(run_id)], "status": _NON_TERMINAL}),
        UpdateSpec(set={
            "status": str(status),
            "finished_at_utc": _now(),
            "error": error,
        }),
    )
