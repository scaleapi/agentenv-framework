"""An eval instance: one run of an eval, recorded before its first submit so that it can be shown, resumed, regraded
and compared. Its slots, one per task and replica, are documents of their own, each listing the attempts (task runs)
that filled it in order; a slot's result is its last attempt's."""

from __future__ import annotations

import secrets
import string
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Optional


class EvalInstanceStatus(StrEnum):
    PLANNING = "planning"
    RUNNING = "running"
    FINISHED = "finished"
    STOPPED_BUDGET = "stopped_budget"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


class SlotState(StrEnum):
    """Where a slot stands. An ``ungraded`` slot's result says why: a grader failure, never retried, or an
    infrastructure failure that outlasted its retries."""

    PLANNED = "planned"
    RUNNING = "running"
    GRADED = "graded"
    UNGRADED = "ungraded"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    NOT_RUN_BUDGET = "not_run_budget"


class SourceKind(StrEnum):
    EVAL = "eval"
    BUNDLE = "bundle"
    TASKS = "tasks"


def new_eval_instance_id(root_id: str) -> str:
    """A fresh id for a run of the eval ``root_id``, derived from it, as a task run's is, so the record keeps the eval's
    namespace."""
    return f"{root_id}-{''.join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8))}"


def slot_id(task_id: str, task_version: int, replica: int) -> str:
    """``task_id@version#replica``, the replica counted from 1."""
    if replica < 1:
        raise ValueError(f"a slot's replica counts from 1, got {replica}")
    return f"{task_id}@{task_version}#{replica}"


@dataclass
class EvalSource:
    """What the eval instance runs: a stored eval, a bundle's eval, or tasks named on their own."""

    kind: SourceKind
    eval_id: Optional[str] = None
    eval_version: Optional[int] = None
    ref: Optional[str] = None
    content_hash: Optional[str] = None
    tasks: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.kind = SourceKind(self.kind)

    def to_dict(self) -> dict:
        return {"kind": str(self.kind), "eval_id": self.eval_id, "eval_version": self.eval_version, "ref": self.ref,
                "content_hash": self.content_hash, "tasks": self.tasks}

    @classmethod
    def from_dict(cls, data: dict) -> EvalSource:
        return cls(kind=data["kind"], eval_id=data.get("eval_id"), eval_version=data.get("eval_version"),
                   ref=data.get("ref"), content_hash=data.get("content_hash"), tasks=data.get("tasks") or [])


@dataclass
class EvalSettings:
    """What the eval instance was asked to do, kept so a resume or rerun does it again: ``overrides`` are the run
    overrides each task run gets (models, judge samples and the like)."""

    k: int = 1
    max_concurrency: Optional[int] = None
    max_cost_usd: Optional[float] = None
    min_passing_score: Optional[float] = None
    grade: Optional[str] = None
    fail_under: Optional[float] = None
    overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"k": self.k, "max_concurrency": self.max_concurrency, "max_cost_usd": self.max_cost_usd,
                "min_passing_score": self.min_passing_score, "grade": self.grade, "fail_under": self.fail_under,
                "overrides": self.overrides}

    @classmethod
    def from_dict(cls, data: dict) -> EvalSettings:
        return cls(k=data.get("k", 1), max_concurrency=data.get("max_concurrency"),
                   max_cost_usd=data.get("max_cost_usd"), min_passing_score=data.get("min_passing_score"),
                   grade=data.get("grade"), fail_under=data.get("fail_under"), overrides=data.get("overrides") or {})


@dataclass
class RunnerDeclarations:
    """The runner an eval instance submits through, with what it declared when the instance started."""

    type: str
    runs_outlive_submitter: bool = False
    retries_infra: bool = False

    @classmethod
    def of(cls, runner) -> RunnerDeclarations:
        return cls(type=runner.type, runs_outlive_submitter=runner.runs_outlive_submitter,
                   retries_infra=runner.retries_infra)

    def to_dict(self) -> dict:
        return {"type": self.type, "runs_outlive_submitter": self.runs_outlive_submitter,
                "retries_infra": self.retries_infra}

    @classmethod
    def from_dict(cls, data: dict) -> RunnerDeclarations:
        return cls(type=data["type"], runs_outlive_submitter=bool(data.get("runs_outlive_submitter")),
                   retries_infra=bool(data.get("retries_infra")))


@dataclass
class Lineage:
    """Where an eval instance comes from: a rerun or a regrade of another, and whether it was let run despite hard
    drift from the one it continues."""

    rerun_of: Optional[str] = None
    regrade_of: Optional[str] = None
    not_like_for_like: bool = False

    def to_dict(self) -> dict:
        return {"rerun_of": self.rerun_of, "regrade_of": self.regrade_of, "not_like_for_like": self.not_like_for_like}

    @classmethod
    def from_dict(cls, data: dict) -> Lineage:
        return cls(rerun_of=data.get("rerun_of"), regrade_of=data.get("regrade_of"),
                   not_like_for_like=bool(data.get("not_like_for_like")))


@dataclass
class EvalInstance:
    """The record of one eval run, without its slots, which ``agent_env.eval.instance_store`` keeps beside it.

    ``intent_lock`` is what every run was pinned to before the first submit, ``framework`` the versions that planned
    it, ``summary`` a cache of the slots' results stamped with the ``slots_rev`` it was computed at, and ``driver``
    the process driving it. ``slots_rev`` counts slot writes; ``rev`` counts writes to this record."""

    id: str
    source: EvalSource
    settings: EvalSettings
    runner: RunnerDeclarations
    created_at: str
    status: EvalInstanceStatus = EvalInstanceStatus.PLANNING
    intent_lock: dict[str, Any] = field(default_factory=dict)
    framework: dict[str, Any] = field(default_factory=dict)
    summary: Optional[dict[str, Any]] = None
    driver: Optional[dict[str, Any]] = None
    lineage: Lineage = field(default_factory=Lineage)
    created_by: Optional[str] = None
    invocation_id: Optional[str] = None
    slots_rev: int = 0
    rev: int = 0

    def __post_init__(self) -> None:
        self.status = EvalInstanceStatus(self.status)

    @property
    def eval_id(self) -> Optional[str]:
        return self.source.eval_id

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "eval_id": self.eval_id,
            "source": self.source.to_dict(),
            "settings": self.settings.to_dict(),
            "runner": self.runner.to_dict(),
            "intent_lock": self.intent_lock,
            "framework": self.framework,
            "status": str(self.status),
            "summary": self.summary,
            "driver": self.driver,
            "lineage": self.lineage.to_dict(),
            "created_at": self.created_at,
            "created_by": self.created_by,
            "invocation_id": self.invocation_id,
            "slots_rev": self.slots_rev,
            "rev": self.rev,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EvalInstance:
        return cls(
            id=data["id"],
            source=EvalSource.from_dict(data["source"]),
            settings=EvalSettings.from_dict(data.get("settings") or {}),
            runner=RunnerDeclarations.from_dict(data["runner"]),
            created_at=data["created_at"],
            status=data.get("status", EvalInstanceStatus.PLANNING),
            intent_lock=data.get("intent_lock") or {},
            framework=data.get("framework") or {},
            summary=data.get("summary"),
            driver=data.get("driver"),
            lineage=Lineage.from_dict(data.get("lineage") or {}),
            created_by=data.get("created_by"),
            invocation_id=data.get("invocation_id"),
            slots_rev=data.get("slots_rev", 0),
            rev=data.get("rev", 0),
        )


@dataclass
class Attempt:
    """One submission of a slot: one task run, with its own task instance."""

    run_id: str
    instance_id: str
    submitted_at: str

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "instance_id": self.instance_id, "submitted_at": self.submitted_at}

    @classmethod
    def from_dict(cls, data: dict) -> Attempt:
        return cls(run_id=data["run_id"], instance_id=data["instance_id"], submitted_at=data["submitted_at"])


@dataclass
class Slot:
    """One planned run of an eval instance: a task at its pinned version and a replica. ``attempts`` are the task runs
    submitted for it, oldest first, and ``result`` is the last one's, as its task instance recorded it."""

    eval_instance_id: str
    task_id: str
    task_version: int
    replica: int
    state: SlotState = SlotState.PLANNED
    attempts: list[Attempt] = field(default_factory=list)
    result: Optional[dict[str, Any]] = None
    rev: int = 0

    def __post_init__(self) -> None:
        self.state = SlotState(self.state)
        self.id = slot_id(self.task_id, self.task_version, self.replica)

    def to_dict(self) -> dict:
        return {
            "eval_instance_id": self.eval_instance_id,
            "id": self.id,
            "task_id": self.task_id,
            "task_version": self.task_version,
            "replica": self.replica,
            "state": str(self.state),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "result": self.result,
            "rev": self.rev,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Slot:
        return cls(
            eval_instance_id=data["eval_instance_id"],
            task_id=data["task_id"],
            task_version=data["task_version"],
            replica=data["replica"],
            state=data.get("state", SlotState.PLANNED),
            attempts=[Attempt.from_dict(attempt) for attempt in data.get("attempts") or []],
            result=data.get("result"),
            rev=data.get("rev", 0),
        )


def planned_slots(eval_instance_id: str, tasks: list[tuple[str, int]], k: int) -> list[Slot]:
    """The slots of running each ``(task_id, task_version)`` ``k`` times: replicas 1 to ``k`` of every task."""
    if k < 1:
        raise ValueError(f"k counts the runs of each task and must be at least 1, got {k}")
    return [Slot(eval_instance_id, task_id, task_version, replica)
            for replica in range(1, k + 1) for task_id, task_version in tasks]
