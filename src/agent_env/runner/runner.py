"""The ``Runner`` seam: dispatch a task run and return a handle to poll. Selected via
``[runner]`` in ``.agentenv/config.toml`` like a store.

``RunStatus`` and ``submit``'s ``(run_id, instance_id)`` return match the Temporal
worker's, so the hub works against either backend unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Optional


class RunStatus(StrEnum):
    """Run lifecycle. Values match Temporal's workflow-execution statuses; ``QUEUED`` is
    added for the pre-start state and is non-terminal."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"

    @property
    def is_terminal(self) -> bool:
        return self in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELED)


@dataclass(frozen=True)
class RunHandle:
    """Returned by ``submit`` before the run starts. ``run_id`` is the hub's
    ``workflow_id`` (a minted id under the local runner); ``instance_id`` is minted at
    submit time so a caller can link to the run immediately."""

    run_id: str
    instance_id: str


@dataclass
class RunRecord:
    """The persisted state of one run."""

    run_id: str
    runner: str
    task_id: str
    task_version: Optional[int]
    instance_id: str
    status: RunStatus = RunStatus.QUEUED
    created_at_utc: Optional[str] = None
    started_at_utc: Optional[str] = None
    finished_at_utc: Optional[str] = None
    error: Optional[str] = None
    overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "runner": self.runner,
            "task_id": self.task_id,
            "task_version": self.task_version,
            "instance_id": self.instance_id,
            "status": str(self.status),
            "created_at_utc": self.created_at_utc,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "error": self.error,
            "overrides": self.overrides,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunRecord":
        return cls(
            run_id=data["run_id"],
            runner=data.get("runner", ""),
            task_id=data["task_id"],
            task_version=data.get("task_version"),
            instance_id=data["instance_id"],
            status=RunStatus(data.get("status", RunStatus.QUEUED)),
            created_at_utc=data.get("created_at_utc"),
            started_at_utc=data.get("started_at_utc"),
            finished_at_utc=data.get("finished_at_utc"),
            error=data.get("error"),
            overrides=data.get("overrides") or {},
        )


class Runner(ABC):
    """Dispatches task runs and reports on them.

    ``type`` is persisted on every run record: a hub restarted under a different
    ``[runner]`` reports on a run it does not own rather than misreporting it.
    """

    type: str = "runner"

    @classmethod
    def from_config(cls, **config) -> Runner:
        """Construct from the resolved ``[runner.config]`` table."""
        return cls(**config)

    @abstractmethod
    async def submit(
        self,
        task_id: str,
        task_version: Optional[int] = None,
        *,
        agent_model: Optional[str] = None,
        agent_artifact_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> RunHandle:
        """Enqueue a run and return its handle immediately, without waiting for it."""

    @abstractmethod
    async def status(self, run_id: str) -> Optional[RunRecord]:
        """The current record for ``run_id``, or None if this runner has no such run."""

    @abstractmethod
    async def cancel(self, run_id: str) -> bool:
        """Request cancellation; True if the run was queued/running and is now canceled."""

    async def start(self) -> None:
        """Begin processing work. No-op for runners backed by an external service."""

    async def stop(self) -> None:
        """Stop processing and release resources."""
