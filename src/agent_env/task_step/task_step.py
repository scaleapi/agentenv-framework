"""Base TaskStep model for AgentEnv."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional

from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef

if TYPE_CHECKING:
    from agent_env.task_step.store import TaskStepQuery


@dataclass
class TaskStepDependency:
    task_step_id: str

    @classmethod
    def from_dict(cls, data: dict) -> "TaskStepDependency":
        return cls(task_step_id=data["task_step_id"])


def attach_retry_config(step: "TaskStep", data: dict) -> "TaskStep":
    """Hydrate ``retry_config`` after ``step_cls.from_dict`` at the generic
    parse sites, so the field survives round-trips on step types whose
    ``__init__`` doesn't accept it. No-op if already set or absent."""
    raw = data.get("retry_config")
    if raw is not None and getattr(step, "retry_config", None) is None:
        step.retry_config = RetryConfig.from_dict(raw)
    return step


@dataclass
class RetryConfig:
    """Where a step resumes from on a retry, and how many retries to allow.

    ``retry_from_step_id`` is required. Set to the step itself it re-runs only
    this step. Point it at an earlier dependency ancestor to roll the world back
    to there and re-dispatch the span (``retry_from_step_id`` .. failed step) as
    ordinary steps on a freshly redeployed world (see ``Task._drive_dag``). The
    task author owns the resume point; the platform does not decide which steps
    are safe to re-run.

    ``max_retries`` (default 1) bounds how many times a step may trigger this
    rollback-and-re-dispatch before the failure is treated as terminal."""

    retry_from_step_id: str
    max_retries: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> "RetryConfig":
        # go_to_step_id = wire alias; retry_from_step_id canonical (only spelling emitted).
        step_id = data.get("retry_from_step_id", data.get("go_to_step_id"))
        if step_id is None:
            raise ValueError("retry_config requires 'retry_from_step_id' (or 'go_to_step_id')")
        max_retries = data.get("max_retries", 1)
        # bool is an int subclass; reject it and non-ints so a bad value fails at
        # parse time, not mid-run inside the scheduler's retry-budget comparison.
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise ValueError(f"retry_config.max_retries must be an int, got {max_retries!r}")
        if max_retries < 0:
            raise ValueError(f"retry_config.max_retries must be >= 0, got {max_retries}")
        return cls(retry_from_step_id=step_id, max_retries=max_retries)


class TaskStep(ABC):
    """Base class for all task steps. Immutable - each modification creates a new version."""

    type: ClassVar[str] = "task_step"
    DEFAULT_TTL_SECONDS: ClassVar[int] = 7200
    DEFAULT_AGENT_NAME: ClassVar[str] = "default-agent"
    # Fields holding env, A2A agent or artifact ids; None means undeclared.
    entity_refs: ClassVar[tuple[EntityRef, ...] | None] = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Never inherited: a subclass may add id fields its parent's declaration misses.
        if "entity_refs" not in vars(cls):
            cls.entity_refs = None

    def __init__(
        self,
        id: str,
        version: Optional[int],
        depends_on: Optional[list] = None,
        fail_task_on_error: bool = True,
        retry_config: Optional["RetryConfig | dict"] = None,
    ):
        self.id = id
        self.version = version
        if depends_on is not None:
            depends_on = [
                d if isinstance(d, TaskStepDependency) else TaskStepDependency.from_dict(d)
                for d in depends_on
            ]
        self.depends_on = depends_on
        self.fail_task_on_error = fail_task_on_error
        if retry_config is not None and not isinstance(retry_config, RetryConfig):
            retry_config = RetryConfig.from_dict(retry_config)
        self.retry_config = retry_config

    @abstractmethod
    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        """Execute this task step."""
        pass

    def preflight(self) -> list[str]:
        """Config problems detectable before a run, one message each; empty when fine.

        Overrides must not need a sandbox or a deployed env.
        """
        return []

    def step_param_overrides(self, context: TaskStepContext) -> dict:
        """Per-run overrides for this step's params, from a run's ``step_overrides``.

        Generic seam: a step opts in by reading the params it supports; an
        override wins over the stored value. Empty dict when none set.
        """
        return (context.metadata.get("user_overrides") or {}).get("step_params", {}).get(self.id) or {}

    def to_dict(self) -> dict:
        base: dict = {"id": self.id, "type": self.type, "version": self.version, "fail_task_on_error": self.fail_task_on_error}
        if self.depends_on is not None:
            base["depends_on"] = [dataclasses.asdict(d) for d in self.depends_on]
        if self.retry_config is not None:
            base["retry_config"] = dataclasses.asdict(self.retry_config)
        return base

    @classmethod
    def _base_from_dict(cls, data: dict) -> dict:
        raw = data.get("depends_on")
        depends_on = (
            [TaskStepDependency.from_dict(d) for d in raw] if raw is not None else None
        )
        return {
            "id": data["id"], "version": data.get("version"),
            "depends_on": depends_on,
            "fail_task_on_error": data.get("fail_task_on_error", True),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskStep":
        raise NotImplementedError(f"{cls.__name__} must implement from_dict")

    @classmethod
    def get(cls, id: str, version: Optional[int] = None) -> "TaskStep":
        from .store import get_task_step_store

        return get_task_step_store().get(id, version)

    @classmethod
    def put(cls, **kwargs) -> "TaskStep":
        from .store import get_task_step_store

        kwargs.setdefault("version", None)
        # Most step __init__s don't accept retry_config (it lives on the base but
        # subclasses don't forward it); construct without it, then hydrate so a
        # retry_config on any step type round-trips through the step store.
        retry_config_raw = kwargs.pop("retry_config", None)
        instance = cls(**kwargs)
        if retry_config_raw is not None:
            attach_retry_config(instance, {"retry_config": retry_config_raw})
        return get_task_step_store().put_document(instance)

    @classmethod
    def query(cls) -> "TaskStepQuery":
        from .store import TaskStepQuery, get_task_step_store

        return TaskStepQuery(get_task_step_store())
