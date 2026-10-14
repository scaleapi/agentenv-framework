"""Base TaskStep model for AgentEnv."""

from __future__ import annotations

import dataclasses
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Self

from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef

if TYPE_CHECKING:
    from agent_env.task_step.store import TaskStepQuery


@dataclass
class TaskStepDependency:
    task_step_id: str

    @classmethod
    def from_dict(cls, data: str | dict[str, Any]) -> "TaskStepDependency":
        """One ``depends_on`` entry: a step id, or ``{"task_step_id": <step id>}``."""
        if isinstance(data, str):
            return cls(task_step_id=data)
        if isinstance(data, dict) and "task_step_id" in data:
            return cls(task_step_id=data["task_step_id"])
        raise ValueError(f'a depends_on entry is a step id or {{"task_step_id": "<step id>"}}, not {data!r}')


def dependencies(raw: Any) -> list[TaskStepDependency] | None:
    """``depends_on`` as written, read into dependencies. None, which runs a step after every earlier one,
    stays None; a lone step id is refused rather than read as a list of its characters."""
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"depends_on is a list of step ids, not {raw!r}")
    return [dep if isinstance(dep, TaskStepDependency) else TaskStepDependency.from_dict(dep) for dep in raw]


def attach_retry_config(step: "TaskStep", data: dict[str, Any]) -> "TaskStep":
    """Hydrate ``retry_config`` after ``step_cls.from_dict`` at the generic
    parse sites, so the field survives round-trips on step types whose
    ``__init__`` doesn't accept it. No-op if already set or absent."""
    raw = data.get("retry_config")
    if raw is not None and getattr(step, "retry_config", None) is None:
        step.retry_config = RetryConfig.from_dict(raw)
    return step


UNFORWARDED_BASE_FIELDS = ("retry_config", "when", "if_dependency_skipped")
SKIPPED_STEPS_KEY = "skipped_steps"
_IF_DEPENDENCY_SKIPPED = ("skip", "run")


def attach_base_fields(step: "TaskStep", data: dict[str, Any]) -> "TaskStep":
    """Hydrate the base fields step ``__init__``s don't forward (``retry_config``, ``when``,
    ``if_dependency_skipped``) after ``step_cls.from_dict``, so they survive every round-trip."""
    attach_retry_config(step, data)
    if data.get("when") is not None and getattr(step, "when", None) is None:
        step.when = check_when(data["when"])
    if data.get("if_dependency_skipped") is not None:
        step.if_dependency_skipped = check_if_dependency_skipped(data["if_dependency_skipped"])
    return step


def check_if_dependency_skipped(value: Any) -> str:
    if value in _IF_DEPENDENCY_SKIPPED:
        return value
    raise ValueError(f"if_dependency_skipped is 'skip' or 'run', not {value!r}")


def skip_reason(step: "TaskStep", context: TaskStepContext) -> str | None:
    """Why ``step`` is skipped in this run, or None to run it: its seed-substituted ``when`` is false, or a
    step it names in ``depends_on`` was skipped (unless it sets ``if_dependency_skipped: run``). Implicit
    ordering (``depends_on`` None) does not carry a skip. Skips are read from ``context.metadata``, so a
    resumed run, or a runner that dispatches steps one at a time, sees the earlier ones."""
    when = getattr(step, "when", None)
    if not evaluate_when(when):
        return f"when={when!r}"
    if getattr(step, "if_dependency_skipped", "skip") == "run":
        return None
    skipped = {s["step_id"] for s in context.metadata.get(SKIPPED_STEPS_KEY) or []}
    for dep in getattr(step, "depends_on", None) or []:
        if dep.task_step_id in skipped:
            return f"dependency {dep.task_step_id!r} was skipped"
    return None


def record_skip(step: "TaskStep", context: TaskStepContext, reason: str) -> None:
    """Add ``step`` to the run's skips once; a retried dispatch rehydrates a context that may already hold it."""
    skipped = context.metadata.setdefault(SKIPPED_STEPS_KEY, [])
    if all(s["step_id"] != step.id for s in skipped):
        skipped.append({"step_id": step.id, "reason": reason})


_WHEN_TRUE = frozenset({"true", "yes", "on", "1"})
_WHEN_FALSE = frozenset({"false", "no", "off", "0", ""})
_UNRESOLVED_PLACEHOLDER = re.compile(r"<[A-Za-z0-9_.-]+>")


def check_when(when: Any) -> str | bool:
    """``when`` as stored: a bool, or a string the run's seed may substitute into one."""
    if isinstance(when, (bool, str)):
        return when
    raise ValueError(f"when is a bool or a string, not {when!r}")


def evaluate_when(when: str | bool | None) -> bool:
    """Whether a step with this ``when`` runs. None always runs; a string must, after seed
    substitution, be true/yes/on/1 or false/no/off/0/empty."""
    if when is None or isinstance(when, bool):
        return when is not False
    value = when.strip().lower()
    if value in _WHEN_TRUE:
        return True
    if value in _WHEN_FALSE:
        return False
    if _UNRESOLVED_PLACEHOLDER.search(when):
        raise ValueError(f"when={when!r} still holds a placeholder: the run's seed has no value for it")
    raise ValueError(f"when={when!r} is not a boolean; use true/yes/on/1 or false/no/off/0")


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
    def from_dict(cls, data: dict[str, Any]) -> "RetryConfig":
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

    def __init_subclass__(cls, **kwargs: Any) -> None:
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
        when: Optional[str | bool] = None,
        if_dependency_skipped: str = "skip",
    ):
        self.id = id
        self.version = version
        self.depends_on = dependencies(depends_on)
        self.fail_task_on_error = fail_task_on_error
        if retry_config is not None and not isinstance(retry_config, RetryConfig):
            retry_config = RetryConfig.from_dict(retry_config)
        self.retry_config = retry_config
        self.when = None if when is None else check_when(when)
        self.if_dependency_skipped = check_if_dependency_skipped(if_dependency_skipped)

    @abstractmethod
    async def execute(self, context: TaskStepContext) -> TaskStepContext | None:
        """Execute this task step."""
        pass

    def preflight(self) -> list[str]:
        """Config problems detectable before a run, one message each; empty when fine.

        Overrides must not need a sandbox or a deployed env.
        """
        return []

    def step_param_overrides(self, context: TaskStepContext) -> dict[str, Any]:
        """Per-run overrides for this step's params, from a run's ``step_overrides``.

        Generic seam: a step opts in by reading the params it supports; an
        override wins over the stored value. Empty dict when none set.
        """
        return (context.metadata.get("user_overrides") or {}).get("step_params", {}).get(self.id) or {}

    def to_dict(self) -> dict[str, Any]:
        base: dict[str, Any] = {"id": self.id, "type": self.type, "version": self.version, "fail_task_on_error": self.fail_task_on_error}
        if self.depends_on is not None:
            base["depends_on"] = [dataclasses.asdict(d) for d in self.depends_on]
        if self.retry_config is not None:
            base["retry_config"] = dataclasses.asdict(self.retry_config)
        if getattr(self, "when", None) is not None:
            base["when"] = self.when
        if getattr(self, "if_dependency_skipped", "skip") != "skip":
            base["if_dependency_skipped"] = self.if_dependency_skipped
        return base

    @classmethod
    def _base_from_dict(cls, data: dict) -> dict:
        return {
            "id": data["id"], "version": data.get("version"),
            "depends_on": dependencies(data.get("depends_on")),
            "fail_task_on_error": data.get("fail_task_on_error", True),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskStep":
        raise NotImplementedError(f"{cls.__name__} must implement from_dict")

    @classmethod
    def get(cls, id: str, version: Optional[int] = None) -> "TaskStep":
        from .store import get_task_step_store

        return get_task_step_store().get(id, version)

    @classmethod
    def put(cls, **kwargs: Any) -> Self:
        from .store import get_task_step_store

        kwargs.setdefault("version", None)
        # Most step __init__s don't accept the unforwarded base fields; construct
        # without them, then hydrate so they round-trip through the step store on any step type.
        base_fields = {k: kwargs.pop(k) for k in UNFORWARDED_BASE_FIELDS if k in kwargs}
        instance = attach_base_fields(cls(**kwargs), base_fields)
        return get_task_step_store().put_document(instance)

    @classmethod
    def query(cls) -> "TaskStepQuery":
        from .store import TaskStepQuery, get_task_step_store

        return TaskStepQuery(get_task_step_store())
