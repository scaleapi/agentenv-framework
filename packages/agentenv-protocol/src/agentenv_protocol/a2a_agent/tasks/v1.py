"""Version 1 of the stable agent task boundary.

The framework constructs :class:`TaskRequest` instances and agent authors return
an immutable :class:`TaskResult`. V1 is additive-only: new request fields must
have defaults and new result capabilities are exposed through optional builder
methods rather than required constructor arguments.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Generic, Literal, TypeAlias, TypeVar, Union

from a2a.types import TaskStatusUpdateEvent
from pydantic import BaseModel, ConfigDict, Field


_WriteOnlyT = TypeVar("_WriteOnlyT")
WriteOnly: TypeAlias = Annotated[
    _WriteOnlyT,
    Field(json_schema_extra={"writeOnly": True}),
]


def _copy_json_value(value: Any) -> Any:
    """Detach JSON-shaped values without changing their serializable types."""
    if isinstance(value, BaseModel):
        return value.model_copy(deep=True)
    if isinstance(value, Mapping):
        return {str(key): _copy_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_copy_json_value(item) for item in value]
    return value


def _thaw(value: Any) -> Any:
    """Convert immutable boundary values into JSON-ready containers."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, frozenset):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class TextPart:
    text: str
    kind: Literal["text"] = "text"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _copy_json_value(self.metadata))


@dataclass(frozen=True, slots=True, kw_only=True)
class FilePart:
    name: str | None = None
    mime_type: str | None = None
    bytes: str | None = None
    uri: str | None = None
    kind: Literal["file"] = "file"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.bytes is None) == (self.uri is None):
            raise ValueError("a file part requires exactly one of bytes or uri")
        object.__setattr__(self, "metadata", _copy_json_value(self.metadata))


@dataclass(frozen=True, slots=True, kw_only=True)
class DataPart:
    data: Mapping[str, Any]
    kind: Literal["data"] = "data"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", _copy_json_value(self.data))
        object.__setattr__(self, "metadata", _copy_json_value(self.metadata))


TaskPart = Union[TextPart, FilePart, DataPart]


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskProgress:
    """A non-terminal status update yielded by a streaming ``run`` method."""

    parts: tuple[TaskPart, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parts", tuple(self.parts))
        object.__setattr__(self, "metadata", _copy_json_value(self.metadata))

    @classmethod
    def text(
        cls, text: str, *, metadata: Mapping[str, Any] | None = None
    ) -> TaskProgress:
        return cls(parts=(TextPart(text=text),), metadata=metadata or {})


class AgentConfig(BaseModel):
    """Base configuration supplied by AgentEnv to every configured agent.

    Subclasses add runtime-specific fields.  These SDK-owned fields are part of
    the control-plane contract and therefore do not need to be repeated by each
    agent author.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str | None = None
    description: str | None = None
    role: str | None = None
    timeout_seconds: int = 600


ConfigT = TypeVar("ConfigT", bound=AgentConfig)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskRequest(Generic[ConfigT]):
    """Frozen request record with detached, JSON-native nested values."""

    task_id: str
    context_id: str
    parts: tuple[TaskPart, ...]
    config: ConfigT
    mcp_servers: Mapping[str, Any] = field(default_factory=dict)
    skills: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    session_ref: str | None = None
    workspace: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parts", tuple(self.parts))
        if not isinstance(self.config, AgentConfig):
            raise TypeError("TaskRequest.config must be an AgentConfig instance")
        # Detach mutable values from framework-owned state while preserving the
        # JSON-native types declared by the config model and request contract.
        object.__setattr__(self, "config", self.config.model_copy(deep=True))
        object.__setattr__(self, "mcp_servers", _copy_json_value(self.mcp_servers))
        object.__setattr__(
            self, "skills", tuple(_copy_json_value(skill) for skill in self.skills)
        )
        object.__setattr__(self, "metadata", _copy_json_value(self.metadata))


class TaskOutcome(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskError:
    code: str
    message: str
    error_type: Literal["agent_error", "infra_error"] = "agent_error"

    def __post_init__(self) -> None:
        if self.error_type not in {"agent_error", "infra_error"}:
            raise ValueError("error_type must be 'agent_error' or 'infra_error'")


@dataclass(frozen=True, slots=True, kw_only=True)
class NativeTrajectory:
    format: str
    payload: Any
    version: int = 1

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("trajectory envelope version must be positive")
        object.__setattr__(self, "payload", _copy_json_value(self.payload))


@dataclass(frozen=True, slots=True, kw_only=True)
class Usage:
    """Common execution metrics plus runtime-specific, JSON-shaped details."""

    tool_call_count: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    provider_details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "tool_call_count",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} cannot be negative")
        if self.cost_usd is not None and (
            isinstance(self.cost_usd, bool)
            or not isinstance(self.cost_usd, (int, float))
            or self.cost_usd < 0
        ):
            raise ValueError("cost_usd cannot be negative")
        if not isinstance(self.provider_details, Mapping):
            raise TypeError("provider_details must be a mapping")
        object.__setattr__(
            self, "provider_details", _copy_json_value(self.provider_details)
        )

    @property
    def is_empty(self) -> bool:
        return (
            self.tool_call_count is None
            and self.input_tokens is None
            and self.output_tokens is None
            and self.total_tokens is None
            and self.cost_usd is None
            and not self.provider_details
        )

    def to_dict(self) -> dict[str, Any]:
        values = {
            "tool_call_count": self.tool_call_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
        }
        result = {name: value for name, value in values.items() if value is not None}
        if self.provider_details:
            result["provider_details"] = _thaw(self.provider_details)
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskResult:
    """Validated, immutable result returned by an agent's ``run`` method."""

    outcome: TaskOutcome
    parts: tuple[TaskPart, ...] = ()
    error: TaskError | None = None
    session_ref: str | None = None
    usage: Usage = field(default_factory=Usage)
    native_trajectory: NativeTrajectory | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parts", tuple(self.parts))
        object.__setattr__(self, "usage", _copy_json_value(self.usage))
        if self.outcome is TaskOutcome.SUCCEEDED and self.error is not None:
            raise ValueError("a successful result cannot contain an error")
        if self.outcome is TaskOutcome.SUCCEEDED and not self.parts:
            raise ValueError("a successful result requires at least one part")
        if self.outcome is TaskOutcome.FAILED and self.error is None:
            raise ValueError("a failed result requires an error")
        if not isinstance(self.usage, Usage):
            raise TypeError("TaskResult.usage must be a Usage instance")

    @classmethod
    def builder(cls) -> "TaskResultBuilder":
        return TaskResultBuilder()

    @classmethod
    def text(cls, text: str, **kwargs: Any) -> "TaskResult":
        return cls.success(parts=(TextPart(text=text),), **kwargs)

    @classmethod
    def success(
        cls,
        *,
        parts: Iterable[TaskPart] = (),
        session_ref: str | None = None,
        usage: Usage | None = None,
        native_trajectory: NativeTrajectory | None = None,
    ) -> "TaskResult":
        builder = (
            cls.builder()
            .succeeded()
            .parts(parts)
            .session_ref(session_ref)
            .usage(usage or Usage())
        )
        if native_trajectory is not None:
            builder.native_trajectory(
                format=native_trajectory.format,
                payload=native_trajectory.payload,
                version=native_trajectory.version,
            )
        return builder.build()

    @classmethod
    def failure(
        cls,
        code: str,
        message: str,
        *,
        error_type: Literal["agent_error", "infra_error"] = "agent_error",
        parts: Iterable[TaskPart] | None = None,
    ) -> "TaskResult":
        return (
            cls.builder()
            .failed(code, message, error_type=error_type)
            .parts(parts if parts is not None else (TextPart(text=message),))
            .build()
        )


class TaskResultBuilder:
    """Mutable assembly object whose ``build`` method returns a valid result."""

    def __init__(self) -> None:
        self._outcome: TaskOutcome | None = None
        self._parts: list[TaskPart] = []
        self._error: TaskError | None = None
        self._session_ref: str | None = None
        self._usage = Usage()
        self._native_trajectory: NativeTrajectory | None = None

    def succeeded(self) -> "TaskResultBuilder":
        self._outcome = TaskOutcome.SUCCEEDED
        self._error = None
        return self

    def failed(
        self,
        code: str,
        message: str,
        *,
        error_type: Literal["agent_error", "infra_error"] = "agent_error",
    ) -> "TaskResultBuilder":
        self._outcome = TaskOutcome.FAILED
        self._error = TaskError(code=code, message=message, error_type=error_type)
        return self

    def parts(self, parts: Iterable[TaskPart]) -> "TaskResultBuilder":
        self._parts = list(parts)
        return self

    def add_text(
        self, text: str, *, metadata: Mapping[str, Any] | None = None
    ) -> "TaskResultBuilder":
        self._parts.append(TextPart(text=text, metadata=metadata or {}))
        return self

    def add_data(
        self, data: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None
    ) -> "TaskResultBuilder":
        self._parts.append(DataPart(data=data, metadata=metadata or {}))
        return self

    def add_structured_output(self, value: Any) -> "TaskResultBuilder":
        """Append the AgentEnv structured-output convention as a data part.

        Structured output is ordinary terminal response data, not a separate
        result field. AgentEnv consumers read ``data["structured_output"]``.
        """
        return self.add_data({"structured_output": value})

    def session_ref(self, value: str | None) -> "TaskResultBuilder":
        self._session_ref = value
        return self

    def usage(self, value: Usage) -> "TaskResultBuilder":
        if not isinstance(value, Usage):
            raise TypeError("usage must be a Usage instance")
        self._usage = value
        return self

    def native_trajectory(
        self, *, format: str, payload: Any, version: int = 1
    ) -> "TaskResultBuilder":
        self._native_trajectory = NativeTrajectory(
            format=format, payload=payload, version=version
        )
        return self

    def build(self) -> TaskResult:
        if self._outcome is None:
            raise ValueError(
                "task result outcome is required; call succeeded() or failed()"
            )
        return TaskResult(
            outcome=self._outcome,
            parts=tuple(self._parts),
            error=self._error,
            session_ref=self._session_ref,
            usage=self._usage,
            native_trajectory=self._native_trajectory,
        )


TaskStreamItem = Union[
    TaskProgress,
    TaskStatusUpdateEvent,
    TaskResult,
]
AgentRunResult = Union[
    Awaitable[TaskResult],
    AsyncIterator[TaskStreamItem],
]
