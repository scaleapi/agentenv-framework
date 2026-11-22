"""Version 1 of the stable agent task boundary.

The framework constructs :class:`TaskRequest` instances and agent authors return
an immutable :class:`TaskResult`. V1 is additive-only: new request fields must
have defaults and new result capabilities are exposed through optional builder
methods rather than required constructor arguments.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Iterable, Mapping
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Generic, Literal, TypeAlias, TypeVar, Union

from a2a.types import TaskStatusUpdateEvent
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...atif import AdapterSpec

logger = logging.getLogger(__name__)

_LARGE_LIVE_TRAJECTORY_BYTES = 100 * 1024 * 1024

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


ToolName = TypeVar("ToolName", bound=str)


class BuiltinTools(BaseModel, Generic[ToolName]):
    """Which of an agent's own tools it may use, as a task selects them: ``disable`` turns off the ones it names, or
    all of them with ``"*"``; ``only`` keeps the ones it names. An agent's config takes it as
    ``builtin_tools: BuiltinTools[<a Literal of its tools>] | None``, so its card's agent-config schema names them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    disable: Literal["*"] | list[ToolName] | None = None
    only: list[ToolName] | None = None

    @model_validator(mode="after")
    def _one_of(self) -> BuiltinTools:
        if (self.disable is None) == (self.only is None):
            raise ValueError("builtin_tools takes exactly one of disable and only")
        return self


ConfigT = TypeVar("ConfigT", bound=AgentConfig)


class TrajectoryLog:
    """Append-only trajectory of one task, readable while the task runs.

    Each event is serialized as it is appended, so an event once read never changes; append
    from one thread at a time. Unless the result carries a ``native_trajectory``, the log is
    the task's final trajectory. An agent whose card declares a ``native`` format starts with
    that format, and setting or returning another fails the turn. With an ATIF adapter
    (``adapter``, its spec), every event and aux record also feeds the context's atif/1 copy."""

    __slots__ = (
        "_events", "_size_bytes", "_format", "_native", "_sealed", "_task_id", "_warned", "_adapter", "_atif",
        "_starts_document",
    )

    def __init__(self, *, native: str | None = None, adapter: AdapterSpec | None = None) -> None:
        self._events: list[bytes] = []
        self._size_bytes = 0
        self._format: str | None = native
        self._native = native
        self._sealed = False
        self._task_id: str | None = None
        self._warned = False
        self._adapter = adapter
        self._atif: Any = None
        self._starts_document = False

    @property
    def format(self) -> str | None:
        return self._format

    @property
    def builds_atif(self) -> bool:
        """Whether this task feeds an atif/1 copy: the agent has an ATIF adapter and the run prescribed atif/1. An agent
        gathers what only its aux files take, such as a log it tails, only then."""
        return self._atif is not None

    @property
    def starts_document(self) -> bool:
        """Whether this task opens a new atif/1 document of its conversation: the conversation's first, or the first
        since the copy was dropped (a restart, an eviction, a snapshot restore). What an agent feeds a document once,
        such as the tools its model is offered, it feeds again then."""
        return self._starts_document

    def set_format(self, format: str) -> None:
        """Name the events' versioned format, such as ``my-cli-events/1``; once an event is
        appended, it stays."""
        if not isinstance(format, str) or not format.strip():
            raise ValueError("trajectory format must be a non-empty string")
        self._hold_to_native(format)
        self._ensure_open()
        if self._events and format != self._format:
            raise RuntimeError("the trajectory's format cannot change once events are appended")
        self._format = format

    def append(self, event: Mapping[str, Any]) -> None:
        """Record one complete event, a JSON object, once the log has a format: the card's
        ``native``, or one ``set_format`` names."""
        if not isinstance(event, Mapping):
            raise TypeError("a trajectory event must be a JSON object")
        self._ensure_open()
        if self._format is None:
            raise RuntimeError("call set_format(...) before appending trajectory events")
        encoded = json.dumps(dict(event), separators=(",", ":"), allow_nan=False).encode()
        self._events.append(encoded)
        self._size_bytes += len(encoded)
        if self._atif is not None:
            self._atif.native(json.loads(encoded))
        if not self._warned and self._size_bytes > _LARGE_LIVE_TRAJECTORY_BYTES:
            self._warned = True
            logger.warning(
                "trajectory of running task %s exceeds %d bytes",
                self._task_id,
                _LARGE_LIVE_TRAJECTORY_BYTES,
            )

    def aux(self, name: str, record: Any) -> None:
        """Feed a record of aux file ``name``, one the agent's ATIF adapter declares, such as a
        hook's note, where it happened among the native events. It reaches the adapter and the
        turn's atif/1 final read; the native log doesn't change."""
        if self._adapter is None:
            raise RuntimeError("aux records feed an ATIF adapter: enable(TRAJECTORY_V1, native=..., atif=...)")
        if name not in self._adapter.aux:
            raise ValueError(f"the ATIF adapter declares no aux file {name!r}")
        self._ensure_open()
        encoded = json.dumps(record, separators=(",", ":"), allow_nan=False)
        if self._atif is not None:
            self._atif.aux(name, json.loads(encoded))

    def __len__(self) -> int:
        return len(self._events)

    def _hold_returned(self, format: str) -> None:
        """Fail the turn when a returned native trajectory breaks the card: its format isn't the
        native one, or the task also appended events while an ATIF adapter builds the atif/1 copy
        from them, so the copy and the native record would come from different sources."""
        self._hold_to_native(format)
        if self._adapter is not None and self._events:
            raise _TaskFailed(
                "framework.trajectory_appended_and_returned",
                "the task appended trajectory events and returned a native trajectory; with an ATIF adapter, "
                "append the events or return the trajectory, not both",
            )

    def _hold_to_native(self, format: str) -> None:
        """Fail the turn as the agent's error when ``format``, set or returned, isn't the native
        format the card declares; a retry would fail the same way."""
        if self._native is not None and format != self._native:
            raise _TaskFailed(
                "framework.trajectory_format_mismatch",
                f"the trajectory format {format!r} is not {self._native!r}, the native format the card declares",
            )

    def _ensure_open(self) -> None:
        if self._sealed:
            raise RuntimeError("the task has ended; its trajectory can no longer change")


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskRequest(Generic[ConfigT]):
    """Frozen request record with detached, JSON-native nested values, plus the
    task's live trajectory log."""

    task_id: str
    context_id: str
    parts: tuple[TaskPart, ...]
    config: ConfigT
    mcp_servers: Mapping[str, Any] = field(default_factory=dict)
    cdp_endpoints: Mapping[str, Any] = field(default_factory=dict)
    skills: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    session_ref: str | None = None
    workspace: Path | None = None
    trajectory: TrajectoryLog = field(
        default_factory=TrajectoryLog, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "parts", tuple(self.parts))
        if not isinstance(self.config, AgentConfig):
            raise TypeError("TaskRequest.config must be an AgentConfig instance")
        # Detach mutable values from framework-owned state while preserving the
        # JSON-native types declared by the config model and request contract.
        object.__setattr__(self, "config", self.config.model_copy(deep=True))
        object.__setattr__(self, "mcp_servers", _copy_json_value(self.mcp_servers))
        object.__setattr__(self, "cdp_endpoints", _copy_json_value(self.cdp_endpoints))
        object.__setattr__(
            self, "skills", tuple(_copy_json_value(skill) for skill in self.skills)
        )
        object.__setattr__(self, "metadata", _copy_json_value(self.metadata))

    def cdp_endpoint(self, name: str | None = None) -> Mapping[str, Any]:
        """The CDP endpoint named ``name``, or the turn's only one. Anything else fails the turn, so an agent
        never falls back to a browser of its own."""
        if name is not None:
            if name in self.cdp_endpoints:
                return self.cdp_endpoints[name]
            raise _TaskFailed(
                "cdp.no_endpoint", f"the task gave no browser named {name!r}", error_type="infra_error"
            )
        if len(self.cdp_endpoints) == 1:
            return next(iter(self.cdp_endpoints.values()))
        if not self.cdp_endpoints:
            raise _TaskFailed("cdp.no_endpoint", "the task gave no browser", error_type="infra_error")
        raise _TaskFailed(
            "cdp.ambiguous",
            f"the task gave several browsers; name one of {', '.join(sorted(self.cdp_endpoints))}",
            error_type="infra_error",
        )


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


class _TaskFailed(Exception):
    """Ends the turn as ``TaskResult.failure(code, message, error_type=error_type)``; ``cdp_endpoint`` raises it."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        error_type: Literal["agent_error", "infra_error"] = "agent_error",
    ) -> None:
        super().__init__(message)
        self.error = TaskError(code=code, message=message, error_type=error_type)


@dataclass(frozen=True, slots=True, kw_only=True)
class NativeTrajectory:
    format: str
    payload: Any
    version: int = 1

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("trajectory envelope version must be positive")
        object.__setattr__(self, "payload", _copy_json_value(self.payload))


_USAGE_COUNTERS = (
    "tool_call_count",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
)


# The keys a model response's usage is read by, in any of the shapes ``Usage.from_model_response`` reads.
_RESPONSE_USAGE_KEYS = frozenset({
    "prompt_tokens", "completion_tokens", "input_tokens", "output_tokens", "total_tokens",
    "prompt_tokens_details", "input_tokens_details", "cache_read_input_tokens", "cache_creation_input_tokens",
})


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


@dataclass(frozen=True, slots=True, kw_only=True)
class Usage:
    """Common execution metrics plus runtime-specific, JSON-shaped details. ``cache_read_tokens`` and
    ``cache_write_tokens`` count the prompt tokens read from and written to the provider's prompt cache;
    ``input_includes_cache`` says whether ``input_tokens`` counts both of them (True), neither (False), or is unknown
    (None), which a price needs to read the counts."""

    tool_call_count: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    input_includes_cache: bool | None = None
    cost_usd: float | None = None
    provider_details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in _USAGE_COUNTERS:
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        if self.input_includes_cache is not None and not isinstance(
            self.input_includes_cache, bool
        ):
            raise ValueError("input_includes_cache must be true, false or null")
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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Usage":
        """The usage ``to_dict`` wrote, from this protocol version or another: a field it leaves out is None, and a
        key this version doesn't know is ignored. Raises ValueError or TypeError for a field it can't hold."""
        if not isinstance(data, Mapping):
            raise TypeError("usage must be a mapping")
        known = {
            f.name: data[f.name]
            for f in fields(cls)
            if f.name != "provider_details" and data.get(f.name) is not None
        }
        details = data.get("provider_details")
        return cls(**known, provider_details={} if details is None else details)

    @classmethod
    def from_model_response(cls, usage: Mapping[str, Any]) -> "Usage":
        """A model call's usage as its response carried it, with ``input_tokens`` counting the cache: OpenAI's Chat
        Completions (``prompt_tokens``) and Responses (``input_tokens``) shapes, and Anthropic's. An OpenAI count,
        or a LiteLLM client's, says what it holds in its ``prompt_tokens_details`` or ``input_tokens_details``, whose
        ``cached_tokens`` it counts, and ``cache_write_tokens`` with them. Anthropic's ``input_tokens``, and the
        ``prompt_tokens`` a LiteLLM proxy sends with Anthropic's ``cache_read_input_tokens`` and
        ``cache_creation_input_tokens`` beside it, count neither, so they're added, to ``total_tokens`` too. A
        provider's split of its cache writes by how long they're kept (Anthropic's ``cache_creation``, OpenAI's
        ``cache_write_tokens_details``), priced apart, is kept in ``provider_details``; so are the keys of a usage
        in none of these shapes, as ``unrecognized``. A value that isn't a count is left out."""
        key = "prompt_tokens" if "prompt_tokens" in usage else "input_tokens"
        details = usage.get(f"{key}_details")
        details = details if isinstance(details, Mapping) else {}
        cached = _count(details.get("cached_tokens"))
        read = cached if cached is not None else _count(usage.get("cache_read_input_tokens"))
        written = _count(details.get("cache_write_tokens"))
        write = written if written is not None else _count(usage.get("cache_creation_input_tokens"))
        cache = (read or 0) + (write or 0)
        input_tokens = _count(usage.get(key))
        counts_cache = cached is not None and input_tokens is not None and input_tokens >= cache
        missing = cache if input_tokens is not None and not counts_cache else 0
        if input_tokens is not None:
            input_tokens += missing
        output = _count(usage.get("completion_tokens", usage.get("output_tokens")))
        if (total := _count(usage.get("total_tokens"))) is not None:
            total += missing
        elif input_tokens is not None and output is not None:
            total = input_tokens + output
        kept = {"cache_creation": usage.get("cache_creation"),
                "cache_write_tokens_details": details.get("cache_write_tokens_details")}
        provider_details: dict[str, Any] = {
            name: counts for name, split in kept.items() if isinstance(split, Mapping)
            if (counts := {k: v for k, v in split.items() if _count(v) is not None})}
        if usage and _RESPONSE_USAGE_KEYS.isdisjoint(usage):
            provider_details["unrecognized"] = sorted(map(str, usage))
        return cls(input_tokens=input_tokens, output_tokens=output, total_tokens=total, cache_read_tokens=read,
                   cache_write_tokens=write, input_includes_cache=True if input_tokens is not None else None,
                   provider_details=provider_details)

    @property
    def is_empty(self) -> bool:
        return all(value is None for value in self._values().values()) and not self.provider_details

    def to_dict(self) -> dict[str, Any]:
        result = {name: value for name, value in self._values().items() if value is not None}
        if self.provider_details:
            result["provider_details"] = _thaw(self.provider_details)
        return result

    def _values(self) -> dict[str, Any]:
        """Every field but ``provider_details``, by name, in declaration order."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "provider_details"}


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
        usage: Usage | None = None,
    ) -> "TaskResult":
        return (
            cls.builder()
            .failed(code, message, error_type=error_type)
            .parts(parts if parts is not None else (TextPart(text=message),))
            .usage(usage or Usage())
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
