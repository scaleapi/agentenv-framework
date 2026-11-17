"""The yatf/1 event models: one tolerant tree that keeps every field, event type and part type it doesn't know, so a
reader writes a newer 1.x stream back out unchanged. A producer's strictness lives in the validator and in the strict
JSON schema generated from these models."""
from __future__ import annotations

import re
from typing import Annotated, Any, ClassVar, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Tag, TypeAdapter, ValidationError, WithJsonSchema

FORMAT = "yatf/1"
SPEC_VERSION = "1.0"

CORE_TYPES = frozenset({
    "turn.start", "message", "final_response", "turn.end", "step", "reasoning", "tool_call", "tool_result", "tools",
    "actor", "error", "attempt", "note",
})
#: Defined by a later 1.x together with its first producer; no 1.0 producer may write them.
RESERVED_TYPES = frozenset({"delegation", "context", "producer", "activation", "usage.checkpoint"})
#: An extension type is ``<namespace>/<name>``; the ``yatf`` namespace belongs to the spec.
EXTENSION_TYPE = re.compile(r"^(?!yatf/)[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*$")

OBSERVERS = ("agent", "gateway", "browser", "framework")
ACTOR_KINDS = ("agent", "subagent", "user", "user_sim", "human", "peer", "trigger_executor", "env", "judge", "framework")
MODEL_SOURCES = ("configured", "reported", "unknown")
ORIGIN_KINDS = ("framework", "user_sim", "trigger", "notification", "peer", "human")
ELISION_REASONS = ("truncated", "redacted", "digest_only", "evicted", "not_captured", "not_requested")
MEDIA_KINDS = ("image", "audio", "video", "document", "file")
ROLES = ("user", "assistant", "system", "developer")
STEP_SOURCES = ("agent", "user", "system")
COST_STATUSES = ("reported", "priced", "unpriced")
BASES = ("steps", "reported")
VISIBILITIES = ("raw", "summary", "omitted", "redacted")
OPAQUE_KINDS = ("signature", "encrypted", "redacted_data")
TOOL_KINDS = ("builtin", "mcp", "function", "custom")
EXECUTORS = ("agent", "env", "provider", "client")
RESULT_STATUSES = ("ok", "error", "canceled", "timeout")
TOOLS_MODES = ("replace", "add", "remove")
TURN_STATES = ("completed", "failed", "canceled")
ERROR_TYPES = ("agent_error", "infra_error")


def _vocabulary(values: tuple[str, ...]) -> WithJsonSchema:
    """A string the strict schema limits to ``values``; the model takes any string, as a later minor may add one."""
    return WithJsonSchema({"type": "string", "enum": list(values)})


Observer = Annotated[str, _vocabulary(OBSERVERS)]
ActorKind = Annotated[str, _vocabulary(ACTOR_KINDS)]
ModelSource = Annotated[str, _vocabulary(MODEL_SOURCES)]
OriginKind = Annotated[str, _vocabulary(ORIGIN_KINDS)]
ElisionReason = Annotated[str, _vocabulary(ELISION_REASONS)]
Role = Annotated[str, _vocabulary(ROLES)]
StepSource = Annotated[str, _vocabulary(STEP_SOURCES)]
CostStatus = Annotated[str, _vocabulary(COST_STATUSES)]
Basis = Annotated[str, _vocabulary(BASES)]
Visibility = Annotated[str, _vocabulary(VISIBILITIES)]
OpaqueKind = Annotated[str, _vocabulary(OPAQUE_KINDS)]
ToolKind = Annotated[str, _vocabulary(TOOL_KINDS)]
Executor = Annotated[str, _vocabulary(EXECUTORS)]
ResultStatus = Annotated[str, _vocabulary(RESULT_STATUSES)]
ToolsMode = Annotated[str, _vocabulary(TOOLS_MODES)]
TurnState = Annotated[str, _vocabulary(TURN_STATES)]
ErrorType = Annotated[str, _vocabulary(ERROR_TYPES)]


class YatfError(ValueError):
    """A stream or event that isn't yatf/1, or an adapter that breaks the adapter contract."""


class YatfModel(BaseModel):
    """Keeps fields it doesn't know through a dump (the validator reports them); the strict schema forbids them."""

    model_config = ConfigDict(extra="allow", protected_namespaces=(), json_schema_extra={"additionalProperties": False})
    #: Whether fields this version doesn't name are part of the model's meaning, not unknowns (V11 skips them).
    OPEN: ClassVar[bool] = False


class _Open(YatfModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=(), json_schema_extra={"additionalProperties": True})
    OPEN: ClassVar[bool] = True


class VirtualTime(YatfModel):
    env_id: str
    env_step_id: str | None = None
    at: str


class Time(YatfModel):
    """When an event happened per the native record. ``at`` is always present and null when the native has no time;
    ``raw`` is the native string whenever it differs from ``at``."""

    at: str | None
    start: str | None = None
    end: str | None = None
    duration_ms: int | None = None
    raw: str | None = None
    virtual: list[VirtualTime] | None = None


class Source(YatfModel):
    """Bytes held by reference: a ``sha256:<hex>`` URL resolved under the run's blob prefix, or an absolute URL."""

    url: str
    media_type: str
    bytes: int
    sha256: str
    width: int | None = None
    height: int | None = None
    duration_ms: int | None = None
    pages: int | None = None
    name: str | None = None


class Elision(YatfModel):
    """Why content is missing, so elided never reads as empty. A redacted secret has its length and no hash."""

    reason: ElisionReason
    bytes: int | None = None
    sha256: str | None = None


class Origin(_Open):
    """Where a user-side part came from; the part's own metadata (a notification's channel, a peer's id) sits beside
    ``kind``."""

    kind: OriginKind


class TextPart(YatfModel):
    type: Literal["text"]
    text: str
    origin: Origin | None = None
    elision: Elision | None = None


class RefPart(YatfModel):
    """Text over the inline limit: a preview and the full text by reference."""

    type: Literal["ref"]
    preview: str
    source: Source
    origin: Origin | None = None


class MediaPart(YatfModel):
    type: Literal["image", "audio", "video", "document", "file"]
    source: Source
    purpose: str | None = None
    origin: Origin | None = None


class DataPart(YatfModel):
    """A structured A2A part, such as a harness's tool-call count or a structured output."""

    type: Literal["data"]
    data: Any
    origin: Origin | None = None


class ElidedPart(YatfModel):
    type: Literal["elided"]
    elision: Elision
    media_type: str | None = None


class UnknownPart(_Open):
    """A part type this version doesn't define: readers keep it, producers may not write it."""

    type: str


_PART_TAGS = {"text": "text", "ref": "ref", "data": "data", "elided": "elided", **{kind: "media" for kind in MEDIA_KINDS}}
_NOTHING = WithJsonSchema({"not": {}, "description": "A type this version doesn't define: readers keep it, producers may "
                                                     "not write it."})


def _part_tag(value: Any) -> str:
    kind = value.get("type") if isinstance(value, dict) else getattr(value, "type", None)
    return _PART_TAGS.get(kind, "unknown") if isinstance(kind, str) else "unknown"


Part = Annotated[
    Union[
        Annotated[TextPart, Tag("text")],
        Annotated[RefPart, Tag("ref")],
        Annotated[MediaPart, Tag("media")],
        Annotated[DataPart, Tag("data")],
        Annotated[ElidedPart, Tag("elided")],
        Annotated[UnknownPart, Tag("unknown"), _NOTHING],
    ],
    Discriminator(_part_tag),
]


class Actor(YatfModel):
    """A participant, declared in ``turn.start`` or an ``actor`` event before any event names it. A sub-agent carries
    its ``parent`` actor and the ``tool_call_id`` that ``spawned_by`` it."""

    id: str
    kind: ActorKind
    name: str | None = None
    version: str | None = None
    model_name: str | None = None
    model_source: ModelSource | None = None
    role: str | None = None
    participant: str | None = None
    a2a_agent_id: str | None = None
    parent: str | None = None
    spawned_by: str | None = None
    stream: str | None = None


class DerivedFrom(YatfModel):
    """The record an event came from: its position in the turn's stored native log, or in the named aux log."""

    index: int
    aux: str | None = None


class AdapterInfo(YatfModel):
    name: str
    version: str
    lossy: list[str] | None = None


class Metrics(YatfModel):
    """One step's usage, inclusive as in ATIF: ``prompt_tokens`` counts cache reads and writes, ``completion_tokens``
    counts reasoning. An unknown count is null or absent, never 0; ``raw`` keeps the native counters."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float | None = None
    cost_status: CostStatus | None = None
    cost_source: str | None = None
    raw: dict[str, Any] | None = None


class UsageTotals(YatfModel):
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_cached_tokens: int | None = None
    total_cache_write_tokens: int | None = None
    total_reasoning_tokens: int | None = None
    total_cost_usd: float | None = None
    cost_status: CostStatus | None = None
    cost_source: str | None = None


class FinalMetrics(UsageTotals):
    """A turn's totals under ATIF's FinalMetrics names. ``basis`` says whether they sum the turn's steps or are what
    the harness reported; ``total_steps`` counts its ``step`` events."""

    basis: Basis
    total_steps: int | None = None
    partial: bool | None = None
    includes_subagents: bool | None = None
    by_actor: dict[str, UsageTotals] | None = None
    by_model: dict[str, UsageTotals] | None = None


class Opaque(YatfModel):
    """A provider payload kept byte for byte, such as a thinking signature, for replay."""

    kind: OpaqueKind
    value: str
    provider: str | None = None


class ToolDef(YatfModel):
    name: str
    kind: ToolKind | None = None
    server: str | None = None
    description: str | None = None
    input_schema: dict[str, Any] | None = None


class Event(YatfModel):
    """The envelope every event carries. Its order is (``turn``, ``seq``), and ``id`` is ``<turn>.<seq>``."""

    seq: int
    id: str
    type: str
    turn: int
    actor: str
    observer: Observer
    time: Time
    step: str | None = None
    attempt: int | None = None
    is_copied_context: bool | None = None
    derived_from: DerivedFrom | None = None
    x: dict[str, Any] | None = None


class TurnStart(Event):
    """Opens a turn, one per A2A task. It names who made the stream and declares the turn's actors."""

    type: Literal["turn.start"]
    spec_version: str
    context_id: str
    task_id: str
    message_id: str | None = None
    native_format: str
    live: bool | None = None
    adapter: AdapterInfo
    usage_input_includes_cache: bool | None
    actors: list[Actor]


class Message(Event):
    type: Literal["message"]
    role: Role
    content: list[Part]


class FinalResponse(Event):
    type: Literal["final_response"]
    content: list[Part]


class TurnEnd(Event):
    type: Literal["turn.end"]
    state: TurnState
    tool_call_count: int | None = None
    final_metrics: FinalMetrics | None = None


class StepEvent(Event):
    """One inference, dispatch or inbound message, under ATIF's step names; the content events that share its
    ``step`` key belong to it."""

    type: Literal["step"]
    source: StepSource
    llm_call_count: int | None = None
    model_name: str | None = None
    reasoning_effort: str | None = None
    metrics: Metrics | None = None
    stop_reason: str | None = None
    provider_ids: dict[str, str] | None = None
    first_token_at: str | None = None
    sampling: dict[str, Any] | None = None


class Reasoning(Event):
    type: Literal["reasoning"]
    visibility: Visibility
    text: str | None = None
    opaque: list[Opaque] | None = None


class ToolCall(Event):
    """A tool call; ``arguments`` is any JSON value, and ``arguments_raw`` the native string when it was one."""

    type: Literal["tool_call"]
    tool_call_id: str
    function_name: str
    arguments: Any
    kind: ToolKind | None = None
    server: str | None = None
    executor: Executor | None = None
    env_id: str | None = None
    arguments_raw: str | None = None


class ToolResult(Event):
    """A result answering any earlier call in the context, kept where it arrived."""

    type: Literal["tool_result"]
    source_call_id: str
    status: ResultStatus
    content: list[Part]
    exit_code: int | None = None


class Tools(Event):
    """The tools the agent can call from here on: the whole set, or ones added or removed."""

    type: Literal["tools"]
    mode: ToolsMode
    tools: list[ToolDef]


class ActorDeclared(Event):
    type: Literal["actor"]
    declared: Actor


class ErrorEvent(Event):
    type: Literal["error"]
    error_type: ErrorType
    message: str
    code: str | None = None
    fatal: bool | None = None


class Attempt(Event):
    """A harness retry within the task; later events carry ``attempt`` equal to ``number``."""

    type: Literal["attempt"]
    number: int
    reason: str | None = None


class Note(Event):
    type: Literal["note"]
    text: str


class ExtensionEvent(Event):
    """A ``<namespace>/<name>`` type, such as an adapter's raw native record: readers keep it and may skip it."""

    model_config = ConfigDict(extra="allow", protected_namespaces=(), json_schema_extra={"additionalProperties": True})
    OPEN: ClassVar[bool] = True
    type: Annotated[str, WithJsonSchema({"type": "string", "pattern": EXTENSION_TYPE.pattern})]


class UnknownEvent(Event):
    """An event type this version doesn't define: readers keep it, producers may not write it."""

    model_config = ConfigDict(extra="allow", protected_namespaces=(), json_schema_extra={"additionalProperties": True})
    OPEN: ClassVar[bool] = True


EVENT_MODELS: dict[str, type[Event]] = {
    "turn.start": TurnStart, "message": Message, "final_response": FinalResponse, "turn.end": TurnEnd,
    "step": StepEvent, "reasoning": Reasoning, "tool_call": ToolCall, "tool_result": ToolResult, "tools": Tools,
    "actor": ActorDeclared, "error": ErrorEvent, "attempt": Attempt, "note": Note,
}


def is_extension_type(kind: str) -> bool:
    """Whether ``kind`` names an extension type an adapter may write."""
    return bool(EXTENSION_TYPE.match(kind))


def _event_tag(value: Any) -> str:
    kind = value.get("type") if isinstance(value, dict) else getattr(value, "type", None)
    if not isinstance(kind, str):
        return "unknown"
    if kind in EVENT_MODELS:
        return kind
    return "extension" if is_extension_type(kind) else "unknown"


AnyEvent = Annotated[
    Union[tuple(Annotated[model, Tag(kind)] for kind, model in EVENT_MODELS.items())
          + (Annotated[ExtensionEvent, Tag("extension")], Annotated[UnknownEvent, Tag("unknown"), _NOTHING])],
    Discriminator(_event_tag),
]
_EVENT = TypeAdapter(AnyEvent)


def parse_event(value: Any) -> Event:
    """One event as its typed model; an unknown type parses as UnknownEvent and a namespaced one as ExtensionEvent."""
    try:
        return _EVENT.validate_python(value)
    except ValidationError as exc:
        raise YatfError(f"not a yatf/1 event: {exc}") from exc


def dump_event(event: Event) -> dict[str, Any]:
    """The event as JSON, with exactly the fields it was given, unknown ones included."""
    return event.model_dump(mode="json", exclude_unset=True)
