"""Writing a yatf/1 adapter: a class that turns an agent's native records into YATF events. It's declared with
``@yatf_adapter``, its ``@native``, ``@aux`` and ``@finish`` methods write through a YatfTurn, and one instance serves
an A2A context. It reads only what the run stores (the turn's context, its native records and its aux records), so an
offline replay makes the events the agent served live."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Iterable, Mapping, Sequence, TypeVar

from pydantic import ValidationError

from .models import (
    SPEC_VERSION,
    Actor,
    ActorDeclared,
    AdapterInfo,
    Attempt,
    DerivedFrom,
    ErrorEvent,
    Event,
    ExtensionEvent,
    FinalMetrics,
    FinalResponse,
    Message,
    Metrics,
    Note,
    Opaque,
    Part,
    Reasoning,
    StepEvent,
    Time,
    ToolCall,
    ToolDef,
    ToolResult,
    Tools,
    TurnEnd,
    TurnStart,
    YatfError,
    dump_event,
    is_extension_type,
)

#: A native format id: ``<name>/<version>``, as ``claude-code-stream-json/1``.
NATIVE_FORMAT = re.compile(r"^[a-z0-9][a-z0-9._-]*/[A-Za-z0-9._-]+$")
#: The step key of a turn's inbound message, ``in-<turn>``.
INBOUND_STEP = "in-{turn}"
_ENVELOPE = frozenset(Event.model_fields)
_MISSING = object()
_CURRENT: Any = object()

A = TypeVar("A", bound="Adapter")


@dataclass(frozen=True)
class Native:
    """What ``from_yatf`` gives back: each turn's native log in turn order, for the round trip, and the files the CLI
    resumes from, by name, or None when this adapter can't seed a session."""

    log: list[list[Any]]
    session: dict[str, bytes] | None = None


@dataclass(frozen=True)
class Selector:
    """The records a handler takes: native ones when ``aux`` is None, else aux file ``aux``'s, whose fields at each
    dotted path in ``conditions`` hold one of the listed values."""

    aux: str | None
    conditions: tuple[tuple[str, tuple[Any, ...]], ...]
    media_type: str | None = field(default=None, compare=False)

    def matches(self, record: Any) -> bool:
        return all(_lookup(record, path) in allowed for path, allowed in self.conditions)


def _lookup(record: Any, path: str) -> Any:
    value = record
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return _MISSING
        value = value[key]
    return value


def _values(value: Any) -> tuple[Any, ...]:
    return tuple(value) if isinstance(value, (tuple, list, set, frozenset)) else (value,)


def _handler(aux_name: str | None, kind: Any, where: Mapping[str, Any] | None,
             media_type: str | None) -> Callable[[Callable], Callable]:
    conditions = ([("type", _values(kind))] if kind is not None else []) + [
        (path, _values(value)) for path, value in (where or {}).items()]
    selector = Selector(aux_name, tuple(conditions), media_type)

    def mark(method: Callable) -> Callable:
        method.__dict__.setdefault("__yatf_selectors__", []).append(selector)
        return method
    return mark


def native(*, type: Any = None, where: Mapping[str, Any] | None = None) -> Callable[[Callable], Callable]:
    """Handle native records whose ``type`` is ``type`` (one value or several) and whose fields at the dotted ``where``
    paths hold the given values. Of the handlers that match a record, the one with the most conditions takes it."""
    return _handler(None, type, where, None)


def aux(name: str, *, media_type: str = "application/jsonl", type: Any = None,
        where: Mapping[str, Any] | None = None) -> Callable[[Callable], Callable]:
    """Handle records of the aux file ``name``, matched as ``native`` matches; this also declares the file. Aux
    records no handler takes are dropped, since the run stores them anyway."""
    return _handler(name, type, where, media_type)


def finish(method: Callable) -> Callable:
    """Run at turn end, after the last record: emit what's held and reset per-turn state."""
    method.__yatf_finish__ = True
    return method


@dataclass(frozen=True)
class AdapterSpec:
    """What ``@yatf_adapter`` declares, and its handlers by selector."""

    name: str
    native_format: str
    version: str
    usage_input_includes_cache: bool | None
    tested_with: tuple[str, ...]
    lossy: tuple[str, ...]
    namespace: str
    aux: Mapping[str, str]
    handlers: tuple[tuple[Selector, str], ...]
    finish: str | None

    def handler(self, aux_name: str | None, record: Any) -> str | None:
        """The method that takes ``record``: the most specific matching handler, None when none matches."""
        matches = [(len(sel.conditions), method) for sel, method in self.handlers
                   if sel.aux == aux_name and sel.matches(record)]
        if not matches:
            return None
        best = max(count for count, _ in matches)
        methods = sorted({method for count, method in matches if count == best})
        if len(methods) > 1:
            raise YatfError(f"a record matches {', '.join(methods)} equally; make one of them more specific")
        return methods[0]


def yatf_adapter(*, native_format: str, version: str, usage_input_includes_cache: bool | None,
                 tested_with: Sequence[str] = (), lossy: Sequence[str] = (), name: str | None = None,
                 namespace: str | None = None) -> Callable[[type[A]], type[A]]:
    """Declare an Adapter subclass for ``native_format``.

    ``usage_input_includes_cache`` says whether the native input token count already includes cache reads and writes
    (None when the native reports no usage). ``tested_with`` names the native producer versions its recorded fixtures
    come from, ``lossy`` only what the native never records, and ``namespace`` its extension types
    (``<namespace>/raw``), by default the native format's name."""
    if not isinstance(native_format, str) or not NATIVE_FORMAT.match(native_format):
        raise YatfError(f"native_format {native_format!r} isn't <name>/<version>")
    if native_format.startswith("yatf/"):
        raise YatfError(f"native_format {native_format!r}: yatf is the format adapters produce, not a native one")
    if not isinstance(version, str) or not version:
        raise YatfError("version is the adapter's own version string")
    if usage_input_includes_cache not in (True, False, None):
        raise YatfError("usage_input_includes_cache is true, false or None")
    ns = namespace or native_format.split("/", 1)[0]
    if not is_extension_type(f"{ns}/raw"):
        raise YatfError(f"namespace {ns!r} can't name extension types")

    def declare(cls: type[A]) -> type[A]:
        if not (isinstance(cls, type) and issubclass(cls, Adapter)):
            raise TypeError("@yatf_adapter decorates an Adapter subclass")
        members: dict[str, Any] = {}
        for klass in reversed(cls.__mro__):
            members.update(vars(klass))
        owners: dict[Selector, str] = {}
        aux_types: dict[str, str] = {}
        finishes = []
        for attr, member in members.items():
            if getattr(member, "__yatf_finish__", False):
                finishes.append(attr)
            for selector in getattr(member, "__yatf_selectors__", ()):
                if selector in owners and owners[selector] != attr:
                    raise YatfError(f"{cls.__name__}.{owners[selector]} and .{attr} take the same records")
                owners[selector] = attr
                if selector.aux is not None:
                    known = aux_types.setdefault(selector.aux, selector.media_type)
                    if known != selector.media_type:
                        raise YatfError(f"aux file {selector.aux!r} is declared as both {known} and "
                                        f"{selector.media_type}")
        if len(finishes) > 1:
            raise YatfError(f"{cls.__name__} has more than one @finish method: {', '.join(finishes)}")
        cls.spec = AdapterSpec(
            name=name or f"{ns}-yatf", native_format=native_format, version=version,
            usage_input_includes_cache=usage_input_includes_cache, tested_with=tuple(tested_with),
            lossy=tuple(lossy), namespace=ns, aux=dict(aux_types),
            handlers=tuple((selector, attr) for selector, attr in owners.items()),
            finish=finishes[0] if finishes else None)
        return cls
    return declare


class Adapter:
    """Base of a yatf/1 adapter, declared with ``@yatf_adapter``. One instance serves an A2A context across its turns,
    so state on ``self`` carries over; per-turn state resets in the ``@finish`` method. A handler may hold an event
    back until another record places it: capture ``turn.source`` and pass it as ``derived_from`` when it's emitted."""

    spec: ClassVar[AdapterSpec]

    def __init__(self, turn: YatfTurn) -> None:
        """Set up for the context whose first turn here is ``turn``."""

    def unmatched(self, record: Any, turn: YatfTurn) -> None:
        """A native record no handler takes, kept as a ``<namespace>/raw`` event so nothing is lost."""
        turn.raw(record)

    @classmethod
    def from_yatf(cls, events: list[Event]) -> Native:
        """A context's events, every turn so far, back to native."""
        raise NotImplementedError(f"{cls.__name__} doesn't convert yatf/1 back to its native format")

    @classmethod
    def normal_form(cls, records: list[Any]) -> list[Any]:
        """What the round trip compares: by default the parsed records themselves."""
        return list(records)


@dataclass(frozen=True, kw_only=True)
class TurnContext:
    """What the SDK knows of a turn from A2A, written into its frame: ``turn.start`` with the declared actors (by
    default the agent and, with an inbound message, its sender), the inbound message from ``inbound_actor``, and
    ``turn.end``. ``actor`` is the agent's own actor id, the default for every event the adapter writes."""

    index: int
    context_id: str
    task_id: str
    message_id: str | None = None
    actor: str = "agent"
    actors: Sequence[Actor | Mapping[str, Any]] = ()
    inbound: Sequence[Part | Mapping[str, Any]] | None = None
    inbound_actor: str = "caller"
    started_at: str | None = None
    live: bool = True


def _time(value: Any) -> Time:
    if value is None or isinstance(value, str):
        return Time(at=value)
    return value if isinstance(value, Time) else Time.model_validate(value)


def _content(content: Iterable[Any] | None, text: str | None) -> list[Any]:
    if (content is None) == (text is None):
        raise YatfError("give either content or text")
    return [{"type": "text", "text": text}] if text is not None else list(content)


class YatfTurn:
    """Builds one turn's events. An adapter calls its methods and writes only what the native says; the turn stamps
    ``seq``, ``id``, ``turn``, ``observer``, the actor (the agent's unless given), the attempt and ``derived_from``
    (the record being handled, unless given). Each method returns the event it emitted. Events are kept as JSON text,
    so what a reader is handed is a copy, and an event that isn't JSON is refused as it's written. Not thread-safe:
    records are fed one at a time."""

    def __init__(self, context: TurnContext, spec: AdapterSpec) -> None:
        self.context = context
        self.spec = spec
        #: The record being handled: capture it to emit an event that comes from it later.
        self.source: DerivedFrom | None = None
        self._events: list[str] = []
        self._drained = 0
        self._calls = 0
        self._attempt = 1
        self._final_metrics: FinalMetrics | None = None

    @property
    def index(self) -> int:
        return self.context.index

    @property
    def events(self) -> list[dict[str, Any]]:
        return [json.loads(text) for text in self._events]

    def message(self, *, role: str, step: str | None = None, content: Iterable[Any] | None = None,
                text: str | None = None, time: Any = None, actor: str | None = None, derived_from: Any = _CURRENT,
                x: Mapping[str, Any] | None = None) -> Message:
        return self._emit(Message, "message", time, actor, derived_from, x, step=step, role=role,
                          content=_content(content, text))

    def reasoning(self, *, visibility: str, step: str | None = None, text: str | None = None,
                  opaque: Iterable[Opaque | Mapping[str, Any]] | None = None, time: Any = None,
                  actor: str | None = None, derived_from: Any = _CURRENT,
                  x: Mapping[str, Any] | None = None) -> Reasoning:
        return self._emit(Reasoning, "reasoning", time, actor, derived_from, x, ("text",), step=step,
                          visibility=visibility, text=text, opaque=None if opaque is None else list(opaque))

    def tool_call(self, *, tool_call_id: str, function_name: str, arguments: Any, step: str | None = None,
                  kind: str | None = None, server: str | None = None, executor: str | None = None,
                  env_id: str | None = None, arguments_raw: str | None = None, time: Any = None,
                  actor: str | None = None, derived_from: Any = _CURRENT,
                  x: Mapping[str, Any] | None = None) -> ToolCall:
        return self._emit(ToolCall, "tool_call", time, actor, derived_from, x, ("arguments",), step=step,
                          tool_call_id=tool_call_id, function_name=function_name, arguments=arguments, kind=kind,
                          server=server, executor=executor, env_id=env_id, arguments_raw=arguments_raw)

    def tool_result(self, *, source_call_id: str, status: str = "ok", content: Iterable[Any] | None = None,
                    text: str | None = None, exit_code: int | None = None, step: str | None = None, time: Any = None,
                    actor: str | None = None, derived_from: Any = _CURRENT,
                    x: Mapping[str, Any] | None = None) -> ToolResult:
        return self._emit(ToolResult, "tool_result", time, actor, derived_from, x, step=step,
                          source_call_id=source_call_id, status=status, content=_content(content, text),
                          exit_code=exit_code)

    def step(self, *, key: str, source: str = "agent", llm_call_count: int | None = 1, model_name: str | None = None,
             reasoning_effort: str | None = None, metrics: Metrics | Mapping[str, Any] | None = None,
             stop_reason: str | None = None, provider_ids: Mapping[str, str] | None = None,
             first_token_at: str | None = None, sampling: Mapping[str, Any] | None = None, time: Any = None,
             actor: str | None = None, derived_from: Any = _CURRENT,
             x: Mapping[str, Any] | None = None) -> StepEvent:
        return self._emit(StepEvent, "step", time, actor, derived_from, x, step=key, source=source,
                          llm_call_count=llm_call_count, model_name=model_name, reasoning_effort=reasoning_effort,
                          metrics=metrics, stop_reason=stop_reason,
                          provider_ids=None if provider_ids is None else dict(provider_ids),
                          first_token_at=first_token_at, sampling=None if sampling is None else dict(sampling))

    def tools(self, tools: Iterable[ToolDef | Mapping[str, Any]], *, mode: str = "replace", time: Any = None,
              actor: str | None = None, derived_from: Any = _CURRENT, x: Mapping[str, Any] | None = None) -> Tools:
        return self._emit(Tools, "tools", time, actor, derived_from, x, mode=mode, tools=list(tools))

    def declare(self, declared: Actor | Mapping[str, Any], *, time: Any = None, actor: str | None = None,
                derived_from: Any = _CURRENT, x: Mapping[str, Any] | None = None) -> ActorDeclared:
        """Declare an actor that appears mid-turn, such as a sub-agent, before any event names it."""
        return self._emit(ActorDeclared, "actor", time, actor, derived_from, x, declared=declared)

    def error(self, *, message: str, error_type: str = "agent_error", code: str | None = None,
              fatal: bool | None = None, step: str | None = None, time: Any = None, actor: str | None = None,
              derived_from: Any = _CURRENT, x: Mapping[str, Any] | None = None) -> ErrorEvent:
        return self._emit(ErrorEvent, "error", time, actor, derived_from, x, step=step, error_type=error_type,
                          message=message, code=code, fatal=fatal)

    def attempt(self, number: int, *, reason: str | None = None, time: Any = None, actor: str | None = None,
                derived_from: Any = _CURRENT, x: Mapping[str, Any] | None = None) -> Attempt:
        """A harness retry within the task: this event and every later one carry ``attempt`` = ``number``."""
        self._attempt = number
        return self._emit(Attempt, "attempt", time, actor, derived_from, x, number=number, reason=reason)

    def note(self, text: str, *, time: Any = None, actor: str | None = None, derived_from: Any = _CURRENT,
             x: Mapping[str, Any] | None = None) -> Note:
        return self._emit(Note, "note", time, actor, derived_from, x, text=text)

    def raw(self, record: Any, *, time: Any = None, actor: str | None = None,
            derived_from: Any = _CURRENT) -> ExtensionEvent:
        """A native record kept whole, as ``<namespace>/raw``."""
        return self.extension(f"{self.spec.namespace}/raw", {"record": record}, time=time, actor=actor,
                              derived_from=derived_from)

    def extension(self, kind: str, body: Mapping[str, Any], *, time: Any = None, actor: str | None = None,
                  derived_from: Any = _CURRENT, x: Mapping[str, Any] | None = None) -> ExtensionEvent:
        """An event of the ``<namespace>/<name>`` type ``kind``, with ``body`` beside the envelope."""
        if not is_extension_type(kind):
            raise YatfError(f"{kind!r} isn't an extension type: <namespace>/<name>, outside the yatf namespace")
        if clash := sorted(_ENVELOPE & set(body)):
            raise YatfError(f"an extension body can't set envelope fields: {', '.join(clash)}")
        return self._emit(ExtensionEvent, kind, time, actor, derived_from, x, tuple(body), **body)

    def final_metrics(self, metrics: FinalMetrics | Mapping[str, Any]) -> None:
        """The turn's totals, written into ``turn.end``."""
        try:
            self._final_metrics = (metrics if isinstance(metrics, FinalMetrics)
                                   else FinalMetrics.model_validate(metrics))
        except ValidationError as exc:
            raise YatfError(f"final_metrics: {exc}") from exc

    def _emit(self, model: type[Event], event_type: str, time: Any, actor: str | None, derived_from: Any,
              x: Mapping[str, Any] | None, nulls: tuple[str, ...] = (), /, **body: Any) -> Any:
        source = self.source if derived_from is _CURRENT else derived_from
        seq = len(self._events)
        envelope: dict[str, Any] = {"seq": seq, "id": f"{self.index}.{seq}",
                                    "type": event_type, "turn": self.index, "actor": actor or self.context.actor,
                                    "observer": "agent", "time": _time(time)}
        if self._attempt > 1:
            envelope["attempt"] = self._attempt
        if source is not None:
            envelope["derived_from"] = source
        if x is not None:
            envelope["x"] = dict(x)
        fields = {key: value for key, value in body.items() if value is not None or key in nulls}
        try:
            event = model(**envelope, **fields)
            text = json.dumps(dump_event(event), separators=(",", ":"), allow_nan=False)
        except ValidationError as exc:
            raise YatfError(f"{event_type} event: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise YatfError(f"{event_type} event isn't JSON: {exc}") from exc
        self._events.append(text)
        self._calls += event_type == "tool_call"
        return event

    def _start(self) -> None:
        context = self.context
        actors = list(context.actors) or [Actor(id=context.actor, kind="agent")] + (
            [Actor(id=context.inbound_actor, kind="framework")] if context.inbound is not None else [])
        self._emit(TurnStart, "turn.start", context.started_at, None, None, None, ("usage_input_includes_cache",),
                   spec_version=SPEC_VERSION, context_id=context.context_id, task_id=context.task_id,
                   message_id=context.message_id, native_format=self.spec.native_format, live=context.live,
                   adapter=AdapterInfo(name=self.spec.name, version=self.spec.version,
                                       lossy=list(self.spec.lossy) or None),
                   usage_input_includes_cache=self.spec.usage_input_includes_cache, actors=actors)
        if context.inbound is not None:
            self._emit(Message, "message", context.started_at, context.inbound_actor, None, None,
                       step=INBOUND_STEP.format(turn=self.index), role="user", content=list(context.inbound))

    def _end(self, state: str, final: Sequence[Any] | None, ended_at: str | None) -> None:
        if final is not None:
            self._emit(FinalResponse, "final_response", ended_at, None, None, None, content=list(final))
        self._emit(TurnEnd, "turn.end", ended_at, None, None, None, state=state, tool_call_count=self._calls,
                   final_metrics=self._final_metrics)

    def _drain(self) -> list[str]:
        new, self._drained = self._events[self._drained:], len(self._events)
        return new
