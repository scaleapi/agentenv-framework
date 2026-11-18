"""Running an adapter: feed one A2A context's records through it, a turn at a time. The run writes each turn's frame and
hands every handler the YatfTurn it writes through. The SDK runs it live and the round-trip kit replays it offline, so
the same records, fed in the same order, make the same events."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from pydantic import ValidationError

from .adapter import Adapter, AdapterSpec
from .models import (
    EVENT_MODELS,
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

#: The step key of a turn's inbound message, ``in-<turn>``.
INBOUND_STEP = "in-{turn}"
_ENVELOPE = frozenset(Event.model_fields)
_TYPE_OF = {model: kind for kind, model in EVENT_MODELS.items()}
_CURRENT: Any = object()


@dataclass(frozen=True, kw_only=True)
class TurnContext:
    """What the SDK knows of a turn from A2A, written into its frame. ``actor`` is the agent's own actor id, the
    default for every event; ``actors`` defaults to the agent and, with an ``inbound`` message, its sender."""

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
    """What a handler writes one turn's events through. It stamps the envelope (``seq``, ``id``, ``turn``,
    ``observer``, the agent's actor, the attempt, ``derived_from``) and keeps each event as JSON text, refusing one
    that isn't JSON as it's written, and any once the turn has closed. Not thread-safe: records are fed one at a
    time."""

    def __init__(self, context: TurnContext, spec: AdapterSpec, out: list[str] | None = None) -> None:
        self.context = context
        self.spec = spec
        #: The record being handled: capture it to emit an event that comes from it later.
        self.source: DerivedFrom | None = None
        self._out = [] if out is None else out
        self._first = len(self._out)
        self._last: int | None = None
        self._calls = 0
        self._attempt = 1
        self._final_metrics: FinalMetrics | None = None

    @property
    def index(self) -> int:
        return self.context.index

    @property
    def events(self) -> list[dict[str, Any]]:
        """This turn's events so far, and no later turn's, as fresh copies."""
        return [json.loads(text) for text in self._out[self._first:self._last]]

    def message(self, *, role: str, step: str | None = None, content: Iterable[Any] | None = None,
                text: str | None = None, time: Any = None, actor: str | None = None, derived_from: Any = _CURRENT,
                x: Mapping[str, Any] | None = None, is_copied_context: bool | None = None) -> Message:
        return self._emit(Message, {"step": step, "role": role, "content": _content(content, text)},
                          time=time, actor=actor, derived_from=derived_from, x=x,
                          is_copied_context=is_copied_context)

    def reasoning(self, *, visibility: str, step: str | None = None, text: str | None = None,
                  opaque: Iterable[Opaque | Mapping[str, Any]] | None = None, time: Any = None,
                  actor: str | None = None, derived_from: Any = _CURRENT,
                  x: Mapping[str, Any] | None = None, is_copied_context: bool | None = None) -> Reasoning:
        return self._emit(Reasoning, {"step": step, "visibility": visibility, "text": text,
                                      "opaque": None if opaque is None else list(opaque)},
                          time=time, actor=actor, derived_from=derived_from, x=x,
                          is_copied_context=is_copied_context)

    def tool_call(self, *, tool_call_id: str, function_name: str, arguments: Any, step: str | None = None,
                  kind: str | None = None, server: str | None = None, executor: str | None = None,
                  env_id: str | None = None, arguments_raw: str | None = None, time: Any = None,
                  actor: str | None = None, derived_from: Any = _CURRENT,
                  x: Mapping[str, Any] | None = None, is_copied_context: bool | None = None) -> ToolCall:
        return self._emit(ToolCall, {"step": step, "tool_call_id": tool_call_id, "function_name": function_name,
                                     "arguments": arguments, "kind": kind, "server": server, "executor": executor,
                                     "env_id": env_id, "arguments_raw": arguments_raw},
                          time=time, actor=actor, derived_from=derived_from, x=x,
                          is_copied_context=is_copied_context)

    def tool_result(self, *, source_call_id: str, status: str = "ok", content: Iterable[Any] | None = None,
                    text: str | None = None, exit_code: int | None = None, step: str | None = None, time: Any = None,
                    actor: str | None = None, derived_from: Any = _CURRENT,
                    x: Mapping[str, Any] | None = None, is_copied_context: bool | None = None) -> ToolResult:
        return self._emit(ToolResult, {"step": step, "source_call_id": source_call_id, "status": status,
                                       "content": _content(content, text), "exit_code": exit_code},
                          time=time, actor=actor, derived_from=derived_from, x=x,
                          is_copied_context=is_copied_context)

    def step(self, *, key: str, source: str = "agent", llm_call_count: int | None = 1, model_name: str | None = None,
             reasoning_effort: str | None = None, metrics: Metrics | Mapping[str, Any] | None = None,
             stop_reason: str | None = None, provider_ids: Mapping[str, str] | None = None,
             first_token_at: str | None = None, sampling: Mapping[str, Any] | None = None, time: Any = None,
             actor: str | None = None, derived_from: Any = _CURRENT,
             x: Mapping[str, Any] | None = None, is_copied_context: bool | None = None) -> StepEvent:
        return self._emit(StepEvent, {"step": key, "source": source, "llm_call_count": llm_call_count,
                                      "model_name": model_name, "reasoning_effort": reasoning_effort,
                                      "metrics": metrics, "stop_reason": stop_reason,
                                      "provider_ids": None if provider_ids is None else dict(provider_ids),
                                      "first_token_at": first_token_at,
                                      "sampling": None if sampling is None else dict(sampling)},
                          time=time, actor=actor, derived_from=derived_from, x=x,
                          is_copied_context=is_copied_context)

    def tools(self, tools: Iterable[ToolDef | Mapping[str, Any]], *, mode: str = "replace", time: Any = None,
              actor: str | None = None, derived_from: Any = _CURRENT, x: Mapping[str, Any] | None = None,
              is_copied_context: bool | None = None) -> Tools:
        return self._emit(Tools, {"mode": mode, "tools": list(tools)},
                          time=time, actor=actor, derived_from=derived_from, x=x,
                          is_copied_context=is_copied_context)

    def declare(self, declared: Actor | Mapping[str, Any], *, time: Any = None, actor: str | None = None,
                derived_from: Any = _CURRENT, x: Mapping[str, Any] | None = None,
                is_copied_context: bool | None = None) -> ActorDeclared:
        """Declare an actor that appears mid-turn, such as a sub-agent, before any event names it."""
        return self._emit(ActorDeclared, {"declared": declared},
                          time=time, actor=actor, derived_from=derived_from, x=x,
                          is_copied_context=is_copied_context)

    def error(self, *, message: str, error_type: str = "agent_error", code: str | None = None,
              fatal: bool | None = None, step: str | None = None, time: Any = None, actor: str | None = None,
              derived_from: Any = _CURRENT, x: Mapping[str, Any] | None = None,
              is_copied_context: bool | None = None) -> ErrorEvent:
        return self._emit(ErrorEvent, {"step": step, "error_type": error_type, "message": message, "code": code,
                                       "fatal": fatal},
                          time=time, actor=actor, derived_from=derived_from, x=x,
                          is_copied_context=is_copied_context)

    def attempt(self, number: int, *, reason: str | None = None, time: Any = None, actor: str | None = None,
                derived_from: Any = _CURRENT, x: Mapping[str, Any] | None = None,
                is_copied_context: bool | None = None) -> Attempt:
        """A harness retry within the task: this event and every later one carry ``attempt`` = ``number``."""
        self._attempt = number
        return self._emit(Attempt, {"number": number, "reason": reason},
                          time=time, actor=actor, derived_from=derived_from, x=x,
                          is_copied_context=is_copied_context)

    def note(self, text: str, *, time: Any = None, actor: str | None = None, derived_from: Any = _CURRENT,
             x: Mapping[str, Any] | None = None, is_copied_context: bool | None = None) -> Note:
        return self._emit(Note, {"text": text}, time=time, actor=actor, derived_from=derived_from, x=x,
                          is_copied_context=is_copied_context)

    def raw(self, record: Any, *, time: Any = None, actor: str | None = None, derived_from: Any = _CURRENT,
            is_copied_context: bool | None = None) -> ExtensionEvent:
        """A native record kept whole, as ``<namespace>/raw``."""
        return self.extension(f"{self.spec.namespace}/raw", {"record": record}, time=time, actor=actor,
                              derived_from=derived_from, is_copied_context=is_copied_context)

    def extension(self, kind: str, body: Mapping[str, Any], *, time: Any = None, actor: str | None = None,
                  derived_from: Any = _CURRENT, x: Mapping[str, Any] | None = None,
                  is_copied_context: bool | None = None) -> ExtensionEvent:
        """An event of the ``<namespace>/<name>`` type ``kind``, with every key of ``body`` beside the envelope."""
        if not is_extension_type(kind):
            raise YatfError(f"{kind!r} isn't an extension type: <namespace>/<name>, outside the yatf namespace")
        if clash := sorted(_ENVELOPE & set(body)):
            raise YatfError(f"an extension body can't set envelope fields: {', '.join(clash)}")
        return self._emit(ExtensionEvent, {**body, "type": kind}, time=time, actor=actor, derived_from=derived_from,
                          x=x, is_copied_context=is_copied_context)

    def final_metrics(self, metrics: FinalMetrics | Mapping[str, Any]) -> None:
        """The turn's totals, written into ``turn.end``."""
        try:
            self._final_metrics = (metrics if isinstance(metrics, FinalMetrics)
                                   else FinalMetrics.model_validate(metrics))
        except ValidationError as exc:
            raise YatfError(f"final_metrics: {exc}") from exc

    def _emit(self, model: type[Event], body: dict[str, Any], *, time: Any = None, actor: str | None = None,
              derived_from: Any = _CURRENT, x: Mapping[str, Any] | None = None,
              is_copied_context: bool | None = None) -> Any:
        """Write one event. A None in ``body`` leaves an optional field out and writes null for a required one or for
        an extension's own key, so whether a null is written follows the model."""
        event_type = body.pop("type", None) or _TYPE_OF[model]
        if self._last is not None:
            raise YatfError(f"turn {self.index} is closed: write through the turn the handler is given")
        source = self.source if derived_from is _CURRENT else derived_from
        seq = len(self._out) - self._first
        envelope: dict[str, Any] = {"seq": seq, "id": f"{self.index}.{seq}", "type": event_type, "turn": self.index,
                                    "actor": actor or self.context.actor, "observer": "agent", "time": _time(time)}
        if self._attempt > 1:
            envelope["attempt"] = self._attempt
        if source is not None:
            envelope["derived_from"] = source
        if x is not None:
            envelope["x"] = dict(x)
        if is_copied_context is not None:
            envelope["is_copied_context"] = is_copied_context
        fields = {key: value for key, value in body.items()
                  if value is not None or key not in model.model_fields or model.model_fields[key].is_required()}
        try:
            event = model(**envelope, **fields)
            text = json.dumps(dump_event(event), separators=(",", ":"), allow_nan=False)
        except ValidationError as exc:
            raise YatfError(f"{event_type} event: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise YatfError(f"{event_type} event isn't JSON: {exc}") from exc
        self._out.append(text)
        self._calls += event_type == "tool_call"
        return event

    def _start(self) -> None:
        context = self.context
        actors = list(context.actors) or [Actor(id=context.actor, kind="agent")] + (
            [Actor(id=context.inbound_actor, kind="framework")] if context.inbound is not None else [])
        adapter = AdapterInfo(name=self.spec.name, version=self.spec.version, lossy=list(self.spec.lossy) or None)
        self._emit(TurnStart, {"spec_version": SPEC_VERSION, "context_id": context.context_id,
                               "task_id": context.task_id, "message_id": context.message_id,
                               "native_format": self.spec.native_format, "live": context.live, "adapter": adapter,
                               "usage_input_includes_cache": self.spec.usage_input_includes_cache, "actors": actors},
                   time=context.started_at, derived_from=None)
        if context.inbound is not None:
            self._emit(Message, {"step": INBOUND_STEP.format(turn=self.index), "role": "user",
                                 "content": list(context.inbound)},
                       time=context.started_at, actor=context.inbound_actor, derived_from=None)

    def _end(self, state: str, final: Sequence[Any] | None, ended_at: str | None) -> None:
        if final is not None:
            self._emit(FinalResponse, {"content": list(final)}, time=ended_at, derived_from=None)
        self._emit(TurnEnd, {"state": state, "tool_call_count": self._calls, "final_metrics": self._final_metrics},
                   time=ended_at, derived_from=None)
        self._last = len(self._out)


@dataclass(frozen=True)
class Fed:
    """One record as it was fed: a native record, or a record of the aux file ``aux``."""

    record: Any
    aux: str | None = None


@dataclass(frozen=True, kw_only=True)
class RecordedTurn:
    """A stored turn: its context, its records in the order they were fed, and how it ended."""

    context: TurnContext
    records: Sequence[Fed] = ()
    state: str = "completed"
    final: Sequence[Part | Mapping[str, Any]] | None = None
    ended_at: str | None = None


class AdapterRun:
    """One A2A context's conversion. Each call returns the events it emitted, so a live reader sees them as they come,
    and ``events`` holds them all; every read is a fresh copy, so a caller can't rewrite history."""

    def __init__(self, adapter: type[Adapter]) -> None:
        spec = vars(adapter).get("spec")
        if not isinstance(spec, AdapterSpec):
            raise YatfError(f"{adapter.__name__} isn't declared with @yatf_adapter")
        self._cls = adapter
        self._spec = spec
        self._adapter: Adapter | None = None
        self._turn: YatfTurn | None = None
        self._events: list[str] = []
        self._sent = 0
        self._native = 0
        self._aux: dict[str, int] = {}

    @property
    def events(self) -> list[dict[str, Any]]:
        return [json.loads(text) for text in self._events]

    def start_turn(self, context: TurnContext) -> list[dict[str, Any]]:
        """Open a turn: write ``turn.start`` and the inbound message. The context's first turn creates the adapter."""
        if self._turn is not None:
            raise YatfError(f"turn {self._turn.index} is still open")
        turn = YatfTurn(context, self._spec, self._events)
        self._turn, self._native, self._aux = turn, 0, {}
        turn._start()
        if self._adapter is None:
            self._adapter = self._cls(turn)
        return self._page()

    def native(self, record: Any) -> list[dict[str, Any]]:
        """Feed the turn's next native record."""
        turn = self._open()
        source, self._native = DerivedFrom(index=self._native), self._native + 1
        self._handle(turn, source, self._spec.handler(None, record) or "unmatched", record)
        return self._page()

    def aux(self, name: str, record: Any) -> list[dict[str, Any]]:
        """Feed the turn's next record of aux file ``name``."""
        turn = self._open()
        if name not in self._spec.aux:
            raise YatfError(f"{self._cls.__name__} declares no aux file {name!r}")
        index = self._aux.get(name, 0)
        self._aux[name] = index + 1
        method = self._spec.handler(name, record)
        if method is not None:
            self._handle(turn, DerivedFrom(index=index, aux=name), method, record)
        return self._page()

    def end_turn(self, *, state: str = "completed", final: Sequence[Part | Mapping[str, Any]] | None = None,
                 ended_at: str | None = None) -> list[dict[str, Any]]:
        """Close the turn: run the adapter's ``@finish`` method, then write ``final_response`` and ``turn.end``. The
        frame is written even when ``@finish`` raises, and the error is raised after it."""
        turn = self._open()
        try:
            if self._spec.finish is not None:
                getattr(self._adapter, self._spec.finish)(turn)
        finally:
            turn._end(state, final, ended_at)
            self._turn = None
            page = self._page()
        return page

    def _open(self) -> YatfTurn:
        if self._turn is None:
            raise YatfError("no turn is open: call start_turn first")
        return self._turn

    def _handle(self, turn: YatfTurn, source: DerivedFrom, method: str, record: Any) -> None:
        turn.source = source
        try:
            getattr(self._adapter, method)(record, turn)
        finally:
            turn.source = None

    def _page(self) -> list[dict[str, Any]]:
        page, self._sent = self._events[self._sent:], len(self._events)
        return [json.loads(text) for text in page]


def replay(adapter: type[Adapter], turns: Iterable[RecordedTurn]) -> list[dict[str, Any]]:
    """A context's stored turns run through ``adapter`` again, as the agent ran them live."""
    run = AdapterRun(adapter)
    for turn in turns:
        run.start_turn(turn.context)
        for fed in turn.records:
            if fed.aux is None:
                run.native(fed.record)
            else:
                run.aux(fed.aux, fed.record)
        run.end_turn(state=turn.state, final=turn.final, ended_at=turn.ended_at)
    return run.events
