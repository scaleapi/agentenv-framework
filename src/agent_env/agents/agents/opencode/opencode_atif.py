"""OpenCode's ATIF adapter: the CLI's ``run --format json`` events, the native log, and the model calls the turn's
relay forwarded, which the bridge feeds as aux records (``calls``), as ATIF. The events hold the root session's steps
whole, each part keyed by its assistant message, but no model, and nothing of the calls the CLI makes outside them:
its title's, its sub-agents' sessions' and its retried attempts'. The relay sees every call, with its model, its times
and its usage; and the bridge feeds each sub-agent session a task call ran, read from the CLI's session store once the
call ended, as the events the CLI would print for it (``subagents``).

A step is one model response, keyed by its message's id: its reasoning, its text, its calls with their results, and
the usage of the relay's call that made it, none for a call answered without usage (the CLI's zeros for it aren't
counts); one no call is known for keeps its own counts. A root step's call is the one whose counts are the step's, else
the latest to start before the step did, and the step is written once that call is known: at its step_finish when the
CLI goes on, else at the next step, the CLI's error or the task's end, so the calls it made alongside, each a step of
its own, its usage alone or, refused, its error, come before it, and the task's last step is its answer. A task call's
sub-agent is embedded with its session's steps, each joined by its counts, its cache read's too, to a call inside the
task call's run, so sub-agents run at once are told apart (where those tie, the latest to start before the step did),
and the steps no counts name (cut short, or on an endpoint that sends no usage) each to a call left on its sub-agent's
model that started before it did, the latest likeliest, as many as can have one; the calls inside the run none of them
made are its too, unless another task call ran at once. A task call whose session wasn't fed is a sub-agent only when
no other task call of its step ran at once: the calls inside its run are its steps, its last holding its answer. One
whose part names no session (it failed without one) is none, the calls inside its run the root's. The CLI's own report
of a failed call is the error of the call the endpoint refused last."""
from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agentenv_protocol.atif import Adapter, Native, atif_adapter, aux, finish, is_token_count, native

NATIVE_FORMAT = "opencode-events/1"
#: The CLI versions the adapter's records come from; a unit test holds package.json's pin to this list.
TESTED_WITH = ("1.17.18",)
#: The step_finish reasons after which the CLI makes another model call.
_CONTINUING = frozenset({"tool-calls", "unknown"})
_TASK = "task"
_STATUSES = {"completed": "ok", "error": "error"}
_COUNTS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
_DATA_URL = re.compile(r"data:(?P<type>[\w.+-]+/[\w.+-]+);base64,(?P<data>.*)", re.S)
_TASK_RESULT = re.compile(r"<task_result>\n?(?P<text>.*?)\n?</task_result>", re.S)
#: The event the CLI prints for each kind of part.
_EVENTS = {"step-start": "step_start", "reasoning": "reasoning", "text": "text", "tool": "tool_use",
           "step-finish": "step_finish"}


def _is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _count(value: Any) -> int | None:
    return value if is_token_count(value) else None


def _moment(value: Any) -> int | None:
    """An event's time: epoch milliseconds."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _iso(ms: Any) -> str | None:
    """An event's time as the builder writes one."""
    if _moment(ms) is None:
        return None
    return f"{datetime.fromtimestamp(ms // 1000, timezone.utc):%Y-%m-%dT%H:%M:%S}.{ms % 1000:03d}Z"


def _ms(stamp: str) -> int:
    """A relay record's time (ISO 8601, ``Z``) as epoch milliseconds, the clock the CLI's events are on."""
    return round(datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp() * 1000)


def _usage(counts: Any) -> dict[str, int]:
    """A relay record's counts: its input holds the cache, its output the reasoning."""
    if not _is_mapping(counts):
        return {}
    return {name: counts[name] for name in _COUNTS if is_token_count(counts.get(name))}


def _native_usage(tokens: Any) -> dict[str, int]:
    """A step_finish's counts as the relay counts a call, the cache in its input and the reasoning in its output,
    which the CLI counts apart."""
    if not _is_mapping(tokens):
        return {}
    cache = tokens.get("cache") if _is_mapping(tokens.get("cache")) else {}
    read, write = _count(cache.get("read")), _count(cache.get("write"))
    inputs, output = _count(tokens.get("input")), _count(tokens.get("output"))
    counts = {"input_tokens": None if inputs is None else inputs + (read or 0) + (write or 0),
              "output_tokens": None if output is None else output + (_count(tokens.get("reasoning")) or 0),
              "cache_read_tokens": read, "cache_write_tokens": write}
    return {name: count for name, count in counts.items() if count is not None}


@dataclass
class _Step:
    """A step's events while its call isn't known: when it started, its content records in order, and its
    step_finish with when it was printed."""

    key: str
    started: int | None
    content: list[dict[str, Any]] = field(default_factory=list)
    finish: dict[str, Any] | None = None
    finished: int | None = None

    def tools(self) -> list[dict[str, Any]]:
        return [record["part"] for record in self.content if record.get("type") == "tool_use"]

    def windows(self) -> dict[str, tuple[int, int]]:
        """Each task call's run, by its call id."""
        found = {}
        for part in self.tools():
            times = (part.get("state") or {}).get("time") or {}
            start, end = _moment(times.get("start")), _moment(times.get("end"))
            if part.get("tool") == _TASK and start is not None and end is not None:
                found[part["callID"]] = (start, end)
        return found


@dataclass
class _Child:
    """A task call's sub-agent: its session's steps, each with the call that made it, and the calls inside the task
    call's run that none of them made; with no steps fed, those calls alone."""

    steps: list[tuple[_Step, dict[str, Any] | None]]
    calls: list[dict[str, Any]]


def _alone(windows: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    """The windows no other window overlaps: the calls inside one of them are its sub-agent's alone."""
    return {call_id: (start, end) for call_id, (start, end) in windows.items()
            if not any(other != call_id and start <= other_end and other_start <= end
                       for other, (other_start, other_end) in windows.items())}


def _inside(record: dict[str, Any], window: tuple[int, int]) -> bool:
    return window[0] <= _ms(record["started"]) and _ms(record["timestamp"]) <= window[1]


def _among(record: dict[str, Any], records: list[dict[str, Any]]) -> bool:
    return any(record is other for other in records)


def _started(record: dict[str, Any]) -> int:
    return _ms(record["started"])


def _same(held: dict[str, Any], counts: dict[str, int], names: tuple[str, ...]) -> bool:
    """Whether the relay counted an answered call as the CLI counted a step, on each of ``names`` the relay gives."""
    usage = _usage(held.get("usage"))
    return held.get("outcome") == "answered" and all(usage.get(name) == counts.get(name) for name in names
                                                     if name in usage)


def _matched(could: dict[Any, list[dict[str, Any]]]) -> dict[Any, dict[str, Any]]:
    """A call for as many steps as can have one, each step of ``could`` given the calls that could have made it, the
    likeliest last: each tries its likeliest first, and takes one another step holds when that step can take another,
    so a call only one step could have made is that step's."""
    holder: dict[int, Any] = {}
    calls = {id(held): held for options in could.values() for held in options}

    def place(key: Any, seen: set[int]) -> bool:
        for held in reversed(could[key]):
            if id(held) not in seen:
                seen.add(id(held))
                if id(held) not in holder or place(holder[id(held)], seen):
                    holder[id(held)] = key
                    return True
        return False

    for key in could:
        place(key, set())
    return {key: calls[call] for call, key in holder.items()}


def _steps(events: list[dict[str, Any]]) -> list[_Step]:
    """A sub-agent session's events as its steps, in order."""
    steps: dict[str, _Step] = {}
    for event in events:
        part = event["part"]
        step = steps.setdefault(part["messageID"], _Step(part["messageID"], event.get("timestamp")))
        if event["type"] == "step_finish":
            step.finish, step.finished = part, event.get("timestamp")
        elif event["type"] in ("text", "reasoning", "tool_use"):
            step.content.append(event)
    return list(steps.values())


@atif_adapter(native_format=NATIVE_FORMAT, version="0.1.0", input_includes_cache=True, tested_with=TESTED_WITH)
class OpenCodeAtif(Adapter):
    """One A2A context's OpenCode session as ATIF, from the CLI's events, the relay's records of its calls, which the
    bridge feeds as the ``calls`` aux file beside the turn's model and effort (``turn``), and the sub-agent sessions
    its task calls ran (``subagents``)."""

    def __init__(self, turn: Any) -> None:
        self._seq = 0
        self._subagents: set[str] = set()
        self._new_task()

    def _new_task(self) -> None:
        self._held: list[dict[str, Any]] = []
        self._sessions: dict[str, list[dict[str, Any]]] = {}
        self._step: _Step | None = None
        self._model: str | None = None
        self._effort: str | None = None

    @aux("turn", type="agentenv.turn")
    def settings(self, record: dict[str, Any], turn: Any) -> None:
        self._model, self._effort = _string(record.get("model")), _string(record.get("effort"))

    @aux("calls", type="agentenv.model_call")
    def call(self, record: dict[str, Any], turn: Any) -> None:
        self._held.append(record)

    @aux("subagents", type="agentenv.opencode_session")
    def session(self, record: dict[str, Any], turn: Any) -> None:
        """A sub-agent session's events, fed before the event of the task call that ran it."""
        self._sessions[record["session"]] = record["events"]

    @native(type="step_start")
    def step_start(self, record: dict[str, Any], turn: Any) -> None:
        self._step_of(record, turn)

    @native(type=("text", "reasoning", "tool_use"))
    def content(self, record: dict[str, Any], turn: Any) -> None:
        self._step_of(record, turn).content.append(record)

    @native(type="step_finish")
    def step_finish(self, record: dict[str, Any], turn: Any) -> None:
        step = self._step_of(record, turn)
        step.finish, step.finished = record["part"], record.get("timestamp")
        if step.finish.get("reason") in _CONTINUING and self._main(step, self._span(step), guess=False) is not None:
            self._flush(turn)

    @native(type="error")
    def error(self, record: dict[str, Any], turn: Any) -> None:
        """The CLI's report of a failure it didn't retry past: on the step it cut short, else on the call the endpoint
        refused last, else on a step of its own."""
        error = record.get("error") if _is_mapping(record.get("error")) else {}
        data = error.get("data") if _is_mapping(error.get("data")) else {}
        name, said = _string(error.get("name")), _string(data.get("message")) or _string(error.get("message"))
        message = (f"{name}: {said}" if name and said and not said.startswith(f"{name}:") else said or name
                   or "the CLI reported an error")
        code = str(data["statusCode"]) if data.get("statusCode") is not None else None
        cut = self._step if self._step is not None and self._step.finish is None else None
        self._flush(turn)
        if cut is not None:
            turn.error(message=message, kind="infra_error", code=code, step=cut.key)
            return
        refused = [held for held in self._held if held.get("outcome") == "refused"]
        last = refused[-1] if refused else None
        for held in self._held:
            if held is not last:
                self._side(held, turn)
        self._held = []
        if last is not None:
            key = self._side(last, turn)
        else:
            self._seq += 1
            key = f"error:{self._seq}"
            turn.step(key=key, llm_call_count=0, time=_iso(record.get("timestamp")))
        turn.error(message=message, kind="infra_error", code=code, step=key)

    @finish
    def end(self, turn: Any) -> None:
        self._flush(turn)
        for held in self._held:
            self._side(held, turn)
        self._new_task()

    def _step_of(self, record: dict[str, Any], turn: Any) -> _Step:
        """The step of the event's message, which a new message's event starts, writing the one before."""
        key = record["part"]["messageID"]
        if self._step is None or self._step.key != key:
            self._flush(turn)
            self._step = _Step(key, record.get("timestamp"))
        return self._step

    def _span(self, step: _Step) -> list[dict[str, Any]]:
        """The calls held that started before the step finished: its own, and those made alongside or before it."""
        return [held for held in self._held if step.finished is None or _started(held) < step.finished]

    def _main(self, step: _Step, span: list[dict[str, Any]], *, guess: bool) -> dict[str, Any] | None:
        """The call that made the step: of the calls outside its task calls' runs, the one whose counts are its
        step_finish's, else, with ``guess``, the latest to start before the step did, its stream having begun after
        its call; a sub-agent's call of a task call the CLI was killed before printing began later."""
        windows = list(step.windows().values())
        candidates = [held for held in span if held.get("outcome") in ("answered", "unreported")
                      and not any(_inside(held, window) for window in windows)]
        native = _native_usage((step.finish or {}).get("tokens"))
        matching = [held for held in candidates if native and _same(held, native, ("input_tokens", "output_tokens"))]
        if matching or not guess:
            return max(matching, key=_started, default=None)
        before = step.started
        if before is None:
            starts = [_moment(((part.get("state") or {}).get("time") or {}).get("start")) for part in step.tools()]
            before = min((start for start in starts if start is not None), default=step.finished)
        earlier = [held for held in candidates if before is None or _started(held) <= before]
        return max(earlier, key=_started, default=None)

    @staticmethod
    def _counted(step: _Step, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Of ``calls``, those whose counts, its cache read's too, are the sub-agent step's, since sub-agents run at
        once can send prompts of one length, the likeliest last: of those that started before the step did, the
        latest likeliest, as counts the CLI prints can tie; with no start, the first to start likeliest."""
        native = _native_usage((step.finish or {}).get("tokens"))
        matching = [held for held in calls
                    if native and _same(held, native, ("input_tokens", "output_tokens", "cache_read_tokens"))]
        if step.started is not None:
            return sorted((held for held in matching if _started(held) <= step.started), key=_started)
        return sorted(matching, key=_started, reverse=True)

    @staticmethod
    def _began(step: _Step, calls: list[dict[str, Any]], model: str | None) -> list[dict[str, Any]]:
        """Of ``calls``, those that could have made a step of a sub-agent on ``model`` that no counts name (cut short,
        or its endpoint sent no usage), the likeliest last: those that started before the step did, its stream having
        begun after its call, the latest likeliest; with no start, those that started before it finished, the first
        likeliest. A step the CLI counted nothing for was made by a call answered without usage, and none by a call
        naming another model."""
        native = _native_usage((step.finish or {}).get("tokens"))
        outcomes = ("unreported",) if native and not any(native.values()) else ("answered", "unreported")
        made = [held for held in calls if held.get("outcome") in outcomes
                and (model is None or _string(held.get("model")) in (None, model))]
        if step.started is not None:
            return sorted((held for held in made if _started(held) <= step.started), key=_started)
        return sorted((held for held in made if step.finished is None or _started(held) < step.finished),
                      key=_started, reverse=True)

    def _children(self, step: _Step, span: list[dict[str, Any]], taken: list[dict[str, Any]]) -> dict[str, _Child]:
        """Each task call's sub-agent, adding the calls of ``span`` it made to ``taken``: with its session fed, its
        steps, each with its call, the steps matched first to the calls their counts name, then the steps left to the
        calls left that began before them, so a step of one sub-agent can't take a call another's counts name, and as
        many as can have one get one; else, when no other task call ran at once, the calls inside its run."""
        windows = step.windows()
        alone = _alone(windows)
        runs = []
        for part in sorted((part for part in step.tools() if part["callID"] in windows),
                           key=lambda part: windows[part["callID"]]):
            call_id = part["callID"]
            metadata = (part.get("state") or {}).get("metadata")
            session = _string(metadata.get("sessionId")) if _is_mapping(metadata) else None
            if session is None:
                continue
            events = self._sessions.pop(session, None)
            if events is None and call_id not in alone:
                continue
            inside = [held for held in span if not _among(held, taken) and _inside(held, windows[call_id])]
            model = metadata.get("model") if _is_mapping(metadata.get("model")) else {}
            runs.append((call_id, inside, _steps(events or []), _string(model.get("modelID"))))
        made = _matched({(call_id, index): self._counted(child, inside) for call_id, inside, steps, _ in runs
                         for index, child in enumerate(steps)})
        taken += made.values()
        timed = _matched({(call_id, index): self._began(child, [held for held in inside if not _among(held, taken)],
                                                        model)
                          for call_id, inside, steps, model in runs for index, child in enumerate(steps)
                          if (call_id, index) not in made})
        made.update(timed)
        taken += timed.values()
        children = {}
        for call_id, inside, steps, _ in runs:
            calls = [held for held in inside if not _among(held, taken)] if call_id in alone else []
            taken += calls
            children[call_id] = _Child([(child, made.get((call_id, index))) for index, child in enumerate(steps)],
                                       calls)
        return children

    def _flush(self, turn: Any) -> None:
        """Write the open step: the calls it made alongside first, then the step with its call's usage, and each task
        call's sub-agent with the calls it made."""
        step, self._step = self._step, None
        if step is None:
            return
        span = self._span(step)
        main = self._main(step, span, guess=True)
        taken = [main] if main is not None else []
        children = self._children(step, span, taken)
        for held in span:
            if not _among(held, taken):
                self._side(held, turn)
        self._held = [held for held in self._held if not _among(held, span)]
        self._write(step, main, turn, children=children)

    def _write(self, step: _Step, main: dict[str, Any] | None, turn: Any, *, agent: str | None = None,
               children: dict[str, _Child] | None = None) -> None:
        """A step with its call's model and usage, none for a call answered without usage, whose zeros the CLI prints
        aren't counts; with no call known, its own counts; and its reasoning, text and calls."""
        usage = _usage(main.get("usage")) if main is not None else _native_usage((step.finish or {}).get("tokens"))
        model = (_string(main.get("model")) if main is not None else None) or self._model \
            or turn.context.agent_model
        key = step.key
        turn.step(key=key, model_name=model, reasoning_effort=self._effort, time=_iso(step.started), agent=agent,
                  **usage)
        texts = []
        for record in step.content:
            part = record["part"]
            if record["type"] == "reasoning" and _string(part.get("text")):
                turn.reasoning(step=key, text=part["text"], visibility="raw", agent=agent)
            elif record["type"] == "text" and isinstance(part.get("text"), str) and part["text"].strip():
                texts.append(part["text"])
            elif record["type"] == "tool_use":
                self._tool(key, part, (children or {}).get(part["callID"]), turn, agent=agent)
        if texts:
            # Joined as the bridge joins the turn's response, so a task's last step holds the response itself.
            turn.message(step=key, text="\n".join(texts).strip(), agent=agent)

    def _tool(self, key: str, part: dict[str, Any], child: _Child | None, turn: Any, *, agent: str | None) -> None:
        state = part.get("state") or {}
        call_id = part["callID"]
        turn.tool_call(step=key, tool_call_id=call_id, function_name=part["tool"], arguments=state.get("input"),
                       agent=agent)
        metadata = state.get("metadata") if _is_mapping(state.get("metadata")) else {}
        if child is not None and (child.steps or child.calls) and _string(metadata.get("sessionId")):
            self._spawn(call_id, state, metadata, child, turn)
        status = _STATUSES.get(state.get("status"), "ok")
        if status == "error":
            content: list[Any] = [_string(state.get("error")) or ""]
        else:
            output = _string(state.get("output"))
            content = ([output] if output else []) + _media(state.get("attachments"), turn) or [""]
        turn.tool_result(source_call_id=call_id, content=content, status=status)

    def _spawn(self, call_id: str, state: dict[str, Any], metadata: dict[str, Any], child: _Child, turn: Any) -> None:
        """The sub-agent a task call ran: its session, its steps with the calls made beside them before each, the last
        holding its answer; with no steps fed, a step per call it made, the last given the task's answer."""
        session = metadata["sessionId"]
        agent_id = session if session not in self._subagents else f"{session}:{call_id}"
        self._subagents.add(agent_id)
        model = metadata.get("model") if _is_mapping(metadata.get("model")) else {}
        turn.subagent(agent_id, spawned_by=call_id, name=_string((state.get("input") or {}).get("subagent_type")),
                      model_name=_string(model.get("modelID")))
        if child.steps:
            beside = sorted(child.calls, key=_started)
            for index, (step, main) in enumerate(child.steps):
                begun = _started(main) if main is not None else step.started
                while beside and (index == len(child.steps) - 1 or begun is None or _started(beside[0]) < begun):
                    self._side(beside.pop(0), turn, agent=agent_id)
                self._write(step, main, turn, agent=agent_id)
            return
        keys = [self._side(held, turn, agent=agent_id) for held in child.calls]
        answer = _TASK_RESULT.search(state.get("output") or "") if state.get("status") == "completed" else None
        if answer and answer["text"].strip() and child.calls[-1].get("outcome") != "refused":
            turn.message(step=keys[-1], text=answer["text"].strip(), agent=agent_id)

    def _side(self, held: dict[str, Any], turn: Any, *, agent: str | None = None) -> str:
        """A call that made no step of its own: its usage alone, or, refused, its error."""
        self._seq += 1
        key = f"call:{self._seq}"
        model, when = _string(held.get("model")), held.get("timestamp")
        if held.get("outcome") == "refused":
            status = held.get("status")
            turn.step(key=key, llm_call_count=0, model_name=model, time=when, agent=agent)
            turn.error(message=f"the endpoint refused the call (HTTP {status})" if status is not None
                       else "the call reached no endpoint", kind="infra_error",
                       code=str(status) if status is not None else None, step=key, agent=agent)
        else:
            turn.step(key=key, model_name=model, reasoning_effort=self._effort, time=when, agent=agent,
                      **_usage(held.get("usage")))
        return key

    @classmethod
    def from_atif(cls, document: dict[str, Any]) -> Native:
        """Each model step as the CLI prints one: step_start, its reasoning and text, a completed tool_use per call
        with its result's text, and step_finish with its counts; best effort, with no session to resume."""
        log: list[dict[str, Any]] = []
        session = document.get("session_id")
        for index, step in enumerate(document.get("steps") or ()):
            if step.get("source") != "agent" or step.get("llm_call_count") == 0:
                continue
            at = _ms(step["timestamp"]) if isinstance(step.get("timestamp"), str) else None
            parts: list[dict[str, Any]] = [{"type": "step-start"}]
            if _string(step.get("reasoning_content")):
                parts.append({"type": "reasoning", "text": step["reasoning_content"]})
            if text := _text(step.get("message")):
                parts.append({"type": "text", "text": text})
            results = {result.get("source_call_id"): result
                       for result in (step.get("observation") or {}).get("results") or ()}
            for call in step.get("tool_calls") or ():
                parts.append({"type": "tool", "callID": call.get("tool_call_id"), "tool": call.get("function_name"),
                              "state": {"status": "completed", "input": call.get("arguments") or {},
                                        "output": _text((results.get(call.get("tool_call_id")) or {}).get("content"))}})
            metrics = step.get("metrics") if _is_mapping(step.get("metrics")) else {}
            prompt, cached = _count(metrics.get("prompt_tokens")) or 0, _count(metrics.get("cached_tokens")) or 0
            parts.append({"type": "step-finish", "reason": "tool-calls" if step.get("tool_calls") else "stop",
                          "cost": 0, "tokens": {"input": max(prompt - cached, 0), "reasoning": 0,
                                                "output": _count(metrics.get("completion_tokens")) or 0,
                                                "cache": {"read": cached, "write": 0}}})
            log += [{"type": _EVENTS[part["type"]], "timestamp": at, "sessionID": session,
                     "part": {"messageID": f"msg_{index}", "sessionID": session, **part}} for part in parts]
        return Native(log=log)


def _media(attachments: Any, turn: Any) -> list[Any]:
    """A result's attachments, each a data URL, as media files."""
    found = []
    for attachment in attachments if isinstance(attachments, list) else ():
        url = attachment.get("url") if _is_mapping(attachment) else None
        match = _DATA_URL.fullmatch(url) if isinstance(url, str) else None
        if match is None:
            continue
        try:
            found.append(turn.media(base64.b64decode(match["data"], validate=True), match["type"]))
        except (binascii.Error, ValueError):
            continue
    return found


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if _is_mapping(part) and part.get("type") == "text")
    return ""
