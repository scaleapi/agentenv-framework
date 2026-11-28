"""Hermes' ATIF adapter: the hook events the agent's Hermes plugin writes, which the bridge appends as they come, as
ATIF. A model call is one ``post_api_request``, with its model, its request's start, its usage (``prompt_tokens``
counting the cache) and its assistant message whole; its calls are the ``pre_tool_call`` events of the tools Hermes ran,
the tool search wrapper unwrapped and each entry of a connector batch its own, and their results the
``post_tool_call`` events by call id. A call the model made that Hermes never ran is written when its step closes. A
failed attempt, which Hermes retries itself, is a step with no model call holding the error.

An auxiliary call (a title, a compression, a vision analysis) is a step with its usage alone, written at its agent's
next step, before a response not yet written, so the turn's last response stays its last step; a summary Hermes
committed makes its compression call's step a compaction. A ``delegate_task`` child, its events under a session of its
own, is a sub-agent under the call that spawned it; a batch's children past the first, which one call spawned, each
under a call of its own. Two things only the server knows come as aux records (``turn``): the turn's reasoning effort,
and Hermes' cost when it priced the turn."""
from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agentenv_protocol.atif import Adapter, Native, atif_adapter, aux, finish, is_token_count, native

NATIVE_FORMAT = "hermes-hook-events/1"
#: The Hermes commit the adapter reads; a unit test holds the Dockerfile's pin to this list.
TESTED_WITH = ("bf0203a794a07aac3ff764972ab9087baa4e4f4e",)
#: The aux file the server feeds: ``{"type": "start", "effort"}`` before the turn, ``{"type": "end", "cost_usd"}``
#: after it.
AUX = "turn"
_DELEGATE = "delegate_task"
_COMPRESSION = "compression"
#: The compactions a model call wrote, whose summary is its compression call's answer.
_SUMMARIES = ("summarize", "micro_summarize")
_COUNTS = (("prompt_tokens", "input_tokens"), ("output_tokens", "output_tokens"),
           ("cache_read_tokens", "cache_read_tokens"), ("cache_write_tokens", "cache_write_tokens"))
_CHILD_STATUSES = {"completed": "ok", "interrupted": "canceled", "failed": "error"}
_DATA_URL = re.compile(r"data:(?P<type>[\w.+-]+/[\w.+-]+);base64,(?P<data>.*)", re.S)
#: Where a child's events go when no delegate_task call is waiting for it: the native log alone.
_UNPLACED = object()


def _is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _usage(counts: Any) -> dict[str, int]:
    """A call's counts as its step takes them; none when it reported none, as the bridge counts such a call."""
    if not _is_mapping(counts) or not (counts.get("prompt_tokens") or counts.get("output_tokens")):
        return {}
    return {name: counts[key] for key, name in _COUNTS if is_token_count(counts.get(key))}


def _millis(record: dict[str, Any]) -> str | None:
    """When the plugin wrote the event: epoch milliseconds, as the builder writes times."""
    millis = record.get("timestamp")
    if not isinstance(millis, int) or isinstance(millis, bool):
        return None
    seconds, rest = divmod(millis, 1000)
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + f".{rest:03d}Z"


def _started(record: dict[str, Any]) -> datetime | str | None:
    """When the call's request was sent (epoch seconds), else when its event was written."""
    started = record.get("started_at")
    if isinstance(started, (int, float)) and not isinstance(started, bool):
        return datetime.fromtimestamp(started, tz=timezone.utc)
    return _millis(record)


def _arguments(raw: Any) -> Any:
    """A model call's arguments, a JSON string, as an object when they parse to one."""
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return raw
        return parsed if _is_mapping(parsed) else raw
    return raw if raw is not None else {}


@dataclass
class _Response:
    """A model call's response, held until a tool runs for it or its agent's next step: its key, model, start, counts,
    text, reasoning (``reasoned`` when the call counted reasoning tokens) and the calls it made, by id."""

    key: str
    model: str | None
    time: datetime | str | None
    usage: dict[str, int]
    text: str | None
    reasoning: str | None
    reasoned: bool
    calls: dict[str, tuple[str, Any]]


def _response(key: str, record: dict[str, Any]) -> _Response:
    """A ``post_api_request`` as a held response. A record from before the plugin kept the message whole has it, cut at
    8000 characters, and the usage in Hermes' record of the response."""
    response = record.get("response") if _is_mapping(record.get("response")) else {}
    message = record.get("message")
    if not _is_mapping(message):
        message = response.get("assistant_message") if _is_mapping(response.get("assistant_message")) else {}
    counts = record.get("usage") if _is_mapping(record.get("usage")) else response.get("usage")
    usage = _usage(counts)
    text = _string(message.get("content"))
    calls = {call["id"]: (call["name"], _arguments(call.get("arguments"))) for call in message.get("tool_calls") or ()
             if _is_mapping(call) and _string(call.get("id")) and _string(call.get("name"))}
    reasoning_tokens = counts.get("reasoning_tokens") if _is_mapping(counts) else None
    return _Response(key=key, model=_string(record.get("model")) or _string(response.get("model")),
                     time=_started(record), usage=usage, text=text.strip() or None if text else None,
                     reasoning=_string(message.get("reasoning")),
                     reasoned=bool(usage) and is_token_count(reasoning_tokens) and reasoning_tokens > 0, calls=calls)


@dataclass
class _Aux:
    """An auxiliary call held until its agent's next step, and the compaction its summary made, if one did."""

    record: dict[str, Any]
    compaction: dict[str, Any] | None = None


@dataclass
class _Agent:
    """One trajectory's place in its task: its open step's key and the model calls of it no tool event took, the
    response not yet written, the auxiliary calls not yet written, and its delegate_task calls still waiting."""

    open: str | None = None
    calls: dict[str, tuple[str, Any]] = field(default_factory=dict)
    held: _Response | None = None
    aux: list[_Aux] = field(default_factory=list)
    delegates: list[str] = field(default_factory=list)


@atif_adapter(native_format=NATIVE_FORMAT, version="0.1.0", input_includes_cache=True, tested_with=TESTED_WITH)
class HermesAtif(Adapter):
    """One A2A context's Hermes session as ATIF, from the plugin's hook events and the server's ``turn`` records. The
    turn is the one whose ``turn_id`` its first model call carries, as the bridge decides it; another turn's events
    are a child's, by its session."""

    def __init__(self, turn: Any) -> None:
        self._seq = 0
        self._new_task()

    def _new_task(self) -> None:
        self._main: str | None = None
        self._session: str | None = None
        self._effort: str | None = None
        self._agents: dict[str | None, _Agent] = {None: _Agent()}
        self._calls: set[str] = set()
        self._spawned: set[str] = set()
        self._fanned: dict[str, str] = {}
        self._unplaced: set[str] = set()

    @aux(AUX, type="start")
    def started(self, record: dict[str, Any], turn: Any) -> None:
        self._effort = _string(record.get("effort"))

    @aux(AUX, type="end")
    def ended(self, record: dict[str, Any], turn: Any) -> None:
        cost = record.get("cost_usd")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            turn.task_cost(cost, source="harness")

    @native(type="post_api_request")
    def response(self, record: dict[str, Any], turn: Any) -> None:
        self._claim(record)
        agent = self._agent(record, turn)
        if agent is _UNPLACED:
            turn.unmatched(record)
            return
        self._close(agent, turn)
        self._agents[agent].held = _response(self._key("call"), record)

    @native(type="api_request_error")
    def failed(self, record: dict[str, Any], turn: Any) -> None:
        """A failed attempt at a call: a step with no model call holding the error, whatever Hermes does next."""
        self._claim(record)
        agent = self._agent(record, turn)
        if agent is _UNPLACED:
            turn.unmatched(record)
            return
        self._close(agent, turn)
        key = self._key("error")
        error = record.get("error") if _is_mapping(record.get("error")) else {}
        status, reason = record.get("status_code"), _string(record.get("reason"))
        turn.step(key=key, llm_call_count=0, model_name=_string(record.get("model")), time=_started(record), agent=agent)
        turn.error(message=_string(error.get("message")) or reason or "the model request failed", kind="infra_error",
                   code=str(status) if status is not None else reason, step=key, agent=agent)

    @native(type="pre_tool_call")
    def tool_call(self, record: dict[str, Any], turn: Any) -> None:
        agent = self._agent(record, turn)
        if agent is _UNPLACED:
            turn.unmatched(record)
            return
        state = self._agents[agent]
        if state.held is not None:
            self._write(agent, turn)
        call_id, name = record["tool_call_id"], record["tool_name"]
        state.calls.pop(call_id, None)
        arguments = record.get("args")
        turn.tool_call(step=state.open, tool_call_id=call_id, function_name=name,
                       arguments=arguments if arguments is not None else {}, time=_millis(record), agent=agent)
        self._calls.add(call_id)
        if name == _DELEGATE:
            state.delegates.append(call_id)

    @native(type="post_tool_call")
    def tool_result(self, record: dict[str, Any], turn: Any) -> None:
        """A result, in its call's step; one whose call the copy doesn't hold, a child's it couldn't place, is left to
        the native log."""
        call_id = record.get("tool_call_id")
        if call_id not in self._calls:
            turn.unmatched(record)
            return
        for state in self._agents.values():
            if call_id in state.delegates:
                state.delegates.remove(call_id)
        turn.tool_result(source_call_id=call_id, content=_content(record.get("result"), turn),
                         status="error" if record.get("status") == "error" else "ok", exit_code=_exit_code(record),
                         time=_millis(record))

    @native(type="post_llm_call")
    def answered(self, record: dict[str, Any], turn: Any) -> None:
        """The answer as the turn returns it, the text of its last response when no tool ran for that one."""
        agent = self._agent(record, turn)
        if agent is _UNPLACED:
            turn.unmatched(record)
            return
        held, answer = self._agents[agent].held, _string(record.get("assistant_response"))
        if held is not None and answer is not None:
            held.text = answer.strip() or None

    @native(type="post_auxiliary_call")
    def auxiliary(self, record: dict[str, Any], turn: Any) -> None:
        agent = self._agent(record, turn)
        if agent is _UNPLACED:
            turn.unmatched(record)
            return
        self._agents[agent].aux.append(_Aux(record))

    @native(type="post_compaction")
    def compacted(self, record: dict[str, Any], turn: Any) -> None:
        """A summary Hermes committed: its compression call, the latest its agent made, becomes the compaction."""
        if record.get("kind") not in _SUMMARIES:
            return
        sessions = (record.get("session_id"), record.get("old_session_id"))
        agent = next((session for session in sessions if session is not None and session in self._agents), None)
        held = next((call for call in reversed(self._agents[agent].aux) if call.compaction is None
                     and call.record.get("aux_task") == _COMPRESSION and not call.record.get("error")), None)
        if held is None:
            return
        reason = record.get("reason")
        compaction: dict[str, Any] = {"trigger": "manual" if reason == "manual" else
                                      None if reason in (None, "unknown") else "auto"}
        for name, key in (("tokens_before", "before_tokens"), ("tokens_after", "after_tokens")):
            if is_token_count(record.get(key)):
                compaction[name] = record[key]
        held.compaction = compaction

    @native(type="subagent_start")
    def spawned(self, record: dict[str, Any], turn: Any) -> None:
        child = _string(record.get("child_session_id"))
        if child is None or child in self._agents or child in self._unplaced:
            return
        parent = record.get("parent_session_id")
        self._declare(child, parent if parent is not None and parent in self._agents else None,
                      _string(record.get("child_subagent_id")), _string(record.get("child_goal")), turn)

    @native(type="subagent_stop")
    def stopped(self, record: dict[str, Any], turn: Any) -> None:
        """A batch's later child ended: its own call's result, the summary Hermes hands its parent."""
        call_id = self._fanned.pop(record.get("child_session_id"), None)
        if call_id is not None:
            turn.tool_result(source_call_id=call_id, text=_string(record.get("child_summary")) or "",
                             status=_CHILD_STATUSES.get(record.get("child_status"), "ok"), time=_millis(record))

    @native(type=("on_session_start", "on_session_end"))
    def session(self, record: dict[str, Any], turn: Any) -> None:
        """A session's start or end, which the user step and the task's state cover."""

    @finish
    def end(self, turn: Any) -> None:
        for agent in list(self._agents):
            self._close(agent, turn)
        self._new_task()

    def _claim(self, record: dict[str, Any]) -> None:
        if self._main is None and _string(record.get("turn_id")):
            self._main = record["turn_id"]

    def _agent(self, record: dict[str, Any], turn: Any) -> Any:
        """The trajectory an event belongs to: a declared child's, by its session; the root for the turn's own and an
        auxiliary call no turn made; else the child its session names, declared on its first event when no
        subagent_start declared it."""
        turn_id, session = record.get("turn_id"), record.get("session_id")
        if _string(session) and session in self._agents:
            return session
        if self._main is None or not turn_id or turn_id == self._main:
            if turn_id and _string(session):
                self._session = session
            return None
        if not _string(session) or session in self._unplaced:
            return _UNPLACED
        if session == self._session:
            return None
        return self._declare(session, None, None, None, turn)

    def _declare(self, child: str, parent: str | None, name: str | None, goal: str | None, turn: Any) -> Any:
        """Declare ``child`` under its parent's first delegate_task call still waiting with no child, else under a call
        of its own beside the parent's latest, as a batch's later children; with none waiting, it isn't placed."""
        holder = self._agents[parent]
        spawner = next((call for call in holder.delegates if call not in self._spawned), None)
        if spawner is None and holder.delegates and holder.open is not None:
            spawner = f"{holder.delegates[-1]}:{child}"
            turn.tool_call(step=holder.open, tool_call_id=spawner, function_name=_DELEGATE,
                           arguments={"goal": goal} if goal else {}, agent=parent)
            self._calls.add(spawner)
            self._fanned[child] = spawner
        if spawner is None:
            self._unplaced.add(child)
            return _UNPLACED
        turn.subagent(child, spawned_by=spawner, name=name)
        self._spawned.add(spawner)
        self._agents[child] = _Agent()
        return child

    def _write(self, agent: str | None, turn: Any) -> None:
        """Write the agent's held response as its open step, after the auxiliary calls held before it."""
        state = self._agents[agent]
        held, state.held = state.held, None
        self._flush(agent, turn)
        turn.step(key=held.key, model_name=held.model, reasoning_effort=self._effort if agent is None else None,
                  time=held.time, agent=agent, **held.usage)
        if held.reasoning is not None or held.reasoned:
            turn.reasoning(step=held.key, text=held.reasoning, visibility="raw" if held.reasoning else "omitted",
                           agent=agent)
        if held.text is not None:
            turn.message(step=held.key, text=held.text, agent=agent)
        state.open, state.calls = held.key, held.calls

    def _close(self, agent: str | None, turn: Any) -> None:
        """End the agent's step: its held response written, the calls it made that no tool ran for, then the
        auxiliary calls made since."""
        state = self._agents[agent]
        if state.held is not None:
            self._write(agent, turn)
        for call_id, (name, arguments) in state.calls.items():
            turn.tool_call(step=state.open, tool_call_id=call_id, function_name=name, arguments=arguments, agent=agent)
            self._calls.add(call_id)
        state.calls = {}
        self._flush(agent, turn)
        state.open = None

    def _flush(self, agent: str | None, turn: Any) -> None:
        state = self._agents[agent]
        held, state.aux = state.aux, []
        for call in held:
            self._auxiliary(call, agent, turn)

    def _auxiliary(self, call: _Aux, agent: str | None, turn: Any) -> None:
        """An auxiliary call's step: its usage, or its error; a compaction's, the summary it wrote."""
        record, key = call.record, self._key("aux")
        model, when = _string(record.get("model")) or _string(record.get("response_model")), _started(record)
        error = _string(record.get("error"))
        if error is not None:
            task = _string(record.get("aux_task"))
            turn.step(key=key, llm_call_count=0, model_name=model, time=when, agent=agent)
            turn.error(message=f"{task}: {error}" if task else error, kind="infra_error", step=key, agent=agent)
            return
        turn.step(key=key, model_name=model, time=when, agent=agent, **_usage(record.get("usage")))
        if call.compaction is not None:
            if summary := _string(record.get("content")):
                turn.message(step=key, text=summary, agent=agent)
            turn.compaction(step=key, written_by="assistant", agent=agent, **call.compaction)

    def _key(self, kind: str) -> str:
        self._seq += 1
        return f"{kind}:{self._seq}"

    @classmethod
    def from_atif(cls, document: dict[str, Any]) -> Native:
        """Each task as the plugin's events: a turn per user step, a post_api_request per model step with its usage,
        text and calls, a pre_tool_call and post_tool_call per call, and the answer; best effort, no session."""
        log: list[dict[str, Any]] = []
        session = _string(document.get("session_id")) or "atif"
        turn_id, count, answer = None, 0, ""

        def end_turn() -> None:
            log.append({"type": "post_llm_call", "session_id": session, "turn_id": turn_id,
                        "assistant_response": answer})
            log.append({"type": "on_session_end", "session_id": session, "turn_id": turn_id, "completed": True,
                        "failed": False, "interrupted": False})

        for step in document.get("steps") or ():
            if step.get("source") == "user":
                if turn_id is not None:
                    end_turn()
                else:
                    log.append({"type": "on_session_start", "session_id": session})
                turn_id, count, answer = f"{session}:atif:{step.get('step_id')}", 0, ""
                continue
            if step.get("source") != "agent" or turn_id is None:
                continue
            text = _text(step.get("message"))
            answer = text or answer
            if step.get("llm_call_count") == 0:
                continue
            count += 1
            calls = [call for call in step.get("tool_calls") or () if _is_mapping(call)]
            log.append({"type": "post_api_request", "session_id": session, "turn_id": turn_id,
                        "api_call_count": count, "model": step.get("model_name"),
                        "finish_reason": "tool_calls" if calls else "stop", "usage": _native_usage(step.get("metrics")),
                        "message": {"content": text, "tool_calls": [
                            {"id": call.get("tool_call_id"), "name": call.get("function_name"),
                             "arguments": json.dumps(call.get("arguments") or {})} for call in calls]}})
            results = {result.get("source_call_id"): result
                       for result in (step.get("observation") or {}).get("results") or ()}
            for call in calls:
                ids = {"session_id": session, "turn_id": turn_id, "tool_call_id": call.get("tool_call_id"),
                       "tool_name": call.get("function_name")}
                log.append({"type": "pre_tool_call", **ids, "args": call.get("arguments") or {}})
                if (result := results.get(call.get("tool_call_id"))) is not None:
                    log.append({"type": "post_tool_call", **ids, "result": _text(result.get("content")),
                                "status": "ok"})
        if turn_id is not None:
            end_turn()
        return Native(log=log)


def _content(result: Any, turn: Any) -> list[Any]:
    """A tool's result as content: its text, or, for one Hermes hands the model with its images, each part, an image
    as a media file."""
    if isinstance(result, str):
        return [result]
    if not (_is_mapping(result) and result.get("_multimodal")):
        return [json.dumps(result, ensure_ascii=False)] if result is not None else [""]
    parts: list[Any] = []
    for block in result.get("content") or ():
        kind = block.get("type") if _is_mapping(block) else None
        url = block.get("image_url") if kind == "image_url" else None
        if kind == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif _is_mapping(url) and (image := _image(url.get("url"))) is not None:
            parts.append(turn.media(*image))
        else:
            parts.append(json.dumps(block, ensure_ascii=False))
    return parts or [""]


def _image(url: Any) -> tuple[bytes, str] | None:
    found = _DATA_URL.fullmatch(url) if isinstance(url, str) else None
    if found is None:
        return None
    try:
        return base64.b64decode(found["data"], validate=True), found["type"]
    except (binascii.Error, ValueError):
        return None


def _exit_code(record: dict[str, Any]) -> int | None:
    """The exit code a terminal call's result names: ``{"output", "exit_code", "error"}``."""
    if record.get("tool_name") != "terminal" or not isinstance(record.get("result"), str):
        return None
    try:
        result = json.loads(record["result"])
    except ValueError:
        return None
    code = result.get("exit_code") if _is_mapping(result) else None
    return code if isinstance(code, int) and not isinstance(code, bool) else None


def _native_usage(metrics: Any) -> dict[str, int] | None:
    """A step's metrics as the plugin's usage: ``prompt_tokens`` counting the cache, ``input_tokens`` without it."""
    metrics = metrics if _is_mapping(metrics) else {}
    prompt, output = metrics.get("prompt_tokens"), metrics.get("completion_tokens")
    if not is_token_count(prompt):
        return None
    profile = (metrics.get("extra") or {}).get("agentenv") or {}
    read = metrics.get("cached_tokens") if is_token_count(metrics.get("cached_tokens")) else 0
    write = profile.get("cache_write_tokens") if is_token_count(profile.get("cache_write_tokens")) else 0
    output = output if is_token_count(output) else 0
    return {"input_tokens": max(prompt - read - write, 0), "output_tokens": output, "cache_read_tokens": read,
            "cache_write_tokens": write, "prompt_tokens": prompt, "total_tokens": prompt + output}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if _is_mapping(part) and part.get("type") == "text")
    return ""
