"""Antigravity's ATIF adapter: the SDK's streamed chunks and hook results, the native log, placed on the model calls
the turn's relay forwarded, which the bridge feeds as the ``calls`` aux file. The chunks carry no model, time or usage,
and only a tool call names its trajectory (its step id is ``<trajectory>:<step index>``), while the relay sees every
call, a sub-agent's included, the conversation it belongs to and whether it holds the turn's own message.

A step is one call, holding what streamed for it: its thoughts as summary reasoning, its text a paragraph per step index
as the bridge joins the answer, and its tool calls with their results. The SDK streams a call's tool calls once the call
has ended, so a tool call is its trajectory's last ended call's. A thought or text streams while its call does and the
bridge holds it until the next chunk, naming no trajectory; each trajectory counts its step indices up, so it is the
call still writing that step, else a call of the trajectory nearest below it that hasn't streamed it or a later one. A
sub-agent runs on after its ``invoke_subagent`` call returns: the first call of a new conversation after one streamed is
that sub-agent's, and so is every later call of that conversation, its steps in a trajectory embedded under the call. A
call that ends while another is its trajectory's open step, as a refused call or one of a conversation no sub-agent
claims, is a step of its own before its trajectory's next one, written once that opens since the open one may still
stream, a refused one holding its error with no model call."""
from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass, field
from typing import Any

from agentenv_protocol.atif import Adapter, Native, atif_adapter, aux, finish, is_token_count, native

NATIVE_FORMAT = "antigravity-sdk/1"
#: The SDK versions the adapter is tested with; a unit test holds requirements.txt's pin to this list.
TESTED_WITH = ("0.1.24",)
_SPAWN = "invoke_subagent"
# The SDK streams a file the model creates as an edit_file step, while its hooks name the call create_file.
_FILE_WRITES = frozenset({"create_file", "edit_file"})
_COUNTS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
_DATA_URL = re.compile(r"data:(?P<type>[\w.+-]+/[\w.+-]+);base64,(?P<data>.*)", re.S)
_ANY: Any = object()


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _step(step_id: Any) -> tuple[str | None, int | None]:
    """The trajectory and step index an SDK step id, ``<trajectory>:<step index>``, names."""
    if not isinstance(step_id, str) or ":" not in step_id:
        return None, None
    trajectory, _, index = step_id.rpartition(":")
    return trajectory or None, int(index) if index.isdigit() else None


def _same_tool(name: str, other: str) -> bool:
    return name == other or {name, other} <= _FILE_WRITES


@dataclass(eq=False)
class _Call:
    """A model call the relay forwarded, by the step it writes: the turn's conversation's (``root``), a sub-agent's
    (``agent``), or neither. Its step takes content only while it's its trajectory's open one, where its text waits, by
    step index, until the step closes."""

    key: str
    root: bool = False
    conversation: str | None = None
    model: str | None = None
    effort: Any = None
    started: str | None = None
    agent: str | None = None
    opened: bool = False
    streamed: bool = False
    texts: dict[Any, str] = field(default_factory=dict)

    @property
    def placed(self) -> bool:
        return self.root or self.agent is not None


@atif_adapter(native_format=NATIVE_FORMAT, version="0.2.0", input_includes_cache=True, tested_with=TESTED_WITH)
class AntigravityAtif(Adapter):
    """One A2A context's Antigravity conversation as ATIF, from the SDK's chunks and hook results and the relay's
    ``calls`` records of the model calls the harness made."""

    def __init__(self, turn: Any) -> None:
        self._n = 0
        self._ids: set[str] = set()
        # Each SDK trajectory's sub-agent, None for the turn's own, whose trajectory keeps its id from task to task, and
        # the latest step index each has streamed.
        self._slots: dict[str, str | None] = {}
        self._tops: dict[str | None, int] = {}
        self._new_task()

    def _new_task(self) -> None:
        self._calls: dict[Any, _Call] = {}
        self._ended: list[_Call] = []
        self._open: dict[str | None, _Call] = {}
        self._held: list[tuple[_Call, dict[str, Any]]] = []
        self._streamed: dict[str, str] = {}
        self._unanswered: list[tuple[str, str, str | None]] = []
        self._unclaimed: list[str] = []
        self._agents: dict[str, str] = {}
        self._roots: set[str | None] = set()
        # Each trajectory's step its latest thought or text is of, with the call that streamed it, until it streams a
        # later step.
        self._writing: dict[str | None, tuple[int, _Call]] = {}

    @aux("calls", type="agentenv.model_request")
    def request(self, record: dict[str, Any], turn: Any) -> None:
        """A call of a conversation a sub-agent has claimed is that sub-agent's, though it holds the turn's message: a
        sub-agent given the user's words. Else one holding the turn's message is the turn's conversation's, unless it's
        of a new conversation while a sub-agent waits for its first call; one of a new conversation otherwise is the
        turn's conversation compacted."""
        call = _Call(self._key(), conversation=_string(record.get("conversation")), model=_string(record.get("model")),
                     effort=record.get("reasoning_effort"), started=record.get("started"))
        self._calls[record.get("seq")] = call
        claimed = self._agents.get(call.conversation) if call.conversation is not None else None
        if claimed is not None:
            call.agent = claimed
        elif record.get("turn") is True and (not self._roots or call.conversation in self._roots
                                             or not self._unclaimed):
            call.root = True
            self._roots.add(call.conversation)
        else:
            self._place(call, turn)

    @aux("calls", type="agentenv.model_call")
    def call(self, record: dict[str, Any], turn: Any) -> None:
        """A call's end, written on its step when that's its trajectory's open one. Else the call is held until its
        trajectory's next step opens, its own included: the call before it may still stream into the open step, since
        the bridge writes a text once the next chunk comes."""
        call = self._calls.pop(record.get("seq"), None) or _Call(self._key(), model=_string(record.get("model")))
        self._place(call, turn)
        if call.placed and record.get("outcome") != "refused":
            self._ended.append(call)
        if not call.placed or self._open.get(call.agent) is not call:
            self._held.append((call, record))
            return
        self._write(call, record, turn)

    @native(type="Thought")
    def thought(self, record: dict[str, Any], turn: Any) -> None:
        if (text := _string(record.get("text"))) is not None:
            call = self._streaming(turn, index=record.get("step_index"), thought=True)
            turn.reasoning(step=call.key, text=text, visibility="summary", agent=call.agent)

    @native(type="Text")
    def text(self, record: dict[str, Any], turn: Any) -> None:
        if (text := _string(record.get("text"))) is not None:
            call = self._streaming(turn, index=record.get("step_index"))
            call.texts[record.get("step_index")] = call.texts.get(record.get("step_index"), "") + text

    @native(type="ToolCall")
    def tool_call(self, record: dict[str, Any], turn: Any) -> None:
        name, streamed = str(record["name"]), str(record["id"])
        trajectory, index = _step(record.get("step_id") or streamed)
        call = self._streaming(turn, trajectory=trajectory)
        if index is not None:
            self._tops[call.agent] = max(index, self._tops.get(call.agent, index))
            if (writing := self._writing.get(call.agent)) is not None and index >= writing[0]:
                del self._writing[call.agent]
        call_id = self._unique(streamed)
        turn.tool_call(step=call.key, tool_call_id=call_id, function_name=name, arguments=_arguments(record),
                       agent=call.agent)
        self._streamed[streamed] = call_id
        self._unanswered.append((call_id, name, trajectory))
        if name == _SPAWN:
            self._n += 1
            agent = f"subagent-{self._n}"
            turn.subagent(agent, spawned_by=call_id)
            self._unclaimed.append(agent)
            waiting = next((held for held, _ in self._held if not held.placed and held.conversation is not None), None)
            if waiting is not None:
                self._place(waiting, turn)

    @native(where={"kind": "tool_result"})
    def tool_result(self, record: dict[str, Any], turn: Any) -> None:
        call_id = self._answered(record)
        if call_id is None:
            # Its call never streamed, as a peer's or one the turn's end cut: the call as the hooks saw it.
            name = str(record.get("name"))
            call_id = self._unique(str(record.get("id") or name))
            holder = self._open.get(None) or self._streaming(turn)
            turn.tool_call(step=holder.key, tool_call_id=call_id, function_name=name, arguments=None,
                           agent=holder.agent)
        error = _string(record.get("error"))
        content = [error] if error else _result_content(record.get("result"), turn)
        turn.tool_result(source_call_id=call_id, content=content, status="error" if error else "ok")

    @finish
    def end(self, turn: Any) -> None:
        for call, record in self._held:
            self._begin(call, turn)
            self._write(call, record, turn)
        for agent in list(self._open):
            self._close(agent, turn)
        self._new_task()

    def _key(self) -> str:
        self._n += 1
        return f"call-{self._n}"

    def _unique(self, call_id: str) -> str:
        """``call_id``, made unique in the document: a stream id may come again in a later task."""
        candidate, n = call_id, 1
        while candidate in self._ids:
            n += 1
            candidate = f"{call_id}#{n}"
        self._ids.add(candidate)
        return candidate

    def _answered(self, record: dict[str, Any]) -> str | None:
        """The streamed call a result answers: the one the bridge bound it to, else the first one still unanswered of
        its tool and trajectory, which the bridge binds by a name the SDK streams differently."""
        bound = self._streamed.get(record.get("call_ref"))
        name, (trajectory, _) = str(record.get("name")), _step(record.get("step_id"))
        found = next((entry for entry in self._unanswered if entry[0] == bound), None) or next(
            (entry for entry in self._unanswered if _same_tool(entry[1], name)
             and (entry[2] == trajectory or None in (entry[2], trajectory))), None)
        if found is None:
            return None
        self._unanswered.remove(found)
        return found[0]

    def _streaming(self, turn: Any, *, trajectory: str | None = None, index: Any = None,
                   thought: bool = False) -> _Call:
        """The call a streamed chunk is of, its step open: a thought's or text's of step ``index``, else the last ended
        call of the trajectory a tool call names, else the one in flight. A chunk of a call whose step has closed goes
        in its trajectory's open one."""
        flying = [call for call in self._calls.values() if call.placed]
        if index is not None:
            call = self._shown(index, flying, thought)
        else:
            slots = self._slot(trajectory) if trajectory is not None else _ANY
            call = _latest(self._ended, slots) or _latest(flying, slots)
        call = call or _latest(self._ended, _ANY) or _latest(flying, _ANY)
        if call is None:
            call = _Call(self._key(), root=True)
            self._ended.append(call)
        if self._open.get(call.agent) is not call:
            if call.opened:
                call = self._open[call.agent]
            else:
                self._open_step(call, turn)
        if index is not None and index >= self._tops.get(call.agent, -1):
            self._tops[call.agent] = index
            self._writing[call.agent] = (index, call)
        call.streamed = True
        return call

    def _shown(self, index: int, flying: list[_Call], thought: bool) -> _Call | None:
        """The call a thought or text of step ``index`` is of. Each trajectory counts its step indices up, so it's the
        call still writing that step in its trajectory: a text after its step's thought, or a step's text the bridge
        wrote as two records, another trajectory's chunk between. Else it's a call of a trajectory that hasn't streamed
        that step or a later one, the one whose latest step is nearest below it first: of its last ended call that has
        streamed nothing and its call in flight, the one in flight for a thought, which the bridge holds until its text
        starts, and the ended one for a text, which it holds until the next chunk; then the latest."""
        calls, ranked = [*self._ended, *flying], []
        for slot in [None, *dict.fromkeys(call.agent for call in calls if call.agent is not None)]:
            if (writing := self._writing.get(slot)) is not None and writing[0] == index:
                return writing[1]
            if (top := self._tops.get(slot, -1)) >= index:
                continue
            ended, live = _latest(self._ended, [slot]), _latest(flying, [slot])
            ended = None if ended is None or ended.streamed else ended
            if (call := (live or ended) if thought else (ended or live)) is not None:
                ranked.append((top, (call in self._ended) != thought, calls.index(call), call))
        return max(ranked, key=lambda rank: rank[:3])[3] if ranked else None

    def _slot(self, trajectory: str) -> Any:
        """The sub-agent an SDK trajectory is, None for the turn's own, as a list of one: that of the call that ended
        last before its first tool call streamed, of those no trajectory is yet; ``_ANY`` while there's none."""
        if trajectory not in self._slots:
            taken = set(self._slots.values())
            call = next((call for call in reversed(self._ended) if call.agent not in taken), None)
            if call is None:
                return _ANY
            self._slots[trajectory] = call.agent
        return [self._slots[trajectory]]

    def _place(self, call: _Call, turn: Any) -> None:
        """A call of another conversation goes in that conversation's sub-agent, the oldest declared one no conversation
        has claimed when the conversation has none, its calls held so far moved into it."""
        if call.placed or call.conversation is None:
            return
        agent = self._agents.get(call.conversation)
        if agent is None and self._unclaimed:
            agent = self._agents[call.conversation] = self._unclaimed.pop(0)
            for held, record in self._held:
                if held.conversation == call.conversation and not held.placed:
                    held.agent = agent
                    if record.get("outcome") != "refused":
                        self._ended.append(held)
        call.agent = agent

    def _open_step(self, call: _Call, turn: Any) -> None:
        """Open ``call``'s step in its trajectory, after the steps of the calls held for it before it, and write its end
        when it's held itself."""
        mine = [entry for entry in self._held if entry[0].agent == call.agent]
        at = next((n for n, (held, _) in enumerate(mine) if held is call), len(mine))
        done = mine[:at + 1]
        self._held = [entry for entry in self._held if all(entry is not written for written in done)]
        for earlier, record in mine[:at]:
            self._begin(earlier, turn)
            self._write(earlier, record, turn)
        self._begin(call, turn)
        if at < len(mine):
            self._write(call, mine[at][1], turn)

    def _begin(self, call: _Call, turn: Any) -> None:
        """``call``'s step, after its trajectory's open one, whose text is written then."""
        self._close(call.agent, turn)
        turn.step(key=call.key, model_name=call.model, reasoning_effort=call.effort, time=call.started,
                  agent=call.agent)
        call.opened = True
        self._open[call.agent] = call

    def _close(self, agent: str | None, turn: Any) -> None:
        """A trajectory's open step's text, a paragraph per step index, as the bridge joins the turn's answer."""
        call = self._open.pop(agent, None)
        if call is None:
            return
        if text := "\n\n".join(part.strip() for part in call.texts.values() if part.strip()):
            turn.message(step=call.key, text=text, agent=agent)
        call.texts.clear()

    def _write(self, call: _Call, record: dict[str, Any], turn: Any) -> None:
        """An ended call's step: its model, effort, time and counts, or, refused, its error on a step with no model
        call, unless content streamed into it."""
        common = {"key": call.key, "model_name": call.model or _string(record.get("model")),
                  "reasoning_effort": call.effort, "time": record.get("timestamp"), "agent": call.agent}
        if record.get("outcome") != "refused":
            turn.step(**common, **_usage(record.get("usage")))
            return
        turn.step(**common, llm_call_count=1 if call.streamed else 0)
        turn.error(message=_refusal(record), kind="infra_error", step=call.key, agent=call.agent)

    @classmethod
    def from_atif(cls, document: dict[str, Any]) -> Native:
        """Each agent step as the SDK streams one, its index the step index: its reasoning as a Thought, its message
        as a Text, and each call as a ToolCall with its result after it; best effort, no conversation store."""
        log: list[dict[str, Any]] = []
        for index, step in enumerate(document.get("steps") or ()):
            if step.get("source") != "agent":
                continue
            if text := _string(step.get("reasoning_content")):
                log.append({"type": "Thought", "step_index": index, "text": text})
            if text := _text(step.get("message")):
                log.append({"type": "Text", "step_index": index, "text": text})
            results = {result.get("source_call_id"): result
                       for result in (step.get("observation") or {}).get("results") or ()}
            for call in step.get("tool_calls") or ():
                call_id, name = call.get("tool_call_id"), call.get("function_name")
                log.append({"type": "ToolCall", "id": call_id, "step_id": call_id, "name": name,
                            "args": call.get("arguments") or {}})
                if (result := results.get(call_id)) is not None:
                    log.append({"kind": "tool_result", "id": call_id, "name": name,
                                "result": _text(result.get("content")), "call_ref": call_id})
        return Native(log=log)


def _latest(calls: list[_Call], slots: Any) -> _Call | None:
    """The last of ``calls`` in one of ``slots``' trajectories, any trajectory's for ``_ANY``."""
    return next((call for call in reversed(calls) if slots is _ANY or call.agent in slots), None)


def _usage(counts: Any) -> dict[str, int]:
    """The relay's counts of a call, its input holding the cache, as the step's."""
    if not isinstance(counts, dict):
        return {}
    return {name: counts[name] for name in _COUNTS if is_token_count(counts.get(name))}


def _refusal(record: dict[str, Any]) -> str:
    if (reason := _string(record.get("refusal"))) is not None:
        return reason
    status = record.get("status")
    return f"the endpoint refused the call (HTTP {status})" if status else "the call reached no endpoint"


def _arguments(record: dict[str, Any]) -> Any:
    """A streamed call's ``args``, a JSON object, or its text when that's what the model gave."""
    arguments = record.get("args")
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except ValueError:
            return arguments
    return arguments


def _result_content(value: Any, turn: Any) -> list[Any]:
    """A tool's result as content: its text, each image it carries (an MCP image block, a blob's ``inline_data`` or a
    data URL) as a media file, an MCP result as its content, and anything else as its JSON."""
    if isinstance(value, list):
        return [part for item in value for part in _result_content(item, turn)] or [""]
    if isinstance(value, dict) and isinstance(value.get("content"), list):
        return _result_content(value["content"], turn)
    if isinstance(value, dict) and value.get("type") == "text" and isinstance(value.get("text"), str):
        return [value["text"]]
    if (image := _image(value)) is not None:
        return [turn.media(*image)]
    if isinstance(value, str):
        return [value]
    return [""] if value is None else [json.dumps(value, ensure_ascii=False)]


def _image(value: Any) -> tuple[bytes, str] | None:
    if isinstance(value, str):
        found = _DATA_URL.fullmatch(value)
        encoded, media_type = (found["data"], found["type"]) if found else (None, None)
    elif isinstance(value, dict):
        blob = value.get("inline_data") or value.get("inlineData") or value
        encoded, media_type = blob.get("data"), blob.get("mime_type") or blob.get("mimeType")
    else:
        return None
    if not isinstance(encoded, str) or not isinstance(media_type, str) or not media_type.startswith("image/"):
        return None
    try:
        return base64.b64decode(encoded, validate=True), media_type
    except (binascii.Error, ValueError):
        return None


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content
                       if isinstance(part, dict) and part.get("type") == "text")
    return ""
