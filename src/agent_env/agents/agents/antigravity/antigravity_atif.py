"""Antigravity's ATIF adapter: the SDK's streamed chunks and hook results, the native log, placed on the model calls
the turn's relay forwarded, which the bridge feeds as the ``calls`` aux file. The chunks carry no call boundary, model,
time or usage, while the relay sees every call, a sub-agent's included, and each call's request names its conversation
and whether it holds the turn's own message.

A step is one call of the turn's conversation, holding what streamed from its request to the next one's: its thoughts
as summary reasoning, its text as the bridge's answer joins it, and its tool calls with their results. A chunk of the
open step's step index streamed before anything of the next call stays in the open step. A call of another
conversation is a sub-agent's when an ``invoke_subagent`` call waits on its result, a step of its usage in a trajectory
embedded under that call; else it, and a refused call of the turn's, is a root step of its own before the turn's next
call, a refused one holding its error with no model call."""
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
_COUNTS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
_DATA_URL = re.compile(r"data:(?P<type>[\w.+-]+/[\w.+-]+);base64,(?P<data>.*)", re.S)


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


@dataclass(eq=False)
class _Call:
    """A model call the relay forwarded, by the step it writes: the turn's conversation's (``root``), a sub-agent's
    (``agent``), or neither. A root call keeps its last streamed step index and its text by step index until its step
    closes."""

    key: str
    root: bool = False
    conversation: str | None = None
    model: str | None = None
    effort: Any = None
    started: str | None = None
    agent: str | None = None
    index: Any = None
    texts: dict[Any, str] = field(default_factory=dict)
    streamed: bool = False


@atif_adapter(native_format=NATIVE_FORMAT, version="0.1.0", input_includes_cache=True, tested_with=TESTED_WITH)
class AntigravityAtif(Adapter):
    """One A2A context's Antigravity conversation as ATIF, from the SDK's chunks and hook results and the relay's
    ``calls`` records of the model calls the harness made."""

    def __init__(self, turn: Any) -> None:
        self._n = 0
        self._ids: set[str] = set()
        self._new_task()

    def _new_task(self) -> None:
        self._calls: dict[Any, _Call] = {}
        self._current: _Call | None = None
        self._open: _Call | None = None
        self._held: list[tuple[_Call, dict[str, Any]]] = []
        self._streamed: dict[str, str] = {}
        self._spawns: list[str] = []
        self._agents: dict[str, str] = {}
        self._root: str | None = None

    @aux("calls", type="agentenv.model_request")
    def request(self, record: dict[str, Any], turn: Any) -> None:
        """A call holding the turn's message is the turn's conversation's, unless it's of another conversation while an
        ``invoke_subagent`` call waits: a sub-agent given the user's words. One of another conversation otherwise is
        the turn's conversation compacted."""
        call = _Call(self._key(), conversation=_string(record.get("conversation")), model=_string(record.get("model")),
                     effort=record.get("reasoning_effort"), started=record.get("started"))
        self._calls[record.get("seq")] = call
        if record.get("turn") is True and (self._root in (None, call.conversation) or not self._spawns):
            call.root, self._root, self._current = True, call.conversation, call
        else:
            self._place(call, turn)

    @aux("calls", type="agentenv.model_call")
    def call(self, record: dict[str, Any], turn: Any) -> None:
        call = self._calls.pop(record.get("seq"), None) or _Call(self._key(), model=_string(record.get("model")))
        if call.root and (call is self._open or record.get("outcome") != "refused"):
            if call is not self._open:
                self._open_root(call, turn)
            self._write(call, record, turn)
            return
        if call is self._current:
            # The harness retries a refused call, and what streams before the retry is the open call's.
            self._current = self._open
        self._place(call, turn)
        if call.agent is None:
            self._held.append((call, record))
        else:
            self._write(call, record, turn)

    @native(type="Thought")
    def thought(self, record: dict[str, Any], turn: Any) -> None:
        if (text := _string(record.get("text"))) is not None:
            call = self._streaming(record, turn)
            turn.reasoning(step=call.key, text=text, visibility="summary")

    @native(type="Text")
    def text(self, record: dict[str, Any], turn: Any) -> None:
        if (text := _string(record.get("text"))) is not None:
            call = self._streaming(record, turn)
            call.texts[record.get("step_index")] = call.texts.get(record.get("step_index"), "") + text

    @native(type="ToolCall")
    def tool_call(self, record: dict[str, Any], turn: Any) -> None:
        call = self._streaming(record, turn)
        name, streamed = str(record["name"]), str(record["id"])
        call_id = self._unique(streamed)
        turn.tool_call(step=call.key, tool_call_id=call_id, function_name=name, arguments=_arguments(record))
        self._streamed[streamed] = call_id
        if name == _SPAWN:
            self._spawns.append(call_id)

    @native(where={"kind": "tool_result"})
    def tool_result(self, record: dict[str, Any], turn: Any) -> None:
        call_id = self._streamed.get(record.get("call_ref"))
        if call_id is None:
            # Its call never streamed, as a peer's or one the turn's end cut: the call as the hooks saw it.
            name = str(record.get("name"))
            call_id = self._unique(str(record.get("id") or name))
            holder = self._open or self._streaming({}, turn)
            turn.tool_call(step=holder.key, tool_call_id=call_id, function_name=name, arguments=None)
        if call_id in self._spawns:
            self._spawns.remove(call_id)
        error = _string(record.get("error"))
        content = [error] if error else _result_content(record.get("result"), turn)
        turn.tool_result(source_call_id=call_id, content=content, status="error" if error else "ok")

    @finish
    def end(self, turn: Any) -> None:
        self._close_root(turn)
        for call, record in self._held:
            self._write(call, record, turn)
        self._new_task()

    def _key(self) -> str:
        self._n += 1
        return f"call-{self._n}"

    def _unique(self, call_id: str) -> str:
        """``call_id``, made unique in the document: the SDK's stream ids may start again each turn."""
        candidate, n = call_id, 1
        while candidate in self._ids:
            n += 1
            candidate = f"{call_id}#{n}"
        self._ids.add(candidate)
        return candidate

    def _streaming(self, record: dict[str, Any], turn: Any) -> _Call:
        """The root call a streamed chunk is of: the open call's when the chunk has its step index and the call made
        since has streamed nothing, else the latest call of the turn's conversation, whose step it opens."""
        index, held, latest = record.get("step_index"), self._open, self._current
        if held is not None and held is not latest and index is not None and index == held.index:
            return held
        if latest is None:
            latest = self._current = _Call(self._key(), root=True)
        if latest is not held:
            self._open_root(latest, turn)
        if index is not None:
            latest.index = index
        latest.streamed = True
        return latest

    def _open_root(self, call: _Call, turn: Any) -> None:
        """Open ``call``'s step after the open one, whose text is written, and the held calls that ended during it."""
        self._close_root(turn)
        for held, record in self._held:
            self._write(held, record, turn)
        self._held = []
        turn.step(key=call.key, model_name=call.model, reasoning_effort=call.effort, time=call.started)
        self._open = call

    def _close_root(self, turn: Any) -> None:
        """The open root step's text, a paragraph per step index, as the bridge joins the turn's answer."""
        call, self._open = self._open, None
        if call is None:
            return
        if text := "\n\n".join(part.strip() for part in call.texts.values() if part.strip()):
            turn.message(step=call.key, text=text)
        call.texts.clear()

    def _place(self, call: _Call, turn: Any) -> None:
        """A call of another conversation goes in that conversation's sub-agent, declared under the oldest
        ``invoke_subagent`` call waiting on one when the conversation has none, its calls held so far moved into it."""
        if call.root or call.agent is not None or call.conversation is None:
            return
        agent = self._agents.get(call.conversation)
        if agent is None and self._spawns:
            self._n += 1
            agent = self._agents[call.conversation] = f"subagent-{self._n}"
            turn.subagent(agent, spawned_by=self._spawns.pop(0), model_name=call.model)
            moved = [(held, record) for held, record in self._held if held.conversation == call.conversation
                     and not held.root]
            self._held = [entry for entry in self._held if entry not in moved]
            for held, record in moved:
                held.agent = agent
                self._write(held, record, turn)
        call.agent = agent

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
                log.append({"type": "ToolCall", "id": call_id, "name": name, "step_index": index,
                            "args": call.get("arguments") or {}})
                if (result := results.get(call_id)) is not None:
                    log.append({"kind": "tool_result", "id": call_id, "name": name,
                                "result": _text(result.get("content")), "call_ref": call_id})
        return Native(log=log)


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
