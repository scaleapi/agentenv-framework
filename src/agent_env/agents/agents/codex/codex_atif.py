"""Codex's ATIF adapter: the CLI's rollout, which the bridge tails and feeds as aux records (``rollout``), as ATIF,
built live. The exec JSON events on stdout are the native log, but they carry no times and no per-call usage, so the
copy is built from the rollout alone: it holds each model response's items with the times the CLI wrote them, then a
token_usage_record with that response's own usage. A step is one model response, its items from the first after the
previous response's usage to its own; its key is its first item's id, and its calls keep the rollout's call ids,
unique across a thread's turns.

A sub-agent runs in a thread of its own, whose records the bridge marks with their parent's thread, and whose first
record names its path, as the parent's spawn_agent result does: that result is held until the sub-agent is declared
under the call, and the sub-agent's records until it is. A sub-agent the CLI continues in a later turn isn't placed
there; its records stay in the native log. A turn's later attempt, after the bridge resumes the thread, starts with
another task_started. A model call the endpoint refused or that failed ends its attempt's task_complete with the error,
which is a step with no call of its own."""
from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agentenv_protocol.atif import Adapter, Native, atif_adapter, aux, finish, is_token_count, native

NATIVE_FORMAT = "codex-exec-json/1"
#: The CLI versions the recorded fixtures come from; a unit test holds package.json's pin to this list.
TESTED_WITH = ("0.172.0",)
_CALLS = ("function_call", "custom_tool_call", "tool_search_call", "web_search_call")
_RESULTS = ("function_call_output", "custom_tool_call_output", "tool_search_output")
_SPAWN = "spawn_agent"
_DATA_URL = re.compile(r"data:(?P<type>[\w.+-]+/[\w.+-]+);base64,(?P<data>.*)", re.S)
_COUNTS = (("input_tokens", "input_tokens"), ("output_tokens", "output_tokens"),
           ("cached_input_tokens", "cache_read_tokens"), ("cache_write_input_tokens", "cache_write_tokens"))


def _is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _usage(counts: Any) -> dict[str, int]:
    if not _is_mapping(counts):
        return {}
    return {name: counts[key] for key, name in _COUNTS if is_token_count(counts.get(key))}


def _texts(content: Any) -> list[str]:
    return [block["text"] for block in content or () if _is_mapping(block) and isinstance(block.get("text"), str)]


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


class _Placed(Enum):
    """Where a sub-agent's records go: held until it's declared under its spawn in this task, then into its embedded
    trajectory; one an earlier task saw has its records left to the native log, and no spawn now takes it, even one
    that reuses its path."""

    HELD = "held"
    EMBEDDED = "embedded"
    EARLIER = "earlier"


@dataclass
class _Sub:
    """A sub-agent's thread: its path and name once its first record gives them, where its records go (``placed``),
    and its records while they're held."""

    path: str | None = None
    name: str | None = None
    held: list[dict[str, Any]] = field(default_factory=list)
    placed: _Placed = _Placed.HELD


@atif_adapter(native_format=NATIVE_FORMAT, version="0.1.0", input_includes_cache=True, tested_with=TESTED_WITH)
class CodexAtif(Adapter):
    """One A2A context's Codex thread as ATIF, from the rollout records the bridge feeds as the ``rollout`` aux file,
    each naming its thread; a sub-agent's thread is an embedded trajectory under the call that spawned it."""

    def __init__(self, turn: Any) -> None:
        self._model: dict[str, str] = {}
        self._effort: dict[str, str] = {}
        self._subs: dict[str, _Sub] = {}
        self._seq = 0
        self._new_task()

    def _new_task(self) -> None:
        self._open: dict[str, tuple[str, Any]] = {}
        self._starts = 0
        self._spawns: set[str] = set()
        self._held_results: dict[str, tuple[str, dict[str, Any]]] = {}

    @native()
    def exec_event(self, record: dict[str, Any], turn: Any) -> None:
        """An exec JSON event or the bridge's usage record: the native log's, which the rollout's records cover."""

    @aux("rollout")
    def rollout(self, record: dict[str, Any], turn: Any) -> None:
        thread = record["thread"]
        if record.get("parent") is not None:
            sub = self._subs.setdefault(thread, _Sub())
            if record.get("type") == "session_meta":
                self._meta(thread, sub, record["payload"], turn)
                return
            if sub.placed is not _Placed.EMBEDDED:
                sub.held.append(record)
                return
        self._apply(record, thread, turn)

    @finish
    def end(self, turn: Any) -> None:
        for _, result in list(self._held_results.values()):
            self._result(result, turn)
        for sub in self._subs.values():
            for record in sub.held:
                turn.unmatched(record)
            sub.held.clear()
            sub.placed = _Placed.EARLIER
        self._new_task()

    def _meta(self, thread: str, sub: _Sub, payload: dict[str, Any], turn: Any) -> None:
        sub.path = _string(payload.get("agent_path"))
        sub.name = _string(payload.get("agent_nickname")) or sub.path
        if sub.placed is _Placed.HELD and sub.path in self._held_results:
            self._declare(thread, *self._held_results.pop(sub.path), turn)

    def _apply(self, record: dict[str, Any], thread: str, turn: Any) -> None:
        kind, payload, when = record.get("type"), record["payload"], record.get("timestamp")
        if kind == "turn_context":
            if _string(payload.get("model")):
                self._model[thread] = payload["model"]
            if _string(payload.get("effort")):
                self._effort[thread] = payload["effort"]
        elif kind == "token_usage_record":
            self._responded(thread, payload, when, turn)
        elif kind == "event_msg":
            self._event(thread, payload, when, turn)
        elif kind == "response_item":
            if payload.get("type") in _RESULTS:
                self._output(payload, record, turn)
            else:
                self._item(thread, payload, when, turn)

    def _event(self, thread: str, payload: dict[str, Any], when: Any, turn: Any) -> None:
        event = payload.get("type")
        if event == "task_started":
            if thread not in self._subs:
                self._starts += 1
                if self._starts > 1:
                    self._open.pop(thread, None)
                    turn.attempt(self._starts)
        elif event == "task_complete":
            error = payload.get("error") if _is_mapping(payload.get("error")) else {}
            if (message := _string(error.get("message"))) is not None:
                self._failed(thread, message, when, turn)

    def _item(self, thread: str, payload: dict[str, Any], when: Any, turn: Any) -> None:
        agent = self._agent(thread)
        step = self._step(thread, payload, when, turn)
        item = payload.get("type")
        if item == "reasoning":
            summary = "\n".join(_texts(payload.get("summary")) + _texts(payload.get("content")))
            ciphertext = _string(payload.get("encrypted_content"))
            visibility = "summary" if summary else "redacted" if ciphertext else "omitted"
            turn.reasoning(step=step, text=summary or None, visibility=visibility,
                           data=ciphertext if visibility == "redacted" else None, time=when, agent=agent)
        elif item == "message":
            text = "".join(_texts(payload.get("content")))
            if text:
                turn.message(step=step, text=text, time=when, agent=agent)
        elif item in _CALLS:
            call_id, function, arguments = _call(payload)
            turn.tool_call(step=step, tool_call_id=call_id, function_name=function, arguments=arguments, time=when,
                           agent=agent)
            if function == _SPAWN:
                self._spawns.add(call_id)
            if item == "web_search_call":
                # The provider runs the search inside the response; the rollout has no result for it, only its status.
                status = _string(payload.get("status")) or "completed"
                turn.tool_result(source_call_id=call_id, text=status, status="ok" if status == "completed" else "error",
                                 time=when)

    def _output(self, payload: dict[str, Any], record: dict[str, Any], turn: Any) -> None:
        call_id = payload.get("call_id")
        path = _spawned_path(payload.get("output")) if call_id in self._spawns else None
        if path is None:
            self._result(record, turn)
            return
        thread = next((t for t, sub in self._subs.items() if sub.path == path and sub.placed is _Placed.HELD), None)
        if thread is None:
            self._held_results[path] = (call_id, record)
        else:
            self._declare(thread, call_id, record, turn)

    def _step(self, thread: str, payload: dict[str, Any], when: Any, turn: Any) -> str:
        """The step of the thread's response being written, opened by this item when none is."""
        if thread not in self._open:
            self._seq += 1
            key = _string(payload.get("id")) or f"{thread}:{self._seq}"
            self._open[thread] = (key, when)
            turn.step(key=key, model_name=self._model.get(thread), reasoning_effort=self._effort.get(thread),
                      time=when, agent=self._agent(thread))
        return self._open[thread][0]

    def _responded(self, thread: str, payload: dict[str, Any], when: Any, turn: Any) -> None:
        """A response's usage, which ends its step; a response that wrote nothing gets a step for its usage alone."""
        key, at = self._open.pop(thread, (None, when))
        if key is None:
            self._seq += 1
            key = _string(payload.get("response_id")) or f"{thread}:{self._seq}"
        turn.step(key=key, model_name=self._model.get(thread), reasoning_effort=self._effort.get(thread), time=at,
                  agent=self._agent(thread), **_usage(payload.get("usage")))

    def _failed(self, thread: str, message: str, when: Any, turn: Any) -> None:
        """A model call that failed or was refused: a step with no call of its own holding the error."""
        self._open.pop(thread, None)
        self._seq += 1
        key = f"{thread}:error:{self._seq}"
        agent = self._agent(thread)
        turn.step(key=key, llm_call_count=0, time=when, agent=agent)
        turn.error(message=message, kind="infra_error", step=key, agent=agent)

    def _declare(self, thread: str, call_id: str, result: dict[str, Any], turn: Any) -> None:
        """Declare sub-agent ``thread`` under the spawn_agent call ``call_id``, then write the call's held result and
        the sub-agent's held records."""
        sub = self._subs[thread]
        turn.subagent(thread, spawned_by=call_id, name=sub.name)
        sub.placed = _Placed.EMBEDDED
        self._result(result, turn)
        held, sub.held = sub.held, []
        for record in held:
            self._apply(record, thread, turn)

    def _result(self, record: dict[str, Any], turn: Any) -> None:
        payload = record["payload"]
        if payload.get("type") == "tool_search_output":
            content: list[Any] = [json.dumps(payload.get("tools") or [], ensure_ascii=False)]
        else:
            content = _content(payload.get("output"), turn)
        turn.tool_result(source_call_id=payload["call_id"], content=content, time=record.get("timestamp"))

    def _agent(self, thread: str) -> str | None:
        return thread if thread in self._subs else None

    @classmethod
    def from_atif(cls, document: dict[str, Any]) -> Native:
        """Each task as exec JSON events: a turn per user step, an item per message, reasoning and call with its
        result, and the turn's usage summed from its steps; best effort, no rollout."""
        log: list[dict[str, Any]] = []
        usage: dict[str, int] = {}
        index = 0

        def item(body: dict[str, Any]) -> None:
            nonlocal index
            log.append({"type": "item.completed", "item": {"id": f"item_{index}", **body}})
            index += 1

        def end_turn() -> None:
            log.append({"type": "turn.completed", "usage": dict(usage)})
            usage.clear()

        for step in document.get("steps") or ():
            if step.get("source") == "user":
                if log:
                    end_turn()
                log.append({"type": "turn.started"})
                continue
            if step.get("source") != "agent":
                continue
            metrics = step.get("metrics") if _is_mapping(step.get("metrics")) else {}
            for key, name in (("prompt_tokens", "input_tokens"), ("cached_tokens", "cached_input_tokens"),
                              ("completion_tokens", "output_tokens")):
                if is_token_count(metrics.get(key)):
                    usage[name] = usage.get(name, 0) + metrics[key]
            if _string(step.get("reasoning_content")):
                item({"type": "reasoning", "text": step["reasoning_content"]})
            results = {result.get("source_call_id"): result
                       for result in (step.get("observation") or {}).get("results") or ()}
            for call in step.get("tool_calls") or ():
                result = results.get(call.get("tool_call_id"))
                item(_native_call(call, _result_text(result.get("content")) if result else None))
            text = _result_text(step.get("message"))
            if text:
                item({"type": "agent_message", "text": text})
        if log:
            end_turn()
        return Native(log=log)


def _call(payload: dict[str, Any]) -> tuple[str, str, Any]:
    """A call item's id, function name and arguments as the model gave them; an MCP tool's name keeps its namespace,
    as the model sees it."""
    item = payload["type"]
    if item == "web_search_call":
        return payload["id"], "web_search", payload.get("action") or {}
    if item == "tool_search_call":
        return payload["call_id"], "tool_search", payload.get("arguments")
    name, namespace = payload["name"], _string(payload.get("namespace"))
    function = f"{namespace}__{name}" if namespace is not None and namespace.startswith("mcp__") else name
    if item == "custom_tool_call":
        return payload["call_id"], function, payload.get("input")
    arguments = payload.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            pass
    return payload["call_id"], function, arguments


def _content(output: Any, turn: Any) -> list[Any]:
    """A call's output as result content: its text, and each image it carries as a media file."""
    if isinstance(output, str):
        return [output]
    if not isinstance(output, list):
        return [json.dumps(output, ensure_ascii=False)]
    parts: list[Any] = []
    for block in output:
        kind = block.get("type") if _is_mapping(block) else None
        if kind in ("input_text", "output_text") and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif kind == "input_image" and (image := _image(block.get("image_url"))) is not None:
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


def _spawned_path(output: Any) -> str | None:
    """The sub-agent path a spawn_agent call's result names (``{"task_name": "/root/<name>"}``)."""
    try:
        named = json.loads(output) if isinstance(output, str) else None
    except ValueError:
        return None
    return _string(named.get("task_name")) if _is_mapping(named) else None


def _native_call(call: dict[str, Any], output: str | None) -> dict[str, Any]:
    name, arguments = call.get("function_name") or "", call.get("arguments") or {}
    if name == "exec_command":
        return {"type": "command_execution", "command": arguments.get("cmd"), "aggregated_output": output or "",
                "status": "completed"}
    server, _, tool = name[len("mcp__"):].partition("__") if name.startswith("mcp__") else ("", "", name)
    return {"type": "mcp_tool_call", "server": server, "tool": tool or name, "arguments": arguments,
            "result": {"content": [{"type": "text", "text": output}]} if output is not None else None,
            "status": "completed"}


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if _is_mapping(part) and part.get("type") == "text")
    return ""
