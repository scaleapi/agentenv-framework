"""Pydantic AI's ATIF adapter: the turn's Pydantic AI messages, the native log, as ATIF. The agent runs in process and
appends each ModelRequest once it's sent and each ModelResponse once it's answered, so a step is one ModelResponse, one
model call, with its own usage, model and time, and the calls it makes are answered by the next request's tool returns.

Three aux files hold what the messages don't, fed only while a task builds an atif/1 copy: the turn's settings
(``turn``: its reasoning effort); the requests and responses of each run delegate_task starts (``delegated``), which
keep messages of their own, each naming the run, the call it runs under and that call's run, so a delegated run is a
sub-agent embedded under its call; and the turn's run failing (``failed``), with the request it had sent, whose tool
returns never reach the messages, and its error, a step with no model call whose message is the task's reply. A call
still unanswered when its run ends, a failed or cancelled one's, is answered canceled. This module imports only the
protocol, so it loads where Pydantic AI isn't installed."""
from __future__ import annotations

import base64
import json
from collections import Counter
from collections.abc import Mapping
from typing import Any

from agentenv_protocol.atif import Adapter, Native, atif_adapter, aux, finish, is_token_count, native

NATIVE_FORMAT = "pydantic-ai-messages/v1"
#: The Pydantic AI versions the adapter was checked against; a unit test holds the image's pin to this list.
TESTED_WITH = ("2.61.0",)
TURN, DELEGATED, FAILED = "turn", "delegated", "failed"
DELEGATE_TOOL = "delegate_task"
#: The sub-agent name delegate_task gives the running agent itself.
_SELF = "self"
_CALLS = ("tool-call", "capability-load-call", "tool-search-call")
_RETURNS = ("tool-return", "capability-load-return", "tool-search-return")
_NOTES = ("user-prompt", "system-prompt", "retry-feedback")
_URLS = ("image-url", "audio-url", "video-url", "document-url")
_STATUSES = {"failed": "error", "denied": "error", "retried": "error", "interrupted": "canceled"}


def counted(prompt: int, output: int, read: int, write: int,
            details: Mapping[str, Any]) -> tuple[int, int, int, int, bool] | None:
    """A model call's input counting its cache, its output, cache reads and writes, and whether its input's holding the
    cache is known; None when it reported no usage. Pydantic AI reads the cache from OpenAI's ``prompt_tokens_details``,
    whose prompt counts it; a LiteLLM proxy's Claude answers send Anthropic's ``cache_read_input_tokens`` and
    ``cache_creation_input_tokens``, kept in ``details``, beside a prompt that doesn't, so those are added when the
    prompt is smaller than them, and the call's flag is unknown otherwise. The turn's usage sums the same counts."""
    if not (prompt or output):
        return None
    if read or write:
        return prompt, output, read, write, True
    read, write = details.get("cache_read_input_tokens", 0), details.get("cache_creation_input_tokens", 0)
    if read + write == 0:
        return prompt, output, read, write, True
    if prompt < read + write:
        return prompt + read + write, output, read, write, True
    return prompt, output, read, write, False


def _is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _usage(usage: Any) -> dict[str, int]:
    """A response's usage as the step's counts, none when it reported no usage."""
    if not _is_mapping(usage):
        return {}
    numbers = [usage.get(key) for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")]
    details = usage.get("details") if _is_mapping(usage.get("details")) else {}
    counts = counted(*(number if is_token_count(number) else 0 for number in numbers), details)
    if counts is None:
        return {}
    return dict(zip(("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"), counts[:4]))


@atif_adapter(native_format=NATIVE_FORMAT, version="0.1.0", input_includes_cache=True, tested_with=TESTED_WITH)
class PydanticAIAtif(Adapter):
    """One A2A context's Pydantic AI conversation as ATIF, from its messages and the aux files the agent feeds; the
    runs delegate_task starts are sub-agents, by their run ids."""

    def __init__(self, turn: Any) -> None:
        self._seq = 0
        self._ids: Counter[str] = Counter()
        self._new_task()

    def _new_task(self) -> None:
        self._effort: str | None = None
        self._prompted: set[str | None] = set()
        self._open: dict[tuple[str | None, str], str] = {}
        self._delegates: dict[str, str | None] = {}
        self._spawned: dict[str, str] = {}

    @native(where={"kind": "request"})
    def request(self, record: dict[str, Any], turn: Any) -> None:
        self._request(record, None, turn)

    @native(where={"kind": "response"})
    def response(self, record: dict[str, Any], turn: Any) -> None:
        self._response(record, None, turn)

    @aux(TURN)
    def configured(self, record: dict[str, Any], turn: Any) -> None:
        self._effort = _string(record.get("effort"))

    @aux(DELEGATED)
    def delegated(self, record: dict[str, Any], turn: Any) -> None:
        run, message = self._run(record, turn), record["message"]
        if message.get("kind") == "request":
            self._request(message, run, turn)
        else:
            self._response(message, run, turn)

    @aux(FAILED)
    def failed(self, record: dict[str, Any], turn: Any) -> None:
        """The turn's run failing: the messages it added after its last node, then a step with no model call holding
        the error and, as its message, the task's reply. A response the failure cut off (``interrupted``) is the
        failed call that step stands for."""
        for message in record["messages"]:
            if message.get("kind") == "request":
                self._request(message, None, turn)
            elif message.get("state") != "interrupted":
                self._response(message, None, turn)
        error = record["error"]
        key = self._key()
        turn.step(key=key, llm_call_count=0, time=record.get("at"))
        turn.message(step=key, text=error["message"])
        turn.error(message=error["message"], kind=error["kind"], code=_string(error.get("code")), step=key)

    @finish
    def end(self, turn: Any) -> None:
        for call_id in self._open.values():
            turn.tool_result(source_call_id=call_id, content=[], status="canceled")
        self._new_task()

    def _key(self) -> str:
        self._seq += 1
        return f"response:{self._seq}"

    def _run(self, record: dict[str, Any], turn: Any) -> str:
        """The delegated run a record is of, declared as a sub-agent under the delegate_task call it runs under on its
        first record; delegated to the running agent itself, it carries the agent's own name."""
        run = record["run"]
        if run not in self._spawned.values():
            spawned_by = self._open[(record.get("parent"), record["spawned_by"])]
            name = self._delegates.get(spawned_by)
            turn.subagent(run, spawned_by=spawned_by, name=turn.context.agent_name if name == _SELF else name,
                          version=turn.context.agent_version)
            self._spawned[spawned_by] = run
        return run

    def _ended(self, run: str, turn: Any) -> None:
        """A delegated run whose call has its result, so has ended: its calls still open never get theirs."""
        for key in [key for key in self._open if key[0] == run]:
            call_id = self._open.pop(key)
            if call_id in self._spawned:
                self._ended(self._spawned[call_id], turn)
            turn.tool_result(source_call_id=call_id, content=[], status="canceled")

    def _request(self, message: dict[str, Any], run: str | None, turn: Any) -> None:
        """A request's tool returns as its calls' results; the prompt that starts a run is the task's user step or its
        spawning call's, and what the harness adds to a later request of the turn's own run is a system step."""
        prompted = run in self._prompted
        self._prompted.add(run)
        for part in message.get("parts") or ():
            kind = part.get("part_kind")
            if kind in _RETURNS:
                call_id = self._open.pop((run, part["tool_call_id"]))
                if call_id in self._spawned:
                    self._ended(self._spawned[call_id], turn)
                turn.tool_result(source_call_id=call_id, content=_result(part.get("content"), turn),
                                 status=_STATUSES.get(part.get("outcome"), "ok"), time=part.get("timestamp"))
            elif kind in _NOTES and prompted and run is None:
                turn.system_message(content=_note(part.get("content"), turn), time=part.get("timestamp"))

    def _response(self, message: dict[str, Any], run: str | None, turn: Any) -> None:
        key = self._key()
        turn.step(key=key, model_name=_string(message.get("model_name")), reasoning_effort=self._effort,
                  time=message.get("timestamp"), agent=run, **_usage(message.get("usage")))
        texts: list[str] = []
        for part in message.get("parts") or ():
            kind = part.get("part_kind")
            if kind == "thinking":
                text = _string(part.get("content"))
                turn.reasoning(step=key, text=text, visibility="raw" if text else "omitted",
                               signature=_string(part.get("signature")), agent=run)
            elif kind == "text":
                texts.append(part.get("content") or "")
            elif kind in _CALLS:
                self._call(part, key, run, turn)
        # The run's output is its last response's texts joined, stripped as the agent's reply is.
        if text := "".join(texts).strip():
            turn.message(step=key, text=text, agent=run)

    def _call(self, part: dict[str, Any], key: str, run: str | None, turn: Any) -> None:
        """A call under its own id, or, for an id the document already holds, as some models reuse ids across turns
        and runs, that id numbered."""
        native_id = part["tool_call_id"]
        self._ids[native_id] += 1
        call_id = native_id if self._ids[native_id] == 1 else f"{native_id}#{self._ids[native_id]}"
        self._open[(run, native_id)] = call_id
        arguments = _arguments(part.get("args"))
        if part.get("tool_name") == DELEGATE_TOOL:
            self._delegates[call_id] = _string(arguments.get("agent_name")) if _is_mapping(arguments) else None
        turn.tool_call(step=key, tool_call_id=call_id, function_name=part["tool_name"], arguments=arguments, agent=run)

    @classmethod
    def from_atif(cls, document: dict[str, Any]) -> Native:
        """Each step as the message Pydantic AI keeps: a user or system step as a request with its text as the prompt,
        and an agent step as a response with its reasoning, text, calls and usage, its results the next request's tool
        returns; the log is the conversation's messages, which a session resumes from. Best effort: sub-agents and
        media are left out."""
        log: list[dict[str, Any]] = []
        for step in document.get("steps") or ():
            source, when = step.get("source"), _string(step.get("timestamp"))
            stamped = {"timestamp": when} if when else {}
            if source in ("user", "system"):
                log.append({"kind": "request", "parts": [{"part_kind": "user-prompt",
                                                          "content": _text(step.get("message")), **stamped}]})
                continue
            if source != "agent":
                continue
            calls = [call for call in step.get("tool_calls") or () if _is_mapping(call)]
            parts: list[dict[str, Any]] = []
            if _string(step.get("reasoning_content")):
                parts.append({"part_kind": "thinking", "content": step["reasoning_content"]})
            if text := _text(step.get("message")):
                parts.append({"part_kind": "text", "content": text})
            parts += [{"part_kind": "tool-call", "tool_name": call.get("function_name"),
                       "args": call.get("arguments") or {}, "tool_call_id": call.get("tool_call_id")} for call in calls]
            if not parts:
                continue
            metrics = step.get("metrics") if _is_mapping(step.get("metrics")) else {}
            usage = {name: metrics[key] for key, name in (("prompt_tokens", "input_tokens"),
                                                          ("cached_tokens", "cache_read_tokens"),
                                                          ("completion_tokens", "output_tokens"))
                     if is_token_count(metrics.get(key))}
            log.append({"kind": "response", "parts": parts, "usage": usage, "model_name": _string(step.get("model_name")),
                        **stamped})
            results = {result.get("source_call_id"): result
                       for result in (step.get("observation") or {}).get("results") or () if _is_mapping(result)}
            returns = [{"part_kind": "tool-return", "tool_name": call.get("function_name"),
                        "content": _text(results[call.get("tool_call_id")].get("content")),
                        "tool_call_id": call.get("tool_call_id")}
                       for call in calls if call.get("tool_call_id") in results]
            if returns:
                log.append({"kind": "request", "parts": returns})
        return Native(log=log, session={"messages.json": json.dumps(log).encode()} if log else None)


def _arguments(args: Any) -> Any:
    """A call's arguments, which Pydantic AI keeps as the model's JSON text or as an object."""
    if not isinstance(args, str):
        return args
    if not args.strip():
        return {}
    try:
        return json.loads(args)
    except ValueError:
        return args


def _is_file(item: Any) -> bool:
    return _is_mapping(item) and (item.get("kind") == "binary" or item.get("kind") in _URLS)


def _items(items: list[Any], turn: Any) -> list[Any]:
    """Content items as parts: text as it is, a file as a media file (Pydantic AI writes bytes in URL-safe base64), a
    URL Pydantic AI hands the model as the URL, and anything else as its JSON text."""
    parts: list[Any] = []
    for item in items:
        if isinstance(item, str):
            parts.append(item)
        elif _is_file(item) and item["kind"] == "binary":
            parts.append(turn.media(base64.urlsafe_b64decode(item["data"]), item["media_type"]))
        elif _is_file(item):
            parts.append(item["url"])
        else:
            parts.append(json.dumps(item, ensure_ascii=False))
    return parts or [""]


def _result(content: Any, turn: Any) -> list[Any]:
    """A tool return's content as its result: text as it is, and with files among it, each as a media file; any other
    value as the JSON text the model was sent."""
    if isinstance(content, str):
        return [content]
    if content is None:
        return [""]
    if isinstance(content, list) and any(_is_file(item) for item in content):
        return _items(content, turn)
    return [json.dumps(content, ensure_ascii=False)]


def _note(content: Any, turn: Any) -> list[Any]:
    return [content] if isinstance(content, str) else _items(content if isinstance(content, list) else [content], turn)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if _is_mapping(part) and part.get("type") == "text")
    return ""
