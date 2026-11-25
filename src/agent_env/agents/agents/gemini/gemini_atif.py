"""Gemini's ATIF adapter: the CLI's session transcript, whose model messages the bridge feeds as aux records
(``session``), and the model calls the agent's proxy forwarded (``agentenv.model_call`` records in the native log), as
ATIF. The stream-json events on stdout are the native log, but they carry no thoughts, no tool results and no usage;
the transcript holds each model response whole once the CLI has run its calls, and the proxy each call's own usage.

A step is one model response, keyed by its message's id: its thoughts as summary reasoning, its text, its calls with
their results, and the usage of the proxy's call whose input it names, else the transcript's own count. A call the
proxy forwarded that wrote no message, as the CLI's routing and next-speaker checks, is a step with its usage alone,
and one the endpoint refused is a step with no model call holding the error, which the CLI's own report of a failed
turn replaces when the turn ended there."""
from __future__ import annotations

import base64
import binascii
import json
from typing import Any

from agentenv_protocol.atif import Adapter, Native, atif_adapter, aux, finish, native

NATIVE_FORMAT = "gemini-stream-json/1"
#: The CLI versions the recorded fixtures come from; a unit test holds package.json's pin to this list.
TESTED_WITH = ("0.38.2",)
_STATUSES = {"success": "ok", "error": "error", "cancelled": "canceled"}
_COUNTS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")


def _is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _token_count(value: Any) -> int | None:
    """``value`` when it counts tokens: the adapter imports only what its protocol floor has."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _usage(counts: Any) -> dict[str, int]:
    """A proxy record's counts as the step's: its input holds the cache, its output the thoughts."""
    if not _is_mapping(counts):
        return {}
    return {name: count for name in _COUNTS if (count := _token_count(counts.get(name))) is not None}


def _transcript_inputs(tokens: Any) -> int | None:
    """A response's input as the proxy counts it: the prompt, its cache included, and a tool-use prompt, which the
    transcript counts apart."""
    inputs = _token_count(tokens.get("input")) if _is_mapping(tokens) else None
    return None if inputs is None else inputs + (_token_count(tokens.get("tool")) or 0)


def _transcript_usage(tokens: Any) -> dict[str, int]:
    """The transcript's count of a response, for one the proxy didn't count, as the proxy counts one: the output holds
    the thoughts."""
    if not _is_mapping(tokens):
        return {}
    output = _token_count(tokens.get("output"))
    counts = {"input_tokens": _transcript_inputs(tokens), "cache_read_tokens": _token_count(tokens.get("cached")),
              "output_tokens": None if output is None else output + (_token_count(tokens.get("thoughts")) or 0)}
    return {name: count for name, count in counts.items() if count is not None}


@atif_adapter(native_format=NATIVE_FORMAT, version="0.1.0", input_includes_cache=True, tested_with=TESTED_WITH)
class GeminiAtif(Adapter):
    """One A2A context's Gemini session as ATIF, from the transcript's model messages the bridge feeds as the
    ``session`` aux file and the proxy's records of the calls the CLI made."""

    def __init__(self, turn: Any) -> None:
        self._seq = 0
        self._new_task()

    def _new_task(self) -> None:
        self._calls: list[dict[str, Any]] = []
        self._error: str | None = None
        self._refused: str | None = None

    @native(type="agentenv.model_call")
    def model_call(self, record: dict[str, Any], turn: Any) -> None:
        self._calls.append(record)

    @native(type="result")
    def result(self, record: dict[str, Any], turn: Any) -> None:
        error = record.get("error") if _is_mapping(record.get("error")) else {}
        if record.get("status") == "error":
            self._error = _string(error.get("message"))

    @native()
    def event(self, record: dict[str, Any], turn: Any) -> None:
        """A stream-json event the transcript covers."""

    @aux("session", type="gemini")
    def message(self, record: dict[str, Any], turn: Any) -> None:
        tokens = record.get("tokens")
        index = self._match(_transcript_inputs(tokens))
        for call in self._calls[:len(self._calls) if index is None else index]:
            self._side(call, turn)
        call = self._calls[index] if index is not None else None
        self._calls = self._calls[index + 1:] if index is not None else []
        usage = _usage(call.get("usage")) if call is not None and call.get("outcome") == "answered" \
            else _transcript_usage(tokens)
        key, when = record["id"], record.get("timestamp")
        turn.step(key=key, model_name=_string(record.get("model")) or (call or {}).get("model"), time=when, **usage)
        for thought in record.get("thoughts") or ():
            if (text := _thought(thought)) is not None:
                turn.reasoning(step=key, text=text, visibility="summary", time=thought.get("timestamp"))
        if text := _text(record.get("content")):
            turn.message(step=key, text=text, time=when)
        for tool in record.get("toolCalls") or ():
            turn.tool_call(step=key, tool_call_id=tool["id"], function_name=tool["name"], arguments=tool.get("args"),
                           time=when)
            content = _content(tool.get("result"), turn)
            status = _STATUSES.get(tool.get("status"), "ok")
            if content or status != "ok":
                turn.tool_result(source_call_id=tool["id"], content=content or [""], status=status,
                                 time=tool.get("timestamp"))
        self._refused = None

    @finish
    def end(self, turn: Any) -> None:
        for call in self._calls:
            self._side(call, turn)
        if self._error is not None and self._refused is not None:
            turn.error(message=self._error, kind="infra_error", step=self._refused)
        self._new_task()

    def _match(self, inputs: int | None) -> int | None:
        """Which of the proxy's calls not yet written made the response counting ``inputs``: the first answered with
        that input, else the first whose counts the proxy couldn't read; the calls before it wrote no message."""
        answered = [i for i, call in enumerate(self._calls) if call.get("outcome") == "answered"
                    and inputs is not None and _usage(call.get("usage")).get("input_tokens") == inputs]
        unreported = [i for i, call in enumerate(self._calls) if call.get("outcome") == "unreported"]
        return (answered or unreported or [None])[0]

    def _side(self, call: dict[str, Any], turn: Any) -> None:
        """A call the proxy forwarded that wrote no message: its usage alone, or, refused, its error."""
        self._seq += 1
        key = f"call:{self._seq}"
        model, when = _string(call.get("model")), call.get("timestamp")
        if call.get("outcome") == "refused":
            status = call.get("status")
            turn.step(key=key, llm_call_count=0, model_name=model, time=when)
            turn.error(message=f"the endpoint refused the call (HTTP {status})" if status else
                       "the call reached no endpoint", kind="infra_error", step=key)
            self._refused = key
        else:
            turn.step(key=key, model_name=model, time=when, **_usage(call.get("usage")))
            self._refused = None

    @classmethod
    def from_atif(cls, document: dict[str, Any]) -> Native:
        """Each task as stream-json events: the user's message, each step's text as an assistant message and its calls
        as tool_use and tool_result events, and a result event with the task's usage summed from its steps; best
        effort, no transcript."""
        log: list[dict[str, Any]] = []
        stats: dict[str, int] = {}

        def end_task() -> None:
            log.append({"type": "result", "status": "success", "stats": dict(stats)})
            stats.clear()

        for step in document.get("steps") or ():
            if step.get("source") == "user":
                if log:
                    end_task()
                log.append({"type": "message", "role": "user", "content": _result_text(step.get("message"))})
                continue
            if step.get("source") != "agent":
                continue
            metrics = step.get("metrics") if _is_mapping(step.get("metrics")) else {}
            for key, name in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens"),
                              ("cached_tokens", "cached")):
                if _token_count(metrics.get(key)) is not None:
                    stats[name] = stats.get(name, 0) + metrics[key]
            results = {result.get("source_call_id"): result
                       for result in (step.get("observation") or {}).get("results") or ()}
            for call in step.get("tool_calls") or ():
                log.append({"type": "tool_use", "tool_name": call.get("function_name"),
                            "tool_id": call.get("tool_call_id"), "parameters": call.get("arguments") or {}})
                if (result := results.get(call.get("tool_call_id"))) is not None:
                    log.append({"type": "tool_result", "tool_id": call.get("tool_call_id"), "status": "success",
                                "output": _result_text(result.get("content"))})
            if text := _result_text(step.get("message")):
                log.append({"type": "message", "role": "assistant", "content": text, "delta": True})
        if log:
            end_task()
        return Native(log=log)


def _thought(thought: Any) -> str | None:
    if not _is_mapping(thought):
        return None
    subject, description = _string(thought.get("subject")), _string(thought.get("description"))
    return f"{subject}: {description}" if subject and description else subject or description


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part["text"] for part in content if _is_mapping(part) and isinstance(part.get("text"), str))
    return ""


def _content(result: Any, turn: Any) -> list[Any]:
    """A call's result as content: the function's output, or its error, and each image beside it as a media file."""
    parts: list[Any] = []
    for item in result if isinstance(result, list) else ():
        if not _is_mapping(item):
            continue
        answer = item.get("functionResponse")
        response = answer.get("response") if _is_mapping(answer) else None
        if _is_mapping(response):
            said = _string(response.get("output")) or _string(response.get("error"))
            parts.append(said if said is not None else json.dumps(response, ensure_ascii=False))
        inline = item.get("inlineData")
        if _is_mapping(inline) and (media := _media(inline, turn)) is not None:
            parts.append(media)
    return parts


def _media(inline: dict[str, Any], turn: Any) -> Any:
    media_type, data = _string(inline.get("mimeType")), _string(inline.get("data"))
    if media_type is None or data is None or "/" not in media_type:
        return None
    try:
        return turn.media(base64.b64decode(data, validate=True), media_type)
    except (binascii.Error, ValueError):
        return None


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if _is_mapping(part) and part.get("type") == "text")
    return ""
