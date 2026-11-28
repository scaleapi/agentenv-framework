"""pi's ATIF adapter: the JSON-mode events pi prints, which the bridge appends as they come, as ATIF. Each model call is
one assistant ``message_end`` holding the whole response (thinking, text, tool calls), the model, the request's start
and its usage, whose input pi counts without the prompt cache, so ``counted`` adds it back as the turn's usage does. A
call's results are the ``toolResult`` messages after it, by call id, and the model's tools each system message's
``toolsAdded``. Two things only the bridge knows come as aux records (``bridge``): when each compaction started, since a
compaction's summary call is reported only in its ``compaction_end``, untimed; and a resumed session's tools when a new
document starts on it, since pi declares tools only as they change. A call pi ended with an error and no usage is one
the endpoint refused, a step with no model call holding the error, and each retry pi starts is a harness attempt."""
from __future__ import annotations

import base64
import binascii
from datetime import datetime, timezone
from typing import Any, Mapping

from agentenv_protocol.atif import Adapter, Native, atif_adapter, aux, finish, is_token_count, native

NATIVE_FORMAT = "pi-json-events/v1"
#: The pi versions the adapter reads; a unit test holds install.sh's pin to this list.
TESTED_WITH = ("1.1.0",)
#: The aux file the bridge feeds: ``{"type": "compaction_start", "at"}`` and ``{"type": "tools", "tools"}``.
AUX = "bridge"
_USAGE = ("input", "output", "cacheRead", "cacheWrite", "totalTokens")
#: What pi's Chat Completions client writes as a thinking block's signature when the text came in one of these fields:
#: the field's name, for replay, not a provider's signature.
_REASONING_FIELDS = frozenset({"reasoning", "reasoning_content", "reasoning_text"})
_ENDED = ("error", "aborted")


def counted(usage: Mapping[str, Any]) -> tuple[int, int, int, int, int, int] | None:
    """A pi usage's input counting its cache, output, cache reads, cache writes and total, and 1 when its total
    confirms which its input was, else 0; None when all are zero, as pi records a call whose endpoint reported none.
    pi's input is the prompt less the cache, which its total counts, so the cache is added back; a total of input and
    output alone says the input holds it already, and one that says neither keeps pi's rule, unconfirmed."""
    input, output, read, write, total = (int(usage.get(key) or 0) for key in _USAGE)
    if not any((input, output, read, write, total)):
        return None
    if read + write == 0 or total == input + output:
        return input, output, read, write, total, 1
    return input + read + write, output, read, write, total, int(total == input + output + read + write)


def _is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _iso(millis: Any) -> str | None:
    """A pi message's time, milliseconds since the epoch, as the builder writes times."""
    if not isinstance(millis, int) or isinstance(millis, bool):
        return None
    seconds, rest = divmod(millis, 1000)
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + f".{rest:03d}Z"


def _counts(usage: Any) -> dict[str, Any]:
    """A usage's counts and cost as a step takes them: none when pi recorded none, its cost only when pi priced it."""
    tokens = counted(usage) if _is_mapping(usage) else None
    if tokens is None:
        return {}
    counts: dict[str, Any] = dict(zip(("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"),
                                      tokens))
    cost = (usage.get("cost") or {}).get("total")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost > 0:
        counts.update(cost_usd=cost, cost_source="harness")
    return counts


@atif_adapter(native_format=NATIVE_FORMAT, version="0.1.0", input_includes_cache=True, tested_with=TESTED_WITH)
class PiAtif(Adapter):
    """One A2A context's pi session as ATIF: a step per model call, from the events the bridge appends."""

    def __init__(self, turn: Any) -> None:
        self._seq = 0
        self._model: str | None = None
        self._new_task()

    def _new_task(self) -> None:
        self._retries = 0
        self._compaction_at: str | None = None

    @native(type="message_end", where={"message.role": "assistant"})
    def assistant(self, record: dict[str, Any], turn: Any) -> None:
        """A model call's step. A response that ended in an error or was aborted has its calls left out: pi runs none of
        them, and one cut off mid-stream may have no id yet."""
        message = record["message"]
        key = self._key("call")
        model = _string(message.get("responseModel")) or _string(message.get("model"))
        self._model = model or self._model
        when, stop, usage = _iso(message.get("timestamp")), message.get("stopReason"), message.get("usage")
        error = message.get("errorMessage") or ("pi aborted the run" if stop == "aborted" else
                                                "pi reported a model error")
        kind = "infra_error" if stop == "error" else "agent_error"
        if stop == "error" and counted(usage or {}) is None:
            turn.step(key=key, llm_call_count=0, model_name=model, time=when)
            turn.error(message=error, kind=kind, step=key)
            return
        effort = _string(message.get("thinkingLevel"))
        turn.step(key=key, model_name=model, reasoning_effort=None if effort == "off" else effort, time=when,
                  **_counts(usage))
        blocks = [block for block in message.get("content") or () if _is_mapping(block)]
        for block in blocks:
            if block.get("type") == "thinking":
                _reasoning(block, key, turn)
        if text := "".join(block.get("text") or "" for block in blocks if block.get("type") == "text"):
            turn.message(step=key, text=text)
        if stop in _ENDED:
            turn.error(message=error, kind=kind, step=key)
            return
        for block in blocks:
            if block.get("type") == "toolCall":
                turn.tool_call(step=key, tool_call_id=block["id"], function_name=block["name"],
                               arguments=block.get("arguments"))

    @native(type="message_end", where={"message.role": "toolResult"})
    def tool_result(self, record: dict[str, Any], turn: Any) -> None:
        message = record["message"]
        turn.tool_result(source_call_id=message["toolCallId"], content=_content(message.get("content"), turn),
                         status="error" if message.get("isError") else "ok", time=_iso(message.get("timestamp")))

    @native(type="message_end", where={"message.role": "system"})
    def system(self, record: dict[str, Any], turn: Any) -> None:
        _define(record["message"].get("toolsAdded"), turn)

    @native(type="compaction_end")
    def compaction(self, record: dict[str, Any], turn: Any) -> None:
        """A compaction's summary call: its summary, its usage, and the session's model, which pi summarizes with."""
        result, when = record.get("result"), self._compaction_at
        self._compaction_at = None
        key = self._key("compaction")
        model = self._model or turn.context.agent_model
        if not _is_mapping(result):
            if _string(record.get("errorMessage")):
                turn.step(key=key, llm_call_count=0, model_name=model, time=when)
                turn.error(message=record["errorMessage"], kind="infra_error", step=key)
            return
        if _string(result.get("summary")):
            turn.message(step=key, text=result["summary"], time=when)
        trigger = "manual" if record.get("reason") == "manual" else "auto"
        turn.compaction(step=key, written_by="assistant", trigger=trigger,
                        tokens_before=_count(result.get("tokensBefore")),
                        tokens_after=_count(result.get("estimatedTokensAfter")))
        turn.step(key=key, model_name=model, time=when, **_counts(result.get("usage")))

    @native(type="auto_retry_start")
    def retry(self, record: dict[str, Any], turn: Any) -> None:
        self._retries += 1
        turn.attempt(self._retries + 1)

    @native()
    def event(self, record: dict[str, Any], turn: Any) -> None:
        """A lifecycle event, the user's message (the SDK writes the user step) or a tool's execution, which its
        toolResult message repeats: nothing ATIF holds beyond what the messages give."""

    @aux(AUX, type="compaction_start")
    def compaction_started(self, record: dict[str, Any], turn: Any) -> None:
        self._compaction_at = _string(record.get("at"))

    @aux(AUX, type="tools")
    def resumed_tools(self, record: dict[str, Any], turn: Any) -> None:
        _define(record.get("tools"), turn)

    @finish
    def end(self, turn: Any) -> None:
        self._new_task()

    def _key(self, kind: str) -> str:
        self._seq += 1
        return f"{kind}:{self._seq}"

    @classmethod
    def from_atif(cls, document: dict[str, Any]) -> Native:
        """Each task as pi's events: the user's message, a message_end per agent step with its reasoning, text, calls
        and usage (pi's input, the prompt less its cache), one per result, and agent_settled; best effort, no session
        file."""
        log: list[dict[str, Any]] = []
        for step in document.get("steps") or ():
            when = _millis(step.get("timestamp"))
            if step.get("source") == "user":
                if log:
                    log.append({"type": "agent_settled", "aborted": False})
                log += [{"type": "agent_start"}, {"type": "message_end", "message": {
                    "role": "user", "content": _text(step.get("message")), "timestamp": when}}]
                continue
            if step.get("source") != "agent":
                continue
            calls = [call for call in step.get("tool_calls") or () if _is_mapping(call)]
            content: list[dict[str, Any]] = []
            if _string(step.get("reasoning_content")):
                content.append({"type": "thinking", "thinking": step["reasoning_content"]})
            if text := _text(step.get("message")):
                content.append({"type": "text", "text": text})
            content += [{"type": "toolCall", "id": call.get("tool_call_id"), "name": call.get("function_name"),
                         "arguments": call.get("arguments") or {}} for call in calls]
            log.append({"type": "message_end", "message": {
                "role": "assistant", "content": content, "model": step.get("model_name"),
                "usage": _native_usage(step.get("metrics")), "stopReason": "toolUse" if calls else "stop",
                "timestamp": when}})
            for result in (step.get("observation") or {}).get("results") or ():
                log.append({"type": "message_end", "message": {
                    "role": "toolResult", "toolCallId": result.get("source_call_id"),
                    "content": [{"type": "text", "text": _text(result.get("content"))}], "isError": False,
                    "timestamp": when}})
        if log:
            log.append({"type": "agent_settled", "aborted": False})
        return Native(log=log)


def _reasoning(block: dict[str, Any], key: str, turn: Any) -> None:
    signature = _string(block.get("thinkingSignature"))
    if block.get("redacted"):
        turn.reasoning(step=key, visibility="redacted", data=signature)
        return
    text = _string(block.get("thinking"))
    turn.reasoning(step=key, text=text, visibility="raw" if text else "omitted",
                   signature=None if signature in _REASONING_FIELDS else signature)


def _define(tools: Any, turn: Any) -> None:
    """Tools as pi declares them to the model, ``{name, description, parameters}``, on the agent block."""
    definitions = [{"type": "function", "function": {name: tool[name] for name in ("name", "description", "parameters")
                                                     if tool.get(name) is not None}}
                   for tool in tools or () if _is_mapping(tool) and _string(tool.get("name"))]
    if definitions:
        turn.tool_definitions(definitions)


def _content(blocks: Any, turn: Any) -> list[Any]:
    """A tool result's content as the model reads it: its text blocks joined by lines, as pi sends them, then each
    image as a media file."""
    blocks = [block for block in blocks or () if _is_mapping(block)]
    parts: list[Any] = []
    if text := "\n".join(block["text"] for block in blocks if block.get("type") == "text"
                         and isinstance(block.get("text"), str)):
        parts.append(text)
    for block in blocks:
        if block.get("type") == "image" and (media := _media(block, turn)) is not None:
            parts.append(media)
    return parts or [""]


def _media(block: dict[str, Any], turn: Any) -> Any:
    media_type, data = _string(block.get("mimeType")), _string(block.get("data"))
    if media_type is None or data is None or "/" not in media_type:
        return None
    try:
        return turn.media(base64.b64decode(data, validate=True), media_type)
    except (binascii.Error, ValueError):
        return None


def _count(value: Any) -> int | None:
    return value if is_token_count(value) else None


def _millis(value: Any) -> int | None:
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else None
    except ValueError:
        return None
    return None if instant is None else round(instant.timestamp() * 1000)


def _native_usage(metrics: Any) -> dict[str, Any]:
    metrics = metrics if _is_mapping(metrics) else {}
    prompt, read, output = (metrics.get(key) if is_token_count(metrics.get(key)) else 0
                            for key in ("prompt_tokens", "cached_tokens", "completion_tokens"))
    cost = metrics.get("cost_usd") if isinstance(metrics.get("cost_usd"), (int, float)) else 0
    return {"input": max(0, prompt - read), "output": output, "cacheRead": read, "cacheWrite": 0,
            "totalTokens": prompt + output, "cost": {"total": cost}}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if _is_mapping(part) and part.get("type") == "text")
    return ""
