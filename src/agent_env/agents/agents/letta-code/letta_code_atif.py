"""Letta Code's ATIF adapter: the CLI's stream-json events, the native log, and the model calls the turn's relay
forwarded, which the bridge feeds as aux records (``calls``, ``agentenv.model_call``), as ATIF. The local backend prints
each model response as a run: a line per streamed text or reasoning delta, then its usage and stop reason, and after the
run the calls the CLI made and their results. Every call the CLI and the processes it starts make goes through the
relay, its sub-agents', compaction summaries' and refused attempts' too, none of which print usage of their own.

A step is one run's response, keyed by its run id and written once the next run starts or the task ends, since its
calls follow the run: its reasoning, its text, its calls with their results, and the model and usage of the relay call
whose counts it printed, else its own counts. The relay's other calls are steps in their place, before the response
whose window they ended in (a task's last ones before its last response, so the answer stays its last step); one the
endpoint refused is a step with no model call holding the CLI's report of the failure, and a compaction's summary is
the step of the call that wrote it. A harness retry, the run's input sent again in a new run, starts an attempt."""
from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass, field
from typing import Any

from agentenv_protocol.atif import Adapter, Native, atif_adapter, aux, finish, is_token_count, native

NATIVE_FORMAT = "letta-code-events/1"
#: The CLI versions the recorded fixtures come from; a unit test holds package.json's pin to this list.
TESTED_WITH = ("0.34.26",)
_STATUSES = {"success": "ok", "error": "error"}
_COUNTS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
_MEDIA_ITEMS = ("image", "audio")


def _is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _count(value: Any) -> int | None:
    return value if is_token_count(value) else None


def _when(record: dict[str, Any]) -> str | None:
    """When the CLI wrote a record: its backend chunk's date, else the line's own stamp."""
    return _string(record.get("date")) or _string(record.get("timestamp"))


def _native_counts(usage: dict[str, Any] | None) -> dict[str, int]:
    """A run's printed usage as the relay counts a call, its input with the cache the CLI counts apart; none when every
    count is 0, as the CLI prints a response that carried no usage."""
    if not usage:
        return {}
    prompt, output = _count(usage.get("prompt_tokens")), _count(usage.get("completion_tokens"))
    read, write = _count(usage.get("cached_input_tokens")) or 0, _count(usage.get("cache_write_tokens")) or 0
    if prompt is None or output is None or not (prompt or output or read or write):
        return {}
    return {"input_tokens": prompt + read + write, "output_tokens": output, "cache_read_tokens": read,
            "cache_write_tokens": write}


def _relay_counts(call: dict[str, Any]) -> dict[str, int]:
    usage = call.get("usage") if call.get("outcome") == "answered" else None
    return {name: usage[name] for name in _COUNTS if is_token_count(usage.get(name))} if _is_mapping(usage) else {}


@dataclass
class _Run:
    """One run of the local backend as the CLI printed it, held until it ends: its text and reasoning by segment, its
    usage, its compactions' summaries, the failures it reported (each in-run retry's, then the run's error) and the
    CLI's final error, and the calls the CLI made after it with their results."""

    key: str
    first: str | None = None
    time: str | None = None
    text: dict[str, list[str]] = field(default_factory=dict)
    reasoning: dict[str, list[str]] = field(default_factory=dict)
    usage: dict[str, Any] | None = None
    summaries: list[dict[str, Any]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    final_error: str | None = None
    stopped: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def responded(self) -> bool:
        return self.usage is not None or bool(self.text or self.reasoning or self.calls)


@atif_adapter(native_format=NATIVE_FORMAT, version="0.1.0", input_includes_cache=True, tested_with=TESTED_WITH)
class LettaCodeAtif(Adapter):
    """One A2A context's Letta Code conversation as ATIF, from the CLI's stream-json events and the relay's records of
    the model calls, which the bridge feeds as the ``calls`` aux file."""

    def __init__(self, turn: Any) -> None:
        self._seq = 0
        self._new_task()

    def _new_task(self) -> None:
        self._model: str | None = None
        self._run: _Run | None = None
        self._calls: list[dict[str, Any]] = []
        self._attempt = 1
        self._retried = False

    @native(type="system")
    def init(self, record: dict[str, Any], turn: Any) -> None:
        self._model = _string(record.get("model")) or self._model

    @native(type="message", where={"message_type": "assistant_message"})
    def assistant(self, record: dict[str, Any], turn: Any) -> None:
        run = self._content(record, turn)
        texts = [part["text"] for part in record.get("content") or () if _is_mapping(part)
                 and part.get("type") == "text" and isinstance(part.get("text"), str)]
        run.text.setdefault(record.get("id") or "", []).extend(texts)

    @native(type="message", where={"message_type": "reasoning_message"})
    def reasoning(self, record: dict[str, Any], turn: Any) -> None:
        run = self._content(record, turn)
        if isinstance(record.get("reasoning"), str):
            run.reasoning.setdefault(record.get("id") or "", []).append(record["reasoning"])

    @native(type="message", where={"message_type": "usage_statistics"})
    def usage(self, record: dict[str, Any], turn: Any) -> None:
        self._content(record, turn).usage = record

    @native(type="message", where={"message_type": "stop_reason"})
    def stop(self, record: dict[str, Any], turn: Any) -> None:
        self._at(record, turn).stopped = True

    @native(type="message", where={"message_type": "event_message"})
    def event(self, record: dict[str, Any], turn: Any) -> None:
        """An in-run retry, after a failed call and before any output, or a compaction's start."""
        run = self._at(record, turn)
        data = record.get("event_data") if _is_mapping(record.get("event_data")) else {}
        if record.get("event_type") == "retry":
            run.failures.append(_string(data.get("message")) or "the call failed and was retried")

    @native(type="message", where={"message_type": "summary_message"})
    def summary(self, record: dict[str, Any], turn: Any) -> None:
        self._at(record, turn).summaries.append(record)

    @native(type="message", where={"message_type": "tool_call_message"})
    def tool_call(self, record: dict[str, Any], turn: Any) -> None:
        self._content(record, turn).calls.append(record)

    @native(type="message", where={"message_type": "tool_return_message"})
    def tool_return(self, record: dict[str, Any], turn: Any) -> None:
        if _string(record.get("tool_call_id")):
            self._at(record, turn).results[record["tool_call_id"]] = record

    @native(type="error")
    def error(self, record: dict[str, Any], turn: Any) -> None:
        """A run's failed call, printed in place of its error chunk, or after the run's stop the CLI's final error."""
        run = self._at(record, turn)
        message = _string(record.get("message")) or "Letta Code failed"
        if run.stopped:
            run.final_error = message
        else:
            run.failures.append(message)

    @native(type="retry")
    def retry(self, record: dict[str, Any], turn: Any) -> None:
        self._attempt += 1
        self._retried = True

    @native(type=("result", "recovery"))
    def known(self, record: dict[str, Any], turn: Any) -> None:
        """The task's result, whose usage is only its own process's, and an approval recovery: nothing to write."""

    @aux("calls", type="agentenv.model_call")
    def call(self, record: dict[str, Any], turn: Any) -> None:
        self._calls.append(record)

    @finish
    def end(self, turn: Any) -> None:
        self._flush(self._run or _Run(key=self._key()), turn, last=True)
        self._new_task()

    def _at(self, record: dict[str, Any], turn: Any) -> _Run:
        """The run ``record`` is part of: the one being printed, or, for a record naming another run, a new one, the
        last written first. A harness retry's attempt starts with the run after it."""
        run_id = _string(record.get("run_id"))
        if self._run is None or run_id is not None and run_id != self._run.key:
            if self._run is not None:
                self._flush(self._run, turn)
            if self._retried:
                turn.attempt(self._attempt)
                self._retried = False
            self._run = _Run(key=run_id or self._key(), first=_when(record))
        return self._run

    def _content(self, record: dict[str, Any], turn: Any) -> _Run:
        run = self._at(record, turn)
        run.time = run.time or _when(record)
        return run

    def _key(self) -> str:
        self._seq += 1
        return f"call:{self._seq}"

    def _flush(self, run: _Run, turn: Any, *, last: bool = False) -> None:
        """Write a run and the relay calls that ended since the last one: those before its own, or, for a run that made
        none, through the refused one of its last failure; at the task's end, every call left."""
        index = self._own(run)
        cut = index if index is not None else self._through_failures(len(run.failures))
        own = self._calls[index] if index is not None else None
        head, rest = self._calls[:cut], self._calls[cut + (own is not None):]
        if last:
            head, rest = head + rest, []
        self._calls = rest
        responds = own is not None or run.responded
        messages = list(run.failures)
        if run.final_error is not None and not responds:
            messages[-1:] = [run.final_error]
        messages.reverse()
        summaries = self._summaries(head, run)
        for position, call in enumerate(head):
            if call.get("outcome") == "refused":
                self._refused(call, messages.pop() if messages else None, turn)
            else:
                self._side(call, summaries.get(position), turn)
        for summary in summaries.get(None, ()):
            self._compaction(self._key(), summary, None, run.first, turn)
        while messages:
            key = self._key()
            turn.step(key=key, llm_call_count=0, time=run.first)
            turn.error(message=messages.pop(), kind="infra_error", step=key)
        if responds:
            self._response(run, own, turn)
            if run.final_error is not None:
                turn.error(message=run.final_error, step=run.key)

    def _own(self, run: _Run) -> int | None:
        """Which queued call made the run's response: the first answered one with the counts it printed, or for a
        response printed without usage, the first the relay saw answered without; None when the relay saw neither."""
        if run.usage is None:
            return None
        counts = _native_counts(run.usage)
        for index, call in enumerate(self._calls):
            if not counts and call.get("outcome") == "unreported":
                return index
            relayed = _relay_counts(call) if counts else {}
            if relayed and all(relayed.get(name) == counts[name] for name in ("input_tokens", "output_tokens")):
                return index
        return None

    def _through_failures(self, count: int) -> int:
        refused = 0
        for index, call in enumerate(self._calls):
            if refused == count:
                return index
            refused += call.get("outcome") == "refused"
        return len(self._calls)

    @staticmethod
    def _summaries(head: list[dict[str, Any]], run: _Run) -> dict[Any, Any]:
        """The run's compaction summaries by the queued call that wrote each, the last answered ones before its own;
        under None, those whose call the relay didn't see."""
        written = [position for position, call in enumerate(head) if call.get("outcome") != "refused"]
        paired = min(len(written), len(run.summaries))
        kept = len(run.summaries) - paired
        by_call: dict[Any, Any] = dict(zip(written[len(written) - paired:], run.summaries[kept:]))
        by_call[None] = run.summaries[:kept]
        return by_call

    def _side(self, call: dict[str, Any], summary: dict[str, Any] | None, turn: Any) -> None:
        """A call whose response the CLI didn't print, a sub-agent's, a summary's or a side call's: its usage alone."""
        key, when = self._key(), _string(call.get("started")) or _string(call.get("timestamp"))
        turn.step(key=key, model_name=_string(call.get("model")) or self._model, time=when, **_relay_counts(call))
        if summary is not None:
            self._compaction(key, summary, call, when, turn)

    def _compaction(self, key: str, summary: dict[str, Any], call: dict[str, Any] | None, when: str | None,
                    turn: Any) -> None:
        if call is None:
            turn.step(key=key, model_name=self._model, time=when)
        stats = summary.get("compaction_stats") if _is_mapping(summary.get("compaction_stats")) else {}
        turn.message(step=key, text=summary.get("summary") if isinstance(summary.get("summary"), str) else "",
                     time=_when(summary))
        turn.compaction(step=key, written_by="assistant", trigger="auto",
                        tokens_before=_count(stats.get("context_tokens_before")),
                        tokens_after=_count(stats.get("context_tokens_after")))

    def _refused(self, call: dict[str, Any], message: str | None, turn: Any) -> None:
        key, status = self._key(), call.get("status")
        turn.step(key=key, llm_call_count=0, model_name=_string(call.get("model")) or self._model,
                  time=_string(call.get("started")) or _string(call.get("timestamp")))
        turn.error(message=message or (f"the endpoint refused the call (HTTP {status})" if status else
                                       "the call reached no endpoint"), kind="infra_error", step=key)

    def _response(self, run: _Run, call: dict[str, Any] | None, turn: Any) -> None:
        key, when = run.key, run.time or run.first
        counts = _relay_counts(call) if call is not None else _native_counts(run.usage)
        model = (_string(call.get("model")) if call is not None else None) or self._model
        turn.step(key=key, model_name=model, time=when, **counts)
        for parts in run.reasoning.values():
            if text := "".join(parts):
                turn.reasoning(step=key, text=text, visibility="raw", time=when)
        if not any("".join(parts) for parts in run.reasoning.values()) and _count(
                (run.usage or {}).get("reasoning_tokens")):
            turn.reasoning(step=key, text=None, visibility="omitted", time=when)
        for parts in run.text.values():
            if text := "".join(parts):
                turn.message(step=key, text=text, time=when)
        for record in run.calls:
            tool = record.get("tool_call") if _is_mapping(record.get("tool_call")) else {}
            call_id, name = tool.get("tool_call_id"), tool.get("name")
            turn.tool_call(step=key, tool_call_id=call_id, function_name=name, arguments=_arguments(tool),
                           time=_when(record))
            if (result := run.results.get(call_id)) is not None:
                content = _result_content(name, result.get("tool_return"), turn)
                turn.tool_result(source_call_id=call_id, content=content,
                                 status=_STATUSES.get(result.get("status"), "ok"), time=_when(result))

    @classmethod
    def from_atif(cls, document: dict[str, Any]) -> Native:
        """Each task as the CLI prints it: an init, a run per agent step (its reasoning, its text, its usage and stop
        reason, then its calls and their results) and a result with the task's answer; best effort, no session."""
        log: list[dict[str, Any]] = []
        model = (document.get("agent") or {}).get("model_name")
        state: dict[str, Any] = {"runs": 0, "answer": ""}

        def end_task() -> None:
            log.append({"type": "result", "subtype": "success", "result": state["answer"],
                        "run_ids": [f"local-run-{n}" for n in range(1, state["runs"] + 1)]})
            state.update(runs=0, answer="")

        for step in document.get("steps") or ():
            if step.get("source") == "user":
                if log:
                    end_task()
                log.append({"type": "system", "subtype": "init", "model": model or ""})
                continue
            if step.get("source") != "agent" or step.get("llm_call_count") == 0:
                continue
            state["runs"] += 1
            run = {"type": "message", "run_id": f"local-run-{state['runs']}"}
            if _string(step.get("reasoning_content")):
                log.append({**run, "message_type": "reasoning_message", "reasoning": step["reasoning_content"]})
            if text := _result_text(step.get("message")):
                log.append({**run, "message_type": "assistant_message", "content": [{"type": "text", "text": text}]})
                state["answer"] = text
            metrics = step.get("metrics") if _is_mapping(step.get("metrics")) else {}
            cached = _count(metrics.get("cached_tokens")) or 0
            prompt = _count(metrics.get("prompt_tokens"))
            log.append({**run, "message_type": "usage_statistics", "step_count": 1,
                        "prompt_tokens": prompt - cached if prompt is not None else 0, "cached_input_tokens": cached,
                        "completion_tokens": _count(metrics.get("completion_tokens")) or 0})
            calls = step.get("tool_calls") or ()
            log.append({**run, "message_type": "stop_reason",
                        "stop_reason": "requires_approval" if calls else "end_turn"})
            results = {result.get("source_call_id"): result
                       for result in (step.get("observation") or {}).get("results") or ()}
            for call in calls:
                call_id = call.get("tool_call_id")
                log.append({"type": "message", "message_type": "tool_call_message", "tool_call": {
                    "name": call.get("function_name"), "tool_call_id": call_id,
                    "arguments": json.dumps(call.get("arguments") or {})}})
                if (result := results.get(call_id)) is not None:
                    log.append({"type": "message", "message_type": "tool_return_message", "tool_call_id": call_id,
                                "status": "success", "tool_return": _result_text(result.get("content"))})
        if log:
            end_task()
        return Native(log=log)


def _arguments(tool: dict[str, Any]) -> Any:
    """A call's arguments as the model gave them: the JSON its string holds, else the string itself."""
    arguments = tool.get("arguments")
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except ValueError:
            return arguments
    return arguments


def _result_content(name: Any, text: Any, turn: Any) -> list[Any]:
    """A result as content. An MCP tool's is its whole CallToolResult as JSON text, as the bridge's mod hands it to the
    model; one holding an image or audio is its text parts and each such item as a media file."""
    text = text if isinstance(text, str) else ""
    if not (isinstance(name, str) and name.startswith("mcp__")):
        return [text]
    try:
        result = json.loads(text)
    except ValueError:
        return [text]
    items = result.get("content") if _is_mapping(result) else None
    if not isinstance(items, list) or not any(_is_mapping(item) and item.get("type") in _MEDIA_ITEMS for item in items):
        return [text]
    parts: list[Any] = []
    for item in items:
        if _is_mapping(item) and item.get("type") == "text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
        elif _is_mapping(item) and item.get("type") in _MEDIA_ITEMS and (media := _media(item, turn)) is not None:
            parts.append(media)
        else:
            parts.append(json.dumps(item, ensure_ascii=False))
    return parts


def _media(item: dict[str, Any], turn: Any) -> Any:
    media_type, data = _string(item.get("mimeType")), _string(item.get("data"))
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
