"""browser-use's ATIF adapter: the Pydantic AI messages the server appends when a turn's run ends
(browser-use-messages/1), as ATIF. The run returns the whole conversation, so a task keeps the messages its own run
stamped, whose ``run_id`` the ``turn`` aux record names. A step is one model response: its thinking, text and calls, the
results the requests after it hold, its time, its own usage and cost, and the effort the conversation's settings put it
at, which browser-use names once, in the conversation's first request, even when the turn configured none.

agent.py keeps what the messages don't, and the server feeds it as aux records before the messages: ``attempt``, each
HTTP request to the model and how it was answered, so a refused call is a step with no model call holding its error and
one that got no response a step with no counts; ``summary``, each summarizer response of a compaction, which the
conversation never holds, a step where it was written; ``images``, the screenshots browser-use pruned from the messages,
by the call that returned them. An answer the run delivered through a finish call is that call's step's message."""
from __future__ import annotations

import base64
import json
from datetime import datetime
from typing import Any

from agentenv_protocol.atif import Adapter, Native, atif_adapter, aux, finish, is_token_count, native

NATIVE_FORMAT = "browser-use-messages/1"
#: The browser-use versions the adapter's records come from; a unit test holds browser-use-requirements.txt's pin to it.
TESTED_WITH = ("1.0.3",)
#: What browser-use puts in a tool result in place of each screenshot it prunes from the conversation.
PRUNED = "[older screenshot removed]"
_OUTCOMES = {"success": "ok", "failed": "error", "denied": "error", "retried": "error", "interrupted": "canceled"}
#: A call still unanswered when the run ended: cut off by its timeout, stopped by its error, else abandoned.
_UNANSWERED = {"timeout": "timeout", "error": "error"}
_COUNTS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")


def _is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _instant(value: str) -> datetime:
    """A time as Pydantic AI and agent.py write it: ISO 8601 in UTC, ``Z`` for its zone."""
    return datetime.fromisoformat(value)


def _usage(usage: Any) -> dict[str, Any]:
    """A response's counts, its input holding the cache, and its cost, Pydantic AI's price of it; none of either when
    the response gave no usage (``unmeasured_requests``), whose counts then mean nothing."""
    if usage.get("unmeasured_requests"):
        return {}
    counts: dict[str, Any] = {name: usage[name] for name in _COUNTS if is_token_count(usage.get(name))}
    if usage.get("cost") is not None:  # a Decimal, which Pydantic AI dumps as a string
        counts.update(cost_usd=float(usage["cost"]), cost_source="harness")
    return counts


def _text(content: Any) -> str:
    """A prompt's or a retry's content as text: its strings, and anything else as compact JSON."""
    items = content if isinstance(content, list) else [content]
    return "\n\n".join(item if isinstance(item, str) else json.dumps(item, ensure_ascii=False) for item in items)


@atif_adapter(native_format=NATIVE_FORMAT, version="0.1.0", input_includes_cache=True, tested_with=TESTED_WITH)
class BrowserUseAtif(Adapter):
    """One A2A context's browser-use conversation as ATIF, from each turn's messages and the ``turn``, ``attempt``,
    ``summary`` and ``images`` aux files."""

    def __init__(self, turn: Any) -> None:
        self._seq = 0
        self._new_task()

    def _new_task(self) -> None:
        self._run: str | None = None
        self._status: str | None = None
        self._answer: str | None = None
        self._model: str | None = None
        self._effort: str | None = None
        #: The thinking effort in effect, as the conversation's last settings change named it.
        self._effort_set: str | None = None
        self._attempts: list[dict[str, Any]] = []
        self._summaries: list[tuple[datetime, dict[str, Any]]] = []
        self._images: dict[str, dict[int, dict[str, Any]]] = {}
        self._prompted = False
        self._pending: dict[str, None] = {}
        #: The task's last step when it's a response's: its key, and whether it holds text of its own.
        self._last: tuple[str, bool] | None = None

    # Aux records, all fed before the messages

    @aux("turn")
    def turn_record(self, record: dict[str, Any], turn: Any) -> None:
        self._run, self._status = _string(record.get("run_id")), _string(record.get("status"))
        self._answer, self._model = _string(record.get("text")), _string(record.get("model"))
        self._effort = _string(record.get("effort"))
        cost = record.get("cost_usd")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            turn.task_cost(float(cost), source="harness")

    @aux("attempt")
    def attempt(self, record: dict[str, Any], turn: Any) -> None:
        self._attempts.append({**record, "at": _instant(record.get("started"))})

    @aux("summary")
    def summary(self, record: dict[str, Any], turn: Any) -> None:
        self._summaries.append((_instant(record["response"].get("timestamp")), record))

    @aux("images")
    def images(self, record: dict[str, Any], turn: Any) -> None:
        self._images[record["tool_call_id"]] = {image["index"]: image for image in record["images"]}

    # The messages

    @native(where={"kind": "request"})
    def request(self, record: dict[str, Any], turn: Any) -> None:
        ours, parts = self._ours(record), record["parts"]
        if ours and _string(record.get("timestamp")):
            # Pydantic AI stamps a request as it prepares it, so a call sent before then got its response already,
            # or never will: one the per-request deadline cut off goes before the retry feedback it led to.
            self._until(_instant(record["timestamp"]), turn, answered=False)
        for part in parts:
            settings = part.get("settings")
            if part.get("part_kind") == "model-settings-delta" and _is_mapping(settings) and "thinking" in settings:
                self._effort_set = _string(settings["thinking"])
        if not ours:
            return
        for part in parts:
            if part.get("part_kind") == "tool-return":
                self._result(part, turn)
        for part in parts:
            kind = part.get("part_kind")
            if kind == "user-prompt" and not self._prompted:
                self._prompted = True  # the task's own prompt, the user step the SDK writes
            elif kind in ("user-prompt", "retry-feedback"):
                turn.system_message(text=_text(part.get("content")), time=part.get("timestamp"))
                self._last = None

    @native(where={"kind": "response"})
    def response(self, record: dict[str, Any], turn: Any) -> None:
        if not self._ours(record):
            return
        when = record.get("timestamp")
        self._until(_instant(when), turn)
        key = self._key("response")
        turn.step(key=key, model_name=_string(record.get("model_name")), reasoning_effort=self._effort_in_effect(),
                  time=when, **_usage(record.get("usage")))
        said = self._content(record, key, when, turn, text=True)
        for part in record["parts"]:
            if part.get("part_kind") == "tool-call":
                arguments = part.get("args") or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except ValueError:
                        pass
                turn.tool_call(step=key, tool_call_id=part["tool_call_id"], function_name=part["tool_name"],
                               arguments=arguments, time=when)
                self._pending[part["tool_call_id"]] = None
        self._last = (key, said)

    @finish
    def end(self, turn: Any) -> None:
        self._until(None, turn)
        status = _UNANSWERED.get(self._status, "canceled")
        for call_id in self._pending:
            turn.tool_result(source_call_id=call_id, content=[], status=status)
        if self._status == "completed" and self._answer is not None and self._last is not None and not self._last[1]:
            turn.message(step=self._last[0], text=self._answer)
        self._new_task()

    # Writing

    def _effort_in_effect(self) -> str | None:
        """Claude's thinking effort as the conversation's settings name it; else the configured one, which an
        OpenAI-compatible route sends as a setting of its own that the messages never record."""
        return self._effort_set or self._effort

    def _ours(self, message: dict[str, Any]) -> bool:
        """Whether the message is this task's: one its run stamped, not an earlier turn's the run returned again."""
        return self._run is not None and message.get("run_id") == self._run

    def _key(self, kind: str) -> str:
        """A step key of the context's own, not the provider's response id, which can repeat: Ollama's OpenAI route
        numbers its ids below 1000, and two responses under one key would be one step."""
        self._seq += 1
        return f"{kind}-{self._seq}"

    def _content(self, response: dict[str, Any], key: str, when: Any, turn: Any, *, text: bool) -> bool:
        """A response's thinking as reasoning, and, with ``text``, its text as the step's message; whether it had
        any."""
        said = False
        for part in response["parts"]:
            kind = part.get("part_kind")
            if kind == "thinking":
                self._thinking(part, key, when, turn)
            elif kind == "text" and text and _string(part.get("content")):
                turn.message(step=key, text=part["content"], time=when)
                said = True
        return said

    @staticmethod
    def _thinking(part: dict[str, Any], key: str, when: Any, turn: Any) -> None:
        """Claude's thinking as browser-use asks for it, summarized and signed; a redacted block keeps its ciphertext,
        which Pydantic AI holds as the signature; unsigned text, as an OpenAI-compatible route returns it, is raw."""
        text, signature = _string(part.get("content")), _string(part.get("signature"))
        if part.get("id") == "redacted_thinking":
            turn.reasoning(step=key, visibility="redacted", data=signature, time=when)
        elif text is not None:
            turn.reasoning(step=key, text=text, visibility="summary" if signature else "raw", signature=signature,
                           time=when)
        elif signature is not None:
            turn.reasoning(step=key, visibility="omitted", signature=signature, time=when)

    def _result(self, part: dict[str, Any], turn: Any) -> None:
        call_id = part.get("tool_call_id")
        if call_id not in self._pending:
            return
        del self._pending[call_id]
        turn.tool_result(source_call_id=call_id, content=self._returned(part.get("content"), call_id, turn),
                         status=_OUTCOMES.get(part.get("outcome"), "ok"), time=part.get("timestamp"))

    def _returned(self, content: Any, call_id: str, turn: Any) -> list[Any]:
        """A tool's return as result content: its text, and each image as a media file, its bytes base64 in the URL-safe
        alphabet as Pydantic AI dumps them, a pruned one from what agent.py kept of it."""
        if isinstance(content, str):
            return [content]
        if not isinstance(content, list):
            return [json.dumps(content, ensure_ascii=False)]
        kept = self._images.get(call_id, {})
        parts: list[Any] = []
        for index, item in enumerate(content):
            if item == PRUNED and index in kept:
                item = {**kept[index], "kind": "binary"}
            if isinstance(item, str):
                parts.append(item)
            elif _is_mapping(item) and item.get("kind") == "binary":
                parts.append(turn.media(base64.urlsafe_b64decode(item["data"]), item["media_type"]))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return parts or [""]

    def _until(self, at: datetime | None, turn: Any, *, answered: bool = True) -> None:
        """Write the model calls that ended before ``at``, every one left when None: each compaction's summary, after
        the requests sent before it, then the requests sent before ``at``, with ``answered`` the last one not refused
        answered by the response ending there."""
        while self._summaries and (at is None or self._summaries[0][0] < at):
            written, record = self._summaries.pop(0)
            self._requests(written, True, turn)
            self._summary(record, turn)
        self._requests(at, answered and at is not None, turn)

    def _requests(self, at: datetime | None, answered: bool, turn: Any) -> None:
        """The requests sent before ``at``: with ``answered``, the last one the endpoint didn't refuse is the call a
        record ending at ``at`` answers, and each other is a step of its own."""
        sent = 0
        while sent < len(self._attempts) and (at is None or self._attempts[sent]["at"] <= at):
            sent += 1
        batch, self._attempts = self._attempts[:sent], self._attempts[sent:]
        answer = max((i for i, attempt in enumerate(batch) if not _refused(attempt)), default=None) if answered \
            else None
        for index, attempt in enumerate(batch):
            if index != answer:
                self._unanswered(attempt, turn)

    def _unanswered(self, attempt: dict[str, Any], turn: Any) -> None:
        """A request no response answered: refused, a step with no model call holding the error; else one whose
        usage nobody counted, cut off or lost on the way."""
        self._seq += 1
        key, status = f"call-{self._seq}", attempt.get("status")
        if _refused(attempt):
            turn.step(key=key, llm_call_count=0, model_name=self._model, time=attempt.get("timestamp"))
            turn.error(message=f"the endpoint refused the call (HTTP {status})", kind="infra_error", step=key)
        else:
            turn.step(key=key, model_name=self._model, reasoning_effort=self._effort_in_effect(),
                      time=attempt.get("started"))
            turn.error(message=f"the call's response (HTTP {status}) never reached the agent" if status else
                       "the call got no answer", kind="infra_error", step=key)
        self._last = None

    def _summary(self, record: dict[str, Any], turn: Any) -> None:
        """A compaction's summarizer call: the step of the summary that replaced the conversation in what the model
        sees, or, when the harness kept the conversation whole after all, a step of its usage noting why."""
        response = record["response"]
        key, when = self._key("summary"), response.get("timestamp")
        turn.step(key=key, model_name=_string(response.get("model_name")), time=when, **_usage(response.get("usage")))
        self._content(response, key, when, turn, text=bool(record.get("compacted")))
        if record.get("compacted"):
            turn.compaction(step=key, written_by="assistant", trigger="auto")
        else:
            turn.error(message="the summary was not used: the conversation was sent whole", kind="agent_error",
                       step=key)
        self._last = None

    # Back to Pydantic AI's messages

    @classmethod
    def from_atif(cls, document: dict[str, Any]) -> Native:
        """Each step as Pydantic AI messages: a user or system step a request with its text, an agent step a response
        with its reasoning, text and calls and then a request with their results; best effort, with no history file,
        which agent.py names by the session and the model."""
        log: list[dict[str, Any]] = []
        for step in document.get("steps") or ():
            text = _message_text(step.get("message"))
            if step.get("source") in ("user", "system"):
                log.append({"kind": "request", "parts": [{"part_kind": "user-prompt", "content": text}]})
                continue
            parts: list[dict[str, Any]] = []
            if _string(step.get("reasoning_content")):
                parts.append({"part_kind": "thinking", "content": step["reasoning_content"]})
            if text:
                parts.append({"part_kind": "text", "content": text})
            calls = {call["tool_call_id"]: call["function_name"] for call in step.get("tool_calls") or ()}
            parts += [{"part_kind": "tool-call", "tool_name": call["function_name"], "args": call.get("arguments") or {},
                       "tool_call_id": call["tool_call_id"]} for call in step.get("tool_calls") or ()]
            if not parts:
                continue
            metrics = step.get("metrics") if _is_mapping(step.get("metrics")) else {}
            usage = {name: metrics[key] for key, name in (("prompt_tokens", "input_tokens"),
                                                          ("cached_tokens", "cache_read_tokens"),
                                                          ("completion_tokens", "output_tokens"))
                     if is_token_count(metrics.get(key))}
            response = {"kind": "response", "parts": parts, "usage": usage}
            response.update({key: step[key] for key in ("model_name", "timestamp") if _string(step.get(key))})
            log.append(response)
            results = [{"part_kind": "tool-return", "tool_name": calls[result["source_call_id"]],
                        "tool_call_id": result["source_call_id"], "content": _message_text(result.get("content"))}
                       for result in (step.get("observation") or {}).get("results") or ()
                       if result.get("source_call_id") in calls]
            if results:
                log.append({"kind": "request", "parts": results})
        return Native(log=log)


def _refused(attempt: dict[str, Any]) -> bool:
    status = attempt.get("status")
    return isinstance(status, int) and status >= 400


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n\n".join(part.get("text", "") for part in content or ()
                       if _is_mapping(part) and part.get("type") == "text")
