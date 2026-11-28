"""OpenHands' ATIF adapter: the SDK's events (each tagged by its ``kind``) as ATIF, built live. The events carry no
usage, model or zoned time, so the runtime reads each model call off the turn's LLMs' metrics and feeds it as an
``llm_call`` record just before its response's first event. A step is one model call: its record opens it, and the
events naming its response id fill it (the response's actions, each answered by its observation there, its message,
or the condenser's summary). A call the SDK failed and made again, the endpoint refusing the conversation and the run's
own failure are steps with no model call holding the error; the harness's own messages to the model are system steps.
The tools the model is offered come as a ``tools`` record, the clock the events' naive times are on as ``clock``, and
the turn's cost as ``turn``."""
from __future__ import annotations

import base64
import binascii
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from agentenv_protocol.atif import Adapter, Native, atif_adapter, aux, finish, is_token_count, native

NATIVE_FORMAT = "openhands-events/v1"
#: The SDK versions the recorded fixtures come from; a unit test holds requirements.txt's pin to this list.
TESTED_WITH = ("1.58.2",)
_COUNTS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
_DATA_URL = re.compile(r"data:(?P<type>[\w.+-]+/[\w.+-]+);base64,(?P<data>.*)", re.S)
_RESULTS = ("ObservationEvent", "AgentErrorEvent", "UserRejectObservation")
#: Events with nothing for the copy: the system prompt, whose tools lack their parameters there and come as a
#: ``tools`` record instead, and an interrupt, whose calls left open are answered as the task ends.
_SILENT = ("SystemPromptEvent", "InterruptEvent")
_FINISH = "finish"
_REFUSED = "the endpoint refused the conversation, so it is condensed before the call is made again"


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _text(content: Any) -> str:
    """The text parts' texts joined as the SDK joins a final reply's (``get_agent_final_response``)."""
    return "".join(part["text"] for part in content or () if isinstance(part, dict) and part.get("type") == "text"
                   and isinstance(part.get("text"), str))


@atif_adapter(native_format=NATIVE_FORMAT, version="0.1.0", input_includes_cache=True, tested_with=TESTED_WITH)
class OpenHandsAtif(Adapter):
    """One A2A context's OpenHands conversation as ATIF, from the SDK's events and the runtime's ``llm_call``,
    ``tools``, ``clock`` and ``turn`` records."""

    def __init__(self, turn: Any) -> None:
        self._seq = 0
        self._zone = timezone.utc
        self._new_task()

    def _new_task(self) -> None:
        self._steps: dict[str, str] = {}
        self._pending: dict[str, None] = {}
        self._said: set[str] = set()
        self._prompted = False

    @aux("llm_call", type="llm_call")
    def call(self, record: dict[str, Any], turn: Any) -> None:
        self._seq += 1
        model, when = _string(record.get("model")), record.get("timestamp")
        if record.get("outcome") == "failed":
            key = f"failed:{self._seq}"
            turn.step(key=key, llm_call_count=0, model_name=model, time=when)
            turn.error(message=_string(record.get("error")) or "the model call failed", kind="infra_error",
                       code=_string(record.get("error_type")), step=key)
            return
        key = f"call:{self._seq}"
        if (response := _string(record.get("response_id"))) is not None:
            self._steps[response] = key
        cost = _number(record.get("cost_usd"))
        turn.step(key=key, model_name=model, reasoning_effort=_string(record.get("reasoning_effort")), time=when,
                  cost_usd=cost, cost_source="harness" if cost is not None else None,
                  **{name: record[name] for name in _COUNTS if is_token_count(record.get(name))})

    @aux("tools", type="tools")
    def tools(self, record: dict[str, Any], turn: Any) -> None:
        turn.tool_definitions(record["tools"])

    @aux("clock", type="clock")
    def clock(self, record: dict[str, Any], turn: Any) -> None:
        self._zone = timezone(timedelta(seconds=record["utc_offset_s"]))

    @aux("turn", type="turn")
    def cost(self, record: dict[str, Any], turn: Any) -> None:
        if (cost := _number(record.get("cost_usd"))) is not None:
            turn.task_cost(cost, source="harness")

    @native(where={"kind": "MessageEvent", "source": "agent"})
    def said(self, record: dict[str, Any], turn: Any) -> None:
        """A response with no call, or the SDK's own word as the agent when no response id names a model call."""
        message, when = record.get("llm_message") or {}, self._time(record)
        text = _text(message.get("content"))
        if _string(record.get("llm_response_id")) is None:
            if text:
                turn.system_message(text=text, time=when)
            return
        key = self._key(record)
        self._reason(message, key, when, turn)
        if text:
            self._said.add(key)
            turn.message(step=key, text=text, time=when)

    @native(where={"kind": "MessageEvent"})
    def told(self, record: dict[str, Any], turn: Any) -> None:
        """The task's prompt, which the runner writes as its user step, then the harness's own messages to the model:
        a malformed call's error, a nudge, the skills' text it adds to a message."""
        when = self._time(record)
        content = _content((record.get("llm_message") or {}).get("content"), turn)
        if record.get("source") == "user" and not self._prompted:
            self._prompted, content = True, []
        content += _content(record.get("extended_content"), turn)
        if content:
            turn.system_message(content=content, time=when)

    @native(where={"kind": "ActionEvent"})
    def action(self, record: dict[str, Any], turn: Any) -> None:
        """One call of a response; the first carries its reasoning and its text. A finish call's message is its step's
        when the model wrote nothing else there, so the task's reply is written once."""
        key, when = self._key(record), self._time(record)
        self._reason(record, key, when, turn)
        name = record["tool_name"]
        text = _text(record.get("thought"))
        if not text and name == _FINISH and key not in self._said:
            text = _string((record.get("action") or {}).get("message")) or ""
        if text:
            self._said.add(key)
            turn.message(step=key, text=text, time=when)
        turn.tool_call(step=key, tool_call_id=record["tool_call_id"], function_name=name,
                       arguments=_arguments(record.get("tool_call")), time=when)
        self._pending[record["tool_call_id"]] = None

    @native(where={"kind": _RESULTS})
    def result(self, record: dict[str, Any], turn: Any) -> None:
        """A call's observation, its error or its rejection. One for a call this task didn't make, as for an action
        the SDK runs again from a conversation persisted before this copy began, stays in the native log."""
        call_id = record.get("tool_call_id")
        if call_id not in self._pending:
            self.unmatched(record, turn)
            return
        del self._pending[call_id]
        kind, when = record["kind"], self._time(record)
        if kind == "AgentErrorEvent":
            turn.tool_result(source_call_id=call_id, text=record.get("error") or "", status="error", time=when)
        elif kind == "UserRejectObservation":
            turn.tool_result(source_call_id=call_id, text=record.get("rejection_reason") or "", status="canceled",
                             time=when)
        else:
            observation = record.get("observation") or {}
            content = _content(observation.get("content"), turn) + _content(record.get("extended_content"), turn)
            turn.tool_result(source_call_id=call_id, content=content, status=_status(observation),
                             exit_code=_exit_code(observation.get("exit_code")), time=when)

    @native(where={"kind": "Condensation"})
    def condensed(self, record: dict[str, Any], turn: Any) -> None:
        """The condenser's summary of the events it drops, written by its copy of the agent's model in the call whose
        record came just before."""
        key = self._key(record)
        if summary := _string(record.get("summary")):
            turn.message(step=key, text=summary, time=self._time(record))
        turn.compaction(step=key, written_by="assistant", trigger="auto")

    @native(where={"kind": "CondensationRequest"})
    def refused(self, record: dict[str, Any], turn: Any) -> None:
        """A call the endpoint refused for its conversation, too long for the model's window or malformed, which got
        no response."""
        self._seq += 1
        key = f"refused:{self._seq}"
        turn.step(key=key, llm_call_count=0, time=self._time(record))
        turn.error(message=_REFUSED, kind="infra_error", code="CondensationRequest", step=key)

    @native(where={"kind": "ConversationErrorEvent"})
    def failed(self, record: dict[str, Any], turn: Any) -> None:
        """The run's failure, typed as the runtime types the task's: the agent's own when the SDK classes it so."""
        key = f"error:{record.get('id')}"
        code = _string(record.get("code"))
        classification = record.get("classification") or {}
        turn.step(key=key, llm_call_count=0, time=self._time(record))
        turn.error(message=_string(record.get("detail")) or code or "the run failed", code=code, step=key,
                   kind="agent_error" if classification.get("kind") == "agent_action" else "infra_error")

    @native(where={"kind": _SILENT})
    def silent(self, record: dict[str, Any], turn: Any) -> None:
        """An event the copy takes nothing from."""

    def unmatched(self, record: Any, turn: Any) -> None:
        turn.unmatched({"type": record.get("kind")} if isinstance(record, dict) else record)

    @finish
    def end(self, turn: Any) -> None:
        for call_id in self._pending:
            turn.tool_result(source_call_id=call_id, content=[], status="canceled")
        self._new_task()

    def _key(self, record: dict[str, Any]) -> str:
        """The step of the model call whose response the event names: its record's, or one of the event's own."""
        response = record.get("llm_response_id")
        return self._steps.get(response) or f"response:{response}"

    def _time(self, record: dict[str, Any]) -> datetime | None:
        """An event's time, written in the agent's local time with no zone, in the zone the clock record gives."""
        try:
            moment = datetime.fromisoformat(record["timestamp"]) if isinstance(record.get("timestamp"), str) else None
        except ValueError:
            return None
        return moment if moment is None or moment.tzinfo is not None else moment.replace(tzinfo=self._zone)

    @staticmethod
    def _reason(holder: dict[str, Any], key: str, when: datetime | None, turn: Any) -> None:
        """A response's reasoning: Anthropic's thinking blocks with their signatures, else an OpenAI Responses
        reasoning item's summary, raw text or ciphertext, else LiteLLM's reasoning text, which repeats the blocks."""
        blocks = holder.get("thinking_blocks") or ()
        item = holder.get("responses_reasoning_item")
        if blocks:
            for block in blocks:
                if block.get("type") == "redacted_thinking":
                    turn.reasoning(step=key, visibility="redacted", data=_string(block.get("data")), time=when)
                else:
                    text = _string(block.get("thinking"))
                    turn.reasoning(step=key, text=text, visibility="raw" if text else "omitted",
                                   signature=_string(block.get("signature")), time=when)
        elif isinstance(item, dict):
            summary = "\n".join(text for text in item.get("summary") or () if isinstance(text, str) and text)
            raw = "\n".join(text for text in item.get("content") or () if isinstance(text, str) and text)
            ciphertext = _string(item.get("encrypted_content"))
            if summary:
                turn.reasoning(step=key, text=summary, visibility="summary", time=when)
            if raw:
                turn.reasoning(step=key, text=raw, time=when)
            if not summary and not raw:
                turn.reasoning(step=key, visibility="redacted" if ciphertext else "omitted", data=ciphertext,
                               time=when)
        elif text := _string(holder.get("reasoning_content")):
            turn.reasoning(step=key, text=text, time=when)

    @classmethod
    def from_atif(cls, document: dict[str, Any]) -> Native:
        """Each step as the SDK's events: the user's and the harness's messages, a step's calls as actions sharing its
        response id with their observations after them, a step with no call as the agent's message; best effort, no
        persisted conversation, which needs the agent's whole spec."""
        log: list[dict[str, Any]] = []
        for step in document.get("steps") or ():
            source, text = step.get("source"), _result_text(step.get("message"))
            if source in ("user", "system"):
                log.append({"kind": "MessageEvent", "source": "user" if source == "user" else "environment",
                            "llm_message": {"role": "user", "content": [{"type": "text", "text": text}]}})
                continue
            response = f"response-{step.get('step_id')}"
            calls = step.get("tool_calls") or ()
            if not calls:
                if text:
                    log.append({"kind": "MessageEvent", "source": "agent", "llm_response_id": response,
                                "llm_message": {"role": "assistant", "content": [{"type": "text", "text": text}]}})
                continue
            for index, call in enumerate(calls):
                log.append({"kind": "ActionEvent", "source": "agent", "id": f"action-{call.get('tool_call_id')}",
                            "llm_response_id": response, "tool_name": call.get("function_name"),
                            "tool_call_id": call.get("tool_call_id"),
                            "thought": [{"type": "text", "text": text}] if index == 0 and text else [],
                            "tool_call": {"id": call.get("tool_call_id"), "name": call.get("function_name"),
                                          "arguments": json.dumps(call.get("arguments") or {}),
                                          "origin": "completion"}})
            for result in (step.get("observation") or {}).get("results") or ():
                status = (result.get("extra") or {}).get("agentenv", {}).get("status")
                log.append({"kind": "ObservationEvent", "source": "environment",
                            "tool_call_id": result.get("source_call_id"),
                            "action_id": f"action-{result.get('source_call_id')}",
                            "observation": {"kind": "MCPToolObservation", "is_error": status == "error",
                                            "content": [{"type": "text", "text": _result_text(result.get("content"))}]}})
        return Native(log=log)


def _arguments(call: Any) -> Any:
    """A call's arguments as the model gave them: the SDK keeps them as a JSON string."""
    arguments = call.get("arguments") if isinstance(call, dict) else None
    if not isinstance(arguments, str):
        return arguments
    try:
        return json.loads(arguments)
    except ValueError:
        return arguments


def _content(items: Any, turn: Any) -> list[Any]:
    """Text and image content as ATIF content: each image the SDK holds as a data URL a media file."""
    parts: list[Any] = []
    for item in items or ():
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
        elif item.get("type") == "image":
            for url in item.get("image_urls") or ():
                image = _image(url)
                parts.append(turn.media(*image) if image is not None else json.dumps({"image_url": url}))
    return parts


def _image(url: Any) -> tuple[bytes, str] | None:
    found = _DATA_URL.fullmatch(url) if isinstance(url, str) else None
    if found is None:
        return None
    try:
        return base64.b64decode(found["data"], validate=True), found["type"]
    except (binascii.Error, ValueError):
        return None


def _status(observation: dict[str, Any]) -> str:
    if observation.get("timeout") is True:
        return "timeout"
    return "error" if observation.get("is_error") is True else "ok"


def _exit_code(value: Any) -> int | None:
    """A command's exit code; the terminal's -1 means it hadn't exited."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value != -1 else None


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
    return ""
