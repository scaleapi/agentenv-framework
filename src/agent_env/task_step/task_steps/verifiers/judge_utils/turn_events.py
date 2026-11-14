"""One agent turn's trajectory as what happened in it: the agent's text and each tool call with its input and result,
from whichever format the agent stored it in (OTel spans, OpenInference's included; Claude Code's or Gemini's
stream-json). What each format repeats on every turn or call (the tool list, the conversation so far, signatures, token
counts) is dropped, and results are made readable: JSON envelopes unwrapped, email bodies decoded, images left out."""

from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from agent_env.task_step.task_steps.verifiers.judge_utils.trajectory_filter import (
    extract_final_response,
    is_tool_span,
    parse_json_attr,
    strip_base64_images,
    tool_span_io,
)

OTEL = "otel"
CLAUDE_CODE = "claude-code-stream-json"
GEMINI = "gemini-stream-json"
UNKNOWN = "unknown"
IMAGE_OMITTED = "<image omitted>"
# How many JSON-in-a-string layers a result is unwrapped through (an MCP result is often two deep).
_UNWRAP_DEPTH = 4
_OUTPUT_TEXT = re.compile(r"^llm\.output_messages\.(\d+)\.message\.content$")


@dataclass(frozen=True)
class TurnEvents:
    """``events`` in turn order, each ``{"type": "text", "text"}``, ``{"type": "tool_call", "tool", "id", "input",
    "result", "error"}`` (``result`` absent when none came back), ``{"type": "prompt", "text"}`` (the turn's input,
    when the format records it), ``{"type": "note", "text"}`` (a message the harness put in the turn) or
    ``{"type": "final", "text", "error"}``; for an unknown format, ``{"type": "raw", "event"}`` per stored item."""

    format: str
    events: list[dict]
    raw_bytes: int


def turn_events(raw: bytes | str) -> TurnEvents:
    """The events of one stored turn trajectory (a JSON list, as agent-env stores them)."""
    size = len(raw) if isinstance(raw, bytes) else len(raw.encode())
    try:
        items = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return TurnEvents(UNKNOWN, [{"type": "raw", "event": raw if isinstance(raw, str) else raw.decode(
            "utf-8", errors="replace")}], size)
    fmt = trajectory_format(items)
    if fmt == OTEL:
        return TurnEvents(fmt, _otel_events(items), size)
    if fmt == CLAUDE_CODE:
        return TurnEvents(fmt, _claude_code_events(items), size)
    if fmt == GEMINI:
        return TurnEvents(fmt, _gemini_events(items), size)
    return TurnEvents(UNKNOWN, [{"type": "raw", "event": e} for e in (items if isinstance(items, list) else [items])],
                      size)


def trajectory_format(items: Any) -> str:
    """Which format a stored trajectory is in, from its items' shape."""
    if not isinstance(items, list) or not items or not all(isinstance(e, dict) for e in items):
        return UNKNOWN
    if all(isinstance(e.get("attributes"), dict) for e in items):
        return OTEL
    types = {e.get("type") for e in items}
    if types & {"assistant", "user", "system"} and any(isinstance(e.get("message"), dict) for e in items):
        return CLAUDE_CODE
    if types & {"tool_use", "tool_result", "message", "init"}:
        return GEMINI
    return UNKNOWN


def readable_result(value: Any, depth: int = 0) -> Any:
    """A tool result as a judge should read it: a JSON string, an MCP ``{"result": "..."}`` or ``{"type": "text"}``
    envelope and a list of content blocks unwrapped; base64 email bodies (a ``body.data`` under a ``text/*`` part)
    decoded into ``body.text``; images replaced by ``IMAGE_OMITTED``."""
    if isinstance(value, str):
        stripped = value.strip()
        if depth < _UNWRAP_DEPTH and stripped[:1] in ("{", "["):
            parsed = parse_json_attr(stripped)
            if not isinstance(parsed, str):
                return readable_result(parsed, depth + 1)
        return strip_base64_images(value, IMAGE_OMITTED)
    if isinstance(value, list):
        if value and all(isinstance(b, dict) and b.get("type") in ("text", "image") for b in value):
            texts = [(b.get("text") or "") if b.get("type") == "text" else IMAGE_OMITTED for b in value]
            return readable_result(texts[0] if len(texts) == 1 else "\n".join(texts), depth)
        return [_readable_inside(v) for v in value]
    if isinstance(value, dict):
        if set(value) == {"result"} and isinstance(value["result"], str):
            return readable_result(value["result"], depth)
        if value.get("type") == "text" and isinstance(value.get("text"), str) and set(value) <= {
                "type", "text", "annotations"}:
            return readable_result(value["text"], depth)
        if value.get("type") == "image":
            return IMAGE_OMITTED
        return _decoded_body({k: _readable_inside(v) for k, v in value.items()})
    return value


def _readable_inside(value: Any) -> Any:
    """A value inside a result: decoded and image-free like the result, but a string in it stays a string."""
    if isinstance(value, str):
        return strip_base64_images(value, IMAGE_OMITTED)
    if isinstance(value, (dict, list)):
        return readable_result(value, _UNWRAP_DEPTH)
    return value


def _decoded_body(part: dict) -> dict:
    body = part.get("body")
    if not (isinstance(body, dict) and isinstance(body.get("data"), str)
            and str(part.get("mimeType") or "text/").startswith("text/")):
        return part
    data = body["data"]
    try:
        text = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8")
    except (binascii.Error, ValueError):
        return part
    return {**part, "body": {**{k: v for k, v in body.items() if k != "data"}, "text": text}}


def _call(tool: str, call_id: Optional[str], tool_input: Any) -> dict:
    return {"type": "tool_call", "tool": tool, "id": call_id, "input": tool_input, "error": False}


def _settle(call: dict, result: Any, error: bool) -> None:
    call["result"] = readable_result(result)
    call["error"] = bool(error) or (isinstance(call["result"], dict) and bool(call["result"].get("error")))


# ---- OTel ------------------------------------------------------------------------------------------------------------
def _otel_events(spans: list[dict]) -> list[dict]:
    """Tool spans (``execute_tool``) and the model's text, in start order; the conversation every model span repeats
    (``llm.input_messages.*``), tool lists and token counts are left out."""
    timed: list[tuple[str, int, dict]] = []
    for i, span in enumerate(spans):
        start = str(span.get("start_time") or "")
        if is_tool_span(span):
            tool, tool_input, output = tool_span_io(span)
            # Harnesses that name every tool span generically ("execute_tool") carry the real name in the prompt.
            prompt = parse_json_attr((span.get("attributes") or {}).get("gen_ai.prompt", "{}"))
            call = _call(str((prompt.get("tool") if isinstance(prompt, dict) else None) or tool), None, tool_input)
            _settle(call, output, (span.get("status") or {}).get("status_code") == "ERROR")
            timed.append((start, i, call))
            continue
        text = _span_text(span.get("attributes") or {})
        if text:
            timed.append((start, i, {"type": "text", "text": text}))
    timed.sort(key=lambda t: (t[0], t[1]))
    events = [e for _, _, e in timed]
    events += [{"type": "final", "text": e["text"], "error": False} for e in extract_final_response(spans)]
    return events


def _span_text(attrs: dict) -> str:
    """A model span's own output text: OpenInference's ``llm.output_messages.N.message.content`` or a GenAI ``chat``
    span's text blocks."""
    numbered = ((_OUTPUT_TEXT.match(k), v) for k, v in attrs.items())
    texts = [str(v) for _, v in sorted((int(m.group(1)), v) for m, v in numbered if m and v)]
    if not texts and attrs.get("gen_ai.operation.name") == "chat":
        completion = parse_json_attr(attrs.get("gen_ai.completion", ""))
        content = completion.get("content") if isinstance(completion, dict) else None
        texts = [b.get("text", "") for b in content or [] if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(t for t in texts if t.strip())


# ---- Claude Code stream-json -----------------------------------------------------------------------------------------
def _claude_code_events(items: list[dict]) -> list[dict]:
    """Assistant text and tool calls paired with their results by ``tool_use_id``; the init event's tool list,
    thinking signatures, token counts and ``tool_use_result`` (a copy of the result) are left out. A subagent's events
    carry ``subagent`` (the tool call that started it)."""
    events: list[dict] = []
    calls: dict[str, dict] = {}
    for item in items:
        kind = item.get("type")
        message = item.get("message") if isinstance(item.get("message"), dict) else {}
        parent = item.get("parent_tool_use_id")
        tag = {"subagent": parent} if parent else {}
        if kind == "assistant":
            for block in message.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and (block.get("text") or "").strip():
                    events.append({"type": "text", "text": block["text"], **tag})
                elif block.get("type") == "tool_use":
                    call = {**_call(block.get("name", ""), block.get("id"), block.get("input")), **tag}
                    calls[block.get("id")] = call
                    events.append(call)
        elif kind == "user":
            content = message.get("content")
            if isinstance(content, str):
                events.append({"type": "note", "text": content, **tag})
                continue
            for block in content or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result" and block.get("tool_use_id") in calls:
                    _settle(calls[block["tool_use_id"]], block.get("content"), block.get("is_error"))
                elif block.get("type") == "text" and (block.get("text") or "").strip():
                    events.append({"type": "note", "text": block["text"], **tag})
        elif kind == "result":
            events.append({"type": "final", "text": item.get("result"), "error": bool(item.get("is_error"))})
    return events


# ---- Gemini stream-json ----------------------------------------------------------------------------------------------
def _gemini_events(items: list[dict]) -> list[dict]:
    """The prompt, assistant text (its ``delta`` chunks joined) and tool calls paired with their results by
    ``tool_id``; the init event and stats are left out."""
    events: list[dict] = []
    calls: dict[str, dict] = {}
    for item in items:
        kind = item.get("type")
        if kind == "message":
            text = item.get("content") if isinstance(item.get("content"), str) else ""
            if item.get("role") == "user":
                events.append({"type": "prompt", "text": text})
            elif item.get("delta") and events and events[-1].get("type") == "text" and events[-1].get("open"):
                events[-1]["text"] += text
            elif text:
                events.append({"type": "text", "text": text, "open": bool(item.get("delta"))})
        elif kind == "tool_use":
            call = _call(item.get("tool_name", ""), item.get("tool_id"), item.get("parameters"))
            calls[item.get("tool_id")] = call
            events.append(call)
        elif kind == "tool_result" and item.get("tool_id") in calls:
            _settle(calls[item["tool_id"]], item.get("output", item.get("error")), item.get("status") == "error")
        elif kind == "result":
            error = item.get("error")
            text = error.get("message") if isinstance(error, dict) else error
            events.append({"type": "final", "text": text if isinstance(text, str) else None,
                           "error": item.get("status") not in (None, "success")})
    for e in events:
        e.pop("open", None)
    return [e for e in events if e.get("type") != "text" or e["text"].strip()]
