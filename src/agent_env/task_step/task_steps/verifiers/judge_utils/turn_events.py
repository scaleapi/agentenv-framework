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
# An OpenInference LLM span's output text: ``message.content`` (chat completions) or
# ``message.contents.N.message_content.text`` (the Responses API, as GPT models are called).
_OUTPUT_TEXT = re.compile(r"^llm\.output_messages\.(\d+)\.message\.(?:content|contents\.(\d+)\.message_content\.text)$")


@dataclass(frozen=True)
class TurnEvents:
    """``events`` in turn order, each ``{"type": "text", "text"}``, ``{"type": "tool_call", "tool", "id", "input",
    "result", "error"}`` (``result`` absent when none came back; ``internal`` for a tool the harness provides itself), ``{"type": "prompt", "text"}`` (the turn's input,
    when the format records it), ``{"type": "note", "text"}`` (a message the harness put in the turn; a compaction's
    also has ``compaction``, what its span recorded: ``wakes_summarized``, ``wakes_kept``, ``first_kept_wake``,
    ``waited_s``) or ``{"type": "final", "text", "error"}``; for an unknown format, ``{"type": "raw", "event"}`` per
    stored item. ``agent_labels``: its spans say which agent made each, or hold the harness's own compaction or
    messages (``has_agent_labels``)."""

    format: str
    events: list[dict]
    raw_bytes: int
    agent_labels: bool = False


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
        return TurnEvents(fmt, _otel_events(items), size, agent_labels=has_agent_labels(items))
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
# The PSI neutral harness labels its spans in the OpenTelemetry GenAI conventions' names: every span carries
# ``gen_ai.agent.name`` (the agent that made it, "harness" for the harness's own) and ``gen_ai.main_agent.name`` (the
# assistant). A model span's ``gen_ai.output.messages`` lists its parts in order: reasoning, text, tool calls (the
# summary call's text is a ``compaction`` part). The harness's own records are ``harness.*`` spans: the compaction
# (``harness.compaction``, from the start of the save turn to the end of the summary) and each message it put before
# the assistant (``harness.message``, with its ``purpose``).
_AGENT = "gen_ai.agent.name"
_MAIN_AGENT = "gen_ai.main_agent.name"
_OUTPUT_MESSAGES = "gen_ai.output.messages"
_COMPACTION = "harness.compaction"
_MESSAGE = "harness.message"
_SAVE = "pre_compaction_save"
_AFTER = "after_compaction"
_FAILED_WAKE = "failed_wake"
# The step a harness message's purpose stands for, as staging and the judge name it.
_STEP_OF_PURPOSE = {"save_request": _SAVE, "after_summary": _AFTER, "failed_wake": _FAILED_WAKE}


def _span_id(span: dict) -> Optional[str]:
    return (span.get("context") or {}).get("span_id") if isinstance(span.get("context"), dict) else span.get("span_id")


def _is_tool(span: dict) -> bool:
    return is_tool_span(span) or (span.get("attributes") or {}).get("openinference.span.kind") == "TOOL"


def _delegating_call(span: dict, by_id: dict) -> Optional[dict]:
    """The tool call ``span`` is nested under (a helper's spans are, under the call that started it), if any."""
    node, seen = by_id.get(span.get("parent_id")), set()
    while isinstance(node, dict) and id(node) not in seen:
        seen.add(id(node))
        if _is_tool(node):
            return node
        node = by_id.get(node.get("parent_id"))
    return None


def _otel_events(spans: list[dict]) -> list[dict]:
    """Tool spans (``execute_tool``) and the model's text, in start order; the conversation every model span repeats
    (``llm.input_messages.*``), tool lists and token counts are left out. A helper's events carry ``subagent`` and
    those of a step the harness asked for carry ``step`` (``_steps``). A model span of an agent other than the main
    one, not nested under a tool call, is the harness's own (the compaction summary): its text and reasoning are not
    the assistant's, so they are left out. The compaction (at its end, once the summary is written) and each message
    the harness sent are notes."""
    by_id = {_span_id(s): s for s in spans if _span_id(s)}
    steps = _steps(spans, by_id)
    timed: list[tuple[str, int, dict]] = []
    for i, span in enumerate(spans):
        start = str(span.get("start_time") or "")
        attrs = span.get("attributes") or {}
        tag = _agent_tag(span, attrs, by_id)
        if i in steps:
            tag["step"] = steps[i]
        if is_tool_span(span):
            tool, tool_input, output = tool_span_io(span)
            # Harnesses that name every tool span generically ("execute_tool") carry the real name in the prompt.
            prompt = parse_json_attr(attrs.get("gen_ai.prompt", "{}"))
            call = _call(str((prompt.get("tool") if isinstance(prompt, dict) else None) or tool), None, tool_input)
            _settle(call, output, (span.get("status") or {}).get("status_code") == "ERROR")
            if attrs.get("harness.tool") is True:   # a tool the harness provides itself (its notes, past wakes)
                call["internal"] = True
            timed.append((start, i, {**call, **tag}))
            continue
        if span.get("name") == _COMPACTION:
            # A compaction with no end time goes after the save turn before it.
            save_end = max((str(s.get("end_time") or s.get("start_time") or "") for j, s in enumerate(spans)
                            if steps.get(j) == _SAVE and s.get("name") != _MESSAGE), default="")
            facts = _compaction_facts(attrs)
            timed.append((str(span.get("end_time") or max(start, save_end)), i,
                          {"type": "note", "text": _compaction_note(attrs, facts), "compaction": facts}))
            continue
        if span.get("name") == _MESSAGE:
            timed.append((start, i, {"type": "note", "text": _harness_message_note(attrs), **tag}))
            continue
        if _is_harness_own(span, attrs, by_id):
            continue
        thinking = _span_thinking(attrs)
        if thinking:
            timed.append((start, i, {"type": "thinking", "text": thinking, **tag}))
        text = _span_text(attrs)
        if text:
            timed.append((start, i, {"type": "text", "text": text, **tag}))
    timed.sort(key=lambda t: (t[0], t[1]))
    events = [e for _, _, e in timed]
    events += [{"type": "final", "text": e["text"], "error": False} for e in extract_final_response(spans)]
    return events


def has_agent_labels(spans: list[dict]) -> bool:
    """Whether a trajectory's spans say which agent made each (``gen_ai.agent.name``) or hold the harness's own
    compaction or messages: the PSI neutral harness's from its fifth version."""
    return any(isinstance(s, dict) and (_AGENT in (s.get("attributes") or {}) or s.get("name") in (_COMPACTION, _MESSAGE))
               for s in spans)


def _is_harness_own(span: dict, attrs: dict, by_id: dict) -> bool:
    """A span of an agent other than the main one that no tool call started: the harness's own (its summary call)."""
    agent, main = attrs.get(_AGENT), attrs.get(_MAIN_AGENT)
    return bool(agent and main and agent != main and _delegating_call(span, by_id) is None)


def _agent_tag(span: dict, attrs: dict, by_id: dict) -> dict:
    """``subagent`` for a helper's event: a span of an agent other than the main one, nested under the tool call that
    started it, named by its agent's id or name and that call's span id. Nothing for a span with no agent labels."""
    agent, main = attrs.get(_AGENT), attrs.get(_MAIN_AGENT)
    if not (agent and main and agent != main):
        return {}
    call = _delegating_call(span, by_id)
    if call is None:
        return {}
    return {"subagent": f"{attrs.get('gen_ai.agent.id') or agent}@{_span_id(call)}"}


def _steps(spans: list[dict], by_id: dict) -> dict[int, str]:
    """The step (by span index) each harness message and each span of a step it asked for is in: a message's own
    (from its ``purpose``), and the save turn's, the main agent's spans that start between the save request and the
    end of the compaction it came before (or, without that end, the message after the summary or the summary call)."""
    steps: dict[int, str] = {}
    order = sorted(range(len(spans)), key=lambda i: str(spans[i].get("start_time") or ""))
    for i in order:
        span = spans[i]
        if span.get("name") != _MESSAGE:
            continue
        step = _STEP_OF_PURPOSE.get(str((span.get("attributes") or {}).get("harness.message.purpose") or ""))
        if step:
            steps[i] = step
        if step != _SAVE:
            continue
        begin = str(span.get("start_time") or "")
        later = [spans[j] for j in order if str(spans[j].get("start_time") or "") >= begin and j != i]
        end = next((str(s.get("end_time") or "") for s in later if s.get("name") == _COMPACTION and s.get("end_time")),
                   None) or next((str(s.get("start_time") or "") for s in later
                                  if (s.get("name") == _MESSAGE and (s.get("attributes") or {}).get(
                                      "harness.message.purpose") == "after_summary")
                                  or _is_harness_own(s, s.get("attributes") or {}, by_id)), "")
        for j in order:
            s, attrs = spans[j], spans[j].get("attributes") or {}
            if (begin <= str(s.get("start_time") or "") < end and attrs.get(_AGENT) and attrs.get(_AGENT) == attrs.get(
                    _MAIN_AGENT)):
                steps[j] = _SAVE
    return steps


def _parts(attrs: dict, kind: str) -> list[str]:
    """The ``kind`` parts' contents of a span's ``gen_ai.output.messages``, in order."""
    messages = parse_json_attr(attrs.get(_OUTPUT_MESSAGES, "[]"))
    if not isinstance(messages, list):  # null, a number or an object: no parts, and the wake still reads
        return []
    return [str(p.get("content") or "") for m in messages if isinstance(m, dict)
            for p in m.get("parts") or [] if isinstance(p, dict) and p.get("type") == kind]


def _compaction_facts(attrs: dict) -> dict:
    """What a ``harness.compaction`` span recorded: the wakes it summarized and kept (the harness's wake numbers, from
    the first kept one), how late (real seconds) the wake it came before started (``waited_s``, the span's
    ``next_wake_waited_s``), and a failed save turn's or compaction's error."""
    named = {"wakes_summarized": "wakes_summarized", "wakes_kept": "wakes_kept", "first_kept_wake": "first_kept_wake",
             "waited_s": "next_wake_waited_s", "save_error": "save_error", "error": "error"}
    return {key: attrs[f"{_COMPACTION}.{attr}"] for key, attr in named.items()
            if attrs.get(f"{_COMPACTION}.{attr}") is not None}


def _wakes(first: int, last: int) -> str:
    return f"wake {first}" if first == last else f"wakes {first}–{last}"


def compaction_words(facts: dict) -> list[str]:
    """A compaction's ``facts`` (a note's ``compaction``) in words: which wakes are now in the summary and which were
    kept word for word (by number when the span named the first kept wake, else as counts), how late this wake
    started waiting for it, and what failed."""
    words = []
    first, kept = facts.get("first_kept_wake"), facts.get("wakes_kept")
    if facts.get("error") is not None:
        words.append(f"it failed, so the history was left whole: {facts['error']}")
    elif isinstance(first, int) and first > 1:
        words.append(f"{_wakes(1, first - 1)} {'is' if first == 2 else 'are'} now in the summary")
        if isinstance(kept, int) and kept > 0:
            words.append(f"{_wakes(first, first + kept - 1)} kept word for word")
    else:
        if facts.get("wakes_summarized") is not None:
            words.append(f"{facts['wakes_summarized']} wakes summarized")
        if kept is not None:
            words.append(f"the last {kept} kept word for word")
    if facts.get("save_error") is not None:
        words.append(f"the save turn before it failed: {facts['save_error']}")
    if facts.get("waited_s") is not None:
        words.append(f"this wake started {facts['waited_s']} s late (real time) waiting for it")
    return words


def _compaction_note(attrs: dict, facts: dict) -> str:
    """A ``harness.compaction`` span as a note: which wakes it summarized, how late it made this wake, and the
    summary (its ``compaction`` part)."""
    head = ("Compaction summary (written by the harness from the model's own summary turn; it replaces the older "
            "history the model sees" + "".join(f"; {w}" for w in compaction_words(facts)) + ")")
    summary = "\n".join(_parts(attrs, "compaction")).strip()
    return f"{head}:\n{summary}" if summary else f"{head}: no summary recorded"


def _harness_message_note(attrs: dict) -> str:
    """A ``harness.message`` span as a note: the text the harness put before the model, labelled by its purpose. The
    message after a compaction repeats the user's first request word for word: it is the harness's, not a new one from
    the user."""
    step = _STEP_OF_PURPOSE.get(str(attrs.get("harness.message.purpose") or ""))
    who = "re-sent by the harness, it repeats the user's earlier words" if step == _AFTER else "not from the user"
    return (f"Message the harness sent the assistant ({f'step {step}; ' if step else ''}{who}):\n"
            + str(attrs.get("harness.message.text") or ""))


def _span_text(attrs: dict) -> str:
    """A model span's own output text: OpenInference's ``llm.output_messages.N.message.content``, a GenAI ``chat``
    span's text blocks, or the text parts of its ``gen_ai.output.messages``."""
    numbered = ((_OUTPUT_TEXT.match(k), v) for k, v in attrs.items())
    texts = [str(v) for _, v in sorted(((int(m.group(1)), int(m.group(2) or 0)), v) for m, v in numbered if m and v)]
    if not texts and attrs.get("gen_ai.operation.name") == "chat":
        completion = parse_json_attr(attrs.get("gen_ai.completion", ""))
        content = completion.get("content") if isinstance(completion, dict) else None
        texts = [b.get("text", "") for b in content or [] if isinstance(b, dict) and b.get("type") == "text"]
    if not texts:
        texts = _parts(attrs, "text")
    return "\n".join(t for t in texts if t.strip())


def _span_thinking(attrs: dict) -> str:
    """A model span's reasoning that has text: the reasoning parts of its ``gen_ai.output.messages``, in order, or a
    GenAI ``chat`` span's ``thinking`` completion blocks (Anthropic's shape). A model that leaves its thinking out (an
    empty block, a signature only) has none."""
    reasoning = [r for r in _parts(attrs, "reasoning") if r.strip()]
    if reasoning:
        return "\n".join(reasoning)
    completion = (parse_json_attr(attrs.get("gen_ai.completion", ""))
                  if attrs.get("gen_ai.operation.name") == "chat" else None)
    blocks = completion.get("content") if isinstance(completion, dict) else None
    return "\n".join(b["thinking"] for b in blocks or []
                     if isinstance(b, dict) and b.get("type") == "thinking" and str(b.get("thinking") or "").strip())


# ---- Claude Code stream-json -----------------------------------------------------------------------------------------
def _claude_code_events(items: list[dict]) -> list[dict]:
    """Assistant text, its thinking (when the CLI recorded any; a block with only a signature has none) and tool calls
    paired with their results by ``tool_use_id``; the init event's tool list, thinking signatures, token counts and ``tool_use_result`` (a copy of the result) are left out. A subagent's events
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
                elif block.get("type") == "thinking" and (block.get("thinking") or "").strip():
                    events.append({"type": "thinking", "text": block["thinking"], **tag})
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
