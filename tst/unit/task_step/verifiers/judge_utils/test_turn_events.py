"""turn_events: one stored turn as what happened in it, from recorded OpenInference spans and Claude Code stream-json
(``tst/unit/task_step/verifiers/recorded``), and from Gemini stream-json; what each format repeats is dropped."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from agent_env.task_step.task_steps.verifiers.judge_utils import trajectory_filter
from agent_env.task_step.task_steps.verifiers.judge_utils.turn_events import (
    CLAUDE_CODE,
    GEMINI,
    IMAGE_OMITTED,
    OTEL,
    UNKNOWN,
    readable_result,
    turn_events,
)

RECORDED = Path(__file__).resolve().parents[1] / "recorded"


def _calls(events) -> list[dict]:
    return [e for e in events.events if e["type"] == "tool_call"]


def test_openinference_spans_keep_the_calls_and_the_model_s_text_and_drop_what_every_span_repeats():
    raw = (RECORDED / "turn_otel_wake4.json").read_bytes()
    events = turn_events(raw)

    assert events.format == OTEL and events.raw_bytes == len(raw)
    calls = _calls(events)
    assert [c["tool"] for c in calls] == ["porchbite_get_order", "porchbite_list_conversations",
                                          "porchbite_spending_by_month", "stay_quiet"]
    assert calls[0]["input"] == {"order_key": "PB-C2SHSLHX"} and calls[0]["result"]["order"]["order_key"] == "PB-C2SHSLHX"
    texts = [e["text"] for e in events.events if e["type"] == "text"]
    assert len(texts) == 2 and texts[-1].startswith("Nothing has changed since Nina")
    staged = json.dumps(events.events)
    for repeated in ("llm.input_messages", "gmail_archive_thread", "harness.usage"):
        assert repeated in raw.decode() and repeated not in staged
    assert len(staged) < len(raw) / 4


def test_claude_code_stream_json_pairs_results_by_id_decodes_email_and_drops_the_tool_list():
    raw = (RECORDED / "turn_claude_code.json").read_text()
    events = turn_events(raw)

    assert events.format == CLAUDE_CODE
    calls = _calls(events)
    thread = next(c for c in calls if c["tool"].endswith("gmail_get_thread"))
    body = thread["result"]["messages"][0]["payload"]["body"]
    assert body["text"].startswith("Hi Northwick,\n\nPlease send 4 x 50 lb boxes") and "data" not in body
    assert all("result" in c for c in calls) and not any(c["error"] for c in calls)
    assert events.events[-1]["type"] == "final" and events.events[-1]["text"]
    staged = json.dumps(events.events)
    assert "signature" not in staged and "cache_read_input_tokens" not in staged and '"init"' not in staged
    assert len(staged) < len(raw) / 2


def test_gemini_stream_json_joins_deltas_and_pairs_results_by_tool_id():
    items = [
        {"type": "init", "session_id": "s", "model": "m"},
        {"type": "message", "role": "user", "content": "book it"},
        {"type": "message", "role": "assistant", "content": "Checking ", "delta": True},
        {"type": "message", "role": "assistant", "content": "the calendar.", "delta": True},
        {"type": "tool_use", "tool_name": "gcal_list_events", "tool_id": "t1", "parameters": {"day": "mon"}},
        {"type": "tool_result", "tool_id": "t1", "status": "error", "output": "calendar unavailable"},
        {"type": "result", "status": "success", "stats": {"tool_calls": 1}},
    ]
    events = turn_events(json.dumps(items))

    assert events.format == GEMINI
    assert [e["type"] for e in events.events] == ["prompt", "text", "tool_call", "final"]
    assert events.events[1]["text"] == "Checking the calendar."
    assert events.events[2] == {"type": "tool_call", "tool": "gcal_list_events", "id": "t1", "input": {"day": "mon"},
                                "result": "calendar unavailable", "error": True}


def test_an_unknown_format_passes_through_one_item_an_event():
    events = turn_events(json.dumps([{"kind": "step", "n": 1}, {"kind": "step", "n": 2}]))
    assert events.format == UNKNOWN and [e["event"]["n"] for e in events.events] == [1, 2]


def test_readable_result_unwraps_envelopes_decodes_text_parts_and_leaves_images_out():
    html = base64.urlsafe_b64encode("<p>Bonjour</p>".encode()).decode().rstrip("=")
    message = {"payload": {"mimeType": "multipart/alternative", "parts": [
        {"mimeType": "text/html", "body": {"size": 14, "data": html}},
        {"mimeType": "image/png", "filename": "a.png", "body": {"attachmentId": "a1"}}]}}
    wrapped = {"result": json.dumps(message)}
    part = readable_result(wrapped)["payload"]["parts"][0]
    assert part["body"] == {"size": 14, "text": "<p>Bonjour</p>"}
    png = "iVBORw0KGgo" + "A" * 300
    assert readable_result([{"type": "text", "text": "page"}, {"type": "image", "source": {"data": png}}]) == (
        f"page\n{IMAGE_OMITTED}")
    assert readable_result({"note": f"see data:image/png;base64,{png}"}) == {"note": f"see {IMAGE_OMITTED}"}
    assert readable_result({"body": {"data": "not base64 at all!"}}) == {"body": {"data": "not base64 at all!"}}


def test_the_compactor_rubrics_verifier_uses_still_reads_tool_spans_the_same_way():
    span = {"name": "email_send", "attributes": {"gen_ai.operation.name": "execute_tool",
                                                  "gen_ai.prompt": json.dumps({"input": {"to": "a@b"}}),
                                                  "gen_ai.completion": json.dumps({"output": {"ok": True}})},
            "start_time": "1", "end_time": "2"}
    events, files = trajectory_filter.compact_otel_trajectory([span], trajectory_filter.TrajectoryFilter())
    assert events == [{"type": "tool_call", "tool": "email_send", "input": {"to": "a@b"}, "timestamp": "1"},
                      {"type": "tool_result", "tool": "email_send", "output": {"ok": True}, "timestamp": "2"}]
    assert files == [] and trajectory_filter.tool_span_io(span) == ("email_send", {"to": "a@b"}, {"ok": True})
