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


def test_a_tool_span_named_generically_takes_its_tool_name_from_the_prompt():
    spans = [{"name": "execute_tool", "start_time": "1", "attributes": {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.prompt": json.dumps({"tool": "gmail_send_message", "input": {"to": "nina@example.com"}}),
        "gen_ai.completion": json.dumps({"output": {"id": "m1"}})}}]

    calls = _calls(turn_events(json.dumps(spans)))

    assert [(c["tool"], c["input"], c["result"]) for c in calls] == [
        ("gmail_send_message", {"to": "nina@example.com"}, {"id": "m1"})]


def test_a_failed_gemini_turn_keeps_its_error_message():
    items = [
        {"type": "message", "role": "user", "content": "book it"},
        {"type": "result", "status": "error", "error": {"type": "unknown", "message": "Quota exceeded for model"}},
    ]
    events = turn_events(json.dumps(items))

    assert events.format == GEMINI
    assert events.events[-1] == {"type": "final", "text": "Quota exceeded for model", "error": True}


def test_the_agent_s_thinking_is_kept_when_it_was_recorded_and_an_empty_block_is_dropped():
    claude_code = [{"type": "assistant", "message": {"content": [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "thinking", "thinking": "Nina said no shellfish, so check the menu first.", "signature": "sig"},
        {"type": "text", "text": "Checking the menu."}]}}]
    otel = [{"name": "chat", "start_time": "1", "attributes": {
        "gen_ai.operation.name": "chat",
        "gen_ai.completion": json.dumps({"content": [{"type": "thinking", "thinking": "Ask before paying $855."},
                                                     {"type": "text", "text": "Asking the user."}]})}}]

    assert [(e["type"], e["text"]) for e in turn_events(json.dumps(claude_code)).events] == [
        ("thinking", "Nina said no shellfish, so check the menu first."), ("text", "Checking the menu.")]
    assert [(e["type"], e["text"]) for e in turn_events(json.dumps(otel)).events if e["type"] != "final"] == [
        ("thinking", "Ask before paying $855."), ("text", "Asking the user.")]


def test_the_psi_harness_s_reasoning_on_an_llm_span_comes_before_that_call_s_text():
    spans = [{"name": "ChatCompletion", "start_time": "1", "attributes": {
        "openinference.span.kind": "LLM",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": "I'll ask Nina first.",
        "psi.reasoning": json.dumps([{"type": "thinking", "thinking": ""},
                                     {"type": "thinking", "thinking": "Paying $855 needs her OK."}])}}]

    events = [(e["type"], e["text"]) for e in turn_events(json.dumps(spans)).events if e["type"] != "final"]

    assert events == [("thinking", "Paying $855 needs her OK."), ("text", "I'll ask Nina first.")]


def test_a_responses_api_llm_span_s_text_and_reasoning_are_read():
    spans = [{"name": "Response", "start_time": "1", "attributes": {
        "openinference.span.kind": "LLM",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.contents.0.message_content.type": "text",
        "llm.output_messages.0.message.contents.0.message_content.text": "Booked for Friday.",
        "psi.reasoning": json.dumps([{"type": "thinking", "thinking": "She said Friday works."}])}}]

    events = [(e["type"], e["text"]) for e in turn_events(json.dumps(spans)).events if e["type"] != "final"]

    assert events == [("thinking", "She said Friday works."), ("text", "Booked for Friday.")]


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


def _v3(name: str) -> list[dict]:
    """A recorded wake of the PSI neutral harness v3 (redacted from a real smoke run): it starts with a compaction."""
    return json.loads((RECORDED / f"turn_otel_v3_compaction_{name}.json").read_text())


def test_a_v3_compaction_wake_keeps_the_assistant_s_words_and_makes_the_harness_s_parts_notes():
    spans = _v3("claude")
    summary = next(s["attributes"]["psi.compaction.summary"] for s in spans if s["name"] == "psi.compaction")

    events = turn_events(json.dumps(spans)).events

    assert [(e["type"], e.get("step")) for e in events] == [
        ("note", "pre_compaction_save"), ("tool_call", "pre_compaction_save"), ("tool_call", "pre_compaction_save"),
        ("text", "pre_compaction_save"), ("note", None), ("note", "after_compaction"), ("thinking", None),
        ("text", None)]
    save, compaction, after = (e["text"] for e in events if e["type"] == "note")
    assert save.startswith("Message the harness sent the assistant (step pre_compaction_save; not from the user):\n"
                           "[Harness: older history is about to be summarized")
    assert compaction.startswith("Compaction summary (written by the harness from the model's own summary turn;")
    assert "2 wakes summarized; the last 3 kept word for word; the next wake started 8.694 s late" in compaction
    assert compaction.endswith(summary.strip()) and after.startswith("Message the harness sent the assistant (step "
                                                                      "after_compaction; not from the user):")
    # The summary turn is the harness's model call: neither its text nor its reasoning is the assistant's.
    own = [e["text"] for e in events if e["type"] in ("text", "thinking")]
    assert not any(summary[:200] in t or "checking the schedule" in t for t in own)
    assert own[-1].startswith("Got today's digest, and nothing in it needs action.")
    assert [c["tool"] for c in _calls(turn_events(json.dumps(spans)))] == ["memory_add", "memory_add"]


def test_a_v3_gpt_save_turn_keeps_its_reasoning_and_memory_calls_marked_as_the_harness_s_step():
    events = turn_events(json.dumps(_v3("gpt"))).events

    save = [e for e in events if e.get("step") == "pre_compaction_save" and e["type"] != "note"]
    assert [(e["type"], e.get("tool")) for e in save] == [
        ("thinking", None), ("tool_call", "memory_view"), ("tool_call", "memory_replace"), ("text", None)]
    assert save[1]["result"].startswith("1. HOA approval")
    assert [e["text"] for e in events if e["type"] == "text"] == [
        "Saved the key details and pending items to my profile.", "Received—no action taken on the inbox digest."]


def test_a_helper_agent_s_spans_carry_its_id():
    spans = [s for s in _v3("claude") if s["attributes"].get("psi.turn") == "pre_compaction_save"
             and s["attributes"].get("openinference.span.kind") in ("LLM", "TOOL")]
    for s in spans:
        s["attributes"].pop("psi.turn")
        s["attributes"].update({"psi.agent.kind": "subagent", "psi.agent.id": "sub-002",
                                "psi.agent.delegate_span_id": "0x1", "gen_ai.agent.name": "helper"})

    events = turn_events(json.dumps(spans)).events

    assert [(e["type"], e.get("subagent")) for e in events] == [
        ("tool_call", "sub-002"), ("tool_call", "sub-002"), ("text", "sub-002")]


def test_spans_without_psi_agent_labels_read_as_before():
    events = turn_events((RECORDED / "turn_otel_wake4.json").read_bytes()).events

    assert not any("step" in e or "subagent" in e for e in events)
    assert "note" not in {e["type"] for e in events}
