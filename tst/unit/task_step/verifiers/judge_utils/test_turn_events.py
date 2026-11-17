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


def test_an_llm_span_s_reasoning_parts_come_in_order_before_that_call_s_text_and_an_empty_one_is_dropped():
    spans = [{"name": "ChatCompletion", "start_time": "1", "attributes": {
        "openinference.span.kind": "LLM",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": "I'll ask Nina first.",
        "gen_ai.output.messages": json.dumps([{"role": "assistant", "finish_reason": "stop", "parts": [
            {"type": "reasoning", "content": ""}, {"type": "reasoning", "content": "Paying $855 needs her OK."},
            {"type": "reasoning", "content": "So ask."}, {"type": "text", "content": "I'll ask Nina first."}]}])}}]

    events = [(e["type"], e["text"]) for e in turn_events(json.dumps(spans)).events if e["type"] != "final"]

    assert events == [("thinking", "Paying $855 needs her OK.\nSo ask."), ("text", "I'll ask Nina first.")]


def test_output_messages_that_aren_t_a_list_add_nothing_and_the_span_still_reads():
    for bad in ("null", "7", json.dumps({"role": "assistant"})):
        spans = [{"name": "ChatCompletion", "start_time": "1", "attributes": {
            "openinference.span.kind": "LLM",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "Ordered the pie.",
            "gen_ai.output.messages": bad}}]

        events = [(e["type"], e["text"]) for e in turn_events(json.dumps(spans)).events if e["type"] != "final"]

        assert events == [("text", "Ordered the pie.")], bad


def test_a_responses_api_llm_span_s_text_and_reasoning_are_read():
    spans = [{"name": "Response", "start_time": "1", "attributes": {
        "openinference.span.kind": "LLM",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.contents.0.message_content.type": "text",
        "llm.output_messages.0.message.contents.0.message_content.text": "Booked for Friday.",
        "gen_ai.output.messages": json.dumps([{"role": "assistant", "finish_reason": "stop", "parts": [
            {"type": "reasoning", "content": "She said Friday works."},
            {"type": "text", "content": "Booked for Friday."}]}])}}]

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


def _v5(name: str) -> list[dict]:
    """A recorded wake of the PSI neutral harness v5 (redacted from a fake-model run): it starts with a compaction."""
    return json.loads((RECORDED / f"turn_otel_v5_compaction_{name}.json").read_text())


def _summary(spans: list[dict]) -> str:
    span = next(s for s in spans if s["name"] == "harness.compaction")
    return json.loads(span["attributes"]["gen_ai.output.messages"])[0]["parts"][0]["content"]


def test_a_v5_compaction_wake_keeps_the_assistant_s_words_and_makes_the_harness_s_parts_notes():
    spans = _v5("claude")
    summary = _summary(spans)

    traj = turn_events(json.dumps(spans))
    events = traj.events

    assert traj.agent_labels is True
    assert [(e["type"], e.get("step")) for e in events] == [
        ("note", "pre_compaction_save"), ("thinking", "pre_compaction_save"), ("text", "pre_compaction_save"),
        ("tool_call", "pre_compaction_save"), ("thinking", "pre_compaction_save"), ("text", "pre_compaction_save"),
        ("note", None), ("note", "after_compaction"),
        ("thinking", None), ("text", None), ("tool_call", None), ("thinking", None), ("text", None)]
    save, compaction, after = (e["text"] for e in events if e["type"] == "note")
    assert save.startswith("Message the harness sent the assistant (step pre_compaction_save; not from the user):\n"
                           "[Harness: older history is about to be summarized")
    assert compaction == ("Compaction summary (written by the harness from the model's own summary turn; it replaces "
                          "the older history the model sees; wakes 1–3 are now in the summary; wakes 4–5 kept word for "
                          f"word):\n{summary}")
    assert next(e for e in events if "compaction" in e)["compaction"] == {
        "wakes_summarized": 3, "wakes_kept": 2, "first_kept_wake": 4}
    assert after.startswith("Message the harness sent the assistant (step after_compaction; re-sent by the harness, "
                            "it repeats the user's earlier words):\n[Harness: older history was summarized")
    # The summary call is the harness's (agent "harness", under no tool call): neither its text nor its reasoning is
    # the assistant's, though its OpenInference output still holds the summary as text.
    own = [e["text"] for e in events if e["type"] in ("text", "thinking")]
    assert not any(summary in t or "step 12" in t for t in own)
    assert [(c["tool"], c.get("internal")) for c in _calls(traj)] == [("memory_add", True), ("slow_search", None)]


def test_a_compaction_with_no_end_time_comes_after_its_save_turn_and_its_numbers_and_errors_are_said():
    spans = _v5("claude")
    span = next(s for s in spans if s["name"] == "harness.compaction")
    del span["end_time"], span["attributes"]["harness.compaction.first_kept_wake"]
    span["attributes"].update({"harness.compaction.next_wake_waited_s": 8.694,
                               "harness.compaction.save_error": "APITimeoutError: timed out"})

    events = turn_events(json.dumps(spans)).events

    kinds = [(e["type"], e.get("step")) for e in events]
    assert kinds.index(("note", None)) == len(kinds) - 1 - kinds[::-1].index(("text", "pre_compaction_save")) + 1
    compaction = next(e["text"] for e in events if "compaction" in e)
    assert ("; 3 wakes summarized; the last 2 kept word for word; the save turn before it failed: APITimeoutError: "
            "timed out; this wake started 8.694 s late (real time) waiting for it):") in compaction


def test_a_v5_gpt_save_turn_keeps_its_reasoning_and_calls_marked_as_the_harness_s_step():
    events = turn_events(json.dumps(_v5("gpt"))).events

    save = [e for e in events if e.get("step") == "pre_compaction_save" and e["type"] != "note"]
    assert [(e["type"], e.get("tool"), e.get("text")) for e in save] == [
        ("thinking", None, "GPT reasoning 10."), ("tool_call", "memory_add", None),
        ("thinking", None, "GPT reasoning 11."), ("text", None, "Done (11).")]
    assert [e["text"] for e in events if e["type"] == "text"] == ["Done (11).", "Done (14)."]


def test_a_helper_agent_s_spans_are_named_by_its_agent_and_the_call_that_started_it():
    spans = _v5("claude")
    delegate = next(s for s in spans if s["name"] == "slow_search")
    helper = [s for s in spans if s["name"] == "generation" and s["start_time"] > delegate["start_time"]]
    for s in helper:
        s["parent_id"] = delegate["context"]["span_id"]
        s["attributes"]["gen_ai.agent.name"] = "researcher"

    events = turn_events(json.dumps(spans)).events

    assert [(e["type"], e.get("subagent")) for e in events if e.get("subagent")] == [
        ("thinking", f"researcher@{delegate['context']['span_id']}"), ("text", f"researcher@{delegate['context']['span_id']}")]


def test_a_failed_wake_s_message_is_a_note_in_its_step():
    span = {"name": "harness.message", "start_time": "1", "attributes": {
        "gen_ai.agent.name": "harness", "gen_ai.main_agent.name": "Orchestrator",
        "harness.message.purpose": "failed_wake",
        "harness.message.text": "[Harness: your last wake ended with an error before it finished (TimeoutError).]"}}

    [note] = turn_events(json.dumps([span])).events

    assert note == {"type": "note", "step": "failed_wake", "text": (
        "Message the harness sent the assistant (step failed_wake; not from the user):\n"
        "[Harness: your last wake ended with an error before it finished (TimeoutError).]")}


def test_spans_without_agent_labels_read_as_before():
    events = turn_events((RECORDED / "turn_otel_wake4.json").read_bytes()).events

    assert not any("step" in e or "subagent" in e for e in events)
    assert "note" not in {e["type"] for e in events}
    assert turn_events((RECORDED / "turn_otel_wake4.json").read_bytes()).agent_labels is False
