"""stage_run: an always_on run as files a judge searches, from a recorded timeline and recorded wake trajectories
(``recorded/``): one file per wake with one event per line, each turn in its wake by the timeline's trajectory_url."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path

import pytest

from agent_env.task_step.task_steps.verifiers.always_on_staging import (
    EVENTS_FILE,
    INDEX_FILE,
    LINE_CHARS,
    TRIGGERS_FILE,
    judge_reads,
    stage_run,
)
from agent_env.task_step.task_steps.verifiers.judge_utils.turn_events import turn_events
from agent_env.task_step.task_steps.verifiers.run_context import Firing

RECORDED = Path(__file__).resolve().parent / "recorded"


def _timeline() -> list[dict]:
    return copy.deepcopy(json.loads((RECORDED / "always_on_timeline.json").read_text())["timeline"])


def _url(timeline: list[dict], wake: int) -> str:
    return next(e["trajectory_url"] for e in timeline if e["event"] == "agent_turn_end" and e["n"] == wake)


def _lines(files: dict, path: str) -> list[dict]:
    return [json.loads(line) for line in files[path].splitlines()]


GEMINI_NUDGE = json.dumps([
    {"type": "message", "role": "user", "content": "Your closing text was not delivered."},
    {"type": "tool_use", "tool_name": "message_user", "tool_id": "n1", "parameters": {"text": "Switch to veggie?"}},
    {"type": "tool_result", "tool_id": "n1", "status": "success", "output": "{\"sent\": true}"},
    {"type": "result", "status": "success"}])


@pytest.fixture
def run():
    """The recorded four wakes: wake 2's turn a Claude Code one and a nudge after it, wake 3's trajectory never
    fetched, wake 1's unreadable, and wake 4 with 250 app changes. ``turns`` is built in an order unlike the wakes',
    so only a turn's url can match it to its wake."""
    timeline = _timeline()
    wake2, wake4 = _url(timeline, 2), _url(timeline, 4)
    for e in timeline:
        if e["event"] == "agent_turn_end" and e["n"] == 3:
            e["trajectory_url"] = None
    at = next(i for i, e in enumerate(timeline) if e["event"] == "message_to_user" and e["wake"] == 2)
    timeline.insert(at + 1, {"event": "nudge", "n": 1, "wake": 2, "virtual_time": "2026-11-06T21:00:00Z",
                             "closing_text": "I asked Nina.", "state": "completed", "outcome": "message_user",
                             "messages_to_user": 1, "plain_reply_after": "", "trajectory_url": "mem://nudge.json"})
    timeline.insert(at + 2, {"event": "message_to_user", "virtual_time": "2026-11-06T21:00:00Z",
                             "text": "Switch to veggie?", "wake": 2, "after_nudge": True})
    changes = next(e for e in timeline if e["event"] == "app_changes" and e["wake"] == 4)
    changes["changes"] = [{**changes["changes"][0], "id": i, "summary": f"Updated: order {i} " + "x" * 300}
                          for i in range(250)]
    turns = {"mem://nudge.json": turn_events(GEMINI_NUDGE),
             wake4: turn_events((RECORDED / "turn_otel_wake4.json").read_bytes()),
             wake2: turn_events((RECORDED / "turn_claude_code.json").read_text())}
    return timeline, turns


def test_each_turn_is_in_its_wake_by_url_and_a_nudge_in_its_wake(run):
    staged = stage_run(*run)
    files = staged.files

    wake2 = _lines(files, "wakes/wake_002.jsonl")
    assert [line["event"] for line in wake2[:2]] == ["woke", "wake_input"]
    tools = [line["tool"] for line in wake2 if line["event"] == "tool_call"]
    assert any(t.endswith("gmail_get_thread") for t in tools)
    nudge_at = next(i for i, line in enumerate(wake2) if line["event"] == "nudge")
    assert wake2[nudge_at + 1]["tool"] == "message_user" and wake2[nudge_at + 1]["turn"] == "nudge"
    assert [line["call"] for line in wake2 if line["event"] == "tool_call"] == list(range(1, len(tools) + 1))
    wake4 = _lines(files, "wakes/wake_004.jsonl")
    assert "porchbite_spending_by_month" in [line.get("tool") for line in wake4]
    assert staged.record["formats"] == {"claude-code-stream-json": 1, "otel": 1, "gemini-stream-json": 1}


def test_a_missing_or_unreadable_trajectory_is_flagged_where_its_calls_would_be(run):
    timeline, turns = run
    staged = stage_run(timeline, turns)

    assert staged.record["missing_trajectories"] == [3] and staged.record["unreadable_trajectories"] == [1]
    wake3 = _lines(staged.files, "wakes/wake_003.jsonl")
    assert wake3[3] == {"event": "trajectory_missing", "wake": 3, "turn": "turn"}
    index = staged.files[INDEX_FILE]
    assert "the turn's trajectory is missing: its tool calls aren't here (wakes/wake_003.jsonl:4)" in index
    assert "Wakes whose tool calls are not here (their trajectory is missing or unreadable): 1, 3." in index


def test_what_woke_the_assistant_comes_first_with_the_user_s_whole_message(run):
    staged = stage_run(*run)

    head = _lines(staged.files, "wakes/wake_003.jsonl")[:3]
    assert [line["event"] for line in head] == ["woke", "user_message", "wake_input"]
    assert head[0]["woke_by"] == ["user_reply"] and head[0]["at"] == "2026-11-06T21:00:00Z"
    assert head[1]["text"].startswith("omg yes good catch, he's doing no-meat november!!")
    assert head[1]["text"].endswith("later is totally fine!")
    assert "the user (Fri Nov 6, 9:00 PM UTC): \"omg yes good catch" in staged.files[INDEX_FILE]


def test_the_agent_s_thinking_is_a_line_in_its_wake(run):
    timeline, turns = run
    wake2 = _url(timeline, 2)
    turns[wake2] = turn_events(json.dumps([{"type": "assistant", "message": {"content": [
        {"type": "thinking", "thinking": "Nina is allergic; ask before ordering.", "signature": "s"},
        {"type": "text", "text": "Asking Nina."}]}}]))

    lines = _lines(stage_run(timeline, turns).files, "wakes/wake_002.jsonl")

    kinds = [line["event"] for line in lines if line["event"] in ("assistant_thinking", "assistant_text")]
    assert kinds == ["assistant_thinking", "assistant_text"]
    assert next(line for line in lines if line["event"] == "assistant_thinking")["text"].startswith("Nina is allergic")


def test_every_app_change_is_kept_with_its_whole_summary(run):
    staged = stage_run(*run)

    changes = [line for line in _lines(staged.files, "wakes/wake_004.jsonl") if line["event"] == "app_change"]
    assert len(changes) == 250 and all(len(c["summary"]) > 300 for c in changes)
    assert "at" not in changes[0] and changes[0]["recorded_at"].startswith("2026-11-13")


def test_long_results_go_to_side_files_and_email_bodies_are_decoded(run):
    staged = stage_run(*run)
    files = staged.files

    thread = next(line for line in _lines(files, "wakes/wake_002.jsonl")
                  if line["event"] == "tool_call" and line["tool"].endswith("gmail_get_thread"))
    assert thread["result_file"].startswith("wakes/results/wake_002_call_") and "result" not in thread
    assert thread["result_file"].endswith("_gmail_get_thread.json") and thread["result_preview"]
    assert "Hi Northwick,\n\nPlease send 4 x 50 lb boxes" in json.loads(files[thread["result_file"]])[
        "messages"][0]["payload"]["body"]["text"]
    assert all(len(line) <= LINE_CHARS for path, text in files.items() if path.startswith("wakes/wake_")
               for line in text.splitlines() if json.loads(line)["event"] == "tool_call")
    assert staged.record["side_files"] >= 1 and staged.record["raw_bytes"] > staged.record["staged_bytes"]


def _changed_by(timeline: list[dict], wake: int, tool: str) -> None:
    """An app change wake ``wake``'s turn made with ``tool``, as the apps' change log records it."""
    event = next(e for e in timeline if e["event"] == "app_changes" and e["wake"] == wake)
    event["changes"].append({**event["changes"][0], "id": 999, "table": "gcal.events", "op": "INSERT", "by": "agent",
                             "actors": [{"kind": "tool_call", "tool": tool, "role": "assistant"}]})


def test_the_index_points_at_the_line_each_action_and_message_is_on(run):
    timeline, turns = run
    _changed_by(timeline, 2, "gcal_create_event")
    staged = stage_run(timeline, turns)
    files = staged.files
    index = files[INDEX_FILE]

    assert index.startswith("# The run, wake by wake\n\n4 wakes, from Fri Nov 6, 8:00 PM UTC to Sat Nov 7, 4:00 AM UTC.")
    assert "Actions below are the assistant's tool calls of a tool that changed the apps' data" in index
    block = index[index.index("## Wake 2 "):index.index("## Wake 3 ")]
    for kind in ("action", "sent the user"):
        ref = next(line for line in block.splitlines() if line.startswith(f"- {kind}")).rsplit("(", 1)[1].rstrip(")")
        path, n = ref.split(":")
        line = _lines(files, path)[int(n) - 1]
        assert line["event"] == ("tool_call" if kind == "action" else "message_to_user")
    actions = [line for line in block.splitlines() if line.startswith("- action")]
    assert len(actions) == 1 and actions[0].startswith("- action: gcal_create_event ")
    assert "- stayed quiet (wakes/wake_004.jsonl:" in index
    assert "- nudged after closing text; then message_user" in block


def test_a_write_tool_no_app_change_names_is_a_labelled_possible_action(run):
    timeline, turns = run
    _changed_by(timeline, 2, "gcal_create_event")
    index = stage_run(timeline, turns).files[INDEX_FILE]

    block = index[index.index("## Wake 2 "):index.index("## Wake 3 ")]
    possible = [line.split("): ", 1)[1].split(" ")[0] for line in block.splitlines() if line.startswith("- possible action")]
    assert possible == ["gsheets_values_update", "gcal_patch_event"]
    assert "A \"possible action\" is a call of a tool no app change in the run names" in index


def test_without_a_change_log_actions_are_guessed_from_tool_names(run):
    timeline, turns = run
    timeline = [e for e in timeline if e["event"] != "app_changes"]
    index = stage_run(timeline, turns).files[INDEX_FILE]

    assert "this run's apps kept no change log, so this is a guess from the name" in index
    block = index[index.index("## Wake 2 "):index.index("## Wake 3 ")]
    actions = [line.split(" ")[2] for line in block.splitlines() if line.startswith("- action")]
    assert actions == ["gsheets_values_update", "gcal_patch_event", "gcal_create_event"]


def test_every_line_too_long_to_read_whole_keeps_a_preview_and_a_file(run):
    timeline, turns = run
    reply = "please book it " * 400
    for e in timeline:
        if e["event"] == "user_reply_delivered":
            e["text"] = reply
        if e["event"] == "message_to_user" and e["wake"] == 1:
            e["text"] = "here's the plan: " * 300
    staged = stage_run(timeline, turns)

    assert all(len(line) <= LINE_CHARS for path, text in staged.files.items() if path.endswith(".jsonl")
               for line in text.splitlines())
    head = _lines(staged.files, "wakes/wake_003.jsonl")[1]
    assert head["event"] == "user_message" and "text" not in head
    side = staged.files[head["text_file"]]
    assert side.replace("\n", "") == reply and all(len(line) <= LINE_CHARS for line in side.splitlines())
    assert head["text_file"].startswith("wakes/results/wake_003_line_002_")
    assert head["text_preview"].startswith("please book it please book it")
    assert "- the user (Fri Nov 6, 9:00 PM UTC): \"please book it please book it" in staged.files[INDEX_FILE]


def test_events_jsonl_holds_every_wake_s_lines_and_the_run_s_start_and_end(run):
    timeline, turns = run
    timeline.insert(-1, {"event": "settle_timeout", "wake": 4, "still_at_work": [{"id": "porchbite-update"}]})
    staged = stage_run(timeline, turns)
    events = _lines(staged.files, EVENTS_FILE)

    assert events[0]["event"] == "start" and events[-1]["event"] == "end"
    per_wake = sum(len(staged.files[f"wakes/wake_00{n}.jsonl"].splitlines()) for n in range(1, 5))
    assert len(events) == per_wake + 2
    assert not any("real_time" in e or "input" in e and e["event"] == "user_sim_read" for e in events)
    assert {"event": "settle_timeout", "wake": 4, "still_at_work": [{"id": "porchbite-update"}]} in events


def test_triggers_jsonl_has_every_firing_and_the_wake_it_came_after(run):
    def firing(trigger: str, at: str, ok: bool = True) -> Firing:
        t = datetime.fromisoformat(at.replace("Z", "+00:00"))
        return Firing(trigger_id=trigger, source="env:world", turn=1, at=t, virtual_time=t, ok=ok, provoked_by=None)

    firings = [firing("tess-text", "2026-11-06T20:30:00Z"), firing("porchbite-update", "2026-11-06T21:30:00Z", False)]
    staged = stage_run(*run, firings=firings, unavailable={"user_agent": "no log"})

    lines = _lines(staged.files, TRIGGERS_FILE)
    assert [(line.get("trigger_id"), line.get("after_wake"), line.get("fired")) for line in lines[:2]] == [
        ("tess-text", 1, True), ("porchbite-update", 4, False)]
    assert lines[2] == {"unavailable": "user_agent", "why": "no log"}
    assert "- porchbite-update (env:world): fired 0, failed 1" in staged.files[INDEX_FILE]


def test_120_wakes_sort_in_wake_order_and_an_unknown_format_is_flagged():
    timeline = [{"event": "start", "virtual_time": "2026-11-06T20:00:00Z"}]
    turns = {}
    for n in range(1, 121):
        timeline += [{"event": "wake", "n": n, "virtual_time": f"2026-11-{6 + n // 24:02d}T{n % 24:02d}:00:00Z",
                      "woke_by": ["notification:Gmail"], "parts_text": f"alert {n}"},
                     {"event": "agent_turn_end", "n": n, "state": "completed", "trajectory_url": f"mem://{n}"}]
        turns[f"mem://{n}"] = turn_events(json.dumps([{"type": "tool_use", "tool_name": f"tool_{n}", "tool_id": "t",
                                                        "parameters": {}},
                                                       {"type": "tool_result", "tool_id": "t", "output": "ok"}]))
    turns["mem://7"] = turn_events(json.dumps([{"kind": "step"}]))

    staged = stage_run(timeline, turns)

    wake_files = sorted(p for p in staged.files if p.startswith("wakes/wake_"))
    assert len(wake_files) == 120 and wake_files[:2] == ["wakes/wake_001.jsonl", "wakes/wake_002.jsonl"]
    assert wake_files[-1] == "wakes/wake_120.jsonl"
    assert [int(p[11:14]) for p in wake_files] == list(range(1, 121))
    assert _lines(staged.files, "wakes/wake_099.jsonl")[2]["tool"] == "tool_99"
    assert [e["wake"] for e in _lines(staged.files, EVENTS_FILE)[1:] if e["event"] == "woke"] == list(range(1, 121))
    assert staged.record["not_compacted"] == [{"wake": 7, "turn": "turn", "format": "unknown",
                                               "raw_bytes": len(json.dumps([{"kind": "step"}])),
                                               "staged_bytes": staged.record["not_compacted"][0]["staged_bytes"]}]


def _judge(*calls: tuple[str, dict, str]) -> list[dict]:
    """A judge's Claude Code trajectory: each call's tool, input and result text."""
    blocks = [{"type": "tool_use", "id": f"t{i}", "name": tool, "input": tool_input}
              for i, (tool, tool_input, _) in enumerate(calls)]
    results = [{"type": "tool_result", "tool_use_id": f"t{i}", "content": text} for i, (_, _, text) in enumerate(calls)]
    return [{"type": "assistant", "message": {"content": blocks}}, {"type": "user", "message": {"content": results}}]


def test_a_firing_a_fraction_of_a_second_after_a_wake_comes_after_it(run):
    t = datetime.fromisoformat("2026-11-06T20:00:00.500000+00:00")
    firing = Firing(trigger_id="tess-text", source="env:world", turn=1, at=t, virtual_time=t, ok=True, provoked_by=None)

    lines = _lines(stage_run(*run, firings=[firing]).files, TRIGGERS_FILE)

    assert lines[0]["after_wake"] == 1


def test_judge_reads_names_the_wake_files_a_judge_opened_with_any_tool():
    judge = turn_events(json.dumps(_judge(
        ("Read", {"file_path": "/tmp/r/index.md"}, "## Wake 4 · … wakes/wake_004.jsonl\n## Wake 5 · … wakes/wake_005.jsonl"),
        ("Grep", {"pattern": "gmail_send", "path": "/tmp/r/events.jsonl"}, '{"event": "tool_call", "wake": 23}'),
        ("Read", {"file_path": "/tmp/r/wakes/wake_012.jsonl", "offset": 40}, "…"),
        ("Bash", {"command": "cat /tmp/r/wakes/results/wake_003_call_004_gmail_get_thread.json"}, "{}"),
        ("Bash", {"command": "cd /tmp/r; for w in 080 075; do grep tool_call wakes/wake_$w.jsonl; done"}, "…"),
        ("Bash", {"command": "cd /tmp/r; grep -n -i kidsy index.md"}, "61:- action: … (wakes/wake_009.jsonl:8)"))))

    assert judge_reads(judge) == {"tool_calls": 6, "wakes_opened": [3, 12, 23, 75, 80], "read_all_wakes": False,
                                  "read_index": True, "searched_events": True, "read_triggers": False,
                                  "helpers_started": 0}


def test_judge_reads_finds_the_wakes_in_a_search_that_returned_several_lines():
    judge = turn_events(json.dumps(_judge(
        ("Grep", {"pattern": "porchbite_place_order", "path": "/tmp/r/events.jsonl", "-n": True},
         '12:{"event": "tool_call", "wake": 7, "tool": "porchbite_place_order"}\n'
         '40:{"event": "tool_call", "wake": 19, "tool": "porchbite_place_order"}'))))

    assert judge_reads(judge)["wakes_opened"] == [7, 19]


def test_judge_reads_tells_one_pass_over_every_wake_file():
    judge = turn_events(json.dumps(_judge(
        ("Bash", {"command": "python3 -c \"import glob\nfor f in sorted(glob.glob('wakes/wake_*.jsonl')): print(f)\""},
         '{"event": "woke", "wake": 1}'),
        ("Agent", {"prompt": "check wake 7"}, "done"))))

    reads = judge_reads(judge)
    assert reads["read_all_wakes"] is True and reads["wakes_opened"] == [1] and reads["helpers_started"] == 1
