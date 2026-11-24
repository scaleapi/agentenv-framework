"""Claude Code's ATIF adapter on records shaped as the CLI writes them: stdout's stream-json events, and the session
transcript's records as the bridge feeds them (``session_aux``). Each case runs through the protocol's adapter kit, so
the live rule holds after every record and the document validates as a producer's."""
from __future__ import annotations

import base64
import json
import random
import re
import struct
import zlib

import pytest
from agentenv_protocol.atif import AtifTurn, Fed, RecordedTask, TurnContext, check_adapter, profile_of, task_document

from agent_env.task_step.task_steps.verifiers.judge_utils.turn_events import atif_turn_events, turn_events
from tst.unit.agents.claude_code.agent import AGENT_DIR, claude_bridge, claude_code_atif

ClaudeCodeAtif = claude_code_atif.ClaudeCodeAtif
MODEL = "anthropic/claude-sonnet-5-5"
USAGE = {"input_tokens": 2, "output_tokens": 75, "cache_read_input_tokens": 68042, "cache_creation_input_tokens": 94}


def png(width=4, height=3) -> bytes:
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    rows = b"".join(b"\x00" + b"\x10\x20\x30" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def assistant(msg, uuid, *blocks, parent=None, model=MODEL):
    return {"type": "assistant", "uuid": uuid, "parent_tool_use_id": parent, "session_id": "s",
            "message": {"id": msg, "model": model, "role": "assistant", "content": list(blocks),
                        "usage": {"input_tokens": 2, "output_tokens": 10}, "stop_reason": None}}


def user(uuid, *blocks, parent=None):
    return {"type": "user", "uuid": uuid, "parent_tool_use_id": parent, "session_id": "s",
            "message": {"role": "user", "content": list(blocks)}}


def text(value):
    return {"type": "text", "text": value}


def call(call_id, name="Bash", **arguments):
    return {"type": "tool_use", "id": call_id, "name": name, "input": arguments or {"command": "ls"}}


def result(call_id, content="ok", is_error=False):
    return {"type": "tool_result", "tool_use_id": call_id, "content": content, "is_error": is_error}


class Feed:
    """One task's records in arrival order: stdout events as natives, transcript records through ``session_aux``."""

    def __init__(self):
        self.records: list[Fed] = []
        self.on_stdout: set[str] = set()
        self.shapes: set = set()

    def out(self, event):
        if isinstance(event.get("uuid"), str):
            self.on_stdout.add(event["uuid"])
        self.records.append(Fed(event))
        return self

    def transcript(self, event, at, usage=USAGE, agent=None):
        """The transcript's record of ``event`` (an assistant's or user's), written at ``at``."""
        record = {key: value for key, value in event.items() if key not in ("parent_tool_use_id", "session_id")}
        record.update(timestamp=at, isSidechain=agent is not None, message=dict(event["message"]))
        if event["type"] == "assistant":
            record["message"].update(usage=usage, stop_reason="end_turn")
        return self.fed(record, agent)

    def fed(self, record, agent=None):
        name_record = claude_bridge.session_aux(record, agent, self.on_stdout, self.shapes)
        if name_record is not None:
            self.records.append(Fed(name_record[1], aux=name_record[0]))
        return self

    def aux(self, name, record):
        self.records.append(Fed(record, aux=name))
        return self


def context(index=0):
    return TurnContext(index=index, context_id="ctx", task_id=f"t{index}", message_id=f"m{index}", inbound=["do it"],
                       agent_name="Claude Code", agent_version="1.0.0", agent_model=MODEL,
                       started_at=f"2026-11-22T02:00:0{index}Z")


def run(*feeds: Feed):
    kit = check_adapter(ClaudeCodeAtif, [RecordedTask(context=context(i), records=feed.records)
                                         for i, feed in enumerate(feeds)])
    kit.raise_for_errors()
    return kit


def agent_steps(document):
    return [step for step in document["steps"] if step["source"] == "agent"]


def test_a_step_waits_for_its_session_record_then_carries_its_time_and_final_usage():
    first, second = assistant("msg-1", "u1", text("looking")), assistant("msg-2", "u2", text("done"))
    feed = Feed().out(first).out(second).transcript(first, "2026-11-22T02:01:00.100Z")
    feed.transcript(second, "2026-11-22T02:01:02.000Z", usage={**USAGE, "output_tokens": 4})
    kit = run(feed)
    steps = agent_steps(kit.document)
    assert [s["timestamp"] for s in steps] == ["2026-11-22T02:01:00.100Z", "2026-11-22T02:01:02.000Z"]
    assert steps[0]["metrics"] == {"prompt_tokens": 2 + 68042 + 94, "cached_tokens": 68042, "completion_tokens": 75,
                                   "extra": {"agentenv": {"cache_write_tokens": 94}}}
    assert steps[0]["model_name"] == MODEL
    assert all(len(agent_steps(page or {"steps": []})) == 0 for page in kit.pages[:3])


def test_a_message_split_over_records_is_one_step_with_its_usage_written_once():
    thinking = assistant("msg-1", "u1", {"type": "thinking", "thinking": "", "signature": "sig-1"})
    calling = assistant("msg-1", "u2", call("call-1"))
    feed = Feed().out(thinking).out(calling).transcript(thinking, "2026-11-22T02:01:00Z")
    feed.transcript(calling, "2026-11-22T02:01:00.050Z").out(user("u3", result("call-1"))).transcript(
        user("u3", result("call-1")), "2026-11-22T02:01:01Z")
    step = agent_steps(run(feed).document)[0]
    assert step["metrics"]["completion_tokens"] == 75 and len(step["tool_calls"]) == 1
    assert profile_of(step)["reasoning"] == [{"chars": 0, "visibility": "omitted", "signature": "sig-1"}]
    assert step["observation"]["results"][0]["content"] == "ok"


def test_a_message_whose_session_record_never_comes_is_written_once_a_later_one_of_its_agent_has_its_own():
    lost, found = assistant("msg-1", "u1", text("first")), assistant("msg-2", "u2", text("second"))
    kit = run(Feed().out(lost).out(found).transcript(found, "2026-11-22T02:01:02Z"))
    steps = agent_steps(kit.document)
    assert [s["message"] for s in steps] == ["first", "second"]
    assert "timestamp" not in steps[0] and "metrics" not in steps[0] and steps[0]["model_name"] == MODEL


def test_the_task_end_writes_what_still_waits_without_a_time():
    steps = agent_steps(run(Feed().out(assistant("msg-1", "u1", text("only stdout")))).document)
    assert steps[0]["message"] == "only stdout" and "timestamp" not in steps[0]


def test_a_session_record_ahead_of_its_stdout_event_writes_the_step_once():
    event = assistant("msg-1", "u1", text("raced"))
    feed = Feed()
    feed.transcript(event, "2026-11-22T02:01:00Z").out(event)
    steps = agent_steps(run(feed).document)
    assert [s["message"] for s in steps] == ["raced"] and steps[0]["timestamp"] == "2026-11-22T02:01:00Z"


def test_a_sub_agent_is_embedded_with_its_tools_and_its_last_answer_which_only_its_transcript_has():
    spawn = assistant("msg-1", "u1", call("spawn-1", "Agent", description="count", prompt="count lines",
                                          subagent_type="general-purpose"))
    prompt = user("u2", text("count lines"), parent="spawn-1")
    sub_call = assistant("msg-s1", "u3", call("call-s1"), parent="spawn-1")
    sub_result = user("u4", result("call-s1", "21 /etc/passwd"), parent="spawn-1")
    sub_answer = assistant("msg-s2", "u5", text("It has 21 lines."), parent="spawn-1")
    handed_back = user("u6", result("spawn-1", [text("It has 21 lines.")]))
    answer = assistant("msg-2", "u7", text("21"))
    tools = {"type": "api-request-shape", "shapeHash": "h-sub", "timestamp": "2026-11-22T02:01:00Z",
             "shape": {"tools": [{"name": "Bash", "description": "Run", "input_schema": {"type": "object"}}]}}
    feed = Feed().out(spawn).out(prompt).transcript(prompt, "2026-11-22T02:01:00.900Z", agent="a7f9")
    feed.fed(tools, "a7f9").transcript(spawn, "2026-11-22T02:01:00.500Z")
    feed.out(sub_call).out(sub_result).transcript(sub_call, "2026-11-22T02:01:01Z", agent="a7f9")
    feed.transcript(sub_result, "2026-11-22T02:01:02Z", agent="a7f9").out(handed_back)
    feed.transcript(sub_answer, "2026-11-22T02:01:03Z", agent="a7f9", usage={**USAGE, "output_tokens": 18})
    feed.transcript(handed_back, "2026-11-22T02:01:04Z").out(answer).transcript(answer, "2026-11-22T02:01:05Z")
    document = run(feed).document
    sub = document["subagent_trajectories"][0]
    assert sub["trajectory_id"] == "spawn-1" and sub["agent"]["name"] == "general-purpose"
    assert [d["function"]["name"] for d in sub["agent"]["tool_definitions"]] == ["Bash"]
    assert [s.get("message") for s in sub["steps"]] == ["", "It has 21 lines."]
    assert sub["steps"][1]["metrics"]["completion_tokens"] == 18
    spawned = agent_steps(document)[0]["observation"]["results"][0]
    assert spawned["subagent_trajectory_ref"][0]["trajectory_id"] == "spawn-1"
    assert "tool_definitions" not in document["agent"]


def test_the_main_agents_request_shape_gives_its_tool_definitions_once():
    shape = {"type": "api-request-shape", "shapeHash": "h1", "timestamp": "2026-11-22T02:01:00Z",
             "shape": {"system": [text("a long system prompt")],
                       "tools": [{"name": "Bash", "description": "Run a command",
                                  "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}},
                                 {"name": "Read"}]}}
    event = assistant("msg-1", "u1", text("hi"))
    feed = Feed().fed(shape).fed(shape).out(event).transcript(event, "2026-11-22T02:01:01Z")
    fed = [record.record for record in feed.records if record.aux == "tools"]
    document = run(feed).document
    assert len(fed) == 1 and "system" not in fed[0]
    assert document["agent"]["tool_definitions"] == [
        {"type": "function", "function": {"name": "Bash", "description": "Run a command", "parameters": {
            "type": "object", "properties": {"command": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "Read"}}]


def test_a_protocol_without_tool_definitions_leaves_the_agent_block_without_them(monkeypatch):
    monkeypatch.delattr(AtifTurn, "tool_definitions")
    shape = {"type": "api-request-shape", "shapeHash": "h1", "shape": {"tools": [{"name": "Bash"}]}}
    event = assistant("msg-1", "u1", text("hi"))
    document = run(Feed().fed(shape).out(event).transcript(event, "2026-11-22T02:01:01Z")).document
    assert "tool_definitions" not in document["agent"]


def test_a_resumed_attempts_error_lands_on_its_last_step_and_later_steps_carry_the_next_attempt():
    first, second = assistant("msg-1", "u1", text("working")), assistant("msg-2", "u2", text("done"))
    failed = {"type": "result", "subtype": "success", "is_error": True, "api_error_status": 529,
              "result": "API Error: 529 Overloaded"}
    feed = Feed().out({"type": "system", "subtype": "init", "uuid": "i1"}).out(first)
    feed.transcript(first, "2026-11-22T02:01:00Z").aux("attempt", failed)
    feed.out({"type": "system", "subtype": "init", "uuid": "i2"}).out(second).transcript(second, "2026-11-22T02:01:30Z")
    steps = agent_steps(run(feed).document)
    assert profile_of(steps[0])["error"] == {"kind": "infra_error", "message": "API Error: 529 Overloaded",
                                             "code": "529"}
    assert "attempt" not in profile_of(steps[0]) and profile_of(steps[1])["attempt"] == 2


def test_an_attempt_that_wrote_no_step_gets_a_step_for_its_error():
    failed = {"type": "result", "subtype": "success", "is_error": True, "api_error_status": None,
              "terminal_reason": "api_error", "result": "Request timed out"}
    answer = assistant("msg-1", "u1", text("done"))
    feed = Feed().out({"type": "system", "subtype": "init"}).aux("attempt", failed)
    feed.out({"type": "system", "subtype": "init"}).out(answer).transcript(answer, "2026-11-22T02:01:30Z")
    steps = agent_steps(run(feed).document)
    assert steps[0]["llm_call_count"] == 0 and profile_of(steps[0])["error"] == {
        "kind": "infra_error", "message": "Request timed out", "code": "api_error"}
    assert steps[1]["message"] == "done"


def test_the_turns_reported_cost_is_the_tasks():
    event = assistant("msg-1", "u1", text("hi"))
    feed = Feed().out(event).transcript(event, "2026-11-22T02:01:01Z").aux("turn", {"type": "turn", "cost_usd": 0.07})
    final = run(feed).document["final_metrics"]
    assert final["total_cost_usd"] == 0.07 and profile_of(final).get("cost_source") == "harness"


@pytest.mark.parametrize("block, expected", [
    ({"type": "thinking", "thinking": "let me see", "signature": "s"}, {"chars": 10, "visibility": "raw",
                                                                        "signature": "s"}),
    ({"type": "thinking", "thinking": "", "signature": "s"}, {"chars": 0, "visibility": "omitted", "signature": "s"}),
    ({"type": "redacted_thinking", "data": "opaque"}, {"chars": 0, "visibility": "redacted", "data": "opaque"}),
])
def test_thinking_keeps_its_visibility_and_signature(block, expected):
    event = assistant("msg-1", "u1", block, text("answer"))
    step = agent_steps(run(Feed().out(event).transcript(event, "2026-11-22T02:01:01Z")).document)[0]
    reasoning = profile_of(step)["reasoning"][0]
    assert {key: reasoning[key] for key in expected} == expected


def test_an_image_result_is_a_media_file_named_by_its_bytes():
    shot = png()
    calling = assistant("msg-1", "u1", call("call-1", "Read", file_path="/app/shot.png"))
    returned = user("u2", result("call-1", [{"type": "image", "source": {
        "type": "base64", "media_type": "image/png", "data": base64.b64encode(shot).decode()}}]))
    feed = Feed().out(calling).transcript(calling, "2026-11-22T02:01:00Z").out(returned)
    feed.transcript(returned, "2026-11-22T02:01:01Z")
    kit = run(feed)
    part = agent_steps(kit.document)[0]["observation"]["results"][0]["content"][0]
    assert part["type"] == "image" and re.fullmatch(r"media/[0-9a-f]{64}\.png", part["source"]["path"])
    assert kit.media[part["source"]["path"]] == shot


def test_a_message_the_cli_wrote_itself_is_an_error_step_with_no_model_call():
    event = assistant("msg-1", "u1", text("API Error: Connection error."), model="<synthetic>")
    event["isApiErrorMessage"] = True
    step = agent_steps(run(Feed().out(event).transcript(event, "2026-11-22T02:01:01Z")).document)[0]
    assert step["llm_call_count"] == 0 and "metrics" not in step
    assert profile_of(step)["error"] == {"kind": "infra_error", "message": "API Error: Connection error."}


def test_a_sub_agent_whose_spawning_call_already_has_its_result_is_left_to_the_native_log():
    spawn = assistant("msg-1", "u1", call("spawn-1", "Agent", prompt="go"))
    handed_back = user("u2", result("spawn-1", "done"))
    late = assistant("msg-s1", "u3", text("late"), parent="spawn-1")
    feed = Feed().out(spawn).transcript(spawn, "2026-11-22T02:01:00Z").out(handed_back)
    feed.transcript(handed_back, "2026-11-22T02:01:01Z").out(late).transcript(late, "2026-11-22T02:01:02Z")
    document = run(feed).document
    assert "subagent_trajectories" not in document and len(agent_steps(document)) == 1


def test_records_no_handler_takes_are_counted_in_the_profile():
    event = assistant("msg-1", "u1", text("hi"))
    feed = Feed().out({"type": "system", "subtype": "hook_started"}).out(event).transcript(event, "2026-11-22T02:01:01Z")
    kit = check_adapter(ClaudeCodeAtif, [RecordedTask(context=context(), records=feed.records, final=["done"])])
    assert profile_of(kit.document)["unmatched"] == {"system": 1}
    assert [f.rule for f in kit.findings] == ["kit-unmatched"]


def test_each_task_of_a_context_continues_the_same_document():
    first, second = assistant("msg-1", "u1", text("one")), assistant("msg-2", "u2", text("two"))
    document = run(Feed().out(first).transcript(first, "2026-11-22T02:01:00Z"),
                   Feed().out(second).transcript(second, "2026-11-22T02:02:00Z")).document
    assert [s["source"] for s in document["steps"]] == ["user", "agent", "user", "agent"]


def test_from_atif_gives_back_stream_json_events():
    calling = assistant("msg-1", "u1", call("call-1"))
    answer = assistant("msg-2", "u3", text("all done"))
    feed = Feed().out(calling).transcript(calling, "2026-11-22T02:01:00Z").out(user("u2", result("call-1", "out")))
    feed.transcript(user("u2", result("call-1", "out")), "2026-11-22T02:01:01Z").out(answer)
    feed.transcript(answer, "2026-11-22T02:01:02Z")
    log = ClaudeCodeAtif.from_atif(run(feed).document).log
    assert [event["type"] for event in log] == ["assistant", "user", "assistant", "result"]
    assert log[0]["message"]["content"][0] == {"type": "tool_use", "id": "call-1", "name": "Bash",
                                               "input": {"command": "ls"}}
    assert log[1]["message"]["content"][0]["content"] == "out" and log[-1]["result"] == "all done"


def test_the_pinned_cli_is_one_the_fixtures_were_recorded_on():
    pinned = re.search(r"^ARG CLAUDE_CODE_VERSION=(\S+)$", (AGENT_DIR / "Dockerfile").read_text(), re.MULTILINE)
    assert pinned and pinned.group(1) in ClaudeCodeAtif.spec.tested_with


def _sub_agent_turn(feed, spawn_id, at, *, parent=None, cli="a7f9", step="s"):
    """A sub-agent's turn under call ``spawn_id``: its prompt, a call and its result on stdout (``parent`` the call
    stdout names), and its last answer only in its transcript."""
    parent = parent or spawn_id
    prompt = user(f"{step}-p", text("go"), parent=parent)
    sub_call = assistant(f"m-{step}1", f"{step}-c", call(f"{step}-call"), parent=parent)
    sub_result = user(f"{step}-r", result(f"{step}-call", "out"), parent=parent)
    answer = assistant(f"m-{step}2", f"{step}-a", text(f"answer {step}"), parent=parent)
    feed.out(prompt).transcript(prompt, at, agent=cli).out(sub_call).transcript(sub_call, at, agent=cli)
    feed.out(sub_result).transcript(sub_result, at, agent=cli).transcript(answer, at, agent=cli)


def test_a_sub_agent_continued_in_a_later_task_is_a_trajectory_of_that_task():
    first, second = Feed(), Feed()
    spawn = assistant("msg-1", "u1", call("spawn-1", "Agent", prompt="go"))
    first.out(spawn).transcript(spawn, "2026-11-22T02:01:00Z")
    first.fed({"type": "api-request-shape", "shapeHash": "h-sub", "shape": {"tools": [{"name": "Bash"}]}}, "a7f9")
    _sub_agent_turn(first, "spawn-1", "2026-11-22T02:01:01Z", step="a")
    back = user("u2", result("spawn-1", "done"))
    first.out(back).transcript(back, "2026-11-22T02:01:02Z")
    resume = assistant("msg-2", "u3", call("send-1", "SendMessage", to="a7f9", message="again"))
    second.out(resume).transcript(resume, "2026-11-22T02:02:00Z")
    _sub_agent_turn(second, "send-1", "2026-11-22T02:02:01Z", step="b")
    again = user("u4", result("send-1", "done again"))
    second.out(again).transcript(again, "2026-11-22T02:02:02Z")
    document = run(first, second).document
    subs = {sub["trajectory_id"]: [s.get("message") for s in sub["steps"]] for sub in document["subagent_trajectories"]}
    assert subs == {"spawn-1": ["", "answer a"], "send-1": ["", "answer b"]}
    assert [[d["function"]["name"] for d in sub["agent"]["tool_definitions"]]
            for sub in document["subagent_trajectories"]] == [["Bash"], ["Bash"]]


def test_a_sub_agent_a_later_task_names_by_its_first_call_is_left_to_the_native_log():
    first, second = Feed(), Feed()
    spawn = assistant("msg-1", "u1", call("spawn-1", "Agent", prompt="go"))
    first.out(spawn).transcript(spawn, "2026-11-22T02:01:00Z")
    _sub_agent_turn(first, "spawn-1", "2026-11-22T02:01:01Z", step="a")
    back = user("u2", result("spawn-1", "done"))
    first.out(back).transcript(back, "2026-11-22T02:01:02Z")
    answer = assistant("msg-2", "u3", text("still here"))
    _sub_agent_turn(second, "spawn-1", "2026-11-22T02:02:01Z", step="b")
    second.out(answer).transcript(answer, "2026-11-22T02:02:02Z")
    document = run(first, second).document
    assert [len(sub["steps"]) for sub in document["subagent_trajectories"]] == [2]
    assert agent_steps(document)[-1]["message"] == "still here"


def _compacting_feed(turn_counts=None):
    """A turn that compacts partway: a call and its result, the CLI's boundary and its summary, its own message, then a
    reply; and the turn's record with ``turn_counts``, when given."""
    first, done = assistant("msg-1", "u1", call("c1")), assistant("msg-2", "u4", text("done"))
    answered = user("u2", result("c1"))
    summary = {"type": "user", "uuid": "u3", "parent_tool_use_id": None, "session_id": "s",
               "message": {"role": "user", "content": "The summary of the conversation so far."}}
    notice = {"type": "user", "uuid": "u5", "parent_tool_use_id": None, "session_id": "s",
              "message": {"role": "user", "content": "<local-command-stdout>Compacted</local-command-stdout>"}}
    feed = Feed().out(first).transcript(first, "2026-11-22T02:01:00.000Z").out(answered)
    feed.transcript(answered, "2026-11-22T02:01:01.000Z")
    feed.out({"type": "system", "subtype": "compact_boundary", "session_id": "s", "uuid": "b1",
              "compact_metadata": {"trigger": "auto", "pre_tokens": 90000, "post_tokens": 3000}})
    feed.out(summary).transcript(summary, "2026-11-22T02:01:02.000Z").out(notice)
    feed.transcript(notice, "2026-11-22T02:01:02.500Z").out(done)
    feed.transcript(done, "2026-11-22T02:01:03.000Z", usage={**USAGE, "output_tokens": 4})
    if turn_counts is not None:
        feed.aux("turn", {"type": "turn", "cost_usd": 0.5, **turn_counts})
    return feed


def test_an_auto_compaction_is_the_step_of_its_summary_call_holding_what_the_turn_counted_beyond_its_steps():
    steps_counted = {"input_tokens": 4, "output_tokens": 79, "cache_read_tokens": 136084, "cache_write_tokens": 188}
    summary_call = {"input_tokens": 500, "output_tokens": 300, "cache_read_tokens": 1000, "cache_write_tokens": 20}
    feed = _compacting_feed({name: steps_counted[name] + summary_call[name] for name in steps_counted})
    kit = run(feed)
    user_step, first, compaction, notice, done = kit.document["steps"][:5]
    assert (compaction["source"], compaction["message"]) == ("agent", "The summary of the conversation so far.")
    assert profile_of(compaction)["compaction"] == {"written_by": "assistant", "trigger": "auto", "tokens_before": 90000,
                                                    "tokens_after": 3000}
    assert compaction["metrics"] == {"prompt_tokens": 500 + 1000 + 20, "cached_tokens": 1000, "completion_tokens": 300,
                                     "extra": {"agentenv": {"cache_write_tokens": 20}}}
    assert (notice["source"], notice["message"]) == ("system", "<local-command-stdout>Compacted</local-command-stdout>")
    assert done["message"] == "done" and "metrics" in done
    # Held for the turn's count, its last record: no live page before it serves the compaction, the one after it does.
    counted = len(feed.records)
    compactions = [any("compaction" in profile_of(step) for step in (page or {}).get("steps", [])) for page in kit.pages]
    assert not any(compactions[:counted]) and compactions[counted]


def test_a_compaction_whose_turn_counted_less_than_its_steps_has_no_usage():
    kit = run(_compacting_feed({"input_tokens": 1, "output_tokens": 1, "cache_read_tokens": 1, "cache_write_tokens": 1}))
    compaction = next(step for step in kit.document["steps"] if "compaction" in profile_of(step))
    assert "metrics" not in compaction and compaction["llm_call_count"] == 1


def test_a_compaction_whose_turn_never_reported_is_written_as_its_task_ends():
    kit = run(_compacting_feed())
    sources = [(step["source"], "compaction" in profile_of(step)) for step in kit.document["steps"]]
    assert sources[2:5] == [("agent", True), ("system", False), ("agent", False)]


def _summary(uuid, text_value):
    return {"type": "user", "uuid": uuid, "parent_tool_use_id": None, "session_id": "s",
            "message": {"role": "user", "content": text_value}}


def _boundary(uuid):
    return {"type": "system", "subtype": "compact_boundary", "session_id": "s", "uuid": uuid,
            "compact_metadata": {"trigger": "auto"}}


COUNTED_TURN = {"type": "turn", "cost_usd": 0.5, "input_tokens": 9000, "output_tokens": 900,
                "cache_read_tokens": 200000, "cache_write_tokens": 2000}


def test_two_compactions_in_one_turn_have_no_usage_since_the_turns_remainder_cant_be_split_between_them():
    first, done = assistant("msg-1", "u1", text("working")), assistant("msg-2", "u6", text("done"))
    one, two = _summary("u3", "The first summary."), _summary("u5", "The second summary.")
    feed = Feed().out(first).transcript(first, "2026-11-22T02:01:00.000Z").out(_boundary("b1")).out(one)
    feed.transcript(one, "2026-11-22T02:01:01.000Z").out(_boundary("b2")).out(two)
    feed.transcript(two, "2026-11-22T02:01:02.000Z").out(done).transcript(done, "2026-11-22T02:01:03.000Z")
    kit = run(feed.aux("turn", COUNTED_TURN))
    compactions = [step for step in kit.document["steps"] if "compaction" in profile_of(step)]
    assert [step["message"] for step in compactions] == ["The first summary.", "The second summary."]
    assert all("metrics" not in step for step in compactions)


def _named(kit):
    return next(step for step in kit.document["steps"] if "compaction" in profile_of(step))["model_name"]


def _compacting_only(called):
    summary = _summary("u9", "The summary.")
    feed = Feed().out(_boundary("b9")).out(summary).transcript(summary, "2026-11-22T02:02:01.000Z")
    return feed.aux("turn", {**COUNTED_TURN, "model": called})


def test_a_compaction_names_its_model_as_the_agents_calls_answered_it_and_never_names_another():
    """The turn's tally keys a model by the name it was called as, a gateway's route prefixing it; the agent's steps
    name it as the API answered. A compaction is named as its model's calls answered, or as called when none has."""
    called = "gateway/" + MODEL
    mid_turn = run(_compacting_feed({name: value for name, value in {**COUNTED_TURN, "model": called}.items()
                                     if name != "type"}))
    working = assistant("msg-1", "u1", text("working"))
    earlier = Feed().out(working).transcript(working, "2026-11-22T02:01:00.000Z")
    earlier.aux("turn", {**COUNTED_TURN, "model": called})
    same_model = run(earlier, _compacting_only(called))
    working = assistant("msg-1", "u1", text("working"))
    earlier = Feed().out(working).transcript(working, "2026-11-22T02:01:00.000Z")
    earlier.aux("turn", {**COUNTED_TURN, "model": called})
    another_model = run(earlier, _compacting_only("gateway/another-model"))
    alone = run(_compacting_only(called))
    assert [_named(kit) for kit in (mid_turn, same_model, another_model, alone)] == [
        MODEL, MODEL, "gateway/another-model", called]


def test_a_summary_in_several_text_blocks_is_one_compaction_as_natively():
    summary = {"type": "user", "uuid": "u3", "parent_tool_use_id": None, "session_id": "s",
               "message": {"role": "user", "content": [text("The first part."), text("The second part.")]}}
    feed = Feed().out(_boundary("b1")).out(summary).transcript(summary, "2026-11-22T02:01:01.000Z")
    document = run(feed.aux("turn", COUNTED_TURN)).document
    [compaction] = [step for step in document["steps"] if "compaction" in profile_of(step)]
    assert compaction["message"] == "The first part.\nThe second part."
    assert not [step for step in document["steps"] if step["source"] == "system"]


def test_a_compaction_has_no_usage_when_one_of_its_turns_calls_lacks_its_counts():
    """A call whose transcript record never came has no counts, which the turn's remainder would hold too."""
    uncounted, done = assistant("msg-1", "u1", text("working")), assistant("msg-2", "u4", text("done"))
    summary = _summary("u3", "The summary.")
    feed = Feed().out(uncounted).out(_boundary("b1")).out(summary).transcript(summary, "2026-11-22T02:01:01.000Z")
    feed.out(done).transcript(done, "2026-11-22T02:01:03.000Z")
    kit = run(feed.aux("turn", COUNTED_TURN))
    [compaction] = [step for step in kit.document["steps"] if "compaction" in profile_of(step)]
    assert "metrics" not in compaction and compaction["message"] == "The summary."


class _RandomConversation:
    """A seed's Claude Code conversation of one to three turns. Each turn holds root text, tool calls with their results
    and compactions in a random order: a boundary whose facts are good or bad, its summary a string or one to three
    text blocks, sometimes the CLI's "Compacted", sometimes two in a turn. It ends with its closing text, the CLI's
    result, and a turn record whose counts are right, short or missing. ``facts``: each turn's compactions' facts."""

    TRIGGERS = ("manual", "auto", None, "later")
    COUNTS = (12345, 0, None, -5, True, "300", 2.5)

    def __init__(self, seed: int):
        self.rng, self.n = random.Random(seed), 0
        self.feeds: list[Feed] = []
        self.facts: list[list[dict]] = []
        for _ in range(self.rng.randint(1, 3)):
            self.turn()

    def uid(self) -> str:
        self.n += 1
        return f"u{self.n}"

    def at(self) -> str:
        return f"2026-11-22T02:{self.n // 60:02d}:{self.n % 60:02d}.000Z"

    def say(self, feed: Feed, words: str) -> str:
        event = assistant(f"m{self.n}", self.uid(), text(words))
        feed.out(event).transcript(event, self.at())
        return words

    def compact(self, feed: Feed) -> dict:
        rng = self.rng
        trigger, before, after = rng.choice(self.TRIGGERS), rng.choice(self.COUNTS), rng.choice(self.COUNTS)
        metadata = {key: value for key, value in (("trigger", trigger), ("pre_tokens", before), ("post_tokens", after))
                    if value is not None}
        feed.out({"type": "system", "subtype": "compact_boundary", "session_id": "s", "uuid": self.uid(),
                  "compact_metadata": metadata})
        words = [f"Summary {self.n}, part {k}." for k in range(rng.randint(1, 3))]
        summary = (_summary(self.uid(), words[0]) if rng.random() < 0.5 else
                   {**_summary(self.uid(), ""), "message": {"role": "user", "content": [text(w) for w in words]}})
        feed.out(summary).transcript(summary, self.at())
        if rng.random() < 0.4:
            notice = _summary(self.uid(), "<local-command-stdout>Compacted</local-command-stdout>")
            feed.out(notice).transcript(notice, self.at())
        counts = {name: value for name, value in (("tokens_before", before), ("tokens_after", after))
                  if isinstance(value, int) and not isinstance(value, bool) and value >= 0}
        return {"written_by": "assistant", **({"trigger": trigger} if trigger in ("manual", "auto") else {}), **counts}

    def turn(self) -> None:
        rng, feed, facts, last = self.rng, Feed(), [], ""
        for _ in range(rng.randint(1, 6)):
            roll = rng.random()
            if roll < 0.35:
                last = self.say(feed, rng.choice(["Looking.", "One moment."]))
            elif roll < 0.7:
                call_id = f"c{self.n}"
                asked = assistant(f"m{self.n}", self.uid(), call(call_id))
                feed.out(asked).transcript(asked, self.at())
                answered = user(self.uid(), result(call_id, rng.choice(["ok", ""])))
                feed.out(answered).transcript(answered, self.at())
            else:
                facts.append(self.compact(feed))
        if last or rng.random() < 0.8:
            last = self.say(feed, "Done.")
        feed.out({"type": "result", "result": last, "is_error": False})
        roll = rng.random()
        if roll < 0.7:
            short = {name: 0 for name in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")}
            feed.aux("turn", COUNTED_TURN if roll < 0.5 else {**COUNTED_TURN, **short})
        self.feeds.append(feed)
        self.facts.append(facts)


def test_random_conversations_that_compact_give_a_judge_the_same_events_natively_and_from_the_copy():
    """Two readers keep one rule for a compaction: the adapter, which writes the copy, and the judges' native reader.
    Over fixed seeds, every turn's events are the same from both, its compaction notes and their facts included."""
    for seed in range(200):
        conversation = _RandomConversation(seed)
        document = run(*conversation.feeds).document
        for i, (feed, facts) in enumerate(zip(conversation.feeds, conversation.facts)):
            native = turn_events(json.dumps([fed.record for fed in feed.records if fed.aux is None])).events
            copied = atif_turn_events(task_document(document, [f"t{i}"])).events
            assert copied == native, f"seed {seed}, turn {i}"
            assert [event["compaction"] for event in native if event.get("compaction")] == facts, f"seed {seed}, turn {i}"
