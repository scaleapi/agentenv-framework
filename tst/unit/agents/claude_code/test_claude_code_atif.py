"""Claude Code's ATIF adapter on records shaped as the CLI writes them: stdout's stream-json events, and the session
transcript's records as the bridge feeds them (``session_aux``). Each case runs through the protocol's adapter kit, so
the live rule holds after every record and the document validates as a producer's."""
from __future__ import annotations

import base64
import re
import struct
import zlib

import pytest
from agentenv_protocol.atif import AtifTurn, Fed, RecordedTask, TurnContext, check_adapter, profile_of

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
    failed = {"type": "result", "subtype": "error_during_execution", "is_error": True, "result": "API Error"}
    answer = assistant("msg-1", "u1", text("done"))
    feed = Feed().out({"type": "system", "subtype": "init"}).aux("attempt", failed)
    feed.out({"type": "system", "subtype": "init"}).out(answer).transcript(answer, "2026-11-22T02:01:30Z")
    steps = agent_steps(run(feed).document)
    assert steps[0]["llm_call_count"] == 0 and profile_of(steps[0])["error"]["code"] == "error_during_execution"
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
