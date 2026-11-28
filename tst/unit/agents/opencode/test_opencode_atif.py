"""OpenCode's ATIF adapter on records shaped as opencode-ai 1.17.18 prints them and the turn's relay hands them over:
the CLI's ``run --format json`` events as natives and the relay's ``agentenv.model_call`` records as the ``calls`` aux
file, beside the turn's model and effort (``turn``) and the sub-agent sessions read from the session store
(``subagents``). Each case runs through the protocol's adapter kit, so the live rule holds after every record and the
document validates as a producer's. Then the bridge and the server, which feed the aux records only to a task that
builds an atif/1 copy, and the card."""
from __future__ import annotations

import asyncio
import base64
import itertools
import json
import random
import sqlite3
import struct
import zlib
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone

import pytest
from agentenv_protocol.a2a_agent import Usage
from agentenv_protocol.atif import (
    Fed,
    RecordedTask,
    TurnContext,
    check_adapter,
    errors_in,
    profile_of,
    strip_profile,
    validate,
)

from tst.unit.agents.opencode.agent import AGENT_DIR, a2a_server, model_relay, opencode_atif, opencode_bridge
from tst.unit.agents.opencode.conftest import Harness, result
from tst.unit.agents.opencode.fake_cli import FakeCLI

OpenCodeAtif = opencode_atif.OpenCodeAtif
MODEL = "anthropic/claude-haiku-4-5"
SESSION = "ses_ddf306cf7ffecYPG37nmfG23Da"
CHILD = "ses_ddf305cf8ffex57XN7Odza3nsW"


def png(width=4, height=3) -> bytes:
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    rows = b"".join(b"\x00" + b"\x10\x20\x30" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def stamp(ms: int) -> str:
    return f"{datetime.fromtimestamp(ms // 1000, timezone.utc):%Y-%m-%dT%H:%M:%S}.{ms % 1000:03d}Z"


class Feed:
    """One task's records in the order the bridge feeds them, each some milliseconds after the last, on the clock the
    CLI and the relay share; or, for a sub-agent's ``session``, the events the bridge reads from the session store."""

    clock = 1_795_846_800_000
    ids = itertools.count()

    def __init__(self, session=SESSION):
        self.session = session
        self.records: list[Fed] = []

    @classmethod
    def now(cls, by=5):
        cls.clock += by
        return cls.clock

    def turn(self, model=MODEL, effort=None):
        self.records.append(Fed({"type": "agentenv.turn", "model": model, "effort": effort}, aux="turn"))
        return self

    def call(self, inputs=1020, output=22, read=600, write=100, *, outcome="answered", status=200, started=None,
             ended=None, model=MODEL):
        """The relay's record of a call, fed as it ends: its input holds the cache, its output the reasoning."""
        started = started if started is not None else self.now()
        ended = ended if ended is not None else max(started, self.now())
        record = {"type": "agentenv.model_call", "via": "relay", "model": model, "status": status, "outcome": outcome}
        if outcome == "answered":
            record["usage"] = {"input_tokens": inputs, "output_tokens": output, "total_tokens": inputs + output,
                               "cache_read_tokens": read, "cache_write_tokens": write, "input_includes_cache": True}
        record["started"], record["timestamp"] = stamp(started), stamp(ended)
        self.records.append(Fed(record, aux="calls"))
        return self

    def event(self, kind, message, part, at=None):
        part = {"id": f"prt_{next(Feed.ids)}", "messageID": message, "sessionID": self.session, **part}
        self.records.append(Fed({"type": kind, "timestamp": at if at is not None else self.now(),
                                 "sessionID": self.session, "part": part}))
        return self

    def subagent(self, child):
        """The sub-agent session ``child``'s events, as the bridge feeds them before its task call's event."""
        self.records.append(Fed({"type": "agentenv.opencode_session", "session": child.session,
                                 "events": [fed.record for fed in child.records]}, aux="subagents"))
        return self

    def start(self, message):
        return self.event("step_start", message, {"type": "step-start"})

    def text(self, message, text):
        at = self.now()
        return self.event("text", message, {"type": "text", "text": text, "time": {"start": at - 2, "end": at - 1}},
                          at)

    def reasoning(self, message, text):
        at = self.now()
        return self.event("reasoning", message, {"type": "reasoning", "text": text,
                                                 "time": {"start": at - 2, "end": at - 1}}, at)

    def tool(self, message, call_id, tool="bash", *, arguments=None, output="first\n", status="completed", error=None,
             attachments=None, metadata=None, start=None, end=None):
        """A tool's part as the CLI prints it once the tool ends: its call and its result together."""
        start = start if start is not None else self.now()
        end = end if end is not None else self.now()
        state = {"status": status, "input": arguments or {"command": "echo first", "description": "echo"},
                 "time": {"start": start, "end": end}}
        if status == "completed":
            state.update({"output": output, "title": tool, "metadata": metadata or {"exit": 0, "truncated": False}})
            if attachments is not None:
                state["attachments"] = attachments
        else:
            state.update({"error": error, "metadata": metadata or {}})
        return self.event("tool_use", message, {"type": "tool", "callID": call_id, "tool": tool, "state": state},
                          max(end, self.now()))

    def finish(self, message, reason="stop", *, inputs=420, output=22, reasoning=0, read=600, write=0):
        """A step's end: its counts as 1.17.18 counts them, the cache and the reasoning apart."""
        return self.event("step_finish", message, {
            "type": "step-finish", "reason": reason, "cost": 0,
            "tokens": {"total": inputs + output + reasoning + read + write, "input": inputs, "output": output,
                       "reasoning": reasoning, "cache": {"write": write, "read": read}}})

    def error(self, message="bad request", name="APIError", status=400):
        self.records.append(Fed({"type": "error", "timestamp": self.now(), "sessionID": SESSION, "error": {
            "name": name, "data": {"message": message, "statusCode": status, "isRetryable": False,
                                   "responseHeaders": {"x-secret": "kept out"}, "responseBody": "{}"}}}))
        return self


def task_metadata(session=CHILD):
    return {"parentSessionId": SESSION, "sessionId": session, "model": {"modelID": MODEL, "providerID": "litellm"},
            "truncated": False}


def task_output(answer, session=CHILD):
    return f'<task id="{session}" state="completed">\n<task_result>\n{answer}\n</task_result>\n</task>'


def context(index=0):
    return TurnContext(index=index, context_id="ctx", task_id=f"t{index}", message_id=f"m{index}", inbound=["do it"],
                       agent_name="OpenCode", agent_version="1.0.0", started_at=f"2026-11-28T06:00:0{index}Z")


def run(*feeds: Feed, state="completed", finals=None):
    finals = finals or [None] * len(feeds)
    kit = check_adapter(OpenCodeAtif, [RecordedTask(context=context(i), records=feed.records, state=state,
                                                    final=None if final is None else [final])
                                       for i, (feed, final) in enumerate(zip(feeds, finals))])
    kit.raise_for_errors()
    assert not kit.findings, kit.findings
    return kit


def agent_steps(trajectory):
    return [step for step in trajectory["steps"] if step["source"] == "agent"]


def coding():
    """A first turn: the title call alongside, a bash call, then the answer."""
    feed = Feed().turn().call(1010, 21).call().start("msg_1").tool("msg_1", "call_1")
    feed.finish("msg_1", "tool-calls").call(1030, 23).start("msg_2").text("msg_2", "5050")
    return feed.finish("msg_2", inputs=430, output=23)


def test_each_model_call_is_a_step_with_its_model_time_and_the_relays_counts():
    kit = run(coding(), finals=["5050"])
    title, ran, answered = agent_steps(kit.document)

    assert [step["message"] for step in (title, ran, answered)] == ["", "", "5050"]
    assert all(step["model_name"] == MODEL and step["timestamp"] for step in (title, ran, answered))
    assert ran["metrics"] == {"prompt_tokens": 1020, "cached_tokens": 600, "completion_tokens": 22,
                              "extra": {"agentenv": {"cache_write_tokens": 100}}}
    assert title["metrics"]["prompt_tokens"] == 1010 and answered["metrics"]["prompt_tokens"] == 1030
    final = kit.document["final_metrics"]
    assert (final["total_prompt_tokens"], final["total_completion_tokens"], final["total_cached_tokens"]) == (
        3060, 66, 1800)


def test_the_answer_is_the_tasks_last_step_so_the_sdk_adds_none():
    title = Feed.now()
    feed = Feed().turn().call().start("msg_1").text("msg_1", "first part").text("msg_1", "\n").text("msg_1", "second")
    kit = run(feed.finish("msg_1").call(900, 5, started=title), finals=["first part\nsecond"])
    steps = agent_steps(kit.document)

    assert [step["message"] for step in steps] == ["", "first part\nsecond"]
    assert steps[-1]["llm_call_count"] == 1


def test_a_tool_call_and_its_result_are_in_the_step_that_made_it_and_an_error_is_the_results():
    feed = Feed().turn().call().start("msg_1").tool("msg_1", "call_1")
    feed.tool("msg_1", "call_2", status="error", error="Error: no such file", arguments={"filePath": "/nope"},
              tool="read")
    kit = run(feed.finish("msg_1", "tool-calls").call(1030, 23).start("msg_2").text("msg_2", "done").finish(
        "msg_2", inputs=430, output=23))
    step = agent_steps(kit.document)[0]

    assert step["tool_calls"] == [
        {"tool_call_id": "call_1", "function_name": "bash",
         "arguments": {"command": "echo first", "description": "echo"}},
        {"tool_call_id": "call_2", "function_name": "read", "arguments": {"filePath": "/nope"}}]
    ran, failed = step["observation"]["results"]
    assert ran == {"source_call_id": "call_1", "content": "first\n"}
    assert failed["content"] == "Error: no such file" and profile_of(failed)["status"] == "error"


def test_an_attached_image_is_a_media_file_after_the_results_text():
    image = png()
    url = "data:image/png;base64," + base64.b64encode(image).decode()
    attachment = {"type": "file", "mime": "image/png", "url": url, "id": "prt_a", "sessionID": SESSION,
                  "messageID": "msg_1"}
    feed = Feed().turn().call().start("msg_1").tool("msg_1", "call_1", "read", arguments={"filePath": "/app/shot.png"},
                                                    output="Image read successfully", attachments=[attachment])
    kit = run(feed.finish("msg_1", "tool-calls").call(1030, 23).start("msg_2").text("msg_2", "a square").finish(
        "msg_2", inputs=430, output=23))

    (result,) = agent_steps(kit.document)[0]["observation"]["results"]
    text, picture = result["content"]
    assert text == {"type": "text", "text": "Image read successfully"}
    assert picture["type"] == "image" and kit.media[picture["source"]["path"]] == image


def test_reasoning_the_cli_prints_is_its_steps_raw_reasoning_with_the_turns_effort():
    feed = Feed().turn(effort="high").call(1020, 22).start("msg_1").reasoning("msg_1", "Plan.").reasoning(
        "msg_1", "Run it.").text("msg_1", "ok")
    (step,) = agent_steps(run(feed.finish("msg_1", output=17, reasoning=5)).document)

    assert step["reasoning_content"] == "Plan.\n\nRun it." and step["reasoning_effort"] == "high"
    assert [block["visibility"] for block in profile_of(step)["reasoning"]] == ["raw", "raw"]
    assert step["metrics"]["completion_tokens"] == 22


def test_a_refused_attempt_is_an_error_step_before_the_step_its_retry_made():
    feed = Feed().turn().call(outcome="refused", status=503).call().start("msg_1").text("msg_1", "ok").finish("msg_1")
    refused, answered = agent_steps(run(feed).document)

    assert refused["llm_call_count"] == 0 and "metrics" not in refused
    assert profile_of(refused)["error"] == {
        "kind": "infra_error", "message": "the endpoint refused the call (HTTP 503)", "code": "503"}
    assert answered["message"] == "ok" and answered["metrics"]["prompt_tokens"] == 1020


def test_the_clis_error_is_the_error_of_the_call_the_endpoint_refused_last():
    feed = Feed().turn().call(outcome="refused", status=400).call(outcome="refused", status=400).error()
    kit = run(feed, state="failed", finals=["APIError: bad request"])
    title, failed, final = agent_steps(kit.document)

    assert profile_of(title)["error"]["message"] == "the endpoint refused the call (HTTP 400)"
    assert profile_of(failed)["error"] == {"kind": "infra_error", "message": "APIError: bad request", "code": "400"}
    assert "x-secret" not in json.dumps(kit.document)
    assert final["llm_call_count"] == 0 and profile_of(final)["state"] == "failed"


def test_an_error_that_cuts_a_step_short_is_that_steps():
    feed = Feed().turn().call().start("msg_1").error("stream broke", name="UnknownError", status=None)
    (step,) = agent_steps(run(feed, state="failed").document)

    assert profile_of(step)["error"] == {"kind": "infra_error", "message": "UnknownError: stream broke"}
    assert step["metrics"]["prompt_tokens"] == 1020


def test_a_task_call_is_a_sub_agent_with_a_step_per_call_inside_its_run_the_last_its_answer():
    feed = Feed().turn().call().start("msg_1")
    begun = Feed.now()
    feed.call(1100, 30).call(1200, 4)
    feed.tool("msg_1", "call_t", "task", output=task_output("19"), metadata=task_metadata(), start=begun,
              arguments={"description": "count", "prompt": "wc -l", "subagent_type": "general"})
    kit = run(feed.finish("msg_1", "tool-calls").call(1300, 8).start("msg_2").text("msg_2", "19").finish(
        "msg_2", inputs=700, output=8))
    spawned = agent_steps(kit.document)[0]
    (sub,) = kit.document["subagent_trajectories"]

    assert spawned["observation"]["results"][0]["subagent_trajectory_ref"] == [{"trajectory_id": CHILD}]
    assert sub["agent"] == {"name": "general", "version": "unknown", "model_name": MODEL}
    assert [(step["message"], step["metrics"]["prompt_tokens"]) for step in sub["steps"]] == [("", 1100), ("19", 1200)]
    assert kit.document["final_metrics"]["total_prompt_tokens"] == 1020 + 1100 + 1200 + 1300


def test_parallel_task_calls_whose_sessions_werent_fed_leave_their_calls_to_root_steps_of_their_own():
    feed = Feed().turn().call().start("msg_1")
    begun = Feed.now()
    feed.call(1100, 30).call(1200, 4)
    for call_id, session in (("call_a", CHILD), ("call_b", "ses_other")):
        feed.tool("msg_1", call_id, "task", arguments={"description": "d", "prompt": "p", "subagent_type": "general"},
                  output=task_output("child", session), metadata=task_metadata(session), start=begun)
    kit = run(feed.finish("msg_1", "tool-calls").call(1300, 8).start("msg_2").text("msg_2", "both").finish(
        "msg_2", inputs=700, output=8))
    steps = agent_steps(kit.document)

    assert "subagent_trajectories" not in kit.document
    assert [step["metrics"]["prompt_tokens"] for step in steps] == [1100, 1200, 1020, 1300]
    assert [step["message"] for step in steps] == ["", "", "", "both"]


def test_sub_agents_run_at_once_are_each_their_sessions_steps_each_with_the_call_its_counts_and_cache_read_name():
    """As recorded: two task calls of one step, whose first calls' prompts are as long, told apart by cache reads."""
    feed = Feed().turn().call(613, 8, read=0, write=0).call(9358, 188, read=0, write=9355).start("msg_1")
    one, two = Feed(CHILD), Feed("ses_two")
    begun = Feed.now(), Feed.now()
    feed.call(8404, 53, read=0, write=8401).call(8404, 53, read=8068, write=333)
    one.start("c1").tool("c1", "call_one", arguments={"command": "echo one"}, output="one\n")
    two.start("d1").tool("d1", "call_two", arguments={"command": "echo two"}, output="two\n")
    one.finish("c1", "tool-calls", inputs=8404, output=53, read=0)
    two.finish("d1", "tool-calls", inputs=336, output=53, read=8068)
    feed.call(8471, 13, read=8401, write=64).call(8471, 9, read=8401, write=64)
    one.start("c2").text("c2", "The output is one.").finish("c2", inputs=70, output=13, read=8401)
    two.start("d2").text("d2", "The output is two.").finish("d2", inputs=70, output=9, read=8401)
    end = Feed.now()
    for child, call_id, start, answer in ((one, "task_one", begun[0], "The output is one."),
                                          (two, "task_two", begun[1], "The output is two.")):
        feed.subagent(child).tool("msg_1", call_id, "task", output=task_output(answer, child.session),
                                  arguments={"description": "d", "prompt": "p", "subagent_type": "general"},
                                  metadata=task_metadata(child.session), start=start, end=end)
    feed.finish("msg_1", "tool-calls", inputs=9358, output=188, read=0)
    kit = run(feed.call(9709, 5, read=9355, write=347).start("msg_2").text("msg_2", "one two").finish(
        "msg_2", inputs=354, output=5, read=9355), finals=["one two"])
    subs = {sub["trajectory_id"]: sub for sub in kit.document["subagent_trajectories"]}

    assert [[(step["metrics"]["prompt_tokens"], step["metrics"]["cached_tokens"],
              [call["tool_call_id"] for call in step.get("tool_calls") or ()], step["message"]) for step in
             subs[session]["steps"]] for session in (CHILD, "ses_two")] == [
        [(8404, 0, ["call_one"], ""), (8471, 8401, [], "The output is one.")],
        [(8404, 8068, ["call_two"], ""), (8471, 8401, [], "The output is two.")]]
    assert subs[CHILD]["steps"][0]["observation"]["results"] == [{"source_call_id": "call_one", "content": "one\n"}]
    assert [step["metrics"]["prompt_tokens"] for step in agent_steps(kit.document)] == [613, 9358, 9709]
    assert kit.document["final_metrics"]["total_prompt_tokens"] == 613 + 9358 + 2 * 8404 + 2 * 8471 + 9709


def test_sub_agents_whose_calls_tie_on_every_count_the_cli_prints_are_told_apart_by_when_each_began():
    """Both first prompts as long, both answers too, neither cached: only the cache write, which the CLI counts in
    the input, differs. The other's call began first, but after this sub-agent's step began only this one's had."""
    feed = Feed().turn().call().start("msg_1")
    one, two = Feed(CHILD), Feed("ses_two")
    begun = Feed.now(), Feed.now()
    theirs = Feed.now()
    two.start("d1")
    ours = Feed.now()
    one.start("c1")
    two.text("d1", "two").finish("d1", inputs=8404, output=53, read=0)
    one.text("c1", "one").finish("c1", inputs=8404, output=53, read=0)
    feed.call(8404, 53, read=0, write=8401, started=ours).call(8404, 53, read=0, write=333, started=theirs)
    for child, call_id, start, answer in ((one, "task_one", begun[0], "one"), (two, "task_two", begun[1], "two")):
        feed.subagent(child).tool("msg_1", call_id, "task", output=task_output(answer, child.session), start=start,
                                  metadata=task_metadata(child.session),
                                  arguments={"description": "d", "prompt": "p", "subagent_type": "general"})
    kit = run(feed.finish("msg_1", "tool-calls").call(1300, 8).start("msg_2").text("msg_2", "ok").finish(
        "msg_2", inputs=700, output=8), finals=["ok"])
    subs = {sub["trajectory_id"]: sub for sub in kit.document["subagent_trajectories"]}

    assert [profile_of(subs[session]["steps"][0]["metrics"])["cache_write_tokens"] for session in (CHILD, "ses_two")] \
        == [8401, 333]


def test_a_fed_sub_agents_refused_attempt_is_its_error_step_before_the_step_its_retry_made():
    feed = Feed().turn().call().start("msg_1")
    begun, child = Feed.now(), Feed(CHILD)
    feed.call(outcome="refused", status=503).call(1100, 30, read=600, write=0)
    child.start("c1").tool("c1", "call_b").finish("c1", "tool-calls", inputs=500, output=30, read=600)
    feed.call(1200, 4, read=600, write=0)
    child.start("c2").text("c2", "19").finish("c2", inputs=600, output=4, read=600)
    feed.subagent(child).tool("msg_1", "call_t", "task", output=task_output("19"), metadata=task_metadata(),
                              start=begun, arguments={"description": "count", "prompt": "wc -l",
                                                      "subagent_type": "general"})
    kit = run(feed.finish("msg_1", "tool-calls").call(1300, 8).start("msg_2").text("msg_2", "19").finish(
        "msg_2", inputs=700, output=8), finals=["19"])
    (sub,) = kit.document["subagent_trajectories"]
    refused, ran, answered = sub["steps"]

    assert refused["llm_call_count"] == 0 and profile_of(refused)["error"]["code"] == "503"
    assert [call["tool_call_id"] for call in ran["tool_calls"]] == ["call_b"]
    assert ran["metrics"]["prompt_tokens"] == 1100
    assert answered["message"] == "19" and answered["metrics"]["prompt_tokens"] == 1200


def test_a_resumed_turn_writes_its_own_steps_and_a_resumed_sub_agent_is_one_of_its_own():
    def spawning(feed, call_id, inputs):
        feed.turn().call(inputs, 22).start(f"m_{call_id}")
        begun = Feed.now()
        feed.call(inputs + 1, 3).tool(f"m_{call_id}", call_id, "task", output=task_output("ok"),
                                      arguments={"description": "d", "prompt": "p", "subagent_type": "general",
                                                 "task_id": CHILD}, metadata=task_metadata(), start=begun)
        feed.finish(f"m_{call_id}", "tool-calls", inputs=inputs - 600).call(inputs + 2, 4)
        return feed.start(f"n_{call_id}").text(f"n_{call_id}", "ok").finish(f"n_{call_id}", inputs=inputs - 598,
                                                                            output=4)

    kit = run(spawning(Feed(), "call_1", 2000), spawning(Feed(), "call_2", 3000), finals=["ok", "ok"])

    assert [sub["trajectory_id"] for sub in kit.document["subagent_trajectories"]] == [CHILD, f"{CHILD}:call_2"]
    assert [step["message"] for step in agent_steps(kit.document)] == ["", "ok", "", "ok"]


def test_a_call_record_that_comes_after_its_step_finished_still_makes_that_step():
    started, ended = Feed.now(), Feed.now()
    feed = Feed().turn().start("msg_1").tool("msg_1", "call_1").finish("msg_1", "tool-calls")
    feed.call(started=started, ended=ended).call(1030, 23).start("msg_2").text("msg_2", "done")
    feed.finish("msg_2", inputs=430, output=23)
    steps = agent_steps(run(feed, finals=["done"]).document)

    assert [step["metrics"]["prompt_tokens"] for step in steps] == [1020, 1030]


def test_a_title_call_that_ends_last_comes_before_the_answer():
    title = Feed.now()
    feed = Feed().turn().call().start("msg_1").text("msg_1", "ok").finish("msg_1").call(1010, 21, started=title)
    steps = agent_steps(run(feed, finals=["ok"]).document)

    assert [(step["message"], step["metrics"]["prompt_tokens"]) for step in steps] == [("", 1010), ("ok", 1020)]


def test_a_step_whose_counts_no_call_shares_takes_the_latest_call_to_start_before_its_tools():
    feed = Feed().turn().call(1010, 21).call(5000, 50).start("msg_1").tool("msg_1", "call_1")
    feed.finish("msg_1", "tool-calls", inputs=1).call(1030, 23).start("msg_2").text("msg_2", "ok")
    steps = agent_steps(run(feed.finish("msg_2", inputs=2), finals=["ok"]).document)

    assert [(bool(step.get("tool_calls")), step["metrics"]["prompt_tokens"]) for step in steps] == [
        (False, 1010), (True, 5000), (False, 1030)]


def test_a_step_no_call_was_recorded_for_keeps_its_own_counts_the_cache_added_to_its_input():
    feed = Feed().start("msg_1").text("msg_1", "ok").finish("msg_1", inputs=420, output=17, reasoning=5, read=600)
    (step,) = agent_steps(run(feed).document)

    assert step["metrics"] == {"prompt_tokens": 1020, "cached_tokens": 600, "completion_tokens": 22,
                               "extra": {"agentenv": {"cache_write_tokens": 0}}}


def test_a_step_the_cli_was_killed_in_takes_the_call_it_was_making():
    feed = Feed().turn().call(1010, 21).call(4000, 9).start("msg_1").text("msg_1", "partial")
    steps = agent_steps(run(feed, state="failed").document)

    assert [(step["message"], step["metrics"]["prompt_tokens"]) for step in steps] == [("", 1010), ("partial", 4000)]


def test_a_step_the_cli_was_killed_in_while_its_sub_agent_ran_takes_the_call_that_began_it():
    """The task call was never printed, so the sub-agent's later call is no task call's: it isn't the step's either."""
    begun = Feed.now()
    feed = Feed().turn().start("msg_1").text("msg_1", "looking").call(4000, 9, started=begun).call(1100, 30)
    steps = agent_steps(run(feed, state="failed").document)

    assert [(step["message"], step["metrics"]["prompt_tokens"]) for step in steps] == [("", 1100), ("looking", 4000)]


def test_a_call_answered_without_usage_leaves_its_step_without_counts_not_the_zeros_the_cli_prints():
    feed = Feed().turn().call(outcome="unreported").start("msg_1").text("msg_1", "ok")
    document = run(feed.finish("msg_1", inputs=0, output=0, read=0), finals=["ok"]).document
    (step,) = agent_steps(document)

    assert step["llm_call_count"] == 1 and "metrics" not in step
    assert "total_prompt_tokens" not in document["final_metrics"]


@pytest.mark.parametrize("other_first", [True, False], ids=["other-began-first", "other-began-while-it-waited"])
def test_a_sub_agents_step_cut_short_takes_its_own_call_not_one_another_sub_agent_made_at_once(other_first):
    """One sub-agent's stream breaks mid-step while another's call runs, begun before the broken call or while the
    broken step waited for its stream: the other's step claims that call by its counts, the broken step its own."""
    feed = Feed().turn().call().start("msg_1")
    one, two = Feed(CHILD), Feed("ses_two")
    begun = Feed.now(), Feed.now()
    feed.call(5000, 40, read=0, write=0)
    one.start("c1").tool("c1", "call_one").finish("c1", "tool-calls", inputs=5000, output=40, read=0)
    broken, other = (Feed.now(), Feed.now())[::-1 if other_first else 1]
    one.start("c2").text("c2", "partial")
    feed.call(outcome="unreported", started=broken)
    two.start("d1").text("d1", "two").finish("d1", inputs=7000, output=50, read=0)
    feed.call(7000, 50, read=0, write=0, started=other)
    feed.subagent(one).tool("msg_1", "task_one", "task", status="error", error="Tool execution aborted",
                            metadata=task_metadata(), start=begun[0],
                            arguments={"description": "d", "prompt": "p", "subagent_type": "general"})
    feed.subagent(two).tool("msg_1", "task_two", "task", output=task_output("two", "ses_two"), start=begun[1],
                            metadata=task_metadata("ses_two"),
                            arguments={"description": "d", "prompt": "p", "subagent_type": "general"})
    feed.finish("msg_1", "tool-calls").call(1300, 8).start("msg_2").text("msg_2", "two").finish("msg_2", inputs=700,
                                                                                                output=8)
    kit = run(feed, finals=["two"])
    subs = {sub["trajectory_id"]: sub for sub in kit.document["subagent_trajectories"]}

    assert [(step["message"], (step.get("metrics") or {}).get("prompt_tokens")) for step in subs[CHILD]["steps"]] == [
        ("", 5000), ("partial", None)]
    assert [(step["message"], step["metrics"]["prompt_tokens"]) for step in subs["ses_two"]["steps"]] == [("two", 7000)]
    assert [step["metrics"]["prompt_tokens"] for step in agent_steps(kit.document)] == [1020, 1300]


def test_a_task_call_whose_part_names_no_session_leaves_the_calls_inside_its_run_to_the_root():
    feed = Feed().turn().call().start("msg_1")
    begun = Feed.now()
    feed.call(1100, 30).tool("msg_1", "call_t", "task", status="error", error="Tool execution aborted", start=begun,
                             arguments={"description": "d", "prompt": "p", "subagent_type": "general"})
    kit = run(feed.finish("msg_1", "tool-calls").call(1300, 8).start("msg_2").text("msg_2", "ok").finish(
        "msg_2", inputs=700, output=8), finals=["ok"])

    assert "subagent_trajectories" not in kit.document
    assert [step["metrics"]["prompt_tokens"] for step in agent_steps(kit.document)] == [1100, 1020, 1300]


@pytest.mark.parametrize("other_first", [True, False], ids=["other-began-first", "other-began-while-it-waited"])
def test_a_sub_agents_step_whose_call_had_no_usage_takes_a_call_answered_without_one(other_first):
    """Its counts are the CLI's zeros: the call that made it is one the relay saw answered without usage, not another
    sub-agent's answered call begun near it, whose session wasn't fed."""
    feed = Feed().turn().call().start("msg_1")
    one = Feed(CHILD)
    begun = Feed.now(), Feed.now()
    own, other = (Feed.now(), Feed.now())[::-1 if other_first else 1]
    one.start("c1").text("c1", "one").finish("c1", inputs=0, output=0, read=0)
    feed.call(outcome="unreported", started=own).call(7000, 50, read=0, write=0, started=other)
    feed.subagent(one).tool("msg_1", "task_one", "task", output=task_output("one"), metadata=task_metadata(),
                            start=begun[0], arguments={"description": "d", "prompt": "p", "subagent_type": "general"})
    feed.tool("msg_1", "task_two", "task", output=task_output("two", "ses_two"), metadata=task_metadata("ses_two"),
              start=begun[1], arguments={"description": "d", "prompt": "p", "subagent_type": "general"})
    kit = run(feed.finish("msg_1", "tool-calls").call(1300, 8).start("msg_2").text("msg_2", "ok").finish(
        "msg_2", inputs=700, output=8), finals=["ok"])
    (sub,) = kit.document["subagent_trajectories"]

    assert [(step["message"], "metrics" in step) for step in sub["steps"]] == [("one", False)]
    assert [step["metrics"]["prompt_tokens"] for step in agent_steps(kit.document)] == [7000, 1020, 1300]


def test_sub_agents_run_at_once_on_an_endpoint_that_sends_no_usage_each_take_the_calls_that_began_their_steps():
    """No counts to join by: each sub-agent step takes the latest call to begin before it did, so none is left
    without a call and none of a sub-agent's calls is the root's."""
    feed = Feed().turn().call(outcome="unreported").start("msg_1")
    one, two = Feed(CHILD), Feed("ses_two")
    begun = Feed.now(), Feed.now()
    first = Feed.now()
    one.start("c1").tool("c1", "call_one").finish("c1", "tool-calls", inputs=0, output=0, read=0)
    feed.call(outcome="unreported", started=first)
    other = Feed.now()
    two.start("d1").text("d1", "two").finish("d1", inputs=0, output=0, read=0)
    feed.call(outcome="unreported", started=other)
    second = Feed.now()
    one.start("c2").text("c2", "one").finish("c2", inputs=0, output=0, read=0)
    feed.call(outcome="unreported", started=second)
    for child, call_id, start, answer in ((one, "task_one", begun[0], "one"), (two, "task_two", begun[1], "two")):
        feed.subagent(child).tool("msg_1", call_id, "task", output=task_output(answer, child.session), start=start,
                                  metadata=task_metadata(child.session),
                                  arguments={"description": "d", "prompt": "p", "subagent_type": "general"})
    feed.finish("msg_1", "tool-calls", inputs=0, output=0, read=0).call(outcome="unreported").start("msg_2")
    kit = run(feed.text("msg_2", "one two").finish("msg_2", inputs=0, output=0, read=0), finals=["one two"])
    subs = {sub["trajectory_id"]: sub for sub in kit.document["subagent_trajectories"]}
    calls = [record for record in feed.records if record.aux == "calls"]

    assert [step["message"] for step in agent_steps(kit.document)] == ["", "one two"]
    assert [step["message"] for step in subs[CHILD]["steps"]] == ["", "one"]
    assert [step["message"] for step in subs["ses_two"]["steps"]] == ["two"]
    assert sum(step["llm_call_count"] for trajectory in [kit.document, *subs.values()]
               for step in agent_steps(trajectory)) == len(calls) == 5


def test_with_no_usage_a_call_only_one_sub_agents_step_could_have_made_is_that_steps():
    """The second sub-agent's only call began while the first's first step waited for its stream, so it is also the
    latest to begin before that step; but only the second's step can have made it, the first's step its own call."""
    feed = Feed().turn().call(outcome="unreported").start("msg_1")
    one, two = Feed(CHILD), Feed("ses_two")
    begun_one, first, begun_two, other = Feed.now(), Feed.now(), Feed.now(), Feed.now()
    one.start("c1").tool("c1", "call_one").finish("c1", "tool-calls", inputs=0, output=0, read=0)
    feed.call(outcome="unreported", started=first)
    two.start("d1").text("d1", "two").finish("d1", inputs=0, output=0, read=0)
    feed.call(outcome="unreported", started=other)
    second = Feed.now()
    one.start("c2").text("c2", "one").finish("c2", inputs=0, output=0, read=0)
    feed.call(outcome="unreported", started=second)
    for child, call_id, start, answer in ((one, "task_one", begun_one, "one"), (two, "task_two", begun_two, "two")):
        feed.subagent(child).tool("msg_1", call_id, "task", output=task_output(answer, child.session), start=start,
                                  metadata=task_metadata(child.session),
                                  arguments={"description": "d", "prompt": "p", "subagent_type": "general"})
    feed.finish("msg_1", "tool-calls", inputs=0, output=0, read=0).call(outcome="unreported").start("msg_2")
    kit = run(feed.text("msg_2", "one two").finish("msg_2", inputs=0, output=0, read=0), finals=["one two"])
    subs = {sub["trajectory_id"]: sub for sub in kit.document["subagent_trajectories"]}
    calls = [record for record in feed.records if record.aux == "calls"]

    assert [step["message"] for step in agent_steps(kit.document)] == ["", "one two"]
    assert sum(step["llm_call_count"] for trajectory in [kit.document, *subs.values()]
               for step in agent_steps(trajectory)) == len(calls) == 5


def test_with_no_usage_three_sub_agents_at_once_each_step_takes_a_call_when_every_step_can_have_one():
    """Each of the first steps of three sub-agents could have made either of two calls, one of the second's and the
    third's only after the first took its own: every step still gets a call and no call is left to the root."""
    feed = Feed().turn().call(outcome="unreported").start("msg_1")
    one, two, three = Feed(CHILD), Feed("ses_two"), Feed("ses_three")
    begun_one, own, begun_two, begun_three, shared, last = (Feed.now() for _ in range(6))
    one.start("a1").text("a1", "one").finish("a1", inputs=0, output=0, read=0)
    three.start("c1").tool("c1", "call_three").finish("c1", "tool-calls", inputs=0, output=0, read=0)
    two.start("b1").text("b1", "two").finish("b1", inputs=0, output=0, read=0)
    feed.call(outcome="unreported", started=own).call(outcome="unreported", started=last)
    arguments = {"description": "d", "prompt": "p", "subagent_type": "general"}
    feed.subagent(one).tool("msg_1", "task_one", "task", output=task_output("one"), metadata=task_metadata(),
                            start=begun_one, arguments=arguments)
    feed.call(outcome="unreported", started=shared)
    feed.subagent(two).tool("msg_1", "task_two", "task", output=task_output("two", "ses_two"), start=begun_two,
                            metadata=task_metadata("ses_two"), arguments=arguments)
    later = Feed.now()
    three.start("c2").text("c2", "three").finish("c2", inputs=0, output=0, read=0)
    feed.call(outcome="unreported", started=later)
    feed.subagent(three).tool("msg_1", "task_three", "task", output=task_output("three", "ses_three"),
                              start=begun_three, metadata=task_metadata("ses_three"), arguments=arguments)
    feed.finish("msg_1", "tool-calls", inputs=0, output=0, read=0).call(outcome="unreported").start("msg_2")
    kit = run(feed.text("msg_2", "done").finish("msg_2", inputs=0, output=0, read=0), finals=["done"])
    subs = kit.document["subagent_trajectories"]
    calls = [record for record in feed.records if record.aux == "calls"]

    assert [step["message"] for step in agent_steps(kit.document)] == ["", "done"]
    assert [[step["message"] for step in sub["steps"]] for sub in subs] == [["one"], ["two"], ["", "three"]]
    assert sum(step["llm_call_count"] for trajectory in [kit.document, *subs]
               for step in agent_steps(trajectory)) == len(calls) == 6


@pytest.mark.parametrize("other_first", [True, False], ids=["other-began-first", "other-began-while-it-waited"])
def test_with_no_usage_a_sub_agents_step_takes_a_call_on_its_own_model(other_first):
    """Two sub-agents at once on other models, on an endpoint that sends no usage: a call naming the other's model
    isn't this one's, whenever it began."""
    sonnet = "anthropic/claude-sonnet-5-5"
    feed = Feed().turn().call(outcome="unreported").start("msg_1")
    one, two = Feed(CHILD), Feed("ses_two")
    begun = Feed.now(), Feed.now()
    own, other = (Feed.now(), Feed.now())[::-1 if other_first else 1]
    one.start("c1").text("c1", "one").finish("c1", inputs=0, output=0, read=0)
    two.start("d1").text("d1", "two").finish("d1", inputs=0, output=0, read=0)
    feed.call(outcome="unreported", started=own).call(outcome="unreported", started=other, model=sonnet)
    arguments = {"description": "d", "prompt": "p", "subagent_type": "general"}
    feed.subagent(one).tool("msg_1", "task_one", "task", output=task_output("one"), metadata=task_metadata(),
                            start=begun[0], arguments=arguments)
    on_sonnet = {**task_metadata("ses_two"), "model": {"modelID": sonnet, "providerID": "litellm"}}
    feed.subagent(two).tool("msg_1", "task_two", "task", output=task_output("two", "ses_two"), start=begun[1],
                            metadata=on_sonnet, arguments=arguments)
    feed.finish("msg_1", "tool-calls", inputs=0, output=0, read=0).call(outcome="unreported").start("msg_2")
    kit = run(feed.text("msg_2", "done").finish("msg_2", inputs=0, output=0, read=0), finals=["done"])

    assert [(step["message"], step["model_name"]) for sub in kit.document["subagent_trajectories"]
            for step in sub["steps"]] == [("one", MODEL), ("two", sonnet)]


def test_a_step_takes_the_turns_model_when_no_call_names_one():
    feed = Feed().turn(model="anthropic/claude-sonnet-5-5").start("msg_1").text("msg_1", "ok").finish("msg_1")
    (step,) = agent_steps(run(feed).document)

    assert step["model_name"] == "anthropic/claude-sonnet-5-5"


def test_from_atif_gives_the_cli_events_back_which_the_adapter_reads_as_the_same_steps():
    document = run(coding(), finals=["5050"]).document
    log = OpenCodeAtif.from_atif(strip_profile(document)).log
    feed = Feed()
    feed.records = [Fed(event) for event in log]
    again = run(feed)

    assert [event["type"] for event in log] == ["step_start", "step_finish", "step_start", "tool_use", "step_finish",
                                                "step_start", "text", "step_finish"]
    assert [(step["message"], step["metrics"]["prompt_tokens"]) for step in agent_steps(again.document)] == [
        (step["message"], step["metrics"]["prompt_tokens"]) for step in agent_steps(document)]


def test_the_tested_cli_is_the_one_the_image_pins():
    pinned = json.loads((AGENT_DIR / "package.json").read_text())["dependencies"]["opencode-ai"]
    assert opencode_atif.TESTED_WITH == (pinned,)


class _Session:
    """A random session, as the CLI and the relay give it: each record with when the bridge is handed it, and what
    the copy must hold."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.clock = 1_795_846_800_000
        self.ids = itertools.count()
        self.timeline: list[tuple[float, int, Fed]] = []
        self.want = {"input": 0, "output": 0, "cached": 0, "model_steps": 0, "refused": 0, "calls": set(),
                     "subagents": 0, "roots": [], "answer_alone": []}

    def tick(self, low=1, high=40) -> int:
        self.clock += self.rng.randint(low, high)
        return self.clock

    def put(self, at: float, fed: Fed) -> None:
        self.timeline.append((at, next(self.ids), fed))

    def relay(self, started: int, ended: int, outcome="answered", status=200) -> dict:
        record = {"type": "agentenv.model_call", "via": "relay", "model": MODEL, "status": status, "outcome": outcome,
                  "started": stamp(started), "timestamp": stamp(ended)}
        if outcome == "answered":
            inputs, output = self.rng.randint(2000, 90000), self.rng.randint(1, 900)
            read = self.rng.randint(0, inputs // 2)
            record["usage"] = {"input_tokens": inputs, "output_tokens": output, "cache_read_tokens": read,
                               "cache_write_tokens": 0, "input_includes_cache": True}
            for name, count in (("input", inputs), ("output", output), ("cached", read), ("model_steps", 1)):
                self.want[name] += count
        else:
            self.want["refused"] += 1
        self.put(ended, Fed(record, aux="calls"))
        return record

    def event(self, at: int, kind: str, message: str, part: dict) -> None:
        self.put(at, Fed({"type": kind, "timestamp": at, "sessionID": SESSION,
                          "part": {"messageID": message, "sessionID": SESSION, **part}}))

    def tool_use(self, at: int, message: str, tool: str, state: dict) -> None:
        call_id = f"call_{next(self.ids)}"
        self.want["calls"].add(call_id)
        self.event(at, "tool_use", message, {"type": "tool", "callID": call_id, "tool": tool, "state": state})

    def child_step(self, session: str, call: dict, ended: int, last: bool) -> list[dict]:
        """A sub-agent's step as the bridge reads it from the session store: maybe a tool call, its answer last, and
        its call's counts."""
        message, usage = f"msg_{next(self.ids)}", call["usage"]
        parts = [("step_start", {"type": "step-start"})]
        if not last and self.rng.random() < 0.6:
            call_id = f"call_{next(self.ids)}"
            self.want["calls"].add(call_id)
            parts.append(("tool_use", {"type": "tool", "callID": call_id, "tool": "bash", "state": {
                "status": "completed", "input": {"command": "ls"}, "output": "out",
                "time": {"start": ended + 1, "end": ended + 2}}}))
        if last:
            parts.append(("text", {"type": "text", "text": "done"}))
        parts.append(("step_finish", {"type": "step-finish", "reason": "stop" if last else "tool-calls", "cost": 0,
                                      "tokens": {"input": usage["input_tokens"] - usage["cache_read_tokens"],
                                                 "output": usage["output_tokens"], "reasoning": 0,
                                                 "cache": {"read": usage["cache_read_tokens"], "write": 0}}}))
        return [{"type": kind, "timestamp": ended + 1, "sessionID": session,
                 "part": {"messageID": message, "sessionID": session, **part}} for kind, part in parts]

    def tasks(self, message: str, children: int) -> None:
        """Task calls run at once, each a sub-agent making its calls one after another, a refused one before
        some; their sessions fed or not."""
        begun, ends, sessions = self.tick(0, 3), [], []
        fed = self.rng.random() < 0.6
        for _ in range(children):
            session, events, at = f"ses_{next(self.ids)}", [], begun + 1
            count = self.rng.randint(1, 3)
            for number in range(count):
                if self.rng.random() < 0.15:
                    self.relay(at, at + 1, "refused", 503)
                    at += 2
                ends.append(at + self.rng.randint(5, 200))
                call = self.relay(at, ends[-1])
                if fed:
                    events += self.child_step(session, call, ends[-1], number == count - 1)
                at = ends[-1] + 1
            sessions.append((session, events))
        end = max(ends) + 1
        for child, (session, events) in enumerate(sessions):
            if fed:
                self.put(end + child - 0.5, Fed({"type": "agentenv.opencode_session", "session": session,
                                                 "events": events}, aux="subagents"))
            self.tool_use(end + child, message, "task", {
                "status": "completed", "input": {"description": "d", "prompt": "p", "subagent_type": "general"},
                "output": task_output("done", session), "metadata": task_metadata(session),
                "time": {"start": begun, "end": end}})
        self.clock = end + children
        self.want["subagents"] += children if fed else children == 1

    def tool(self, message: str, kind: str, image: str) -> None:
        begun, end = self.tick(0, 3), self.tick()
        state = {"status": "error", "input": {}, "error": "failed"} if kind == "error" else {
            "status": "completed", "input": {"command": "ls"}, "output": "out",
            **({"attachments": [{"type": "file", "mime": "image/png", "url": image}]} if kind == "image" else {})}
        self.tool_use(end, message, "bash", {**state, "time": {"start": begun, "end": end}})

    def step(self, last: bool, texts: list[str], image: str) -> None:
        """A root step: maybe a refused attempt, its call, then its events; its counts sometimes not its call's, and its
        call's record sometimes handed over only after it finished."""
        message, rng = f"msg_{next(self.ids)}", self.rng
        if rng.random() < 0.2:
            started = self.tick()
            self.relay(started, started + 1, "refused", rng.choice([429, 500, 503]))
            self.tick(100, 300)
        started = self.tick()
        main_end = started + rng.randint(50, 400)
        main = self.relay(started, main_end)
        usage = main["usage"]
        self.want["roots"].append(usage["input_tokens"])
        self.clock = started + rng.randint(1, 400)
        self.event(self.clock, "step_start", message, {"type": "step-start"})
        if rng.random() < 0.3:
            self.event(self.tick(), "reasoning", message, {"type": "reasoning", "text": "Plan."})
        if rng.random() < 0.5 or last:
            texts.append(rng.choice(["ok", "looking", "5050"]))
            self.event(self.tick(), "text", message, {"type": "text", "text": texts[-1]})
        for kind in [] if last else rng.choices(["bash", "image", "error", "task", "parallel"], k=rng.randint(1, 2)):
            if kind in ("task", "parallel"):
                self.tasks(message, 1 if kind == "task" else 2)
            else:
                self.tool(message, kind, image)
        finished = self.clock = max(self.clock, main_end) + 1
        own = rng.random() < 0.85
        tokens = {"input": usage["input_tokens"] - usage["cache_read_tokens"] if own else 1,
                  "output": usage["output_tokens"], "reasoning": 0,
                  "cache": {"read": usage["cache_read_tokens"], "write": 0}}
        self.event(finished, "step_finish", message, {"type": "step-finish", "reason": "stop" if last else "tool-calls",
                                                      "tokens": tokens, "cost": 0})
        if rng.random() < 0.3:
            moved = next(item for item in self.timeline if item[2].record is main)
            self.timeline.remove(moved)
            self.timeline.append((finished + 0.5, moved[1], moved[2]))

    def task(self, first: bool, image: str) -> tuple[Feed, str]:
        """One task: the turn's settings, the first task's title call ending anywhere, its steps; and its answer."""
        self.timeline = []
        self.put(self.tick(), Fed({"type": "agentenv.turn", "model": MODEL, "effort": None}, aux="turn"))
        if first:
            started = self.tick()
            self.relay(started, started + self.rng.randint(10, 3000))
        texts: list[str] = []
        count = self.rng.randint(1, 4)
        for number in range(count):
            self.step(number == count - 1, texts, image)
        feed = Feed()
        feed.records = [fed for _, _, fed in sorted(self.timeline, key=lambda item: (item[0], item[1]))]
        self.want["answer_alone"].append(len(texts) == 1)
        return feed, "\n".join(texts)


def seeded_session(seed: int) -> tuple[list[Feed], list[str], dict]:
    """A random session of one to three tasks: steps of reasoning, text and calls, some returning images or errors,
    some running a sub-agent or two at once, their sessions fed or not; with each task's answer and what the copy must
    hold."""
    session = _Session(seed)
    image = "data:image/png;base64," + base64.b64encode(png()).decode()
    tasks = [session.task(index == 0, image) for index in range(session.rng.randint(1, 3))]
    return [feed for feed, _ in tasks], [answer for _, answer in tasks], session.want


@pytest.mark.parametrize("seed", range(150))
def test_a_seeded_sessions_copy_holds_every_call_the_relay_forwarded_and_every_tool_call(seed):
    feeds, finals, want = seeded_session(seed)
    document = run(*feeds, finals=finals).document
    steps = [step for trajectory in [document, *(document.get("subagent_trajectories") or [])]
             for step in agent_steps(trajectory)]
    model = [step for step in steps if step.get("llm_call_count") != 0]
    final = document["final_metrics"]

    assert (final["total_prompt_tokens"], final["total_completion_tokens"], final["total_cached_tokens"]) == (
        want["input"], want["output"], want["cached"])
    assert len(model) == want["model_steps"]
    assert sum("error" in profile_of(step) for step in steps) == want["refused"]
    assert len(document.get("subagent_trajectories") or []) == want["subagents"]
    roots = [step["metrics"]["prompt_tokens"] for step in agent_steps(document)
             if step.get("llm_call_count") != 0 and (step["message"] or step.get("tool_calls"))]
    assert roots == want["roots"]
    calls = [call["tool_call_id"] for step in steps for call in step.get("tool_calls") or ()]
    results = {result["source_call_id"] for step in steps for result in (step.get("observation") or {}).get(
        "results") or ()}
    assert sorted(calls) == sorted(want["calls"]) and set(calls) <= results
    tasks: list[list[dict]] = []
    for step in document["steps"]:
        tasks.append([]) if step["source"] == "user" else tasks[-1].append(step)
    for steps_of_task, alone in zip(tasks, want["answer_alone"]):
        assert not alone or steps_of_task[-1]["llm_call_count"] == 1


@asynccontextmanager
async def _relay_that_forwards_one_call(seen, upstream, calls, **kwargs):
    seen["calls"] = calls
    calls.add(200, Usage(input_tokens=1020, output_tokens=22, cache_read_tokens=600, input_includes_cache=True), MODEL,
              started=datetime(2026, 11, 28, 6, 20, 5, 161000, tzinfo=timezone.utc))
    yield upstream


@pytest.fixture
def bridge(monkeypatch, tmp_path):
    """run_opencode with a relay that forwards one call, a CLI that prints one step, or the events ``seen["events"]``
    names, and contexts' data dirs under ``tmp_path``; what each saw."""
    seen: dict = {"events": Feed().start("msg_1").text("msg_1", "ok").finish("msg_1").records}
    monkeypatch.setattr(opencode_bridge, "model_relay",
                        lambda upstream, calls, **kwargs: _relay_that_forwards_one_call(seen, upstream, calls))

    async def spawn(*cmd, **kwargs):
        seen["cmd"] = list(cmd)
        return FakeCLI("".join(json.dumps(fed.record) + "\n" for fed in seen["events"]).encode())

    monkeypatch.setattr(opencode_bridge.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(opencode_bridge, "_context_xdg", lambda _context: {})
    monkeypatch.setattr(opencode_bridge, "SESSIONS_ROOT", str(tmp_path))
    return seen


def session_store(context_id, messages):
    """A context's session store with the tables opencode-ai 1.17.18 keeps messages and parts in, holding
    ``messages``: each its id, session, creation time, role and parts, each part its id, update time and data."""
    path = opencode_bridge.context_data_dir(context_id) / opencode_bridge.SESSION_STORE
    path.parent.mkdir(parents=True)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE `message` (`id` text PRIMARY KEY, `session_id` text NOT NULL, `time_created` integer "
                   "NOT NULL, `time_updated` integer NOT NULL, `data` text NOT NULL)")
        db.execute("CREATE TABLE `part` (`id` text PRIMARY KEY, `message_id` text NOT NULL, `session_id` text NOT "
                   "NULL, `time_created` integer NOT NULL, `time_updated` integer NOT NULL, `data` text NOT NULL)")
        for message, session, created, role, parts in messages:
            db.execute("INSERT INTO message VALUES (?, ?, ?, ?, ?)",
                       (message, session, created, created, json.dumps({"role": role, "time": {"created": created}})))
            for part, updated, data in parts:
                db.execute("INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
                           (part, message, session, updated, updated, json.dumps(data)))
    return path


def test_the_bridge_feeds_the_turn_and_each_relayed_call_on_real_time_and_the_cli_prints_reasoning(bridge):
    fed = []
    turned = asyncio.run(opencode_bridge.run_opencode("go", "ctx", model=MODEL, effort="high", web_tools=True,
                                                      aux_callback=lambda name, record: fed.append((name, record))))

    assert fed[0] == ("turn", {"type": "agentenv.turn", "model": MODEL, "effort": "high"})
    ((name, record),) = fed[1:]
    assert name == "calls" and record["outcome"] == "answered" and record["started"] == "2026-11-28T06:20:05.161Z"
    assert record["usage"]["input_tokens"] == 1020 == turned.usage.input_tokens
    assert bridge["calls"].clock is model_relay.utc_now and "--thinking" in bridge["cmd"]


def test_without_an_aux_callback_the_relay_hands_its_calls_to_no_one_and_the_cli_prints_no_reasoning(bridge):
    turned = asyncio.run(opencode_bridge.run_opencode("go", "ctx", model=MODEL, web_tools=True))

    assert bridge["calls"].sink is None and "--thinking" not in bridge["cmd"]
    assert turned.response == "ok" and turned.usage.input_tokens == 1020


def test_the_bridge_feeds_an_ended_task_calls_sub_agent_session_from_the_session_store_before_its_event(bridge):
    begun = Feed.now()
    tool = {"type": "tool", "tool": "bash", "callID": "call_wc", "state": {
        "status": "completed", "input": {"command": "wc -l /etc/passwd"}, "output": "21 /etc/passwd\n",
        "metadata": {"exit": 0}, "time": {"start": begun + 30, "end": begun + 35}}}
    finished = {"type": "step-finish", "reason": "tool-calls", "cost": 0,
                "tokens": {"input": 8430, "output": 59, "reasoning": 0, "cache": {"read": 0, "write": 0}}}
    session_store("ctx", [
        ("msg_earlier", CHILD, begun - 9000, "assistant", [("prt_0", begun - 8000, {"type": "text", "text": "old"})]),
        ("msg_root", SESSION, begun - 100, "assistant", [("prt_1", begun - 90, {"type": "text", "text": "root's"})]),
        ("msg_prompt", CHILD, begun + 9, "user", [("prt_2", begun + 9, {"type": "text", "text": "Run wc -l"})]),
        ("msg_c1", CHILD, begun + 20, "assistant", [("prt_3", begun + 21, {"type": "step-start"}),
                                                    ("prt_4", begun + 36, tool), ("prt_5", begun + 40, finished),
                                                    ("prt_6", begun + 41, {"type": "patch", "files": []})])])
    printed = Feed().start("msg_1").tool("msg_1", "call_t", "task", output=task_output("21"), metadata=task_metadata(),
                                         start=begun, arguments={"prompt": "Run wc -l", "subagent_type": "general"})
    bridge["events"] = printed.finish("msg_1", "tool-calls").start("msg_2").text("msg_2", "21").finish("msg_2").records
    order = []
    asyncio.run(opencode_bridge.run_opencode(
        "go", "ctx", model=MODEL, web_tools=True, aux_callback=lambda name, record: order.append((name, record)),
        event_callback=lambda event: order.append((event["type"], event))))

    assert [name for name, _ in order] == ["turn", "calls", "step_start", "subagents", "tool_use", "step_finish",
                                           "step_start", "text", "step_finish"]
    (session,) = [record for name, record in order if name == "subagents"]
    assert session == {"type": "agentenv.opencode_session", "session": CHILD, "events": [
        {"type": kind, "timestamp": at, "sessionID": CHILD,
         "part": {**part, "id": part_id, "messageID": "msg_c1", "sessionID": CHILD}}
        for kind, at, part_id, part in (("step_start", begun + 21, "prt_3", {"type": "step-start"}),
                                        ("tool_use", begun + 36, "prt_4", tool),
                                        ("step_finish", begun + 40, "prt_5", finished))]}


def test_a_session_store_the_bridge_cant_read_feeds_no_session_and_the_turn_goes_on(bridge, caplog):
    printed = Feed().start("msg_1").tool("msg_1", "call_t", "task", output=task_output("21"), metadata=task_metadata())
    bridge["events"] = printed.finish("msg_1", "tool-calls").start("msg_2").text("msg_2", "21").finish("msg_2").records
    fed, events = [], []
    turned = asyncio.run(opencode_bridge.run_opencode("go", "ctx", model=MODEL, web_tools=True,
                                                      aux_callback=lambda name, record: fed.append(name),
                                                      event_callback=events.append))

    assert fed == ["turn", "calls"] and len(events) == 6 and turned.response == "21" and turned.error is None
    assert "could not read sub-agent session" in caplog.text


@pytest.fixture
def served(monkeypatch):
    """The agent's app with a turn that prints coding()'s events and feeds its relay's calls, if it's handed where to;
    and the keywords each turn got."""
    turns = []
    recorded = coding()

    async def run_opencode(**kwargs):
        turns.append(kwargs)
        for fed in recorded.records:
            if fed.aux is None:
                kwargs["event_callback"](fed.record)
            elif kwargs["aux_callback"] is not None:
                kwargs["aux_callback"](fed.aux, fed.record)
        return result(response="5050", session_id=SESSION, tool_call_count=1)

    monkeypatch.setattr(a2a_server, "run_opencode", run_opencode)
    return Harness(a2a_server.OpenCodeAgent()), turns


@pytest.mark.asyncio
async def test_a_task_that_builds_no_atif_copy_is_fed_nothing_beside_the_clis_events(served):
    oc, turns = served
    async with oc.client() as client:
        (await client.post("/ext/trajectory/prescribe", json={"formats": []})).raise_for_status()
        task = await oc.send(client, "sum it")

    assert task["status"]["state"] == "completed" and turns[0]["aux_callback"] is None


@pytest.mark.asyncio
async def test_a_task_that_builds_an_atif_copy_is_fed_its_relays_calls_and_serves_a_valid_one(served):
    oc, turns = served
    async with oc.client() as client:
        (await client.post("/ext/trajectory/prescribe", json={"formats": ["atif/1"]})).raise_for_status()
        task = await oc.send(client, "sum it")
        read = (await client.post("/ext/trajectory", json={"task_id": task["id"], "format": "atif/1"})).json()

    assert task["status"]["state"] == "completed" and callable(turns[0]["aux_callback"])
    assert [entry["name"] for entry in read["aux"]] == ["turn", "calls", "calls", "calls"]
    document = read["trajectory"]
    assert not errors_in(validate(document))
    assert [step["message"] for step in agent_steps(document)] == ["", "", "5050"]
    assert document["final_metrics"]["total_prompt_tokens"] == 3060


def test_the_card_offers_atif_beside_the_clis_events_which_stay_the_default(oc):
    (trajectory,) = [ext for ext in oc.card["capabilities"]["extensions"] if ext["uri"].endswith("trajectory/v1")]

    assert trajectory["params"]["formats"] == ["opencode-events/1", "atif/1"]
    assert trajectory["params"]["native"] == trajectory["params"]["default"] == "opencode-events/1"
