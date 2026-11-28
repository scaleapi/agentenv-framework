"""OpenCode's ATIF adapter on records shaped as opencode-ai 1.17.18 prints them and the turn's relay hands them over:
the CLI's ``run --format json`` events as natives and the relay's ``agentenv.model_call`` records as the ``calls`` aux
file, beside the turn's model and effort (``turn``). Each case runs through the protocol's adapter kit, so the live
rule holds after every record and the document validates as a producer's. Then the bridge and the server, which feed
the aux records only to a task that builds an atif/1 copy, and the card."""
from __future__ import annotations

import asyncio
import base64
import itertools
import json
import random
import struct
import zlib
from contextlib import asynccontextmanager
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
    CLI and the relay share."""

    clock = 1_795_846_800_000
    ids = itertools.count()

    def __init__(self):
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
        self.records.append(Fed({"type": kind, "timestamp": at if at is not None else self.now(), "sessionID": SESSION,
                                 "part": {"id": f"prt_{next(Feed.ids)}", "messageID": message, "sessionID": SESSION,
                                          **part}}))
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


def test_parallel_task_calls_leave_their_calls_to_root_steps_of_their_own():
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

    def tasks(self, message: str, children: int) -> None:
        """Task calls run at once, each a sub-agent making its calls one after another, a refused one before
        some."""
        begun, ends = self.tick(0, 3), []
        for _ in range(children):
            at = begun + 1
            for _ in range(self.rng.randint(1, 3)):
                if self.rng.random() < 0.15:
                    self.relay(at, at + 1, "refused", 503)
                    at += 2
                ends.append(at + self.rng.randint(5, 200))
                self.relay(at, ends[-1])
                at = ends[-1] + 1
        end = max(ends) + 1
        for child in range(children):
            session = f"ses_{next(self.ids)}"
            self.tool_use(end + child, message, "task", {
                "status": "completed", "input": {"description": "d", "prompt": "p", "subagent_type": "general"},
                "output": task_output("done", session), "metadata": task_metadata(session),
                "time": {"start": begun, "end": end}})
        self.clock = end + children
        self.want["subagents"] += children == 1

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
    some running a sub-agent or two at once; with each task's answer and what the copy must hold."""
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
def bridge(monkeypatch):
    """run_opencode with a relay that forwards one call and a CLI that prints one step; what each saw."""
    seen: dict = {}
    monkeypatch.setattr(opencode_bridge, "model_relay",
                        lambda upstream, calls, **kwargs: _relay_that_forwards_one_call(seen, upstream, calls))
    events = Feed().start("msg_1").text("msg_1", "ok").finish("msg_1").records

    async def spawn(*cmd, **kwargs):
        seen["cmd"] = list(cmd)
        return FakeCLI("".join(json.dumps(fed.record) + "\n" for fed in events).encode())

    monkeypatch.setattr(opencode_bridge.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(opencode_bridge, "_context_xdg", lambda _context: {})
    return seen


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
