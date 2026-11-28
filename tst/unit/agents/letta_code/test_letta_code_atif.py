"""Letta Code's ATIF adapter on records shaped as the CLI and the bridge write them: stdout's stream-json events, a line
per streamed delta of each local-backend run, as natives, and the model relay's ``agentenv.model_call`` records, made
here by the relay's own ``Calls`` from Chat Completions usage, as the ``calls`` aux file. Each case runs through the
protocol's adapter kit, so the live rule holds after every record and the document validates as a producer's. Then the
bridge and the server, which feed the relay's calls only to a task that builds an atif/1 copy, and the card."""
from __future__ import annotations

import base64
import json
import random
import struct
import uuid
import zlib
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from itertools import count

import pytest
from agentenv_protocol.atif import Fed, RecordedTask, TurnContext, check_adapter, errors_in, profile_of, validate
from starlette.testclient import TestClient

from tst.unit.agents.letta_code.agent import AGENT_DIR, a2a_server, letta_code_atif, model_relay
from tst.unit.agents.letta_code.agent import letta_code_bridge as bridge
from tst.util.atif_recorded import (
    assert_every_call_has_its_result,
    assert_every_model_call_has_its_time_and_usage,
    assert_steps_sum_to_reported_usage,
)

LettaCodeAtif = letta_code_atif.LettaCodeAtif
MODEL = "openai/gpt-4.1-mini"
AGENT = "agent-2b7c1d4e-9f30-4a8e-8c55-0e1f2a3b4c5d"
CONV = "local-conv-1"
START = datetime(2026, 11, 27, 16, 0, tzinfo=timezone.utc)


def png(width=4, height=3) -> bytes:
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    rows = b"".join(b"\x00" + b"\x10\x20\x30" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


class Clock:
    """The CLI's and the relay's one clock, as the bridge stamps the relay's calls: a millisecond a record."""

    tick = 0

    @classmethod
    def now(cls) -> datetime:
        cls.tick += 1
        return START + timedelta(milliseconds=cls.tick)

    @classmethod
    def stamp(cls) -> str:
        return model_relay.stamp(cls.now())


class Feed:
    """One task's records in arrival order: the CLI's lines, and the relay's call records as its ``Calls`` hands them
    to the bridge's sink, whose ``usage`` is then the turn's."""

    def __init__(self):
        self.records: list[Fed] = []
        self.calls = model_relay.Calls(sink=lambda record: self.records.append(Fed(record, aux="calls")),
                                       clock=Clock.now)
        self.runs, self.run, self.seq, self.ids, self.tool_calls = 0, None, 0, count(1), 0
        self.answer = ""

    def usage(self) -> dict:
        return self.calls.usage(self.tool_calls, MODEL).to_dict()

    def line(self, record: dict) -> Feed:
        self.records.append(Fed({**record, "timestamp": Clock.stamp()}))
        return self

    def init(self) -> Feed:
        """The CLI's first line, which names the model by the handle the bridge creates the agent with."""
        return self.line({"type": "system", "subtype": "init", "session_id": AGENT, "agent_id": AGENT,
                          "conversation_id": CONV, "model": f"openai-compatible/{MODEL}",
                          "tools": ["Bash", "Read", "Edit", "Glob", "Grep", "Write", "Agent", "Task"],
                          "cwd": "/app", "mcp_servers": [], "permission_mode": "", "slash_commands": [],
                          "memfs_enabled": True, "skill_sources": ["bundled"], "system_info_reminder_enabled": True,
                          "reflection_trigger": "step-count", "reflection_step_count": 25, "uuid": f"init-{AGENT}"})

    def new_run(self) -> Feed:
        self.runs += 1
        self.run, self.seq, self.message = f"local-run-{self.runs}", 0, f"chatcmpl-{uuid.uuid4().hex[:12]}"
        return self

    def chunk(self, kind: str, *, stored: bool = True, **fields) -> Feed:
        """A backend chunk as the headless loop prints it: a stored one with its id and date, all with the run's."""
        self.seq += 1
        at = Clock.stamp()
        head = {"type": "message", "message_type": kind}
        if stored:
            head |= {"id": fields.pop("id", f"letta-msg-{next(self.ids)}"), "date": at, "agent_id": AGENT,
                     "conversation_id": CONV}
        record = {**head, **fields, "run_id": self.run, "seq_id": self.seq, "session_id": AGENT}
        self.records.append(Fed({**record, "uuid": record.get("id") or uuid.uuid4().hex, "timestamp": at}))
        return self

    def say(self, *deltas: str) -> Feed:
        for delta in deltas:
            self.chunk("assistant_message", id=self.message, content=[{"type": "text", "text": delta}])
        return self

    def think(self, *deltas: str) -> Feed:
        for delta in deltas:
            self.chunk("reasoning_message", id=f"{self.message}:reasoning:0", reasoning=delta)
        return self

    def printed(self, prompt: int, completion: int, cached: int = 0, reasoning: int = 0) -> Feed:
        """The run's usage_statistics, as pi-ai counts a Chat Completions usage: its prompt without the cache."""
        return self.chunk("usage_statistics", step_count=1, prompt_tokens=prompt - cached,
                          completion_tokens=completion, total_tokens=prompt + completion, cached_input_tokens=cached,
                          cache_write_tokens=0, reasoning_tokens=reasoning, context_tokens=prompt + completion)

    def stop(self, reason: str = "end_turn") -> Feed:
        return self.chunk("stop_reason", stored=False, stop_reason=reason)

    def relay(self, prompt: int | None = None, completion: int = 0, cached: int = 0, *, status: int | None = 200,
              model: str = MODEL) -> Feed:
        """A call the relay forwarded: answered with this usage, answered without any (``prompt`` None), or refused."""
        usage = None if prompt is None else model_relay.answer_usage(json.dumps({"usage": {
            "prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": cached}}}).encode())
        self.calls.add(status, usage, model, started=Clock.now())
        return self

    def respond(self, *deltas: str, prompt=22860, completion=72, cached=0, reasoning=(), stop="end_turn",
                relay="before", hidden=0) -> Feed:
        """One model response as the CLI prints it, the relay's record of its call ending before its usage is printed
        or after its stop."""
        self.new_run().think(*reasoning).say(*deltas)
        if relay == "before":
            self.relay(prompt, completion, cached)
        self.printed(prompt, completion, cached, hidden).stop(stop)
        if relay == "after":
            self.relay(prompt, completion, cached)
        self.answer = "".join(deltas) or self.answer
        return self

    def tools(self, *calls: tuple) -> Feed:
        """The run's calls, as the headless loop prints them once it approved them, then their results: each
        ``(call_id, name, arguments, output[, status])``."""
        date = Clock.stamp()
        for call_id, name, arguments, *_ in calls:
            self.line({"type": "message", "message_type": "tool_call_message", "id": f"tool-call-{call_id}",
                       "date": date, "tool_call": {"name": name, "tool_call_id": call_id,
                                                   "arguments": json.dumps(arguments)},
                       "session_id": AGENT, "uuid": f"tool-call-{call_id}"})
            self.tool_calls += 1
        date = Clock.stamp()
        for call_id, _, _, output, *status in calls:
            self.line({"type": "message", "message_type": "tool_return_message", "id": f"tool-return-{call_id}",
                       "date": date, "status": status[0] if status else "success", "tool_call_id": call_id,
                       "tool_return": output, "stdout": [output] if isinstance(output, str) else [], "stderr": [],
                       "session_id": AGENT, "uuid": f"tool-return-{call_id}"})
        return self

    def retried(self, message: str = "HTTP 503: upstream unavailable", attempt: int = 1) -> Feed:
        """An in-run retry of a failed call, before any output."""
        return self.chunk("event_message", event_type="retry", event_data={
            "attempt": attempt, "max_attempts": 3, "delay_ms": 1000, "message": message})

    def failed(self, message: str = "400 Bad Request: context too long") -> Feed:
        """A run's failed call, which the headless loop prints in place of its error chunk."""
        return self.line({"type": "error", "message": message, "stop_reason": "error", "run_id": self.run,
                          "session_id": AGENT, "uuid": uuid.uuid4().hex,
                          "api_error": {"message_type": "error_message", "message": message, "error_type": "llm_error",
                                        "detail": message, "run_id": self.run}})

    def harness_retry(self, attempt: int = 1) -> Feed:
        return self.line({"type": "retry", "reason": "llm_api_error", "attempt": attempt, "max_attempts": 3,
                          "delay_ms": 1000, "run_id": self.run, "session_id": AGENT, "uuid": f"retry-{self.run}"})

    def gave_up(self, message: str) -> Feed:
        """The CLI's last word on a failed turn, after which it exits 1 with no result."""
        return self.line({"type": "error", "message": message, "stop_reason": "error", "run_id": self.run,
                          "session_id": AGENT, "uuid": f"error-{self.run}"})

    def transcript(self, agent: str, *messages: dict) -> Feed:
        """What a sub-agent wrote to its transcript, pi-ai messages each in an entry, as the bridge feeds them once the
        turn's processes are stopped: a session entry first, then each message with the time it was written."""
        self.records.append(Fed({"type": "session", "version": 3, "id": "default", "timestamp": Clock.stamp(),
                                 "cwd": "/app", "agent_id": agent}, aux="subagents"))
        for message in messages:
            millis = int(Clock.now().timestamp() * 1000)
            self.records.append(Fed({"type": "message", "id": uuid.uuid4().hex[:8], "timestamp": Clock.stamp(),
                                     "message": {**message, "timestamp": millis}, "agent_id": agent},
                                    aux="subagents"))
        return self

    def result(self) -> Feed:
        return self.line({"type": "result", "subtype": "success", "session_id": AGENT, "duration_ms": 4210,
                          "duration_api_ms": 3120, "num_turns": self.runs, "result": self.answer, "agent_id": AGENT,
                          "conversation_id": CONV, "run_ids": [f"local-run-{n}" for n in range(1, self.runs + 1)],
                          "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                                    "step_count": self.runs, "cached_input_tokens": 0, "cache_write_tokens": 0,
                                    "reasoning_tokens": 0}, "uuid": f"result-{self.run}"})


def context(index=0):
    return TurnContext(index=index, context_id="ctx", task_id=f"t{index}", message_id=f"m{index}", inbound=["do it"],
                       agent_name="Letta Code", agent_version="1.0.0", agent_model=MODEL,
                       started_at=START + timedelta(seconds=index))


def run(*feeds: Feed, state="completed", final=True):
    tasks = [RecordedTask(context=context(i), records=feed.records, state=state,
                          final=[feed.answer] if final and feed.answer else None) for i, feed in enumerate(feeds)]
    kit = check_adapter(LettaCodeAtif, tasks)
    kit.raise_for_errors()
    return kit


def agent_steps(document):
    return [step for step in document["steps"] if step["source"] == "agent"]


def coding() -> Feed:
    """The two calls of a recorded gpt-4.1-mini turn (test_bridge.py's GPT_TURN): a Bash call, then the answer."""
    feed = Feed().init().respond("I'll run ", "it.", stop="requires_approval")
    feed.tools(("call_HbQ1", "Bash", {"command": "python3 hello.py"}, "5050\n"))
    return feed.respond("50", "50", prompt=22993, completion=2, cached=22784).result()


def test_a_run_is_one_step_with_its_deltas_joined_its_calls_results_and_its_relay_calls_model_and_usage():
    feed = coding()
    kit = run(feed)
    first, second = agent_steps(kit.document)

    assert first["message"] == "I'll run it." and first["model_name"] == MODEL and first["timestamp"]
    assert first["tool_calls"] == [{"tool_call_id": "call_HbQ1", "function_name": "Bash",
                                    "arguments": {"command": "python3 hello.py"}}]
    assert first["observation"]["results"] == [{"source_call_id": "call_HbQ1", "content": "5050\n"}]
    assert first["metrics"] == {"prompt_tokens": 22860, "cached_tokens": 0, "completion_tokens": 72}
    assert (second["message"], second["metrics"]["prompt_tokens"], second["metrics"]["cached_tokens"]) == (
        "5050", 22993, 22784)
    assert kit.document["final_metrics"]["total_prompt_tokens"] == 45853 == feed.usage()["input_tokens"]
    assert_steps_sum_to_reported_usage(kit.document, [feed.usage()])
    assert_every_model_call_has_its_time_and_usage(kit.document)
    assert_every_call_has_its_result(kit.document)
    assert not errors_in(validate(kit.document, media=kit.media))


def test_the_answer_is_the_last_step_so_no_final_one_is_added():
    steps = agent_steps(run(coding()).document)

    assert steps[-1]["message"] == "5050" and steps[-1]["llm_call_count"] == 1 and len(steps) == 2


@pytest.mark.parametrize("relayed", ["before", "after"])
def test_a_responses_relay_call_is_its_own_whichever_ends_first(relayed):
    feed = Feed().init().respond("ok", prompt=9000, completion=4, cached=8000, relay=relayed).result()
    (step,) = agent_steps(run(feed).document)

    assert step["metrics"]["prompt_tokens"] == 9000 and step["metrics"]["cached_tokens"] == 8000


def test_calls_only_the_relay_saw_are_steps_with_their_usage_in_their_place_the_tasks_last_before_its_answer():
    feed = Feed().init().respond("Starting a task.", stop="requires_approval")
    feed.tools(("call_T1", "Task", {"description": "count lines", "prompt": "wc -l /etc/passwd"},
                "Task running in background with task ID: task-1\nAgent ID: agent-sub\nConversation ID: c-sub"))
    feed.relay(4100, 30, model="openai/gpt-4.1-nano")
    feed.respond("Waiting on it.", prompt=23100, completion=5, stop="end_turn").relay(4200, 12, cached=4096)
    kit = run(feed.result())
    steps = agent_steps(kit.document)

    assert [step["message"] for step in steps] == ["Starting a task.", "", "", "Waiting on it."]
    assert [step["model_name"] for step in steps] == [MODEL, "openai/gpt-4.1-nano", MODEL, MODEL]
    assert [step["metrics"]["prompt_tokens"] for step in steps] == [22860, 4100, 4200, 23100]
    assert_steps_sum_to_reported_usage(kit.document, [feed.usage()])
    assert_every_model_call_has_its_time_and_usage(kit.document)


SPAWNED = ("Task running in background with task ID: task_1\nAgent ID: agent-local-sub\nConversation ID: default\n"
           "Output file: /tmp/letta-background-0COLHm/task_1.log\n\nYou will be notified automatically when this task "
           "completes.")


def spawning() -> Feed:
    """A turn whose Agent call starts a sub-agent, as the recorded one did: the sub-agent's two calls end while the
    parent runs on, one before its next response is printed and one after, beside a reflection's call; its transcript
    comes once the turn's processes are stopped."""
    feed = Feed().init().respond("Starting it.", stop="requires_approval")
    feed.tools(("call_A0", "Agent", {"description": "count", "prompt": "wc -l /etc/passwd"},
                "Task tool missing required parameter: subagent_type.", "error"),
               ("call_A1", "Agent", {"subagent_type": "general-purpose", "description": "count",
                                     "prompt": "Run wc -l /etc/passwd and report the count."}, SPAWNED))
    feed.relay(10812, 34)
    feed.respond(prompt=23065, completion=20, cached=22912, stop="requires_approval")
    feed.tools(("call_B1", "Bash", {"command": "sleep 2; cat /tmp/letta-background-0COLHm/task_1.log"}, "21\n"))
    feed.relay(10860, 12, cached=10752).relay(4100, 30, model="openai/gpt-4.1-nano")
    feed.respond("21", prompt=23269, completion=2, cached=23040).result()
    return feed.transcript(
        "agent-local-sub",
        {"id": "ui-msg-1", "role": "user", "content": [{"type": "text", "text": "Run wc -l /etc/passwd."}]},
        {"id": "ui-msg-2", "role": "assistant", "model": MODEL, "stopReason": "toolUse",
         "content": [{"type": "toolCall", "id": "call_387I", "name": "Bash",
                      "arguments": {"command": "wc -l /etc/passwd"}}],
         "usage": {"input": 10812, "output": 34, "cacheRead": 0, "cacheWrite": 0, "reasoning": 0}},
        {"id": "ui-msg-3", "role": "toolResult", "toolCallId": "call_387I", "toolName": "Bash",
         "content": [{"type": "text", "text": "21 /etc/passwd\n"}], "isError": False},
        {"id": "ui-msg-4", "role": "assistant", "model": MODEL, "stopReason": "stop",
         "content": [{"type": "text", "text": "The file /etc/passwd contains 21 lines."}],
         "usage": {"input": 108, "output": 12, "cacheRead": 10752, "cacheWrite": 0, "reasoning": 0}})


def test_a_sub_agent_the_agent_tool_started_is_embedded_under_its_call_with_the_relay_calls_it_made():
    feed = spawning()
    kit = run(feed)
    steps = agent_steps(kit.document)
    (sub,) = kit.document["subagent_trajectories"]

    assert sub["trajectory_id"] == "agent-local-sub" and sub["agent"]["name"] == "general-purpose"
    failed, started = steps[0]["observation"]["results"]
    assert "subagent_trajectory_ref" not in failed
    assert started["subagent_trajectory_ref"] == [{"trajectory_id": "agent-local-sub"}]
    called, answered = sub["steps"]
    assert called["tool_calls"][0]["arguments"] == {"command": "wc -l /etc/passwd"}
    assert called["observation"]["results"] == [{"source_call_id": "call_387I", "content": "21 /etc/passwd\n"}]
    assert answered["message"] == "The file /etc/passwd contains 21 lines."
    assert [(step["model_name"], step["metrics"]["prompt_tokens"], step["metrics"]["cached_tokens"])
            for step in sub["steps"]] == [(MODEL, 10812, 0), (MODEL, 10860, 10752)]
    assert [step["metrics"]["prompt_tokens"] for step in steps] == [22860, 23065, 4100, 23269]
    assert steps[-1]["message"] == "21"
    assert_steps_sum_to_reported_usage(kit.document, [feed.usage()])
    assert_every_model_call_has_its_time_and_usage(kit.document)
    assert_every_call_has_its_result(kit.document)


def test_without_its_transcript_a_sub_agents_calls_are_the_parents_side_steps_before_its_answer():
    """As with a bridge that feeds no transcript, or a sub-agent stopped before it wrote one."""
    feed = spawning()
    feed.records = [fed for fed in feed.records if fed.aux != "subagents"]
    kit = run(feed)
    steps = agent_steps(kit.document)

    assert "subagent_trajectories" not in kit.document
    assert "subagent_trajectory_ref" not in steps[0]["observation"]["results"][1]
    assert [step["metrics"]["prompt_tokens"] for step in steps] == [22860, 23065, 10812, 10860, 4100, 23269]
    assert_steps_sum_to_reported_usage(kit.document, [feed.usage()])


def test_a_sub_agent_one_started_is_embedded_under_its_agent_call_with_the_relay_calls_it_made():
    """A depth-one sub-agent may start its own (the CLI's MAX_SUBAGENT_DEPTH is 2), whose transcript the bridge feeds
    beside its parent's; the child's call ends between its parent's."""
    feed = Feed().init().respond("Starting it.", stop="requires_approval")
    feed.tools(("call_A1", "Agent", {"subagent_type": "general-purpose", "description": "count", "prompt": "p"},
                SPAWNED))
    feed.relay(10812, 34).relay(5100, 21)
    feed.respond("Waiting on it.", prompt=23065, completion=20, cached=22912)
    feed.relay(10860, 12, cached=10752).result()
    started = SPAWNED.replace("task_1", "task_2").replace("agent-local-sub", "agent-local-nested")
    feed.transcript(
        "agent-local-sub",
        {"id": "ui-msg-1", "role": "user", "content": [{"type": "text", "text": "Count them."}]},
        {"id": "ui-msg-2", "role": "assistant", "model": MODEL, "stopReason": "toolUse",
         "content": [{"type": "toolCall", "id": "call_N1", "name": "Agent",
                      "arguments": {"subagent_type": "general-purpose", "description": "wc", "prompt": "wc -l"}}],
         "usage": {"input": 10812, "output": 34, "cacheRead": 0, "cacheWrite": 0, "reasoning": 0}},
        {"id": "ui-msg-3", "role": "toolResult", "toolCallId": "call_N1", "toolName": "Agent",
         "content": [{"type": "text", "text": started}], "isError": False},
        {"id": "ui-msg-4", "role": "assistant", "model": MODEL, "stopReason": "stop",
         "content": [{"type": "text", "text": "A helper is counting."}],
         "usage": {"input": 108, "output": 12, "cacheRead": 10752, "cacheWrite": 0, "reasoning": 0}})
    feed.transcript(
        "agent-local-nested",
        {"id": "ui-msg-1", "role": "user", "content": [{"type": "text", "text": "wc -l /etc/passwd"}]},
        {"id": "ui-msg-2", "role": "assistant", "model": MODEL, "stopReason": "stop",
         "content": [{"type": "text", "text": "21"}],
         "usage": {"input": 5100, "output": 21, "cacheRead": 0, "cacheWrite": 0, "reasoning": 0}})
    kit = run(feed)
    sub, nested = kit.document["subagent_trajectories"]

    assert (sub["trajectory_id"], nested["trajectory_id"]) == ("agent-local-sub", "agent-local-nested")
    assert nested["agent"]["name"] == "general-purpose"
    (spawn,) = sub["steps"][0]["observation"]["results"]
    assert spawn["subagent_trajectory_ref"] == [{"trajectory_id": "agent-local-nested"}]
    assert [(step["message"], step["metrics"]["prompt_tokens"]) for step in nested["steps"]] == [("21", 5100)]
    assert [step["metrics"]["prompt_tokens"] for step in sub["steps"]] == [10812, 10860]
    assert [step["metrics"]["prompt_tokens"] for step in agent_steps(kit.document)] == [22860, 23065]
    assert_steps_sum_to_reported_usage(kit.document, [feed.usage()])
    assert_every_call_has_its_result(kit.document)


def test_reasoning_streamed_is_joined_raw_and_hidden_reasoning_is_omitted():
    feed = Feed().init().respond("42", reasoning=("Six ", "times seven."), stop="end_turn")
    (thought,) = agent_steps(run(feed.result()).document)
    assert thought["reasoning_content"] == "Six times seven." and "reasoning" not in profile_of(thought)

    (hidden,) = agent_steps(run(Feed().init().respond("42", hidden=64).result()).document)
    assert "reasoning_content" not in hidden
    assert profile_of(hidden)["reasoning"] == [{"chars": 0, "visibility": "omitted"}]


def test_an_mcp_results_image_is_a_media_file_and_a_text_only_one_stays_as_the_model_saw_it():
    image = png()
    shot = json.dumps({"content": [{"type": "text", "text": "### Page\n- Example Domain"},
                                   {"type": "image", "data": base64.b64encode(image).decode(),
                                    "mimeType": "image/png"}]})
    snapshot = json.dumps({"content": [{"type": "text", "text": "- heading \"Example Domain\""}]})
    feed = Feed().init().respond(stop="requires_approval")
    feed.tools(("call_S1", "mcp__browser__browser_take_screenshot", {}, shot),
               ("call_S2", "mcp__browser__browser_snapshot", {}, snapshot),
               ("call_S3", "mcp__browser__browser_click", {"ref": "e9"}, "Error: element not found", "error"))
    kit = run(feed.respond("Example Domain", prompt=24000, completion=3).result())

    shot_result, snapshot_result, failed = agent_steps(kit.document)[0]["observation"]["results"]
    text, picture = shot_result["content"]
    assert text == {"type": "text", "text": "### Page\n- Example Domain"}
    assert picture["type"] == "image" and kit.media[picture["source"]["path"]] == image
    assert snapshot_result["content"] == snapshot
    assert profile_of(failed)["status"] == "error"


def test_an_in_run_retry_is_a_step_with_no_model_call_holding_the_clis_report_before_the_answer():
    feed = Feed().init().new_run().relay(status=503).retried("HTTP 503: upstream unavailable")
    feed.say("ok").relay(22860, 2).printed(22860, 2).stop()
    feed.answer = "ok"
    kit = run(feed.result())
    refused, answered = agent_steps(kit.document)

    assert refused["llm_call_count"] == 0 and "metrics" not in refused and refused["model_name"] == MODEL
    assert profile_of(refused)["error"] == {"kind": "infra_error", "message": "HTTP 503: upstream unavailable"}
    assert answered["message"] == "ok" and answered["metrics"]["prompt_tokens"] == 22860


def test_a_turn_that_failed_at_a_refused_call_holds_the_clis_last_word_on_it():
    feed = Feed().init().new_run().relay(status=400).failed("400 Bad Request: context too long").stop("error")
    feed.gave_up("Error: 400 Bad Request: context too long (run local-run-1)")
    (step,) = agent_steps(run(feed, state="failed").document)

    assert step["llm_call_count"] == 0
    assert profile_of(step)["error"] == {"kind": "infra_error",
                                         "message": "Error: 400 Bad Request: context too long (run local-run-1)"}
    assert profile_of(step)["state"] == "failed"


def test_a_harness_retry_starts_an_attempt_after_the_runs_every_failed_call():
    feed = Feed().init().new_run()
    for attempt in (1, 2, 3):
        feed.relay(status=503).retried(f"HTTP 503: try {attempt}", attempt)
    feed.relay(status=503).failed("HTTP 503: upstream unavailable").stop("llm_api_error").harness_retry()
    kit = run(feed.respond("ok", prompt=22860, completion=2).result())
    steps = agent_steps(kit.document)

    assert [profile_of(step).get("error", {}).get("message") for step in steps] == [
        "HTTP 503: try 1", "HTTP 503: try 2", "HTTP 503: try 3", "HTTP 503: upstream unavailable", None]
    assert [profile_of(step).get("attempt") for step in steps] == [None, None, None, None, 2]
    assert steps[-1]["message"] == "ok"
    assert_steps_sum_to_reported_usage(kit.document, [feed.usage()])


def test_a_run_whose_stream_broke_after_its_output_began_is_its_step_holding_the_failure_with_its_call():
    """The CLI retries a stream that broke ("terminated") after its first deltas in a new run; the relay answered that
    call, without usage. It made one model call, not one plus a refused one."""
    feed = Feed().init().new_run().say("Look", "ing").relay(None).failed("terminated").stop("llm_api_error")
    feed.harness_retry()
    kit = run(feed.respond("ok", prompt=22860, completion=2).result())
    broken, answered = agent_steps(kit.document)

    assert (broken["message"], broken["llm_call_count"], broken["model_name"]) == ("Looking", 1, MODEL)
    assert "metrics" not in broken
    assert profile_of(broken)["error"] == {"kind": "infra_error", "message": "terminated"}
    assert answered["message"] == "ok" and profile_of(answered)["attempt"] == 2


def test_a_compactions_summary_is_the_step_of_the_relay_call_that_wrote_it():
    feed = Feed().init().respond("Reading.", stop="requires_approval")
    feed.tools(("call_R1", "Read", {"file_path": "/app/big.log"}, "x" * 200))
    feed.new_run().relay(98000, 900)
    feed.chunk("event_message", event_type="compaction", event_data={"trigger": "context_window_limit"})
    feed.chunk("summary_message", summary="The user asked for the log's errors; it was read.", compaction_stats={
        "trigger": "context_window_limit", "context_tokens_before": 121000, "context_tokens_after": 9000,
        "context_window": 128000, "messages_count_before": 40, "messages_count_after": 3})
    feed.say("Two errors.").relay(9400, 4).printed(9400, 4).stop()
    feed.answer = "Two errors."
    kit = run(feed.result())
    _, summary, answer = agent_steps(kit.document)

    assert summary["message"] == "The user asked for the log's errors; it was read."
    assert summary["metrics"]["prompt_tokens"] == 98000
    assert profile_of(summary)["compaction"] == {"written_by": "assistant", "trigger": "auto",
                                                 "tokens_before": 121000, "tokens_after": 9000}
    assert answer["message"] == "Two errors."
    assert_steps_sum_to_reported_usage(kit.document, [feed.usage()])


def test_a_compactions_summary_is_the_call_that_ended_before_it_was_printed_not_one_ending_during_the_run():
    """The CLI prints a compaction's event and summary once the summarizer's call has ended; a side call that ends
    while the run's own call streams is a step of its own."""
    feed = Feed().init().respond("Reading.", stop="requires_approval")
    feed.tools(("call_R1", "Read", {"file_path": "/app/big.log"}, "x" * 200))
    feed.new_run().relay(98000, 900)
    feed.chunk("event_message", event_type="compaction", event_data={"trigger": "context_window_limit"})
    feed.chunk("summary_message", summary="The log was read.", compaction_stats={
        "trigger": "context_window_limit", "context_tokens_before": 121000, "context_tokens_after": 9000})
    feed.relay(4100, 30, model="openai/gpt-4.1-nano")
    feed.say("Two errors.").relay(9400, 4).printed(9400, 4).stop()
    feed.answer = "Two errors."
    kit = run(feed.result())
    steps = agent_steps(kit.document)

    assert [(step["message"], step["model_name"], step["metrics"]["prompt_tokens"]) for step in steps] == [
        ("Reading.", MODEL, 22860), ("The log was read.", MODEL, 98000), ("", "openai/gpt-4.1-nano", 4100),
        ("Two errors.", MODEL, 9400)]
    assert "compaction" in profile_of(steps[1]) and "compaction" not in profile_of(steps[2])


def test_a_response_printed_without_usage_takes_the_relays_call_answered_without_any():
    feed = Feed().init().new_run().say("ok").relay(None).printed(0, 0).stop()
    feed.answer = "ok"
    kit = run(feed.result())
    (step,) = agent_steps(kit.document)

    assert step["message"] == "ok" and step["llm_call_count"] == 1 and "metrics" not in step
    assert feed.usage()["provider_details"] == {"model_calls": 1, "unreported_calls": 1}


def test_a_response_printed_without_usage_takes_the_call_answered_without_after_its_refused_ones():
    """A side call answered without usage that ended first isn't the response's: the run's call was sent only after
    its refused attempt."""
    feed = Feed().init().respond("Reading.", stop="requires_approval")
    feed.tools(("call_R1", "Read", {"file_path": "/app/a"}, "a"))
    feed.relay(None, model="openai/gpt-4.1-nano")
    feed.new_run().relay(status=503).retried("HTTP 503: upstream unavailable")
    feed.say("ok").relay(None).printed(0, 0).stop("requires_approval")
    feed.tools(("call_R2", "Read", {"file_path": "/app/b"}, "b"))
    kit = run(feed.respond("done", prompt=23000, completion=2).result())
    steps = agent_steps(kit.document)

    assert [(step["message"], step["model_name"], step["llm_call_count"]) for step in steps] == [
        ("Reading.", MODEL, 1), ("", "openai/gpt-4.1-nano", 1), ("", MODEL, 0), ("ok", MODEL, 1), ("done", MODEL, 1)]
    assert profile_of(steps[2])["error"]["message"] == "HTTP 503: upstream unavailable"


def test_without_the_relays_calls_a_run_takes_its_own_counts_and_the_model_as_called_not_the_clis_handle():
    feed = Feed().init().respond("5050", prompt=22993, completion=2, cached=22784, relay="none").result()
    (step,) = agent_steps(run(feed).document)

    assert step["metrics"]["prompt_tokens"] == 22993 and step["metrics"]["cached_tokens"] == 22784
    assert step["model_name"] == MODEL


def test_a_resumed_conversations_next_task_writes_its_own_steps_from_its_own_runs():
    second = Feed().init().respond("500", "500", prompt=23300, completion=3, cached=22900).result()
    kit = run(coding(), second)
    steps = agent_steps(kit.document)

    assert [step["message"] for step in steps] == ["I'll run it.", "5050", "500500"]
    assert [step["step_id"] for step in kit.document["steps"] if step["source"] == "user"] == [1, 4]
    assert kit.document["final_metrics"]["total_prompt_tokens"] == 45853 + 23300


def test_the_tested_cli_is_the_one_the_image_pins():
    pinned = json.loads((AGENT_DIR / "package.json").read_text())["dependencies"]["@letta-ai/letta-code"]
    assert letta_code_atif.TESTED_WITH == (pinned,)


def seeded_session(seed: int) -> tuple[list[Feed], dict]:
    """A random conversation of one to three tasks, as the CLI and the relay write them: responses of reasoning, text
    and calls, some returning images or failing; the relay's record of each response's call, ending before its usage
    is printed or after its stop; refused attempts the run retried before some; and calls only the relay saw (a
    sub-agent's, a reflection's) between them and after the last. With what the copy must hold."""
    rng = random.Random(seed)
    image = base64.b64encode(png()).decode()
    want = {"refused": 0, "calls": set(), "texts": []}
    n = count()
    feeds = []
    for _ in range(rng.randint(1, 3)):
        feed = Feed().init()
        responses = rng.randint(1, 4)
        for index in range(responses):
            if rng.random() < 0.2:
                feed.relay(rng.randint(100, 5000), rng.randint(1, 40), model="openai/gpt-4.1-nano")
            last = index == responses - 1
            feed.new_run()
            for attempt in range(rng.choice([0, 0, 0, 1, 2])):
                feed.relay(status=rng.choice([429, 500, 503])).retried(f"HTTP 503: try {attempt}", attempt + 1)
                want["refused"] += 1
            feed.think(*(["Plan", " it."] if rng.random() < 0.3 else []))
            text = "done" if last else rng.choice(["", "", "looking"])
            feed.say(*([text[:2], text[2:]] if text else []))
            prompt, completion = rng.randint(1000, 90000), rng.randint(1, 200)
            cached = rng.randint(0, prompt)
            unreported = rng.random() < 0.1
            relayed = rng.choice(["before", "after"])
            if relayed == "before":
                feed.relay(None if unreported else prompt, completion, cached)
            feed.printed(*((0, 0) if unreported else (prompt, completion, cached)))
            calls = [] if last else [(f"call_{next(n)}", rng.choice(["Bash", "mcp__browser__browser_take_screenshot"]))
                                     for _ in range(rng.randint(1, 2))]
            feed.stop("requires_approval" if calls else "end_turn")
            if relayed == "after":
                feed.relay(None if unreported else prompt, completion, cached)
            if calls:
                feed.tools(*[(call_id, name, {}, json.dumps({"content": [
                    {"type": "image", "data": image, "mimeType": "image/png"}]}) if name.startswith("mcp__")
                    else "ok", rng.choice(["success", "success", "error"])) for call_id, name in calls])
                want["calls"] |= {call_id for call_id, _ in calls}
                if rng.random() < 0.3:
                    feed.relay(rng.randint(100, 5000), rng.randint(1, 40))
            want["texts"].append(text)
        if rng.random() < 0.3:
            feed.relay(rng.randint(100, 5000), rng.randint(1, 40))
        feed.answer = "done"
        feeds.append(feed.result())
    return feeds, want


@pytest.mark.parametrize("seed", range(120))
def test_a_seeded_conversations_copy_holds_every_relay_call_once_and_every_response_in_order(seed):
    """A step whose call reported no usage has no counts, so its document gives no totals; its steps' sums are the
    turns' all the same."""
    feeds, want = seeded_session(seed)
    kit = run(*feeds)
    steps = agent_steps(kit.document)

    usages = [feed.usage() for feed in feeds]
    if any(feed.calls.unreported for feed in feeds):
        for metric, name in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens"),
                             ("cached_tokens", "cache_read_tokens")):
            assert sum(step.get("metrics", {}).get(metric, 0) for step in steps) == sum(
                usage.get(name) or 0 for usage in usages)
    else:
        assert_steps_sum_to_reported_usage(kit.document, usages)
    calls = sum(len(feed.calls.answered) + feed.calls.unreported for feed in feeds)
    assert sum(step["llm_call_count"] for step in steps) == calls
    assert sum(1 for step in steps if step["llm_call_count"] == 0) == want["refused"]
    assert [step["message"] for step in steps if step["message"]] == [text for text in want["texts"] if text]
    assert {call["tool_call_id"] for step in steps for call in step.get("tool_calls") or ()} == want["calls"]
    assert_every_call_has_its_result(kit.document)
    lasts = [step for step, after in zip(kit.document["steps"], kit.document["steps"][1:] + [{"source": "user"}])
             if after["source"] == "user"]
    assert [(step["message"], step["llm_call_count"]) for step in lasts] == [("done", 1)] * len(feeds)


def test_from_atif_gives_the_clis_events_back():
    kit = run(coding())
    native = LettaCodeAtif.from_atif(kit.document)

    kinds = [(event["type"], event.get("message_type")) for event in native.log]
    assert kinds[0] == ("system", None) and kinds[-1] == ("result", None) and native.session is None
    assert ("message", "tool_call_message") in kinds and native.log[-1]["result"] == "5050"


@pytest.fixture
def turn(monkeypatch, tmp_path):
    """A turn of the bridge against a relay that forwards one call, with what it handed the relay."""
    monkeypatch.setattr(bridge, "SESSIONS_ROOT", str(tmp_path / "sessions"))
    seen = {}

    @asynccontextmanager
    async def relay(upstream, calls, **kwargs):
        seen["calls"] = calls
        calls.add(200, model_relay.answer_usage(b'{"usage": {"prompt_tokens": 9, "completion_tokens": 1}}'),
                  "gpt-4.1-mini", started=calls.clock())
        yield "http://127.0.0.1:41000/token"

    async def command(*args, **kwargs):
        return b'{"id":"agent-local-1"}'

    class Proc:
        returncode = 0

        async def wait(self):
            return 0

    async def spawn(*args, **kwargs):
        return Proc()

    async def stream(proc, on_line):
        on_line(json.dumps({"type": "result", "subtype": "success", "result": "ok", "conversation_id": "c-1"}))
        return b""

    monkeypatch.setattr(bridge, "model_relay", relay)
    monkeypatch.setattr(bridge, "_command", command)
    monkeypatch.setattr(bridge, "_configure_agent", lambda *args: None)
    monkeypatch.setattr(bridge.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(bridge, "_stream_cli", stream)
    return seen


@pytest.mark.asyncio
async def test_the_bridge_feeds_each_relay_call_as_aux_on_the_clis_clock(turn, monkeypatch, tmp_path):
    """On a followed clock the CLI's records carry the env's time, which the relay's server, on real time, reads from
    the follower's clock file."""
    universe = datetime(2031, 5, 6, 7, 8, 9, tzinfo=timezone.utc)
    monkeypatch.setattr(bridge, "follows", lambda: True)
    monkeypatch.setattr(bridge, "session_time", universe.timestamp)
    fed = []

    result = await bridge.run_letta_code("hi", "ctx", model="gpt-4.1-mini", workspace=str(tmp_path),
                                         aux_callback=lambda name, record: fed.append((name, record)))

    ((name, record),) = fed
    assert name == "calls" and record["type"] == "agentenv.model_call" and record["outcome"] == "answered"
    assert record["started"] == record["timestamp"] == "2031-05-06T07:08:09.000Z"
    assert record["usage"]["input_tokens"] == 9 == result.usage.input_tokens


def write_conversation(home, name: str, agent: str, *entries: dict) -> None:
    folder = home / bridge._CONVERSATIONS / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "conversation.json").write_text(json.dumps({"id": "default", "agent_id": agent}))
    with (folder / "messages.jsonl").open("a") as transcript:
        transcript.writelines(json.dumps(entry) + "\n" for entry in entries)


@pytest.mark.asyncio
async def test_the_bridge_feeds_what_the_turns_sub_agents_wrote_once_its_processes_are_stopped(turn, monkeypatch,
                                                                                              tmp_path):
    """Each entry a sub-agent wrote during the turn, with its agent's id, after the turn's calls: not the context's own
    conversation, which stdout prints, nor what a sub-agent wrote in an earlier turn."""
    home = bridge.context_data_dir("ctx")
    earlier = {"type": "message", "id": "e0", "message": {"role": "assistant", "content": []}}
    write_conversation(home, "old", "agent-local-old", earlier)
    session = {"type": "session", "version": 3, "id": "default"}
    answer = {"type": "message", "id": "e1", "message": {"role": "assistant", "content": [{"type": "text",
                                                                                          "text": "21"}]}}

    async def stream(proc, on_line):
        write_conversation(home, "main", "agent-local-1", answer)
        write_conversation(home, "old", "agent-local-old", answer)
        write_conversation(home, "sub", "agent-local-sub", session, answer)
        on_line(json.dumps({"type": "result", "subtype": "success", "result": "ok", "conversation_id": "c-1"}))
        return b""

    stopped = []
    monkeypatch.setattr(bridge, "_stream_cli", stream)
    monkeypatch.setattr(bridge, "_stop_leftovers", lambda marker: stopped.append(marker) or [])
    fed = []

    await bridge.run_letta_code("hi", "ctx", model="gpt-4.1-mini", workspace=str(tmp_path),
                                aux_callback=lambda name, record: fed.append((name, record, list(stopped))))

    assert [name for name, _, _ in fed] == ["calls", "subagents", "subagents", "subagents"]
    assert [record for name, record, _ in fed if name == "subagents"] == [
        {**answer, "agent_id": "agent-local-old"}, {**session, "agent_id": "agent-local-sub"},
        {**answer, "agent_id": "agent-local-sub"}]
    assert all(seen for name, _, seen in fed if name == "subagents")


@pytest.mark.asyncio
async def test_without_an_aux_callback_the_relay_hands_its_calls_to_no_one(turn, tmp_path):
    result = await bridge.run_letta_code("hi", "ctx", model="gpt-4.1-mini", workspace=str(tmp_path))

    assert turn["calls"].sink is None and result.usage.input_tokens == 9


def send(client: TestClient, text: str, context_id: str = "ctx-1") -> dict:
    message = {"kind": "message", "role": "user", "messageId": uuid.uuid4().hex, "contextId": context_id,
               "parts": [{"kind": "text", "text": text}]}
    response = client.post("/a2a", json={"jsonrpc": "2.0", "id": 1, "method": "message/send",
                                         "params": {"message": message}})
    response.raise_for_status()
    return response.json()["result"]


@pytest.fixture
def server(monkeypatch):
    """The agent's app with a turn that prints a coding run's events and feeds its relay's calls, if it's handed where
    to; and the keywords each turn got."""
    turns = []
    recorded = coding()

    async def run_letta_code(**kwargs):
        turns.append(kwargs)
        for fed in recorded.records:
            if fed.aux is None:
                kwargs["event_callback"](fed.record)
            elif kwargs["aux_callback"] is not None:
                kwargs["aux_callback"](fed.aux, fed.record)
        return bridge.LettaCodeResult(response="5050", tool_call_count=1, session_id=CONV,
                                      usage=recorded.calls.usage(1, MODEL))

    monkeypatch.setattr(a2a_server, "run_letta_code", run_letta_code)
    return TestClient(a2a_server.build_app(a2a_server.LettaCodeAgent())), turns


def test_a_task_that_builds_no_atif_copy_is_fed_nothing_beside_the_clis_events(server):
    client, turns = server
    client.post("/ext/trajectory/prescribe", json={"formats": []}).raise_for_status()

    assert send(client, "sum it")["status"]["state"] == "completed"
    assert turns[0]["aux_callback"] is None


def test_a_task_that_builds_an_atif_copy_is_fed_its_relays_calls_and_serves_one_step_per_call(server):
    client, turns = server
    client.post("/ext/trajectory/prescribe", json={"formats": ["atif/1"]}).raise_for_status()
    task = send(client, "sum it")

    assert task["status"]["state"] == "completed" and callable(turns[0]["aux_callback"])
    read = client.post("/ext/trajectory", json={"task_id": task["id"], "format": "atif/1"}).json()
    assert [entry["name"] for entry in read["aux"]] == ["calls", "calls"]
    steps = agent_steps(read["trajectory"])
    assert [step["message"] for step in steps] == ["I'll run it.", "5050"]
    assert read["trajectory"]["final_metrics"]["total_prompt_tokens"] == 45853


def test_the_card_offers_atif_beside_the_clis_events_which_stay_the_default():
    card = a2a_server.build_app(a2a_server.LettaCodeAgent()).state.agentenv_a2a.card.model_dump(
        mode="json", exclude_none=True, by_alias=True)
    (trajectory,) = [ext for ext in card["capabilities"]["extensions"] if ext["uri"].endswith("trajectory/v1")]

    assert trajectory["params"]["formats"] == ["letta-code-events/1", "atif/1"]
    assert trajectory["params"]["native"] == trajectory["params"]["default"] == "letta-code-events/1"
