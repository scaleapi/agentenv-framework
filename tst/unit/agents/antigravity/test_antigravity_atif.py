"""Antigravity's ATIF adapter on records made as the agent makes them: the SDK's chunks and hook results, in the shapes
the recorded runs show (a tool call's id its step's, ``<trajectory>:<step index>``, a result's step id its trajectory's),
through the bridge's own TrajectoryRecorder, the native log, and the turn's model calls through its CallFeed and the
relay's Calls, the ``calls`` aux file. Each case runs through the protocol's adapter kit, so the live rule holds after
every record and the document validates as a producer's. Then the bridge and the server feeding them."""

from __future__ import annotations

import asyncio
import base64
import contextvars
import json
import random
import struct
import zlib
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest
from agentenv_protocol.a2a_agent import Usage
from agentenv_protocol.atif import Fed, RecordedTask, TurnContext, check_adapter, profile_of, strip_profile

from tst.unit.agents.antigravity import fake_sdk
from tst.unit.agents.antigravity.agent import AGENT_DIR, antigravity_atif, antigravity_bridge, model_relay
from tst.unit.agents.antigravity.fake_sdk import Text, Thought, ToolCall, ToolResult, Turn

AntigravityAtif = antigravity_atif.AntigravityAtif
MODEL = "anthropic/claude-haiku-4-5"
PROMPT = "Create hello.py that prints the sum of 1 to 100, run it, and reply with only the number."
EPOCH = datetime(2026, 11, 27, 16, 0, tzinfo=timezone.utc)
# The SDK's trajectory ids: the turn's conversation's, and a sub-agent's.
CONV = "776149fe0db66d5ab57ac7d61f21b3ca"
SUB = "576eae4cd9534c6b55bb8a6a5f78927d"
TASK = "Create notes.txt holding the word forty-two, then report back."


def png(width=4, height=3) -> bytes:
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    rows = b"".join(b"\x00" + b"\x10\x20\x30" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def usage(prompt=1200, completion=40, cached=0) -> Usage:
    """A call's usage as the relay reads a LiteLLM Chat Completions answer's."""
    return Usage.from_model_response({"prompt_tokens": prompt, "completion_tokens": completion,
                                      "total_tokens": prompt + completion,
                                      "prompt_tokens_details": {"cached_tokens": cached}})


def turn_messages(prompt=PROMPT) -> list[dict]:
    return [{"role": "system", "content": "You are Antigravity, a coding agent."},
            {"role": "user", "content": f"<USER_REQUEST>\n{prompt}\n</USER_REQUEST>\n"
                                        "The current local time is: 2026-11-27T16:00:00Z."}]


def other_messages(task: str) -> list[dict]:
    return [{"role": "system", "content": "You are a subagent."}, {"role": "user", "content": task}]


def streamed(name, index, trajectory=CONV, args=None) -> ToolCall:
    """A tool call as the SDK streams one, once its model call has ended: named for its step, with the arguments the
    SDK reads off the step."""
    step = f"{trajectory}:{index}"
    return ToolCall(name=name, args=args if args is not None else {}, id=step, step_id=step, canonical_path=None,
                    server_name=None)


def _drive(coroutine) -> bool:
    """Run a rewrite that never waits, as the relay awaits it; whether it refused the call."""
    try:
        coroutine.send(None)
    except StopIteration:
        return False
    except model_relay.Refusal:
        return True
    raise AssertionError("the rewrite waited")


class Feed:
    """One task's records in arrival order: the recorder's native records, and each model call's request and end as
    the bridge's CallFeed feeds them while the relay forwards it, a millisecond apart."""

    clock = 0

    def __init__(self, prompt=PROMPT, effort=None):
        self.records: list[Fed] = []
        self.started = Feed.now()
        self.feed = antigravity_bridge.CallFeed(lambda name, record: self.records.append(Fed(record, aux=name)),
                                                prompt, effort)
        self.feed.clock = Feed.now
        self.calls = model_relay.Calls(sink=self.feed.sink, clock=Feed.now)
        self.recorder = antigravity_bridge.TrajectoryRecorder(lambda record: self.records.append(Fed(record)))

    @staticmethod
    def now() -> datetime:
        Feed.clock += 1
        return EPOCH + timedelta(milliseconds=Feed.clock)

    def request(self, messages=None, *, model=MODEL, refuse=None):
        """A call's request reaching the relay, which rewrites it in the task that forwards it."""
        async def rewrite(body: bytes) -> bytes:
            if refuse is not None:
                raise model_relay.Refusal(refuse)
            return body
        body = json.dumps({"model": model, "messages": messages or turn_messages(), "stream": True}).encode()
        forwarding, started = contextvars.copy_context(), self.calls.clock()
        refused = forwarding.run(_drive, self.feed.rewrite(rewrite)(body))
        return forwarding, started, model, refused

    def end(self, call, *, status=200, counted=None):
        """The call's answer ending, the relay adding it to its calls in the same task."""
        forwarding, started, model, refused = call
        status = 400 if refused else status
        forwarding.run(self.calls.add, status, counted if status < 400 else None, model, started=started)
        return self

    def answered(self, messages=None, counted=None):
        return self.end(self.request(messages), counted=counted or usage())

    def chunk(self, chunk):
        self.recorder.chunk(chunk)
        return self

    def hooked(self, call_id, name):
        self.recorder.hooked_call(ToolCall(id=call_id, name=name))
        return self

    def result(self, call_id, name, result, trajectory=CONV):
        self.recorder.result({"kind": "tool_result", **antigravity_bridge._plain(ToolResult(
            name=name, id=call_id, step_id=f"{trajectory}:99", result=result, error=None, server_name=None))})
        return self

    def failed(self, call_id, name, error):
        self.recorder.result({"kind": "tool_result", "id": call_id, "name": name, "server_name": None, "result": None,
                              "error": error})
        return self

    def tool(self, call_id, name, index, result, trajectory=CONV, args=None):
        """A call the hooks decide, streamed, then its result."""
        self.hooked(call_id, name).chunk(streamed(name, index, trajectory,
                                                  {"command_line": "python3 hello.py"} if args is None else args))
        return self.result(call_id, name, result, trajectory)

    def native(self, record):
        """A native record fed as an older bridge wrote it."""
        self.records.append(Fed(record))
        return self

    def close(self):
        self.recorder.close()
        return self


def context(index, started) -> TurnContext:
    return TurnContext(index=index, context_id="ctx", task_id=f"t{index}", message_id=f"m{index}", inbound=[PROMPT],
                       agent_name="Antigravity", agent_version="1.0.0", agent_model=MODEL, started_at=started)


def run(*feeds: Feed, state="completed"):
    kit = check_adapter(AntigravityAtif, [
        RecordedTask(context=context(index, feed.started), records=feed.records, state=state,
                     final=[feed.recorder.response()], ended_at=Feed.now()) for index, feed in enumerate(feeds)])
    assert not kit.findings, "\n".join(str(finding) for finding in kit.findings)
    return kit


def agent_steps(trajectory):
    return [step for step in trajectory["steps"] if step["source"] == "agent"]


def every_step(document):
    return agent_steps(document) + [step for sub in document.get("subagent_trajectories") or ()
                                    for step in agent_steps(sub)]


def model_steps(trajectory):
    """A trajectory's steps of the relay's calls, not the closing reply the SDK adds."""
    return [step for step in agent_steps(trajectory) if "model_name" in step]


def results(step) -> dict:
    return {result["source_call_id"]: result for result in (step.get("observation") or {}).get("results") or ()}


def test_a_call_is_one_step_with_its_thought_text_and_calls_and_the_relays_model_time_and_usage():
    feed = Feed(effort="high")
    first = feed.request()
    feed.chunk(Thought(step_index=1, text="I'll write the script, then run it."))
    feed.chunk(Text(step_index=1, text="Let me ")).chunk(Text(step_index=1, text="write it."))
    feed.end(first, counted=usage(1200, 40))
    ended = feed.records[-1].record["timestamp"]
    feed.tool("toolu_1", "run_command", 2, "Created At: 2026-11-27T16:00:01Z\nOutput:\n5050")
    second = feed.request()
    feed.end(second, counted=usage(1300, 4, cached=1200)).chunk(Text(step_index=3, text="5050")).close()
    kit = run(feed)
    one, two = agent_steps(kit.document)

    assert (one["model_name"], one["reasoning_effort"], one["timestamp"]) == (MODEL, "high", ended)
    assert one["reasoning_content"] == "I'll write the script, then run it."
    assert profile_of(one)["reasoning"] == [{"chars": 35, "visibility": "summary"}]
    assert one["message"] == "Let me write it."
    assert one["tool_calls"] == [{"tool_call_id": f"{CONV}:2", "function_name": "run_command",
                                  "arguments": {"command_line": "python3 hello.py"}}]
    assert results(one)[f"{CONV}:2"]["content"] == "Created At: 2026-11-27T16:00:01Z\nOutput:\n5050"
    assert one["metrics"] == {"prompt_tokens": 1200, "cached_tokens": 0, "completion_tokens": 40}
    assert (two["message"], two["metrics"]["cached_tokens"]) == ("5050", 1200)
    assert kit.document["final_metrics"]["total_prompt_tokens"] == feed.calls.usage(1).input_tokens == 2500


def test_a_calls_text_the_bridge_flushes_after_the_next_request_is_still_that_calls():
    """The recorded shape: the bridge holds a text until the next chunk, which comes once the harness has asked again,
    as when a sub-agent's message comes in after the answer."""
    feed = Feed().answered(counted=usage(900, 30)).tool("toolu_1", "run_command", 1, "forty-two")
    feed.answered(counted=usage(1000, 3))
    third = feed.request()
    feed.chunk(Text(step_index=2, text="forty-two\n")).chunk(Text(step_index=4, text="forty-two"))
    feed.end(third, counted=usage(1100, 2)).close()
    steps = model_steps(run(feed).document)

    assert [(step["message"], step["metrics"]["prompt_tokens"]) for step in steps] == [
        ("", 900), ("forty-two", 1000), ("forty-two", 1100)]


def test_a_thought_the_bridge_flushes_while_the_next_call_is_in_flight_is_that_calls():
    """A response's thought is held until its text begins, while the call still streams."""
    feed = Feed().answered(counted=usage(900, 30)).tool("toolu_1", "run_command", 1, "5050")
    second = feed.request()
    feed.chunk(Thought(step_index=2, text="It printed 5050.")).chunk(Text(step_index=2, text="5050"))
    feed.end(second, counted=usage(1000, 3)).close()
    one, two = agent_steps(run(feed).document)

    assert ("reasoning_content" in one, one["tool_calls"][0]["tool_call_id"]) == (False, f"{CONV}:1")
    assert (two["reasoning_content"], two["message"], two["metrics"]["prompt_tokens"]) == ("It printed 5050.", "5050",
                                                                                          1000)


def test_a_thought_the_bridge_flushes_while_its_call_is_in_flight_is_that_calls_after_a_call_that_streamed_nothing():
    """The coding recording's first call streamed nothing: the next call's thought, held until its text begins, is
    still the call in flight's, not the ended one's."""
    feed = Feed().answered(counted=usage(900, 30))
    second = feed.request()
    feed.chunk(Thought(step_index=2, text="Write it first.")).chunk(Text(step_index=2, text="Writing hello.py."))
    feed.end(second, counted=usage(1000, 40)).tool("toolu_1", "run_command", 3, "5050")
    feed.answered(counted=usage(1100, 3)).chunk(Text(step_index=4, text="5050")).close()
    one, two, three = agent_steps(run(feed).document)

    assert (one["message"], "reasoning_content" in one, one["metrics"]["prompt_tokens"]) == ("", False, 900)
    assert (two["reasoning_content"], two["message"], two["tool_calls"][0]["tool_call_id"]) == (
        "Write it first.", "Writing hello.py.", f"{CONV}:3")
    assert three["message"] == "5050"


def test_a_calls_text_the_bridge_writes_once_the_next_call_ended_is_still_that_calls():
    """The recorded shape of the turn waiting on a sub-agent, one chunk later: a text-only call's thought, then its
    text, which the bridge holds until the next chunk, comes once the turn's next call has ended too."""
    feed = Feed().answered(counted=usage(1000, 20)).tool("toolu_1", "invoke_subagent", 1, "started", args={})
    feed.answered(other_messages(TASK), usage(400, 10)).chunk(Text(step_index=1, text="Creating notes.txt."))
    waiting = feed.request()
    feed.chunk(Thought(step_index=2, text="Wait for it.")).end(waiting, counted=usage(1100, 5))
    feed.chunk(Text(step_index=2, text="Waiting for the sub-agent."))
    feed.answered(counted=usage(1200, 5)).tool("toolu_2", "run_command", 3, "forty-two")
    feed.answered(counted=usage(1300, 2)).chunk(Text(step_index=4, text="forty-two")).close()
    steps = model_steps(run(feed).document)

    assert [(step.get("reasoning_content"), step["message"], step["metrics"]["prompt_tokens"]) for step in steps] == [
        (None, "", 1000), ("Wait for it.", "Waiting for the sub-agent.", 1100), (None, "", 1200),
        (None, "forty-two", 1300)]
    assert steps[2]["tool_calls"][0]["tool_call_id"] == f"{CONV}:3"


def test_a_refused_call_is_a_step_with_no_model_call_holding_its_error_before_its_retry():
    feed = Feed()
    first = feed.request()
    feed.end(first, counted=usage(900, 30)).chunk(Text(step_index=1, text="Running it."))
    feed.tool("toolu_1", "run_command", 2, "5050")
    feed.end(feed.request(), status=503)
    feed.answered(counted=usage(1000, 3)).chunk(Text(step_index=3, text="5050")).close()
    one, refused, retried = agent_steps(run(feed).document)

    assert one["tool_calls"][0]["tool_call_id"] == f"{CONV}:2" and one["message"] == "Running it."
    assert (refused["llm_call_count"], refused["model_name"], "metrics" in refused) == (0, MODEL, False)
    assert profile_of(refused)["error"] == {"kind": "infra_error",
                                            "message": "the endpoint refused the call (HTTP 503)"}
    assert (retried["message"], retried["metrics"]["prompt_tokens"]) == ("5050", 1000)


def test_a_call_the_relay_refused_holds_the_relays_reason_and_a_turn_failed_there_is_failed():
    reason = "the model call isn't JSON this agent can read, so the model can't be told the env's time"
    feed = Feed()
    feed.end(feed.request(refuse=reason)).close()
    (step,) = agent_steps(run(feed, state="failed").document)

    assert step["llm_call_count"] == 0
    assert profile_of(step)["error"] == {"kind": "infra_error", "message": reason}
    assert profile_of(step)["state"] == "failed"


def test_a_call_answered_without_usage_is_a_step_with_its_model_and_no_counts_so_no_totals():
    feed = Feed()
    first = feed.request()
    feed.end(first).chunk(Text(step_index=1, text="ok")).close()
    kit = run(feed)
    (step,) = agent_steps(kit.document)

    assert (step["model_name"], step["llm_call_count"], "metrics" in step) == (MODEL, 1, False)
    assert "total_prompt_tokens" not in kit.document["final_metrics"]


@pytest.mark.parametrize("returned", [
    pytest.param(lambda data: {"content": [{"type": "text", "text": "### Result"},
                                           {"type": "image", "data": data, "mimeType": "image/png"}]}, id="mcp"),
    pytest.param(lambda data: ["### Result", {"inline_data": {"mime_type": "image/png", "data": data}}],
                 id="inline-data"),
    pytest.param(lambda data: ["### Result", f"data:image/png;base64,{data}"], id="data-url"),
])
def test_an_image_a_tool_returns_is_a_media_file_named_by_its_bytes(returned):
    image = png()
    feed = Feed().answered()
    feed.tool("toolu_1", "browser_take_screenshot", 1, returned(base64.b64encode(image).decode()))
    feed.answered().chunk(Text(step_index=2, text="Example Domain")).close()
    kit = run(feed)

    result = results(agent_steps(kit.document)[0])[f"{CONV}:1"]
    text, picture = result["content"]
    assert text == {"type": "text", "text": "### Result"} and kit.media[picture["source"]["path"]] == image
    assert profile_of(result)["media"][picture["source"]["path"]]["width"] == 4


def test_a_failed_tool_call_is_its_result_with_the_error_and_a_result_whose_call_never_streamed_is_its_own_call():
    feed = Feed().answered()
    feed.hooked("toolu_1", "run_command").chunk(streamed("run_command", 1))
    feed.failed("toolu_1", "run_command", "exit status 1")
    feed.result("toolu_9", "peer_agents_send", "the reviewer says yes")
    feed.answered().chunk(Text(step_index=2, text="done")).close()
    step = agent_steps(run(feed).document)[0]

    assert [(call["tool_call_id"], call["function_name"]) for call in step["tool_calls"]] == [
        (f"{CONV}:1", "run_command"), ("toolu_9", "peer_agents_send")]
    assert (results(step)[f"{CONV}:1"]["content"], profile_of(results(step)[f"{CONV}:1"])["status"]) == (
        "exit status 1", "error")
    assert results(step)["toolu_9"]["content"] == "the reviewer says yes"


def test_a_file_the_sdk_streams_as_edit_file_takes_the_create_file_result_the_bridge_left_unbound_in_its_trajectory():
    """An older bridge bound a result by name alone, and the SDK names a created file's step edit_file and its hook's
    call create_file: the result came unbound at the turn's end, its step id naming its trajectory."""
    feed = Feed().answered()
    feed.chunk(streamed("edit_file", 1, args={"file_path": "/app/hello.py"}))
    feed.answered().chunk(Text(step_index=2, text="5050")).close()
    feed.native({"kind": "tool_result", "name": "create_file", "id": "toolu_1", "step_id": f"{CONV}:4",
                 "result": "Create hello.py", "error": None, "server_name": None, "call_ref": None})
    one, two = agent_steps(run(feed).document)

    assert [call["function_name"] for call in one["tool_calls"]] == ["edit_file"] and "tool_calls" not in two
    assert results(one)[f"{CONV}:1"]["content"] == "Create hello.py"


def test_a_sub_agent_that_runs_on_after_its_spawn_returned_is_embedded_with_its_calls_and_what_it_streamed():
    """The recorded shape: invoke_subagent returns at once, the sub-agent's calls run beside the turn's, its tool calls
    name its own trajectory, and the turn's answer and its last text come while the turn's next call is in flight."""
    feed = Feed().answered(counted=usage(1000, 20))
    feed.tool("toolu_1", "invoke_subagent", 1, "Subagent File Creation", args={})
    root = feed.request()
    sub = feed.request(other_messages(TASK))
    feed.end(root, counted=usage(1100, 10)).tool("toolu_2", "run_command", 2, "cat: notes.txt: No such file")
    root = feed.request()
    feed.end(sub, counted=usage(400, 10))
    feed.hooked("toolu_3", "create_file").chunk(streamed("edit_file", 1, SUB, {"file_path": "/app/notes.txt"}))
    sub = feed.request(other_messages(TASK))
    feed.end(root, counted=usage(1200, 10)).tool("toolu_4", "run_command", 3, "forty-two")
    feed.result("toolu_3", "create_file", "File creation", SUB)
    root = feed.request()
    feed.end(sub, counted=usage(450, 12))
    sub = feed.request(other_messages(TASK))
    feed.end(root, counted=usage(1300, 2))
    root = feed.request()
    feed.chunk(Text(step_index=4, text="forty-two"))
    feed.chunk(Text(step_index=3, text="Task complete: notes.txt holds forty-two."))
    feed.end(sub, counted=usage(500, 8))
    feed.chunk(Text(step_index=6, text="forty-two")).end(root, counted=usage(1400, 2)).close()
    kit = run(feed)
    document = kit.document

    (trajectory,) = document["subagent_trajectories"]
    spawn = results(agent_steps(document)[0])[f"{CONV}:1"]
    assert spawn["subagent_trajectory_ref"] == [{"trajectory_id": trajectory["trajectory_id"]}]
    subs = agent_steps(trajectory)
    assert [step["metrics"]["prompt_tokens"] for step in subs] == [400, 450, 500]
    assert [call["tool_call_id"] for call in subs[0]["tool_calls"]] == [f"{SUB}:1"]
    assert results(subs[0])[f"{SUB}:1"]["content"] == "File creation"
    assert subs[2]["message"] == "Task complete: notes.txt holds forty-two."
    assert [(step["message"], step["metrics"]["prompt_tokens"]) for step in model_steps(document)] == [
        ("", 1000), ("", 1100), ("", 1200), ("forty-two", 1300), ("forty-two", 1400)]
    assert document["final_metrics"]["total_prompt_tokens"] == feed.calls.usage(1).input_tokens == 7350


def test_a_sub_agent_given_the_users_words_is_a_sub_agent_and_the_turns_conversation_compacted_stays_the_turns():
    quoted = other_messages(f"The user asked: {PROMPT} Write hello.py for them.")
    compacted = [{"role": "user", "content": f"Summary of the work so far.\n\n{PROMPT}"}]
    feed = Feed().answered(counted=usage(1000, 20))
    feed.tool("toolu_1", "invoke_subagent", 1, "started", args={})
    feed.answered(quoted, usage(400, 10))
    feed.answered(compacted, usage(300, 5)).chunk(Text(step_index=2, text="5050")).close()
    document = run(feed).document

    (sub,) = document["subagent_trajectories"]
    assert [step["metrics"]["prompt_tokens"] for step in agent_steps(sub)] == [400]
    assert [(step["message"], step["metrics"]["prompt_tokens"]) for step in agent_steps(document)] == [
        ("", 1000), ("5050", 300)]


def test_every_call_of_a_sub_agent_given_the_users_words_is_its_own_and_the_turns_calls_stay_the_turns():
    """Its later calls hold the turn's message too, while no sub-agent waits for a first call: its claimed conversation
    keeps them its own, not the turn's conversation compacted."""
    quoted = other_messages(f"The user asked: {PROMPT} Write hello.py for them.")
    feed = Feed().answered(counted=usage(1000, 20)).tool("toolu_1", "invoke_subagent", 1, "started", args={})
    feed.answered(quoted, usage(400, 10)).answered(quoted, usage(410, 10))
    feed.answered(counted=usage(1100, 5)).answered(quoted, usage(420, 10))
    feed.answered(counted=usage(1200, 5)).chunk(Text(step_index=3, text="5050")).close()
    document = run(feed).document

    (sub,) = document["subagent_trajectories"]
    assert [step["metrics"]["prompt_tokens"] for step in agent_steps(sub)] == [400, 410, 420]
    assert [(step["message"], step["metrics"]["prompt_tokens"]) for step in agent_steps(document)] == [
        ("", 1000), ("", 1100), ("5050", 1200)]


def test_a_steps_text_the_bridge_writes_as_two_records_another_trajectorys_text_between_is_that_steps():
    """The bridge joins a step's text until another chunk comes: the sub-agent's answer, the turn's text between its
    parts, is two records of one step, both the sub-agent's."""
    feed = Feed().answered(counted=usage(1000, 20)).tool("toolu_1", "invoke_subagent", 1, "started", args={})
    sub, root = feed.request(other_messages(TASK)), feed.request()
    feed.chunk(Text(step_index=1, text="Task ")).chunk(Text(step_index=2, text="Waiting."))
    feed.chunk(Text(step_index=1, text="complete.")).end(sub, counted=usage(400, 10))
    feed.end(root, counted=usage(1100, 5)).close()
    document = run(feed).document

    (sub_step,) = agent_steps(document["subagent_trajectories"][0])
    assert (sub_step["message"], sub_step["metrics"]["prompt_tokens"]) == ("Task complete.", 400)
    assert [(step["message"], step["metrics"]["prompt_tokens"]) for step in model_steps(document)] == [
        ("", 1000), ("Waiting.", 1100)]


def test_the_turns_thought_its_tool_call_flushes_once_its_call_ended_is_its_beside_a_sub_agent_streaming_nothing_yet():
    """Each trajectory counts its step indices up from its start: the step after the turn's latest is the turn's next,
    not the first of a sub-agent in flight that has streamed nothing; and the sub-agent's own first thought stays its
    own."""
    feed = Feed().answered(counted=usage(1000, 20)).tool("toolu_1", "invoke_subagent", 1, "started", args={})
    sub, root = feed.request(other_messages(TASK)), feed.request()
    feed.chunk(Thought(step_index=2, text="Check the file.")).end(root, counted=usage(1100, 5))
    feed.tool("toolu_2", "run_command", 2, "cat: notes.txt: No such file")
    feed.chunk(Thought(step_index=1, text="Create notes.txt.")).chunk(Text(step_index=1, text="Creating it."))
    feed.end(sub, counted=usage(400, 10)).answered(counted=usage(1200, 5)).close()
    document = run(feed).document

    (sub_step,) = agent_steps(document["subagent_trajectories"][0])
    assert (sub_step["reasoning_content"], sub_step["message"]) == ("Create notes.txt.", "Creating it.")
    one, two, three = model_steps(document)
    assert (two["reasoning_content"], two["tool_calls"][0]["tool_call_id"], two["metrics"]["prompt_tokens"]) == (
        "Check the file.", f"{CONV}:2", 1100)


def test_a_sub_agents_calls_before_its_spawn_streamed_move_into_it():
    feed = Feed().answered()
    feed.hooked("toolu_1", "invoke_subagent")
    feed.answered(other_messages("Count the files."), usage(400, 10))
    feed.chunk(streamed("invoke_subagent", 1))
    feed.answered(other_messages("Count the files."), usage(450, 12))
    feed.result("toolu_1", "invoke_subagent", "1")
    feed.answered().chunk(Text(step_index=2, text="1")).close()
    document = run(feed).document

    (sub,) = document["subagent_trajectories"]
    assert [step["metrics"]["prompt_tokens"] for step in agent_steps(sub)] == [400, 450]
    assert len(agent_steps(document)) == 2


def test_a_call_of_another_conversation_with_no_sub_agent_waiting_is_a_root_step_before_the_turns_next():
    feed = Feed()
    first = feed.request()
    feed.chunk(Thought(step_index=1, text="Run it.")).chunk(Text(step_index=1, text="Running it."))
    feed.answered(other_messages("Write a title for this conversation."), usage(300, 6))
    feed.end(first, counted=usage(900, 30)).tool("toolu_1", "run_command", 2, "5050")
    feed.answered(counted=usage(1000, 3)).chunk(Text(step_index=3, text="5050")).close()
    steps = agent_steps(run(feed).document)

    assert [(step["message"], step["metrics"]["prompt_tokens"]) for step in steps] == [
        ("Running it.", 900), ("", 300), ("5050", 1000)]


def test_each_task_of_a_resumed_conversation_writes_its_own_steps_its_streams_ids_kept_unique():
    first = Feed().answered().tool("toolu_1", "run_command", 1, "5050")
    first.answered().chunk(Text(step_index=2, text="5050")).close()
    prompt = "Now sum 1 to 1000 instead."
    second = Feed(prompt=prompt)
    second.answered(turn_messages(PROMPT) + turn_messages(prompt)[1:])
    second.tool("toolu_2", "run_command", 1, "500500")
    second.answered(turn_messages(PROMPT) + turn_messages(prompt)[1:]).chunk(Text(step_index=4, text="500500"))
    kit = run(first, second.close())
    steps = agent_steps(kit.document)

    assert [step["message"] for step in steps] == ["", "5050", "", "500500"]
    assert [call["tool_call_id"] for step in steps for call in step.get("tool_calls") or ()] == [
        f"{CONV}:1", f"{CONV}:1#2"]
    assert [step["source"] for step in kit.document["steps"]].count("user") == 2


def test_a_turn_cut_at_its_time_keeps_the_results_its_hooks_gave_and_the_call_the_relay_ended_after():
    feed = Feed().answered()
    feed.hooked("toolu_1", "run_command").result("toolu_1", "run_command", "partial")
    cut = feed.request()
    feed.close().end(cut, counted=usage(1000, 2))
    step, last = agent_steps(run(feed, state="failed").document)

    assert step["tool_calls"][0]["tool_call_id"] == "toolu_1" and results(step)["toolu_1"]["content"] == "partial"
    assert (last["metrics"]["prompt_tokens"], profile_of(last)["state"]) == (1000, "failed")


def test_from_atif_gives_the_sdks_chunks_and_results_back():
    feed = Feed().answered().chunk(Text(step_index=1, text="Running it."))
    feed.tool("toolu_1", "run_command", 2, "5050")
    feed.answered().chunk(Text(step_index=3, text="5050")).close()
    log = AntigravityAtif.from_atif(strip_profile(run(feed).document)).log

    assert [record.get("type") or record["kind"] for record in log] == ["Text", "ToolCall", "tool_result", "Text"]
    assert log[2] == {"kind": "tool_result", "id": f"{CONV}:2", "name": "run_command", "result": "5050",
                      "call_ref": f"{CONV}:2"}


def seeded_session(seed: int) -> tuple[list[Feed], dict]:
    """A random conversation of one to three tasks, in the order the recorded runs show: calls of thoughts and text,
    held by the bridge until the next chunk, and tool calls streamed once their call has ended, with results, some
    failed, some images, some files the SDK streams as edit_file, some spawning a sub-agent that runs its calls of its
    own trajectory and ends on a text; refused attempts before some calls, calls answered without usage, calls of other
    conversations between. With what the copy must hold: a task's answer only where no sub-agent ran, since the SDK
    names no trajectory for a text, and one held behind a call of the turn's that ended after can be either's."""
    rng = random.Random(seed)
    image = base64.b64encode(png()).decode()
    want = {"calls": 0, "refused": 0, "tools": 0, "subagents": [], "answers": []}
    feeds = []
    ids = iter(range(10**6))
    index = iter(range(1, 10**6))
    for _ in range(rng.randint(1, 3)):
        feed, answer, spawned = Feed(), "", False
        count = rng.randint(1, 4)
        for call_index in range(count):
            last = call_index == count - 1
            while rng.random() < 0.2:
                feed.end(feed.request(), status=rng.choice([429, 500, 503]))
                want["calls"] += 1
                want["refused"] += 1
            call = feed.request()
            want["calls"] += 1
            step = next(index)
            text = rng.choice(["5050", "It printed 42."]) if last else rng.choice(["", "", "Running it."])
            live = rng.random() < 0.5
            if rng.random() < 0.5:
                feed.chunk(Thought(step_index=step, text="Plan the next step."))
            if text and live:
                feed.chunk(Text(step_index=step, text=text))
            feed.end(call, counted=None if rng.random() < 0.1 else usage(rng.randint(500, 9000), rng.randint(1, 90)))
            if text and not live:
                feed.chunk(Text(step_index=step, text=text))
            answer = text if last else answer
            for _ in range(0 if last else rng.choice([1, 1, 2])):
                name = rng.choice(["run_command", "view_file", "browser_take_screenshot", "create_file",
                                   "invoke_subagent"])
                call_id = f"toolu_{next(ids)}"
                feed.hooked(call_id, name).chunk(streamed("edit_file" if name == "create_file" else name, next(index)))
                if rng.random() < 0.2:
                    feed.failed(call_id, name, "exit status 1")
                else:
                    feed.result(call_id, name, [{"type": "image", "data": image, "mimeType": "image/png"}]
                                if name == "browser_take_screenshot" else "ok")
                want["tools"] += 1
                if name == "invoke_subagent":
                    spawned = True
                    task, trajectory, sub_index = f"Subtask {next(ids)}.", f"sub{next(ids)}", iter(range(1, 10**6))
                    calls = rng.randint(1, 3)
                    for sub_call in range(calls):
                        feed.answered(other_messages(task), usage(rng.randint(100, 900), rng.randint(1, 20)))
                        if sub_call < calls - 1:
                            sub_id = f"toolu_{next(ids)}"
                            feed.hooked(sub_id, "run_command").chunk(streamed("run_command", next(sub_index),
                                                                              trajectory))
                            feed.result(sub_id, "run_command", "ok", trajectory)
                            want["tools"] += 1
                        else:
                            feed.chunk(Text(step_index=next(sub_index), text="Subtask done."))
                    want["calls"] += calls
                    want["subagents"].append(calls)
            if not last and rng.random() < 0.2:
                feed.answered(other_messages("Write a title for this conversation."), usage(300, 6))
                want["calls"] += 1
        want["answers"].append(None if spawned else answer)
        feeds.append(feed.close())
    return feeds, want


@pytest.mark.parametrize("seed", range(150))
def test_a_seeded_conversations_copy_holds_every_call_the_relay_forwarded_and_every_tool_call(seed):
    feeds, want = seeded_session(seed)
    document = run(*feeds).document
    steps = every_step(document)
    turns = [feed.calls.usage(0) for feed in feeds]

    assert len([step for step in steps if "model_name" in step]) == want["calls"]
    for count, total in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens")):
        assert sum(step.get("metrics", {}).get(count, 0) for step in steps) == sum(
            getattr(turn, total) or 0 for turn in turns)
    refused = [step for step in steps if "model_name" in step and step["llm_call_count"] == 0]
    assert refused == [step for step in steps if "error" in profile_of(step)] and len(refused) == want["refused"]
    calls = [call["tool_call_id"] for step in steps for call in step.get("tool_calls") or ()]
    assert len(calls) == want["tools"] and set(calls) == {key for step in steps for key in results(step)}
    subs = document.get("subagent_trajectories") or ()
    assert [len(agent_steps(sub)) for sub in subs] == want["subagents"]
    assert all(call["tool_call_id"].startswith("sub") for sub in subs for step in agent_steps(sub)
               for call in step.get("tool_calls") or ())
    assert sum(step["message"].count("Subtask done.") for step in steps if "model_name" in step) == len(subs)
    tasks: list[list[dict]] = []
    for step in document["steps"]:
        tasks.append([]) if step["source"] == "user" else tasks[-1].append(step)
    assert [next(step["message"] for step in reversed(task) if step.get("llm_call_count") == 1)
            for task, answer in zip(tasks, want["answers"]) if answer is not None] == [
        answer for answer in want["answers"] if answer is not None]


def test_the_tested_sdk_is_the_one_the_image_pins():
    pinned = [line.split("==", 1)[1] for line in (AGENT_DIR / "requirements.txt").read_text().splitlines()
              if line.startswith("google-antigravity==")]
    assert antigravity_atif.TESTED_WITH == tuple(pinned) == (antigravity_bridge.SDK_VERSION,)


def test_the_bridge_binds_a_create_file_result_to_the_edit_file_call_the_sdk_streamed():
    events: list[dict] = []
    recorder = antigravity_bridge.TrajectoryRecorder(events.append)
    recorder.hooked_call(ToolCall(id="toolu_1", name="create_file"))
    recorder.chunk(streamed("edit_file", 2, args={"file_path": "/app/hello.py"}))
    recorder.result({"kind": "tool_result", "id": "toolu_1", "name": "create_file", "result": "Create hello.py"})
    recorder.close()

    assert [(event.get("type") or event["kind"], event.get("call_ref")) for event in events] == [
        ("ToolCall", None), ("tool_result", f"{CONV}:2")]


@pytest.mark.parametrize("sent, options", [
    ({"stream": True}, {"include_usage": True}),
    ({"stream": True, "stream_options": {"include_obfuscation": False}},
     {"include_obfuscation": False, "include_usage": True}),
    ({"stream": True, "stream_options": {"include_usage": True}}, {"include_usage": True}),
    ({}, None),
])
def test_a_streamed_model_call_asks_for_its_usage_which_the_relay_counts_it_by(sent, options):
    """The harness asks for none, and LiteLLM's stream then carries no usage: every call would be unreported."""
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}], **sent}).encode()

    rewritten = json.loads(asyncio.run(antigravity_bridge.turn_rewrite(None, [])(body)))

    assert rewritten.get("stream_options") == options


def test_a_model_call_the_endpoint_refuses_fails_the_turn_with_the_sdks_reason_as_an_infra_error():
    """The SDK ends such a turn with its connection error, which isn't its execution error."""
    reason = 'Agent execution terminated due to error. ("request failed (code 400): refused")'
    fake_sdk.Agent.turns.append(Turn(steps=[("raise", fake_sdk.AntigravityConnectionError(reason))]))

    result = _bridge()

    assert (result.error_code, result.error, result.infra) == ("antigravity_error", reason, True)
    assert result.usage is not None


class Relay:
    """The turn's model relay without its loopback server: a call it forwards has its request rewritten and is added to
    its calls in the one task, as the relay does."""

    def __init__(self) -> None:
        self.opened: list[dict] = []

    @asynccontextmanager
    async def __call__(self, upstream, calls, **kwargs):
        self.opened.append({"upstream": upstream, "calls": calls, **kwargs})
        yield "http://relay.test/token"

    async def forward(self, messages, counted, status=200):
        opened = self.opened[-1]
        body = json.dumps({"model": MODEL, "messages": messages}).encode()
        started = opened["calls"].clock()
        try:
            await opened["rewrite"](body)
        except model_relay.Refusal:
            status = 400
        opened["calls"].add(status, counted if status < 400 else None, MODEL, started=started)


@pytest.fixture
def forwarding(monkeypatch) -> Relay:
    relay = Relay()
    monkeypatch.setattr(antigravity_bridge, "model_relay", relay)
    return relay


def _turn(relay: Relay, prompt: str) -> Turn:
    return Turn(steps=[
        ("await", lambda: relay.forward(turn_messages(prompt), usage(1200, 40))),
        ("chunk", Text(step_index=1, text="Let me run it.")),
        ("call", ToolCall(id="toolu_1", name="run_command")),
        ("chunk", streamed("run_command", 2)),
        ("result", ToolResult(id="toolu_1", name="run_command", step_id=f"{CONV}:4", result="5050")),
        ("await", lambda: relay.forward(turn_messages(prompt), usage(1300, 4, cached=1200))),
        ("chunk", Text(step_index=3, text="5050")),
    ])


def _bridge(aux_callback=None, effort=None) -> antigravity_bridge.AntigravityResult:
    return asyncio.run(antigravity_bridge.run_antigravity(
        PROMPT, model=MODEL, effort=effort, session_id=None, mcp_servers=None, system_prompt=None, timeout_s=30,
        role=None, builtin_tools=None, web_tools=True, max_turns=None, workspace="/w", event_callback=lambda _: None,
        aux_callback=aux_callback))


def test_the_bridge_feeds_each_call_its_request_then_its_end_with_the_same_number(forwarding):
    fake_sdk.Agent.turns.append(_turn(forwarding, PROMPT))
    fed: list[tuple[str, dict]] = []
    result = _bridge(lambda name, record: fed.append((name, record)), effort="xhigh")

    assert {name for name, _ in fed} == {"calls"}
    records = [record for _, record in fed]
    assert [(record["type"], record["seq"]) for record in records] == [
        ("agentenv.model_request", 1), ("agentenv.model_call", 1), ("agentenv.model_request", 2),
        ("agentenv.model_call", 2)]
    request, ended = records[:2]
    assert {key: request[key] for key in ("via", "turn", "model", "reasoning_effort")} == {
        "via": "relay", "turn": True, "model": MODEL, "reasoning_effort": "high"}
    assert request["conversation"] == antigravity_bridge.message_digest(turn_messages()[1]["content"])
    assert request["started"].endswith("Z") and ended["started"] <= ended["timestamp"]
    assert (ended["outcome"], ended["usage"]["input_tokens"]) == ("answered", 1200)
    assert result.usage.input_tokens == 2500


def test_a_call_of_another_conversation_is_fed_as_not_the_turns_and_a_refusal_with_its_reason():
    async def refuse(_body):
        raise model_relay.Refusal("no env time")

    fed: list[dict] = []
    feed = antigravity_bridge.CallFeed(lambda _, record: fed.append(record), PROMPT, None)

    async def calls():
        await feed.rewrite(lambda body: asyncio.sleep(0, body))(json.dumps(
            {"model": MODEL, "messages": other_messages("Count the files.")}).encode())
        model_relay.Calls(sink=feed.sink).add(200, usage(), MODEL)
        with pytest.raises(model_relay.Refusal):
            await feed.rewrite(refuse)(b"{}")
        model_relay.Calls(sink=feed.sink).add(400, None, None)

    asyncio.run(calls())

    assert (fed[0]["turn"], fed[0]["conversation"]) == (False, antigravity_bridge.message_digest("Count the files."))
    assert (fed[3]["seq"], fed[3]["outcome"], fed[3]["refusal"]) == (2, "refused", "no env time")


def test_a_call_whose_request_the_feed_never_saw_is_fed_without_a_number():
    fed: list[dict] = []
    feed = antigravity_bridge.CallFeed(lambda _, record: fed.append(record), PROMPT, None)

    model_relay.Calls(sink=feed.sink).add(None, None, None)

    assert "seq" not in fed[0] and fed[0]["outcome"] == "refused"


def test_without_an_aux_callback_the_relay_only_counts_the_calls(forwarding):
    fake_sdk.Agent.turns.append(_turn(forwarding, PROMPT))
    result = _bridge()

    (opened,) = forwarding.opened
    assert opened["calls"].sink is None
    assert result.usage.input_tokens == 2500


def test_the_feed_stamps_calls_on_the_session_clock_where_the_image_follows_the_envs(monkeypatch):
    monkeypatch.setattr(antigravity_bridge, "follows", lambda: True)
    monkeypatch.setattr(antigravity_bridge, "session_time", lambda: 1_000_000_000.0)

    feed = antigravity_bridge.CallFeed(lambda *_: None, PROMPT, None)

    assert feed.clock() == datetime(2001, 9, 9, 1, 46, 40, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_a_task_that_builds_atif_is_fed_its_calls_and_serves_one_step_per_call(ag, forwarding):
    fake_sdk.Agent.turns.append(_turn(forwarding, PROMPT))
    async with ag.client() as client:
        (await client.post("/ext/trajectory/prescribe", json={"formats": ["atif/1"]})).raise_for_status()
        task = await ag.send(client, PROMPT)
        read = (await client.post("/ext/trajectory", json={"task_id": task["id"], "format": "atif/1"})).json()

    document = read["trajectory"]
    one, two = agent_steps(document)
    assert task["status"]["state"] == "completed"
    assert {entry["name"] for entry in read["aux"]} == {"calls"}
    assert (one["message"], one["tool_calls"][0]["tool_call_id"], results(one)[f"{CONV}:2"]["content"]) == (
        "Let me run it.", f"{CONV}:2", "5050")
    assert (two["message"], two["metrics"]["prompt_tokens"]) == ("5050", 1300)
    assert profile_of(document["steps"][-1])["state"] == "completed"
    assert document["final_metrics"]["total_prompt_tokens"] == 2500


@pytest.mark.asyncio
async def test_a_task_that_builds_no_atif_copy_feeds_its_relay_no_sink(ag, forwarding):
    fake_sdk.Agent.turns.append(_turn(forwarding, PROMPT))
    async with ag.client() as client:
        task = await ag.send(client, PROMPT)

    assert task["status"]["state"] == "completed"
    assert forwarding.opened[0]["calls"].sink is None
