"""Antigravity's ATIF adapter on records made as the agent makes them: the SDK's chunks and hook results through the
bridge's own TrajectoryRecorder, the native log, and the turn's model calls through its CallFeed and the relay's Calls,
the ``calls`` aux file. Each case runs through the protocol's adapter kit, so the live rule holds after every record
and the document validates as a producer's. Then the bridge and the server feeding them."""

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

    def result(self, call_id, name, result):
        self.recorder.result({"kind": "tool_result", **antigravity_bridge._plain(
            ToolResult(id=call_id, name=name, result=result))})
        return self

    def failed(self, call_id, name, error):
        self.recorder.result({"kind": "tool_result", "id": call_id, "name": name, "server_name": None, "result": None,
                              "error": error})
        return self

    def tool(self, call_id, stream_id, name, result, step_index=0, args=None):
        """A call the hooks decide, streamed, then its result."""
        self.hooked(call_id, name).chunk(ToolCall(id=stream_id, name=name, step_index=step_index,
                                                  args=args or {"CommandLine": "python3 hello.py"}))
        return self.result(call_id, name, result)

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


def results(step) -> dict:
    return {result["source_call_id"]: result for result in (step.get("observation") or {}).get("results") or ()}


def test_a_call_is_one_step_with_its_thought_text_and_calls_and_the_relays_model_time_and_usage():
    feed = Feed(effort="high")
    first = feed.request()
    feed.chunk(Thought(step_index=0, text="I'll write the script, then run it."))
    feed.chunk(Text(step_index=0, text="Let me ")).chunk(Text(step_index=0, text="write it."))
    feed.end(first, counted=usage(1200, 40))
    ended = feed.records[-1].record["timestamp"]
    feed.tool("toolu_1", "c:1", "run_command", "Created At: 2026-11-27T16:00:01Z\nOutput:\n5050")
    second = feed.request()
    feed.chunk(Text(step_index=2, text="5050")).end(second, counted=usage(1300, 4, cached=1200)).close()
    kit = run(feed)
    one, two = agent_steps(kit.document)

    assert (one["model_name"], one["reasoning_effort"], one["timestamp"]) == (MODEL, "high", ended)
    assert one["reasoning_content"] == "I'll write the script, then run it."
    assert profile_of(one)["reasoning"] == [{"chars": 35, "visibility": "summary"}]
    assert one["message"] == "Let me write it."
    assert one["tool_calls"] == [{"tool_call_id": "c:1", "function_name": "run_command",
                                  "arguments": {"CommandLine": "python3 hello.py"}}]
    assert results(one)["c:1"]["content"] == "Created At: 2026-11-27T16:00:01Z\nOutput:\n5050"
    assert one["metrics"] == {"prompt_tokens": 1200, "cached_tokens": 0, "completion_tokens": 40}
    assert (two["message"], two["metrics"]["cached_tokens"]) == ("5050", 1200)
    assert kit.document["final_metrics"]["total_prompt_tokens"] == feed.calls.usage(1).input_tokens == 2500


def test_what_streams_after_a_calls_end_and_before_the_next_request_is_that_calls():
    feed = Feed()
    first = feed.request()
    feed.end(first, counted=usage(900, 30)).chunk(Text(step_index=0, text="Checking."))
    feed.tool("toolu_1", "c:1", "view_file", "hello")
    feed.answered().chunk(Text(step_index=1, text="hello")).close()
    one, two = agent_steps(run(feed).document)

    assert (one["message"], one["metrics"]["prompt_tokens"], [c["tool_call_id"] for c in one["tool_calls"]]) == (
        "Checking.", 900, ["c:1"])
    assert two["message"] == "hello"


def test_a_chunk_of_the_open_steps_index_streamed_after_the_next_request_stays_in_its_step():
    """The SDK's stream can trail the relay: the call's text, which the recorder holds until its tool call, arrives
    once the harness has asked for the next response."""
    feed = Feed()
    first = feed.request()
    feed.chunk(Thought(step_index=0, text="Run it.")).chunk(Text(step_index=0, text="Running it."))
    feed.end(first, counted=usage(900, 30))
    second = feed.request()
    feed.tool("toolu_1", "c:1", "run_command", "5050")
    feed.chunk(Text(step_index=2, text="5050")).end(second, counted=usage(1000, 3)).close()
    one, two = agent_steps(run(feed).document)

    assert (one["message"], one["tool_calls"][0]["tool_call_id"]) == ("Running it.", "c:1")
    assert (two["message"], two["metrics"]["prompt_tokens"]) == ("5050", 1000)


def test_a_refused_call_is_a_step_with_no_model_call_holding_its_error_before_its_retry():
    feed = Feed()
    first = feed.request()
    feed.chunk(Text(step_index=0, text="Running it.")).end(first, counted=usage(900, 30))
    feed.end(feed.request(), status=503)
    feed.tool("toolu_1", "c:1", "run_command", "5050")
    feed.answered(counted=usage(1000, 3)).chunk(Text(step_index=2, text="5050")).close()
    one, refused, retried = agent_steps(run(feed).document)

    assert one["tool_calls"][0]["tool_call_id"] == "c:1" and one["message"] == "Running it."
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
    feed.chunk(Text(step_index=0, text="ok")).end(first).close()
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
    feed.tool("toolu_1", "c:1", "browser_take_screenshot", returned(base64.b64encode(image).decode()))
    feed.answered().chunk(Text(step_index=2, text="Example Domain")).close()
    kit = run(feed)

    result = results(agent_steps(kit.document)[0])["c:1"]
    text, picture = result["content"]
    assert text == {"type": "text", "text": "### Result"} and kit.media[picture["source"]["path"]] == image
    assert profile_of(result)["media"][picture["source"]["path"]]["width"] == 4


def test_a_failed_tool_call_is_its_result_with_the_error_and_a_result_whose_call_never_streamed_is_its_own_call():
    feed = Feed().answered()
    feed.hooked("toolu_1", "run_command").chunk(ToolCall(id="c:1", name="run_command", step_index=0))
    feed.failed("toolu_1", "run_command", "exit status 1")
    feed.result("toolu_9", "peer_agents_send", "the reviewer says yes")
    feed.answered().chunk(Text(step_index=2, text="done")).close()
    step = agent_steps(run(feed).document)[0]

    assert [(call["tool_call_id"], call["function_name"]) for call in step["tool_calls"]] == [
        ("c:1", "run_command"), ("toolu_9", "peer_agents_send")]
    assert (results(step)["c:1"]["content"], profile_of(results(step)["c:1"])["status"]) == ("exit status 1", "error")
    assert results(step)["toolu_9"]["content"] == "the reviewer says yes"


def test_a_sub_agent_is_embedded_under_the_invoke_subagent_call_that_spawned_it_with_its_calls_usage():
    feed = Feed().answered(counted=usage(1000, 20))
    feed.hooked("toolu_1", "invoke_subagent")
    feed.chunk(ToolCall(id="c:1", name="invoke_subagent", step_index=0, args={"Task": "List the files in /app."}))
    feed.answered(other_messages("List the files in /app."), usage(400, 10))
    feed.answered(other_messages("List the files in /app."), usage(450, 12))
    feed.result("toolu_1", "invoke_subagent", "hello.py")
    feed.answered(counted=usage(1100, 5)).chunk(Text(step_index=2, text="hello.py")).close()
    kit = run(feed)
    document = kit.document

    (sub,) = document["subagent_trajectories"]
    spawn = results(agent_steps(document)[0])["c:1"]
    assert spawn["subagent_trajectory_ref"] == [{"trajectory_id": sub["trajectory_id"]}]
    assert [step["metrics"]["prompt_tokens"] for step in agent_steps(sub)] == [400, 450]
    assert sub["agent"]["model_name"] == MODEL
    assert document["final_metrics"]["total_prompt_tokens"] == feed.calls.usage(1).input_tokens == 2950


def test_a_sub_agent_given_the_users_words_is_a_sub_agent_and_the_turns_conversation_compacted_stays_the_turns():
    quoted = other_messages(f"The user asked: {PROMPT} Write hello.py for them.")
    compacted = [{"role": "user", "content": f"Summary of the work so far.\n\n{PROMPT}"}]
    feed = Feed().answered(counted=usage(1000, 20))
    feed.hooked("toolu_1", "invoke_subagent").chunk(ToolCall(id="c:1", name="invoke_subagent", step_index=0))
    feed.answered(quoted, usage(400, 10))
    feed.result("toolu_1", "invoke_subagent", "written")
    feed.answered(compacted, usage(300, 5)).chunk(Text(step_index=2, text="5050")).close()
    document = run(feed).document

    (sub,) = document["subagent_trajectories"]
    assert [step["metrics"]["prompt_tokens"] for step in agent_steps(sub)] == [400]
    assert [(step["message"], step["metrics"]["prompt_tokens"]) for step in agent_steps(document)] == [
        ("", 1000), ("5050", 300)]


def test_a_sub_agents_calls_before_its_spawn_streamed_move_into_it():
    feed = Feed().answered()
    feed.hooked("toolu_1", "invoke_subagent")
    feed.answered(other_messages("Count the files."), usage(400, 10))
    feed.chunk(ToolCall(id="c:1", name="invoke_subagent", step_index=0))
    feed.answered(other_messages("Count the files."), usage(450, 12))
    feed.result("toolu_1", "invoke_subagent", "1")
    feed.answered().chunk(Text(step_index=2, text="1")).close()
    document = run(feed).document

    (sub,) = document["subagent_trajectories"]
    assert [step["metrics"]["prompt_tokens"] for step in agent_steps(sub)] == [400, 450]
    assert len(agent_steps(document)) == 2


def test_a_call_of_another_conversation_with_no_spawn_waiting_is_a_root_step_after_the_call_it_ran_during():
    feed = Feed()
    first = feed.request()
    feed.chunk(Thought(step_index=0, text="Run it.")).chunk(Text(step_index=0, text="Running it."))
    feed.answered(other_messages("Write a title for this conversation."), usage(300, 6))
    feed.end(first, counted=usage(900, 30)).tool("toolu_1", "c:1", "run_command", "5050")
    feed.answered(counted=usage(1000, 3)).chunk(Text(step_index=2, text="5050")).close()
    steps = agent_steps(run(feed).document)

    assert [(step["message"], step["metrics"]["prompt_tokens"]) for step in steps] == [
        ("Running it.", 900), ("", 300), ("5050", 1000)]


def test_each_task_of_a_resumed_conversation_writes_its_own_steps_its_streams_ids_kept_unique():
    first = Feed().answered().tool("toolu_1", "c:1", "run_command", "5050")
    first.answered().chunk(Text(step_index=2, text="5050")).close()
    prompt = "Now sum 1 to 1000 instead."
    second = Feed(prompt=prompt)
    second.answered(turn_messages(PROMPT) + turn_messages(prompt)[1:])
    second.tool("toolu_2", "c:1", "run_command", "500500")
    second.answered(turn_messages(PROMPT) + turn_messages(prompt)[1:]).chunk(Text(step_index=2, text="500500"))
    kit = run(first, second.close())
    steps = agent_steps(kit.document)

    assert [step["message"] for step in steps] == ["", "5050", "", "500500"]
    assert [call["tool_call_id"] for step in steps for call in step.get("tool_calls") or ()] == ["c:1", "c:1#2"]
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
    feed = Feed().answered().chunk(Text(step_index=0, text="Running it."))
    feed.tool("toolu_1", "c:1", "run_command", "5050")
    feed.answered().chunk(Text(step_index=2, text="5050")).close()
    log = AntigravityAtif.from_atif(strip_profile(run(feed).document)).log

    assert [record.get("type") or record["kind"] for record in log] == ["Text", "ToolCall", "tool_result", "Text"]
    assert log[2] == {"kind": "tool_result", "id": "c:1", "name": "run_command", "result": "5050", "call_ref": "c:1"}


def seeded_session(seed: int) -> tuple[list[Feed], dict]:
    """A random conversation of one to three tasks: calls of thoughts, text streamed before or after the call ends,
    and tool calls with results, some failed, some images, some spawning a sub-agent; refused attempts before some
    calls, calls answered without usage, calls of other conversations between; the stream ids starting again each
    task. With what the copy must hold."""
    rng = random.Random(seed)
    image = base64.b64encode(png()).decode()
    want = {"calls": 0, "refused": 0, "tools": 0}
    feeds = []
    ids = iter(range(10**6))
    for _ in range(rng.randint(1, 3)):
        feed, stream = Feed(), iter(range(1, 10**6))
        count = rng.randint(1, 4)
        for index in range(count):
            last = index == count - 1
            while rng.random() < 0.2:
                feed.end(feed.request(), status=rng.choice([429, 500, 503]))
                want["calls"] += 1
                want["refused"] += 1
            call = feed.request()
            want["calls"] += 1
            if rng.random() < 0.5:
                feed.chunk(Thought(step_index=2 * index, text="Plan the next step."))
            text = rng.choice(["5050", "It printed 42."]) if last else rng.choice(["", "", "Running it."])
            early = rng.random() < 0.5
            if text and early:
                feed.chunk(Text(step_index=2 * index, text=text))
            feed.end(call, counted=None if rng.random() < 0.1 else usage(rng.randint(500, 9000), rng.randint(1, 90)))
            if text and not early:
                feed.chunk(Text(step_index=2 * index, text=text))
            for _ in range(0 if last else rng.choice([1, 1, 2])):
                name = rng.choice(["run_command", "view_file", "browser_take_screenshot", "invoke_subagent"])
                call_id, stream_id = f"toolu_{next(ids)}", f"c:{next(stream)}"
                feed.hooked(call_id, name).chunk(ToolCall(id=stream_id, name=name, step_index=2 * index))
                if name == "invoke_subagent":
                    task = f"Subtask {next(ids)}."
                    for _ in range(rng.randint(1, 2)):
                        feed.answered(other_messages(task), usage(rng.randint(100, 900), rng.randint(1, 20)))
                        want["calls"] += 1
                if rng.random() < 0.2:
                    feed.failed(call_id, name, "exit status 1")
                else:
                    feed.result(call_id, name, [{"type": "image", "data": image, "mimeType": "image/png"}]
                                if name == "browser_take_screenshot" else "ok")
                want["tools"] += 1
            if not last and rng.random() < 0.2:
                feed.answered(other_messages("Write a title for this conversation."), usage(300, 6))
                want["calls"] += 1
        feeds.append(feed.close())
    return feeds, want


@pytest.mark.parametrize("seed", range(150))
def test_a_seeded_conversations_copy_holds_every_call_the_relay_forwarded_and_every_tool_call(seed):
    feeds, want = seeded_session(seed)
    document = run(*feeds).document
    steps = every_step(document)
    turns = [feed.calls.usage(0) for feed in feeds]

    assert len(steps) == want["calls"]
    for count, total in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens")):
        assert sum(step.get("metrics", {}).get(count, 0) for step in steps) == sum(
            getattr(turn, total) or 0 for turn in turns)
    assert [step for step in steps if step["llm_call_count"] == 0] == [
        step for step in steps if "error" in profile_of(step)]
    assert sum(1 for step in steps if step["llm_call_count"] == 0) == want["refused"]
    calls = [call["tool_call_id"] for step in steps for call in step.get("tool_calls") or ()]
    assert len(calls) == want["tools"] and set(calls) == {key for step in steps for key in results(step)}


def test_the_tested_sdk_is_the_one_the_image_pins():
    pinned = [line.split("==", 1)[1] for line in (AGENT_DIR / "requirements.txt").read_text().splitlines()
              if line.startswith("google-antigravity==")]
    assert antigravity_atif.TESTED_WITH == tuple(pinned) == (antigravity_bridge.SDK_VERSION,)


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
        ("chunk", Text(step_index=0, text="Let me run it.")),
        ("call", ToolCall(id="toolu_1", name="run_command")),
        ("chunk", ToolCall(id="c:1", name="run_command", step_index=0)),
        ("result", ToolResult(id="toolu_1", name="run_command", result="5050")),
        ("await", lambda: relay.forward(turn_messages(prompt), usage(1300, 4, cached=1200))),
        ("chunk", Text(step_index=2, text="5050")),
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
    assert (one["message"], one["tool_calls"][0]["tool_call_id"], results(one)["c:1"]["content"]) == (
        "Let me run it.", "c:1", "5050")
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
