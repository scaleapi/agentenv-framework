"""Gemini's ATIF adapter on records shaped as the CLI and the agent write them: stdout's stream-json events and the
proxy's ``agentenv.model_call`` records as natives, and the session transcript's model messages as the bridge feeds
them (``session_aux``). Each case runs through the protocol's adapter kit, so the live rule holds after every record and
the document validates as a producer's."""
from __future__ import annotations

import base64
import json
import random
import struct
import zlib

import pytest
from agentenv_protocol.atif import Fed, RecordedTask, TurnContext, check_adapter, profile_of

from tst.unit.agents.gemini.agent import AGENT_DIR, gemini_atif, gemini_transcript

GeminiAtif = gemini_atif.GeminiAtif
MODEL = "gemini-2.5-flash"


def png(width=4, height=3) -> bytes:
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    rows = b"".join(b"\x00" + b"\x10\x20\x30" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


class Feed:
    """One task's records in arrival order, each a millisecond after the last."""

    clock = 0

    def __init__(self):
        self.records: list[Fed] = []

    @staticmethod
    def at():
        Feed.clock += 1
        return f"2026-11-25T16:16:{Feed.clock // 1000:02d}.{Feed.clock % 1000:03d}Z"

    def native(self, event):
        self.records.append(Fed({**event, "timestamp": event.get("timestamp") or self.at()}))
        return self

    def call(self, inputs=7338, output=182, cached=0, outcome="answered", status=200, model=MODEL):
        record = {"type": "agentenv.model_call", "via": "passthrough", "model": model, "status": status,
                  "outcome": outcome}
        if outcome == "answered":
            record["usage"] = {"input_tokens": inputs, "output_tokens": output, "cache_read_tokens": cached,
                               "reasoning_tokens": 105}
        return self.native(record)

    def tool_use(self, tool_id, name="run_shell_command"):
        return self.native({"type": "tool_use", "tool_name": name, "tool_id": tool_id, "parameters": {}})

    def message(self, message_id, content="", *, tokens=None, thoughts=(), calls=(), model=MODEL, at=None):
        """A model message, written at ``at`` (else now), fed now: the bridge feeds one once a later one follows it."""
        message = {"id": message_id, "timestamp": at or self.at(), "type": "gemini", "content": content,
                   "thoughts": [{"subject": subject, "description": description, "timestamp": self.at()}
                                for subject, description in thoughts],
                   "tokens": tokens or {"input": 7338, "output": 77, "cached": 0, "thoughts": 105, "tool": 0,
                                        "total": 7520},
                   "model": model, "toolCalls": list(calls)}
        self.records.append(Fed(gemini_transcript.session_aux(message), aux="session"))
        return self

    def result(self, error=None):
        event = {"type": "result", "status": "error" if error else "success", "stats": {"tool_calls": 0}}
        if error:
            event["error"] = {"type": "unknown", "message": error}
        return self.native(event)


def tool(call_id, output="Output: 5050", *, name="run_shell_command", status="success", error=None, image=None):
    response = {"error": error} if error else {"output": output}
    result = [{"functionResponse": {"id": call_id, "name": name, "response": response}}]
    if image is not None:
        result.append({"inlineData": {"mimeType": "image/png", "data": base64.b64encode(image).decode()}})
    return {"id": call_id, "name": name, "args": {"command": "python3 hello.py"}, "result": result, "status": status,
            "timestamp": Feed.at(), "resultDisplay": [[{"text": "5050", "bold": False}]], "displayName": "Shell",
            "renderOutputAsMarkdown": False}


def context(index=0):
    return TurnContext(index=index, context_id="ctx", task_id=f"t{index}", message_id=f"m{index}", inbound=["do it"],
                       agent_name="Gemini", agent_version="1.0.0", agent_model=MODEL,
                       started_at=f"2026-11-25T16:00:0{index}Z")


def run(*feeds: Feed, state="completed"):
    kit = check_adapter(GeminiAtif, [RecordedTask(context=context(i), records=feed.records, state=state)
                                     for i, feed in enumerate(feeds)])
    kit.raise_for_errors()
    return kit


def agent_steps(document):
    return [step for step in document["steps"] if step["source"] == "agent"]


def test_a_response_is_one_step_with_its_thoughts_calls_results_and_its_proxy_calls_usage():
    feed = Feed().native({"type": "init", "session_id": "s-1"}).call().tool_use("shell_1")
    feed.native({"type": "tool_result", "tool_id": "shell_1", "status": "success"})
    feed.message("m-1", thoughts=[("Running it", "I'll run the script.")], calls=[tool("shell_1")])
    feed.call(inputs=7521, output=4, cached=7003).message("m-2", "5050", tokens={"input": 7521, "output": 4,
                                                                                 "cached": 7003}).result()
    first, second = agent_steps(run(feed).document)

    assert first["model_name"] == MODEL and first["reasoning_content"] == "Running it: I'll run the script."
    assert profile_of(first)["reasoning"][0]["visibility"] == "summary"
    assert first["tool_calls"] == [{"tool_call_id": "shell_1", "function_name": "run_shell_command",
                                    "arguments": {"command": "python3 hello.py"}}]
    assert first["observation"]["results"] == [{"source_call_id": "shell_1", "content": "Output: 5050"}]
    assert first["metrics"] == {"prompt_tokens": 7338, "cached_tokens": 0, "completion_tokens": 182}
    assert second["message"] == "5050" and second["metrics"]["cached_tokens"] == 7003


def test_a_tool_use_prompt_counts_as_the_proxy_counts_it():
    feed = Feed().call(inputs=7400).message("m-1", "ok", tokens={"input": 7338, "output": 4, "tool": 62})
    (step,) = agent_steps(run(feed.result()).document)

    assert step["metrics"]["prompt_tokens"] == 7400 and step["metrics"]["completion_tokens"] == 182


def test_a_call_that_wrote_no_message_is_a_step_with_its_usage_alone_in_its_place():
    feed = Feed().call()
    written = Feed.at()
    feed.call(inputs=900, output=3).call(inputs=7500, output=10)
    feed.message("m-1", "first", tokens={"input": 7338, "output": 77, "thoughts": 105}, at=written)
    feed.message("m-2", "second", tokens={"input": 7500, "output": 10}).call(inputs=950, output=2).result()
    steps = agent_steps(run(feed).document)

    assert [step["message"] for step in steps] == ["first", "", "second", ""]
    assert [step["metrics"]["prompt_tokens"] for step in steps] == [7338, 900, 7500, 950]


def test_a_refused_call_is_a_step_with_no_model_call_before_the_retry_that_answered():
    feed = Feed().call(outcome="refused", status=429).call().message("m-1", "ok").result()
    refused, answered = agent_steps(run(feed).document)

    assert refused["llm_call_count"] == 0 and "metrics" not in refused
    assert profile_of(refused)["error"] == {"kind": "infra_error", "message": "the endpoint refused the call (HTTP 429)"}
    assert answered["message"] == "ok"


def test_the_clis_report_of_a_turn_failed_at_a_refused_call_replaces_its_error():
    error = '[API Error: {"error":{"message":"refused","code":400}}]'
    feed = Feed().native({"type": "init", "session_id": "s-1"}).call(outcome="refused", status=400).result(error)
    (step,) = agent_steps(run(feed, state="failed").document)

    assert profile_of(step)["error"] == {"kind": "infra_error", "message": error}


def test_an_image_beside_a_result_is_a_media_file_and_an_error_is_the_results():
    image = png()
    calls = [tool("shot_1", "### Result", name="mcp_browser_browser_take_screenshot", image=image),
             tool("shot_2", name="mcp_browser_browser_take_screenshot", status="error", error="scale is required")]
    kit = run(Feed().call().message("m-1", calls=calls).call(inputs=8000).message(
        "m-2", "Example Domain", tokens={"input": 8000, "output": 4}).result())

    shot, failed = agent_steps(kit.document)[0]["observation"]["results"]
    text, picture = shot["content"]
    assert text == {"type": "text", "text": "### Result"} and kit.media[picture["source"]["path"]] == image
    assert failed["content"] == "scale is required" and profile_of(failed)["status"] == "error"


def test_a_cancelled_call_without_a_result_is_canceled():
    cancelled = {**tool("shell_1"), "status": "cancelled"}
    del cancelled["result"]
    (step,) = agent_steps(run(Feed().call().message("m-1", calls=[cancelled]).result()).document)

    (result,) = step["observation"]["results"]
    assert profile_of(result)["status"] == "canceled"


@pytest.mark.parametrize("proxied", [
    pytest.param(lambda feed: feed.call(outcome="unreported"), id="unreported"),
    pytest.param(lambda feed: feed, id="unseen"),
])
def test_a_response_whose_call_the_proxy_didnt_count_takes_the_transcripts(proxied):
    feed = proxied(Feed()).message("m-1", "ok", tokens={"input": 7338, "output": 77, "cached": 10, "thoughts": 105})
    (step,) = agent_steps(run(feed.result()).document)

    assert step["metrics"] == {"prompt_tokens": 7338, "cached_tokens": 10, "completion_tokens": 182}


def test_each_task_of_a_session_writes_its_own_steps():
    first = Feed().call().message("m-1", "5050").result()
    second = Feed().call(inputs=7700).message("m-2", "500500", tokens={"input": 7700, "output": 6}).result()
    kit = run(first, second)

    assert [step["message"] for step in agent_steps(kit.document)] == ["5050", "500500"]


def test_the_tested_cli_is_the_one_the_image_pins():
    pinned = json.loads((AGENT_DIR / "package.json").read_text())["dependencies"]["@google/gemini-cli"]
    assert gemini_atif.TESTED_WITH == (pinned,)


def seeded_session(seed: int) -> tuple[list[Feed], dict]:
    """A random session of one to three tasks, as the CLI and the proxy write them: responses of thoughts, text and
    calls, some returning images or errors; the proxy's record of each response's call, a refused attempt before some,
    calls that wrote no message between them, and some the proxy couldn't count; each message fed once a later one
    follows it, or all at the task's end. With what the copy must hold."""
    rng = random.Random(seed)
    image = png()
    want = {"input": 0, "output": 0, "steps": 0, "calls": set(), "messages": [], "unreported": 0}
    n = iter(range(10**6))
    feeds = []
    for _ in range(rng.randint(1, 3)):
        feed, held, lagged, previous = Feed(), [], rng.random() < 0.5, None
        for _ in range(rng.randint(1, 4)):
            if rng.random() < 0.2:
                feed.call(outcome="refused", status=rng.choice([429, 500, 503]))
                want["steps"] += 1
            if rng.random() < 0.2:
                side = rng.randint(100, 900)
                feed.call(inputs=side, output=1)
                want["input"] += side
                want["output"] += 1
                want["steps"] += 1
            # An uncounted call's next one sometimes has its very input, so only times tell them apart.
            inputs = previous if previous is not None and rng.random() < 0.5 else rng.randint(1000, 9000)
            tool_tokens, previous = rng.choice([0, 0, 40]), None
            # The transcript counts a response as its call did, its thoughts in its output here.
            output = rng.randint(1, 50)
            if rng.random() < 0.1:
                feed.call(outcome="unreported")
                want["unreported"] += 1
                previous = inputs
            else:
                feed.call(inputs=inputs + tool_tokens, output=output, cached=rng.randint(0, inputs))
            want["output"] += output
            want["input"] += inputs + tool_tokens
            calls = []
            for _ in range(rng.choice([0, 1, 1, 2])):
                call_id = f"call_{next(n)}"
                calls.append(tool(call_id, image=image if rng.random() < 0.3 else None,
                                  status=rng.choice(["success", "success", "error"]),
                                  error="failed" if rng.random() < 0.2 else None))
                want["calls"].add(call_id)
            text = rng.choice(["", "", "done", "looking"])
            thoughts = [("Plan", "Run it.")] * rng.randint(0, 2)
            message_id = f"m-{next(n)}"
            message = (message_id, text, {"input": inputs, "output": output, "cached": 0, "thoughts": 0,
                                          "tool": tool_tokens}, thoughts, calls, Feed.at())
            want["messages"].append(text)
            want["steps"] += 1
            held.append(message)
            while len(held) > (1 if lagged else 10**6):
                message_id, text, tokens, thoughts, calls, at = held.pop(0)
                feed.message(message_id, text, tokens=tokens, thoughts=thoughts, calls=calls, at=at)
        for message_id, text, tokens, thoughts, calls, at in held:
            feed.message(message_id, text, tokens=tokens, thoughts=thoughts, calls=calls, at=at)
        if rng.random() < 0.2:
            feed.call(inputs=500, output=1)
            want["input"] += 500
            want["output"] += 1
            want["steps"] += 1
        feeds.append(feed.result())
    return feeds, want


@pytest.mark.parametrize("seed", range(150))
def test_a_seeded_sessions_copy_holds_every_response_and_every_call_the_proxy_forwarded(seed):
    feeds, want = seeded_session(seed)
    steps = agent_steps(run(*feeds).document)

    assert len(steps) == want["steps"]
    assert sum(step.get("metrics", {}).get("prompt_tokens", 0) for step in steps) == want["input"]
    assert sum(step.get("metrics", {}).get("completion_tokens", 0) for step in steps) == want["output"]
    assert [step["message"] for step in steps if step["message"]] == [text for text in want["messages"] if text]
    calls = [call["tool_call_id"] for step in steps for call in step.get("tool_calls") or ()]
    results = {result["source_call_id"] for step in steps for result in (step.get("observation") or {}).get(
        "results") or ()}
    assert sorted(calls) == sorted(want["calls"]) and set(calls) <= results


def test_a_response_takes_the_last_call_to_end_before_its_message_one_the_proxy_couldnt_count_too():
    feed = Feed().call(inputs=900, output=2).call(outcome="unreported")
    feed.message("m-1", "ok", tokens={"input": 7338, "output": 77, "thoughts": 105}).result()
    steps = agent_steps(run(feed).document)

    assert [step["message"] for step in steps] == ["", "ok"]
    assert [step["metrics"]["prompt_tokens"] for step in steps] == [900, 7338]


def test_a_call_that_ended_after_the_responses_own_one_but_before_its_message_doesnt_take_it():
    feed = Feed().call(inputs=7338, output=20).call(inputs=900, output=2)
    feed.message("m-1", "ok", tokens={"input": 7338, "output": 20}).result()
    steps = agent_steps(run(feed).document)

    assert [(step["message"], step["metrics"]["prompt_tokens"]) for step in steps] == [("ok", 7338), ("", 900)]


def test_a_response_takes_its_own_uncounted_call_not_the_next_answered_one_of_the_same_input():
    feed = Feed().call(outcome="unreported")
    written = Feed.at()
    feed.call(inputs=7338, output=20)
    feed.message("m-1", "first", tokens={"input": 7338, "output": 7, "thoughts": 3}, at=written)
    feed.message("m-2", "second", tokens={"input": 7338, "output": 18, "thoughts": 2}).result()
    first, second = agent_steps(run(feed).document)

    assert (first["message"], first["metrics"]["completion_tokens"]) == ("first", 10)
    assert (second["message"], second["metrics"]["completion_tokens"]) == ("second", 20)


def test_a_file_in_a_results_own_parts_is_a_media_file_after_its_text():
    image = png(6, 5)
    call = tool("read_1", "Read image file: shot.png", name="read_file")
    call["result"][0]["functionResponse"]["parts"] = [
        {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(image).decode()}}]
    kit = run(Feed().call().message("m-1", calls=[call]).call(inputs=8000).message(
        "m-2", "a red square", tokens={"input": 8000, "output": 4}).result())

    (result,) = agent_steps(kit.document)[0]["observation"]["results"]
    text, picture = result["content"]
    assert text == {"type": "text", "text": "Read image file: shot.png"}
    assert picture["type"] == "image" and kit.media[picture["source"]["path"]] == image
