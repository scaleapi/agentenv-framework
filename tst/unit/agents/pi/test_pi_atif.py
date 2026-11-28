"""pi's ATIF adapter on events shaped as pi 1.1.0 prints them in JSON mode (``message_end`` of each system, user,
assistant and toolResult message, tool executions, compactions and retries) and on the aux records the bridge feeds,
each case run through the protocol's adapter kit, so the live rule holds after every record and the document validates
as a producer's; then the bridge, with the fake pi, feeding the copy and the card offering it."""
from __future__ import annotations

import base64
import re
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

import pytest
from agentenv_protocol.a2a_agent.tasks.v1 import TrajectoryLog
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
from starlette.testclient import TestClient

from tst.unit.agents.pi.agent import AGENT_DIR, pi_agent, pi_atif

PiAtif = pi_atif.PiAtif
MODEL = "anthropic/claude-haiku-4-5"
FAKE_PI = Path(__file__).with_name("fake_pi.py")
BASH = {"name": "bash", "description": "Run a shell command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}
SCREENSHOT = {"name": "mcp__browser__browser_take_screenshot", "description": "Take a screenshot of the page.",
              "parameters": {"type": "object", "properties": {}}}


def png(width=4, height=3) -> bytes:
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    rows = b"".join(b"\x00" + b"\x10\x20\x30" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def usage(prompt=0, output=0, read=0, write=0, cost=0.0) -> dict[str, Any]:
    """A usage as pi's Chat Completions client parses one: its input the prompt less the cache, its total all four."""
    uncached = max(0, prompt - read - write)
    return {"input": uncached, "output": output, "cacheRead": read, "cacheWrite": write, "reasoning": 0,
            "totalTokens": uncached + output + read + write,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": cost}}


def thinking(text="", signature=None, redacted=False) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "thinking", "thinking": text}
    if signature is not None:
        block["thinkingSignature"] = signature
    if redacted:
        block["redacted"] = True
    return block


def text(value) -> dict[str, Any]:
    return {"type": "text", "text": value}


def call(call_id, name="bash", **arguments) -> dict[str, Any]:
    return {"type": "toolCall", "id": call_id, "name": name, "arguments": arguments or {"command": "ls"}}


class Feed:
    """One task's records in the order pi prints them, each message a millisecond after the last."""

    clock = 1_795_795_200_000

    def __init__(self):
        self.records: list[Fed] = []

    @classmethod
    def now(cls) -> int:
        cls.clock += 1
        return cls.clock

    def event(self, event, aux=None):
        self.records.append(Fed(event, aux=aux))
        return self

    def message(self, message):
        self.event({"type": "message_start", "message": message})
        return self.event({"type": "message_end", "message": message})

    def start(self, *tools, prompt="do it"):
        self.event({"type": "session", "version": 3, "id": "ctx", "timestamp": "2026-11-27T16:00:00.000Z",
                    "cwd": "/workspace"})
        self.event({"type": "agent_start"}).event({"type": "turn_start"})
        if tools:
            self.message({"role": "system", "content": "", "sections": {"preamble": "You are pi."},
                          "toolsAdded": list(tools), "timestamp": self.now()})
        return self.message({"role": "user", "content": prompt, "timestamp": self.now()})

    def assistant(self, *content, stop="stop", counts=None, model=MODEL, effort="off", **extra):
        message = {"role": "assistant", "content": list(content), "api": "openai-completions", "provider": "agentenv",
                   "model": model, "responseId": f"chatcmpl-{self.clock}", "thinkingLevel": effort,
                   "usage": counts if counts is not None else usage(1109, 21),
                   "stopReason": stop, "timestamp": self.now(), **extra}
        if stop == "toolUse":
            message["rawStopReason"] = "tool_calls"
        return self.message(message)

    def result(self, call_id, *content, name="bash", error=False):
        self.event({"type": "tool_execution_start", "toolCallId": call_id, "toolName": name, "args": {}})
        blocks = list(content) or [text("ok")]
        self.event({"type": "tool_execution_end", "toolCallId": call_id, "toolName": name,
                    "result": {"content": blocks, "details": {}}, "isError": error, "durationMs": 12})
        return self.message({"role": "toolResult", "toolCallId": call_id, "toolName": name, "content": blocks,
                             "details": {}, "isError": error, "timestamp": self.now()})

    def compaction(self, summary_usage, at="2026-11-27T16:00:05.000Z", reason="threshold"):
        self.event({"type": "compaction_start", "reason": reason})
        self.event({"type": "compaction_start", "at": at}, aux=pi_atif.AUX)
        return self.event({"type": "compaction_end", "reason": reason, "aborted": False, "willRetry": False, "result": {
            "summary": "The user asked for the sum; hello.py prints 5050.", "firstKeptEntryId": "a1b2c3d4",
            "tokensBefore": 7616, "estimatedTokensAfter": 1210, "usage": summary_usage,
            "details": {"readFiles": [], "modifiedFiles": ["hello.py"]}}})

    def end(self):
        self.event({"type": "turn_end", "message": {}, "toolResults": []})
        return self.event({"type": "agent_end", "messages": [], "willRetry": False}).event(
            {"type": "agent_settled", "aborted": False})


def context(index=0, model=MODEL) -> TurnContext:
    return TurnContext(index=index, context_id="ctx", task_id=f"t{index}", message_id=f"m{index}", inbound=["do it"],
                       agent_name="pi", agent_version="1.1.0", agent_model=model,
                       started_at=f"2026-11-27T16:00:0{index}Z")


def run(*feeds: Feed, finals=None, state="completed"):
    finals = finals or [None] * len(feeds)
    kit = check_adapter(PiAtif, [RecordedTask(context=context(i), records=feed.records, state=state, final=final)
                                 for i, (feed, final) in enumerate(zip(feeds, finals))])
    kit.raise_for_errors()
    assert not [finding for finding in kit.findings if finding.rule == "kit-unmatched"]
    return kit


def agent_steps(document):
    return [step for step in document["steps"] if step["source"] == "agent"]


def iso(millis: int) -> str:
    return pi_atif._iso(millis)


def test_each_model_call_is_a_step_with_its_model_time_reasoning_text_calls_and_results():
    feed = Feed().start(BASH)
    feed.assistant(thinking("I'll write it, then run it."), text("Writing it."), call("call_1", command="ls"),
                   stop="toolUse", effort="low", counts=usage(1109, 21))
    first_at = Feed.clock
    feed.result("call_1", text("hello.py"))
    feed.assistant(text("50"), text("50"), effort="low", counts=usage(1139, 2, read=1024)).end()
    kit = run(feed, finals=[["5050"]])
    first, second = agent_steps(kit.document)

    assert first["model_name"] == MODEL and first["timestamp"] == iso(first_at)
    assert first["reasoning_effort"] == "low" and first["reasoning_content"] == "I'll write it, then run it."
    assert first["message"] == "Writing it."
    assert first["tool_calls"] == [{"tool_call_id": "call_1", "function_name": "bash", "arguments": {"command": "ls"}}]
    assert first["observation"]["results"] == [{"source_call_id": "call_1", "content": "hello.py"}]
    assert first["metrics"] == {"prompt_tokens": 1109, "cached_tokens": 0, "completion_tokens": 21,
                                "extra": {"agentenv": {"cache_write_tokens": 0}}}
    assert second["message"] == "5050" and second["metrics"]["prompt_tokens"] == 1139
    assert second["metrics"]["cached_tokens"] == 1024
    assert kit.document["agent"]["tool_definitions"] == [{"type": "function", "function": BASH}]


def test_the_steps_sum_to_the_usage_the_turn_reports():
    """The turn's usage and the copy's totals come from the same records through the same count of the cache,
    whichever its total says pi's input holds; a compaction's summary call counted in both."""
    held = {"input": 1139, "output": 2, "cacheRead": 1024, "cacheWrite": 0, "totalTokens": 1141,
            "cost": {"total": 0.002}}
    feed = Feed().start(BASH)
    feed.assistant(call("call_1"), stop="toolUse", counts=usage(1109, 21, cost=0.001)).result("call_1")
    feed.assistant(call("call_2"), stop="toolUse", counts=usage(2300, 40, read=1024, write=200, cost=0.003))
    feed.result("call_2").assistant(call("call_3"), stop="toolUse", counts=held).result("call_3")
    feed.compaction(usage(329, 156, cost=0.004)).assistant(text("done"), counts=usage(900, 4, cost=0.001)).end()
    final = run(feed, finals=[["done"]]).document["final_metrics"]
    turn = pi_agent._Run()
    for fed in feed.records:
        if fed.aux is None:
            pi_agent._record(turn, fed.record)
    reported = pi_agent._usage(turn)

    assert (final["total_prompt_tokens"], final["total_completion_tokens"], final["total_cached_tokens"]) == (
        reported.input_tokens, reported.output_tokens, reported.cache_read_tokens)
    assert final["extra"]["agentenv"]["total_cache_write_tokens"] == reported.cache_write_tokens == 200
    assert final["total_cost_usd"] == pytest.approx(reported.cost_usd)
    assert final["extra"]["agentenv"]["cost_source"] == "harness"


def test_a_call_whose_endpoint_reported_no_usage_is_unknown_not_zero_and_pis_zero_cost_is_none():
    feed = Feed().start().assistant(text("hi"), counts=usage()).end()
    document = run(feed, finals=[["hi"]]).document
    (step,) = agent_steps(document)

    assert step["llm_call_count"] == 1 and "metrics" not in step
    assert "total_prompt_tokens" not in document["final_metrics"]
    unpriced = run(Feed().start().assistant(text("hi"), counts=usage(1109, 21)).end()).document
    assert "cost_usd" not in agent_steps(unpriced)[0]["metrics"] and "total_cost_usd" not in unpriced["final_metrics"]


def test_an_image_a_tool_returns_is_a_media_file_after_its_text():
    image = png()
    feed = Feed().start(SCREENSHOT)
    feed.assistant(call("call_1", name=SCREENSHOT["name"]), stop="toolUse")
    feed.result("call_1", text("### Result"), text("Took the screenshot."),
                {"type": "image", "data": base64.b64encode(image).decode(), "mimeType": "image/png"},
                name=SCREENSHOT["name"])
    kit = run(feed.assistant(text("Example Domain")).end())
    (result,) = agent_steps(kit.document)[0]["observation"]["results"]

    said, picture = result["content"]
    assert said == {"type": "text", "text": "### Result\nTook the screenshot."}
    assert picture["type"] == "image" and kit.media[picture["source"]["path"]] == image
    assert profile_of(result)["media"][picture["source"]["path"]]["width"] == 4


def test_a_failed_tool_is_an_error_result():
    feed = Feed().start().assistant(call("call_1"), stop="toolUse")
    feed.result("call_1", text("command not found"), error=True).assistant(text("it failed")).end()
    (result,) = agent_steps(run(feed).document)[0]["observation"]["results"]

    assert result["content"] == "command not found" and profile_of(result)["status"] == "error"


@pytest.mark.parametrize(("signature", "redacted", "want"), [
    pytest.param(None, False, {"visibility": "raw"}, id="raw"),
    pytest.param("reasoning_content", False, {"visibility": "raw"}, id="field-name-is-no-signature"),
    pytest.param('[{"type":"reasoning.text","text":"t","signature":"sig"}]', False,
                 {"visibility": "raw", "signature": '[{"type":"reasoning.text","text":"t","signature":"sig"}]'},
                 id="reasoning-details"),
    pytest.param("ciphertext", True, {"visibility": "redacted", "data": "ciphertext"}, id="redacted"),
])
def test_thinking_is_reasoning_with_its_visibility(signature, redacted, want):
    block = thinking("" if redacted else "Plan: run it.", signature=signature, redacted=redacted)
    (step,) = agent_steps(run(Feed().start().assistant(block, text("ok"), effort="medium").end()).document)

    blocks = profile_of(step).get("reasoning") or [{"chars": len(step["reasoning_content"]), "visibility": "raw"}]
    assert {key: value for key, value in blocks[0].items() if key != "chars"} == want
    assert step["reasoning_effort"] == "medium"


def test_a_thinking_block_with_no_text_is_omitted():
    (step,) = agent_steps(run(Feed().start().assistant(thinking(""), text("ok")).end()).document)

    assert "reasoning_content" not in step and profile_of(step)["reasoning"] == [{"chars": 0, "visibility": "omitted"}]
    assert "reasoning_effort" not in step


def test_a_refused_call_is_a_step_with_no_model_call_and_the_retry_that_answered_a_second_attempt():
    """As pi ran in pi-retry-faults: the proxy answered 503, pi ended the call as an error with no usage, hid it from
    the conversation and asked again after its backoff."""
    feed = Feed().start(BASH)
    feed.assistant(stop="error", counts=usage(), errorMessage="503 service unavailable", rawStopReason=None)
    feed.event({"type": "turn_end", "message": {}, "toolResults": []})
    feed.event({"type": "agent_end", "messages": [], "willRetry": True})
    feed.event({"type": "auto_retry_start", "attempt": 1, "maxAttempts": 3, "delayMs": 2000,
                "errorMessage": "503 service unavailable"})
    feed.event({"type": "entry_appended", "entry": {"type": "context_edit", "id": "e1", "parentId": "p1",
                                                    "timestamp": "2026-11-27T16:00:01.000Z"}})
    feed.event({"type": "agent_start"}).event({"type": "turn_start"})
    feed.assistant(text("ok"), counts=usage(1109, 21))
    feed.event({"type": "auto_retry_end", "success": True, "attempt": 1}).end()
    document = run(feed, finals=[["ok"]]).document
    refused, answered = agent_steps(document)

    assert refused["llm_call_count"] == 0 and "metrics" not in refused and refused["model_name"] == MODEL
    assert profile_of(refused)["error"] == {"kind": "infra_error", "message": "503 service unavailable"}
    assert "attempt" not in profile_of(refused) and profile_of(answered)["attempt"] == 2
    assert answered["message"] == "ok" and document["final_metrics"]["total_prompt_tokens"] == 1109


def test_a_response_that_ended_in_an_error_keeps_its_usage_and_runs_none_of_its_calls():
    """A stream cut off after the endpoint counted it: pi keeps the partial blocks, a call among them perhaps without
    its id yet, and runs none of them."""
    feed = Feed().start().assistant(thinking("Let me"), text("Running"), call(""), stop="error",
                                    counts=usage(1109, 3), errorMessage="terminated")
    (step,) = agent_steps(run(feed.end(), state="failed").document)

    assert "tool_calls" not in step and step["metrics"]["prompt_tokens"] == 1109
    assert profile_of(step)["error"] == {"kind": "infra_error", "message": "terminated"}


def test_a_stream_cut_off_before_its_usage_is_a_model_call_holding_what_it_streamed():
    """As pi's Chat Completions client ends a stream that broke after some deltas (an overload mid-answer, say): the
    blocks so far, and zeros for the usage chunk that never came. The endpoint answered, so the call isn't one it
    refused; pi retries it all the same."""
    feed = Feed().start(BASH)
    feed.assistant(thinking("Let me"), text("Running"), call(""), stop="error", counts=usage(),
                   errorMessage="overloaded_error: Overloaded", responseId=None)
    feed.event({"type": "auto_retry_start", "attempt": 1, "maxAttempts": 3, "delayMs": 2000})
    feed.assistant(text("ok")).end()
    cut, answered = agent_steps(run(feed, finals=[["ok"]]).document)

    assert cut["llm_call_count"] == 1 and "metrics" not in cut and "tool_calls" not in cut
    assert cut["message"] == "Running" and cut["reasoning_content"] == "Let me"
    assert profile_of(cut)["error"] == {"kind": "infra_error", "message": "overloaded_error: Overloaded"}
    assert profile_of(answered)["attempt"] == 2


def test_an_aborted_response_is_an_agent_error():
    feed = Feed().start().assistant(text("Runn"), call("call_1"), stop="aborted", counts=usage())
    (step,) = agent_steps(run(feed.end(), state="canceled").document)

    assert step["llm_call_count"] == 1 and "tool_calls" not in step
    assert profile_of(step)["error"] == {"kind": "agent_error", "message": "pi aborted the run"}


def test_every_call_is_named_by_the_model_as_called_answered_or_refused():
    """As the endpoint answered in the recorded retried run: its answers name the model without the routing prefix,
    and a refused call names none, so the steps take the name pi called, the turn's own."""
    feed = Feed().start(BASH)
    feed.assistant(call("call_1"), stop="toolUse", responseModel="claude-haiku-4-5").result("call_1")
    feed.assistant(stop="error", counts=usage(), errorMessage="503 service unavailable", responseId=None)
    feed.event({"type": "auto_retry_start", "attempt": 1, "maxAttempts": 3, "delayMs": 2000})
    feed.assistant(text("first second"), responseModel="claude-haiku-4-5").end()
    document = run(feed, finals=[["first second"]]).document

    assert [step["model_name"] for step in agent_steps(document)] == [MODEL] * 3
    assert document["agent"]["model_name"] == MODEL


def test_a_compaction_is_its_summary_calls_step_timed_by_the_bridge_and_named_by_the_sessions_model():
    feed = Feed().start(BASH).assistant(call("call_1"), stop="toolUse", model="claude-haiku-4-5",
                                        responseModel="claude-haiku-4-5-20251001").result("call_1")
    feed.compaction(usage(329, 156), at="2026-11-27T16:00:05.000Z").assistant(text("5050")).end()
    steps = agent_steps(run(feed, finals=[["5050"]]).document)
    first, summary, answer = steps

    assert first["model_name"] == "claude-haiku-4-5"
    assert summary["message"] == "The user asked for the sum; hello.py prints 5050."
    assert summary["timestamp"] == "2026-11-27T16:00:05.000Z" and summary["model_name"] == first["model_name"]
    assert profile_of(summary)["compaction"] == {"written_by": "assistant", "trigger": "auto",
                                                 "tokens_before": 7616, "tokens_after": 1210}
    assert summary["metrics"]["prompt_tokens"] == 329 and summary["metrics"]["completion_tokens"] == 156
    assert answer["message"] == "5050"


def test_a_compaction_before_a_later_tasks_prompt_is_named_by_the_model_that_task_runs():
    """pi checks for a compaction before it sends a later task's prompt, and summarizes with the model the task runs
    (its --model, the turn's), not the one the task before called."""
    first = Feed().start(BASH).assistant(text("5050")).end()
    second = Feed().event({"type": "session", "version": 3, "id": "ctx", "timestamp": "2026-11-27T16:00:00.000Z",
                           "cwd": "/workspace"})
    second.compaction(usage(329, 156)).event({"type": "agent_start"}).event({"type": "turn_start"})
    second.message({"role": "user", "content": "and again", "timestamp": Feed.now()})
    second.assistant(text("500500"), model="openai/gpt-4.1-mini").end()
    kit = check_adapter(PiAtif, [RecordedTask(context=context(0), records=first.records, final=["5050"]),
                                 RecordedTask(context=context(1, model="openai/gpt-4.1-mini"), records=second.records,
                                              final=["500500"])])
    kit.raise_for_errors()
    _, summary, answer = agent_steps(kit.document)

    assert "compaction" in profile_of(summary)
    assert summary["model_name"] == answer["model_name"] == "openai/gpt-4.1-mini"


def test_a_failed_compaction_is_a_step_with_no_model_call_and_an_aborted_one_none():
    feed = Feed().start().assistant(text("hi"))
    feed.event({"type": "compaction_start", "reason": "overflow"})
    feed.event({"type": "compaction_end", "reason": "overflow", "aborted": False, "willRetry": False,
                "errorMessage": "Compaction failed: 400 bad request"})
    feed.event({"type": "compaction_start", "reason": "threshold"})
    feed.event({"type": "compaction_end", "reason": "threshold", "aborted": True, "willRetry": False})
    _, failed = agent_steps(run(feed.end()).document)

    assert failed["llm_call_count"] == 0 and profile_of(failed)["error"]["message"].startswith("Compaction failed")


def test_a_resumed_task_writes_its_own_steps_and_a_new_documents_tools_come_from_the_bridge():
    first = Feed().start(BASH).assistant(call("call_1"), stop="toolUse").result("call_1").assistant(text("5050")).end()
    second = Feed().event({"type": "tools", "tools": [BASH, SCREENSHOT]}, aux=pi_atif.AUX).start()
    second.assistant(call("call_2"), stop="toolUse").result("call_2").assistant(text("500500")).end()
    added = Feed().start({**BASH, "description": "Run a command in bash."}).assistant(text("ok")).end()
    kit = run(first, second, added, finals=[["5050"], ["500500"], ["ok"]])

    assert [step["message"] for step in agent_steps(kit.document) if step["message"]] == ["5050", "500500", "ok"]
    assert [step["source"] for step in kit.document["steps"]].count("user") == 3
    assert kit.document["final_metrics"]["total_steps"] == len(kit.document["steps"])
    assert [tool["function"] for tool in kit.document["agent"]["tool_definitions"]] == [
        {**BASH, "description": "Run a command in bash."}, SCREENSHOT]


def test_from_atif_gives_pi_events_the_turns_usage_reads_back():
    feed = Feed().start(BASH).assistant(thinking("hm"), call("call_1"), stop="toolUse", counts=usage(1109, 21))
    feed.result("call_1").assistant(text("5050"), counts=usage(1139, 2, read=1024)).end()
    native = PiAtif.from_atif(strip_profile(run(feed).document))
    turn = pi_agent._Run()
    for event in native.log:
        pi_agent._record(turn, event)

    assert native.session is None
    assert [event["message"]["role"] for event in native.log if event["type"] == "message_end"] == [
        "user", "assistant", "toolResult", "assistant"]
    assert (pi_agent._usage(turn).input_tokens, pi_agent._usage(turn).output_tokens) == (2248, 23)


def test_the_tested_pi_is_the_one_install_pins():
    pinned = re.search(r"^PI_VERSION=(\S+)$", (AGENT_DIR / "install.sh").read_text(), re.M).group(1)
    assert pi_atif.TESTED_WITH == (pinned,) == (pi_agent.PI_VERSION,)


@pytest.fixture
def record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "record.json"
    monkeypatch.setenv("FAKE_PI_RECORD", str(path))
    monkeypatch.setenv("LITELLM_BASE_URL", "http://litellm.test/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    return path


def _client(tmp_path: Path) -> TestClient:
    agent = pi_agent.PiAgent(home=tmp_path / "home", workspace=tmp_path / "ws",
                             pi_command=(sys.executable, str(FAKE_PI)))
    return TestClient(agent.create_app(), client=("127.0.0.1", 50000))


@pytest.fixture
def client(tmp_path: Path, record: Path):
    with _client(tmp_path) as client:
        client.post("/ext/agent-config", json={"model": MODEL, "effort": "low"}).raise_for_status()
        yield client


def _send(client: TestClient, text: str = "go", context_id: str = "ctx-1") -> dict[str, Any]:
    response = client.post("/a2a", json={"jsonrpc": "2.0", "id": "1", "method": "message/send", "params": {
        "message": {"kind": "message", "messageId": f"m-{text}", "role": "user", "contextId": context_id,
                    "parts": [{"kind": "text", "text": text}]},
        "configuration": {"blocking": True}}})
    return response.json()["result"]


def _usage_of(task: dict[str, Any]) -> dict[str, Any]:
    return next(part["data"]["usage"] for part in task["status"]["message"]["parts"] if part["kind"] == "data")


def _atif(client: TestClient, task: dict[str, Any]) -> dict[str, Any]:
    read = client.post("/ext/trajectory", json={"task_id": task["id"], "format": "atif/1"})
    assert read.status_code == 200, read.text
    return read.json()


def _prescribe(client: TestClient) -> None:
    client.post("/ext/trajectory/prescribe", json={"formats": ["atif/1"]}).raise_for_status()


def test_the_card_offers_atif_beside_pi_json_events_which_stays_the_default(client):
    card = client.get("/.well-known/agent.json").json()
    (trajectory,) = [e for e in card["capabilities"]["extensions"] if e["uri"] == "urn:agentenv:trajectory/v1"]

    assert trajectory["params"]["formats"] == ["pi-json-events/v1", "atif/1"]
    assert trajectory["params"]["native"] == trajectory["params"]["default"] == "pi-json-events/v1"


def test_a_prescribed_task_serves_an_atif_copy_built_from_the_events_as_they_came(client):
    _prescribe(client)
    task = _send(client, "fix the bug")
    read = _atif(client, task)
    document, usage_reported = read["trajectory"], _usage_of(task)
    native = client.post("/ext/trajectory", json={"task_id": task["id"]}).json()["trajectory"]

    assert task["status"]["state"] == "completed" and errors_in(validate(document)) == []
    assert [step["source"] for step in document["steps"]] == ["user", "agent", "agent"]
    called, answered = agent_steps(document)
    assert called["tool_calls"][0]["function_name"] == "bash" and answered["message"] == "Done."
    assert called["observation"]["results"][0]["content"] == "ok"
    assert (document["final_metrics"]["total_prompt_tokens"], document["final_metrics"]["total_cached_tokens"],
            document["final_metrics"]["total_cost_usd"]) == (
        usage_reported["input_tokens"], usage_reported["cache_read_tokens"], usage_reported["cost_usd"])
    assert document["agent"]["tool_definitions"][0]["function"]["name"] == "bash"
    assert read["aux"] == [] and "message_update" not in [event["type"] for event in native]


def test_each_compaction_start_is_stamped_for_the_copy(client, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "compaction")
    _prescribe(client)
    task = _send(client)
    read = _atif(client, task)

    (stamp,) = [entry["record"] for entry in read["aux"] if entry["name"] == "bridge"]
    assert stamp["type"] == "compaction_start" and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", stamp["at"])
    (summary,) = [step for step in agent_steps(read["trajectory"]) if "compaction" in profile_of(step)]
    assert summary["timestamp"] == stamp["at"] and summary["metrics"]["prompt_tokens"] == 300
    assert read["trajectory"]["final_metrics"]["total_prompt_tokens"] == _usage_of(task)["input_tokens"] == 512


def test_a_new_document_on_a_resumed_session_is_fed_the_sessions_tools_once(tmp_path, record):
    with _client(tmp_path) as before:
        before.post("/ext/agent-config", json={"model": MODEL}).raise_for_status()
        _send(before, "one")
    with _client(tmp_path) as after:
        after.post("/ext/agent-config", json={"model": MODEL}).raise_for_status()
        _prescribe(after)
        resumed, again = _send(after, "two"), _send(after, "three")
        first, second = _atif(after, resumed), _atif(after, again)

    assert [entry["record"] for entry in first["aux"]] == [{"type": "tools", "tools": [BASH]}]
    assert first["trajectory"]["agent"]["tool_definitions"] == [{"type": "function", "function": BASH}]
    assert second["aux"] == []


def test_a_task_that_builds_no_atif_copy_feeds_nothing_beside_its_events(tmp_path, record, monkeypatch):
    fed = []
    monkeypatch.setattr(TrajectoryLog, "aux", lambda self, name, record: fed.append((name, record)))
    monkeypatch.setenv("FAKE_PI_SCENARIO", "compaction")
    with _client(tmp_path) as client:
        client.post("/ext/agent-config", json={"model": MODEL}).raise_for_status()
        _send(client, "one")
        task = _send(client, "two")
        native = client.post("/ext/trajectory", json={"task_id": task["id"]}).json()["trajectory"]

    assert fed == []
    assert [event["type"] for event in native].count("compaction_end") == 1
