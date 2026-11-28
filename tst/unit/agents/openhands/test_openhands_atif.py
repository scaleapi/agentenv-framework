"""OpenHands' ATIF adapter on events shaped as openhands-sdk 1.58.2 dumps them (``kind``-tagged, naive local times),
fed with the aux records the runtime's ``CallFeed`` writes from the LLMs' metrics as the SDK fills them: a response's
latency, then its cost, then its usage under its id. Each case runs through the protocol's adapter kit, so the live rule
holds after every record and the document validates as a producer's; the A2A side is driven through its server with a
runtime that feeds the way ``SdkRuntime`` does."""
from __future__ import annotations

import base64
import itertools
import json
import math
import random
import struct
import uuid
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from agentenv_protocol.a2a_agent import Usage
from agentenv_protocol.atif import Fed, RecordedTask, TurnContext, check_adapter, profile_of
from starlette.testclient import TestClient

from tst.unit.agents.openhands.agent import AGENT_DIR, openhands_agent, openhands_atif

OpenHandsAtif = openhands_atif.OpenHandsAtif
CallFeed = openhands_agent.CallFeed
MODEL = "anthropic/claude-haiku-4-5"
TERMINAL = {"type": "function", "function": {
    "name": "terminal", "description": "Execute a bash command in the terminal.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"},
                                                    "security_risk": {"type": "string"}},
                   "required": ["command", "security_risk"]}}}
FINISH = {"type": "function", "function": {
    "name": "finish", "description": "Signals the completion of the current task or conversation.",
    "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}}}
#: The effort the SDK's LLM calls with when the task's config sets none (``LLM.reasoning_effort``'s default).
SDK_EFFORT = "high"
_TICKS = itertools.count(1)


def png(width=4, height=3) -> bytes:
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    rows = b"".join(b"\x00" + b"\x10\x20\x30" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def text(value: str) -> dict[str, Any]:
    return {"cache_prompt": False, "type": "text", "text": value}


def image(data: bytes, media_type: str = "image/png") -> dict[str, Any]:
    return {"cache_prompt": False, "type": "image",
            "image_urls": [f"data:{media_type};base64,{base64.b64encode(data).decode()}"]}


def event(kind: str, source: str, *, at: str | None = None, **fields: Any) -> dict[str, Any]:
    """An event as the SDK dumps it: its class as ``kind``, and ``datetime.now().isoformat()``, naive, as its time."""
    when = at or (datetime(2026, 11, 27, 10, 0) + timedelta(milliseconds=10 * next(_TICKS))).isoformat()
    return {"kind": kind, "id": str(uuid.uuid4()), "timestamp": when, "source": source, "parent_id": None, **fields}


def llm_message(role: str, content: list[dict[str, Any]], **reasoning: Any) -> dict[str, Any]:
    return {"role": role, "content": content, "tool_calls": None, "tool_call_id": None, "name": None,
            "reasoning_content": None, "thinking_blocks": [], "responses_reasoning_item": None, **reasoning}


def message(source: str, content: list[dict[str, Any]], response: str | None = None, *, role: str = "user",
            extended: tuple[str, ...] = (), **reasoning: Any) -> dict[str, Any]:
    return event("MessageEvent", source, llm_message=llm_message(role, content, **reasoning), llm_response_id=response,
                 activated_skills=[], extended_content=[text(t) for t in extended], sender=None, critic_result=None)


def user(prompt: str, **kw: Any) -> dict[str, Any]:
    return message("user", [text(prompt)], **kw)


def reply(response: str, said: str, **reasoning: Any) -> dict[str, Any]:
    return message("agent", [text(said)] if said else [], response, role="assistant", **reasoning)


def system_prompt() -> dict[str, Any]:
    return event("SystemPromptEvent", "agent", system_prompt=text("You are OpenHands agent."), dynamic_context=None,
                 tools=[{"description": "Execute a bash command", "action_type": "TerminalAction",
                         "observation_type": "TerminalObservation", "annotations": None, "meta": None,
                         "kind": "TerminalTool", "title": "terminal"}])


def action(response: str, call_id: str, name: str = "terminal", arguments: Any = None, *, thought: str = "",
           act: dict[str, Any] | None = None, raw: str | None = None, **reasoning: Any) -> dict[str, Any]:
    arguments = {"command": "python3 hello.py", "security_risk": "LOW"} if arguments is None else arguments
    dumped = {"thought": [text(thought)] if thought else [], "reasoning_content": None, "thinking_blocks": [],
              "responses_reasoning_item": None}
    dumped.update(reasoning)
    return event("ActionEvent", "agent", **dumped,
                 action=act if act is not None else {"kind": "TerminalAction", "command": arguments.get("command"),
                                                     "is_input": False, "timeout": None, "reset": False},
                 tool_name=name, tool_call_id=call_id,
                 tool_call={"id": call_id, "responses_item_id": None, "name": name,
                            "arguments": raw if raw is not None else json.dumps(arguments), "origin": "completion"},
                 llm_response_id=response, security_risk="LOW", critic_result=None, summary="run the script")


def observed(call_id: str, output: str = "5050\n", *, exit_code: int = 0, is_error: bool = False,
             timeout: bool = False) -> dict[str, Any]:
    observation = {"kind": "TerminalObservation", "content": [text(output)], "is_error": is_error,
                   "command": "python3 hello.py", "exit_code": exit_code, "timeout": timeout,
                   "metadata": {"exit_code": exit_code, "pid": -1, "username": "agent", "hostname": "oh",
                                "working_dir": "/workspace", "py_interpreter_path": "/usr/bin/python3", "prefix": "",
                                "suffix": ""},
                   "full_output_save_dir": None}
    return event("ObservationEvent", "environment", tool_name="terminal", tool_call_id=call_id,
                 observation=observation, action_id=f"act-{call_id}", extended_content=[])


def mcp_observed(call_id: str, name: str, content: list[dict[str, Any]], *, is_error: bool = False) -> dict[str, Any]:
    return event("ObservationEvent", "environment", tool_name=name, tool_call_id=call_id, action_id=f"act-{call_id}",
                 observation={"kind": "MCPToolObservation", "content": content, "is_error": is_error,
                              "tool_name": name},
                 extended_content=[])


def finish(response: str, call_id: str, said: str, *, thought: str = "") -> dict[str, Any]:
    return action(response, call_id, "finish", {"message": said}, thought=thought,
                  act={"kind": "FinishAction", "message": said})


def finished(call_id: str, said: str) -> dict[str, Any]:
    return event("ObservationEvent", "environment", tool_name="finish", tool_call_id=call_id,
                 action_id=f"act-{call_id}", extended_content=[],
                 observation={"kind": "FinishObservation", "content": [text(said)], "is_error": False})


def agent_error(call_id: str, name: str, error: str) -> dict[str, Any]:
    return event("AgentErrorEvent", "agent", tool_name=name, tool_call_id=call_id, error=error,
                 classification={"kind": "agent_action", "retryable": True, "user_action": "retry", "error_id": None})


def conversation_error(code: str, detail: str, kind: str, **kw: Any) -> dict[str, Any]:
    return event("ConversationErrorEvent", "environment", code=code, detail=detail, **kw,
                 classification={"kind": kind, "retryable": False, "user_action": "none", "error_id": None})


class Clock:
    """The runtime's UTC clock, a quarter second on at each read."""

    def __init__(self) -> None:
        self.at = datetime(2026, 11, 27, 10, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        self.at += timedelta(milliseconds=250)
        return self.at


class Run:
    """One task's records in the order SdkRuntime.run feeds them: each event through a CallFeed over fake metrics,
    which ``respond`` fills as the SDK's telemetry does when a response comes back, with the aux records the feed
    writes before it. The turn's usage is ``turn_usage`` over the same metrics, as the runtime reports it."""

    def __init__(self, *, responses_api: bool = False, effort: str | None = None, offset_s: int = 0,
                 tools: tuple[dict[str, Any], ...] = (TERMINAL, FINISH)) -> None:
        self.records: list[Fed] = []
        self.metrics = SimpleNamespace(response_latencies=[], costs=[], token_usages=[])
        self.responses_api = responses_api
        self.feed = CallFeed(lambda record: self.records.append(Fed(json.loads(json.dumps(record)))),
                             lambda name, record: self.records.append(Fed(json.loads(json.dumps(record)), aux=name)),
                             model=MODEL, now=Clock())
        self.feed.llms = [SimpleNamespace(metrics=self.metrics, reasoning_effort=effort,
                                          uses_responses_api=lambda: responses_api)]
        self.records.append(Fed({"type": "clock", "utc_offset_s": offset_s}, aux="clock"))
        self.actions = 0
        if tools:
            self.feed.tools(tools)

    def respond(self, response_id: str, prompt: int = 100, completion: int = 10, read: int = 0, write: int = 0,
                *, cost: float | None = 0.001, usage: bool = True) -> Run:
        model = f"litellm_proxy/{MODEL}"
        self.metrics.response_latencies.append(SimpleNamespace(model=model, latency=1.5, response_id=response_id))
        if cost:
            self.metrics.costs.append(SimpleNamespace(model=model, cost=cost, timestamp=0.0))
        if usage:
            self.metrics.token_usages.append(SimpleNamespace(
                model=model, prompt_tokens=prompt, completion_tokens=completion, cache_read_tokens=read,
                cache_write_tokens=write, reasoning_tokens=0, context_window=0, per_turn_token=prompt + completion,
                response_id=response_id))
        return self

    def event(self, record: dict[str, Any]) -> Run:
        self.actions += record["kind"] == "ActionEvent"
        self.feed.event(record)
        return self

    def retried(self, attempt: int, error: BaseException) -> Run:
        self.feed.retried(attempt, 5, error)
        return self

    def usage(self) -> Usage:
        cost = sum(entry.cost for entry in self.metrics.costs)
        return openhands_agent.turn_usage([(usage, self.responses_api) for usage in self.metrics.token_usages],
                                          responses=len(self.metrics.response_latencies), tool_calls=self.actions,
                                          cost=cost or None)

    def ended(self, *, cost: bool = True) -> list[Fed]:
        """The task's records once its run ended: the feed's last read, and the turn's cost as the server feeds it."""
        self.feed.end()
        usage = self.usage()
        if cost and usage.cost_usd is not None:
            self.records.append(Fed({"type": "turn", "cost_usd": usage.cost_usd}, aux="turn"))
        return self.records


def context(index: int = 0) -> TurnContext:
    return TurnContext(index=index, context_id="ctx", task_id=f"t{index}", message_id=f"m{index}", inbound=["do it"],
                       agent_name="OpenHands", agent_version="1.58.2", agent_model=MODEL,
                       started_at=f"2026-11-27T09:59:5{index}Z")


def kit(*tasks: tuple[Run, str] | tuple[Run, str, str], cost: bool = True):
    """The tasks, each a run with its final reply and state, through the adapter kit."""
    recorded = [RecordedTask(context=context(index), records=task[0].ended(cost=cost), final=[task[1]],
                             state=task[2] if len(task) > 2 else "completed") for index, task in enumerate(tasks)]
    result = check_adapter(OpenHandsAtif, recorded)
    result.raise_for_errors()
    return result


def agent_steps(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for step in document["steps"] if step["source"] == "agent"]


def results(step: dict[str, Any]) -> list[dict[str, Any]]:
    return (step.get("observation") or {}).get("results") or []


def assert_sums_to(document: dict[str, Any], *runs: Run) -> None:
    """The copy's totals are the turns' reported usage, and its cost theirs when every turn has one."""
    final, usages = document["final_metrics"], [run.usage() for run in runs]
    assert final["total_prompt_tokens"] == sum(usage.input_tokens for usage in usages)
    assert final["total_completion_tokens"] == sum(usage.output_tokens for usage in usages)
    assert final["total_cached_tokens"] == sum(usage.cache_read_tokens for usage in usages)
    assert profile_of(final)["total_cache_write_tokens"] == sum(usage.cache_write_tokens for usage in usages)
    if all(usage.cost_usd is not None for usage in usages):
        assert math.isclose(final["total_cost_usd"], sum(usage.cost_usd for usage in usages))


def coding(run: Run | None = None) -> Run:
    """A first task: the system prompt, the prompt, a shell call and its output, then the reply."""
    run = run or Run()
    run.event(system_prompt()).event(user("Create hello.py"))
    run.respond("chatcmpl-1", 12, 40, write=8800, cost=0.004)
    run.event(action("chatcmpl-1", "toolu_1", thought="Running it.")).event(observed("toolu_1"))
    run.respond("chatcmpl-2", 120, 3, read=8800, cost=0.0007).event(reply("chatcmpl-2", "5050"))
    return run


def test_each_model_call_is_one_step_with_its_model_time_counts_and_cost():
    run = coding()
    document = kit((run, "5050")).document
    first, second = agent_steps(document)

    assert first["model_name"] == second["model_name"] == MODEL
    assert first["timestamp"] == "2026-11-27T10:00:00.250000Z" and second["timestamp"] == "2026-11-27T10:00:00.500000Z"
    assert first["message"] == "Running it."
    assert first["tool_calls"] == [{"tool_call_id": "toolu_1", "function_name": "terminal",
                                    "arguments": {"command": "python3 hello.py", "security_risk": "LOW"}}]
    assert results(first) == [{"source_call_id": "toolu_1", "content": "5050\n",
                               "extra": {"agentenv": {"exit_code": 0}}}]
    assert first["metrics"] == {"prompt_tokens": 8812, "cached_tokens": 0, "completion_tokens": 40,
                                "cost_usd": 0.004,
                                "extra": {"agentenv": {"cache_write_tokens": 8800, "cost_source": "harness"}}}
    assert second["message"] == "5050" and second["metrics"]["prompt_tokens"] == 8920
    assert [step["source"] for step in document["steps"]] == ["user", "agent", "agent"]
    assert_sums_to(document, run)


def test_a_calls_input_counts_its_cache_as_turn_usage_counts_it():
    """A Chat Completions prompt smaller than its cache leaves it out, and gets it added; a Responses prompt holds it."""
    chat, responses = Run(), Run(responses_api=True)
    chat.event(user("go")).respond("r1", 7803, 22).respond("r2", 300, 3, read=7680).event(reply("r2", "ok"))
    responses.event(user("go")).respond("r1", 7803, 22).respond("r2", 7878, 3, read=7680).event(reply("r2", "ok"))
    chat_document, responses_document = (kit((run, "ok")).document for run in (chat, responses))

    assert [step["metrics"]["prompt_tokens"] for step in agent_steps(chat_document)] == [7803, 7980]
    assert [step["metrics"]["prompt_tokens"] for step in agent_steps(responses_document)] == [7803, 7878]
    assert_sums_to(chat_document, chat)
    assert_sums_to(responses_document, responses)


def test_a_response_that_wrote_no_event_is_a_step_of_its_own():
    run = Run().event(user("go")).respond("r1", 50, 0)
    run.event(message("user", [text("Error validating tool 'terminal': missing command")]))
    run.respond("r2").event(reply("r2", "ok"))
    document = kit((run, "ok")).document

    assert [step["source"] for step in document["steps"]] == ["user", "agent", "system", "agent"]
    assert document["steps"][1]["metrics"]["prompt_tokens"] == 50 and not document["steps"][1]["message"]
    assert document["steps"][2]["message"] == "Error validating tool 'terminal': missing command"
    assert_sums_to(document, run)


def test_the_calls_of_one_response_share_its_step_and_each_takes_its_result_there():
    run = Run().event(user("go")).respond("r1")
    run.event(action("r1", "call_a", arguments={"command": "echo a"}, thought="Both."))
    run.event(action("r1", "call_b", arguments={"command": "echo b"}))
    run.event(observed("call_a", "a\n")).event(observed("call_b", "b\n", exit_code=1, is_error=True))
    run.respond("r2").event(reply("r2", "a b"))
    first = agent_steps(kit((run, "a b")).document)[0]

    assert [call["tool_call_id"] for call in first["tool_calls"]] == ["call_a", "call_b"]
    assert [(result["source_call_id"], result["content"], profile_of(result)) for result in results(first)] == [
        ("call_a", "a\n", {"exit_code": 0}), ("call_b", "b\n", {"status": "error", "exit_code": 1})]


def test_a_timed_out_command_is_a_timeout_and_one_still_running_has_no_exit_code():
    run = Run().event(user("go")).respond("r1").event(action("r1", "call_1"))
    run.event(observed("call_1", "", exit_code=-1, timeout=True)).respond("r2").event(reply("r2", "slow"))
    (result,) = results(agent_steps(kit((run, "slow")).document)[0])

    assert profile_of(result) == {"status": "timeout"}


def test_an_image_a_tool_returns_is_a_media_file_named_by_its_bytes():
    shot = png()
    run = Run().event(user("look")).respond("r1")
    run.event(action("r1", "call_1", "browser_take_screenshot", {}, act={"kind": "MCPToolAction", "data": {}}))
    run.event(mcp_observed("call_1", "browser_take_screenshot",
                           [text("[Tool 'browser_take_screenshot' executed.]"), image(shot)]))
    run.respond("r2").event(reply("r2", "Example Domain"))
    result = kit((run, "Example Domain"))

    (observation,) = results(agent_steps(result.document)[0])
    note, picture = observation["content"]
    assert note == {"type": "text", "text": "[Tool 'browser_take_screenshot' executed.]"}
    assert picture["type"] == "image" and picture["source"]["media_type"] == "image/png"
    assert result.media[picture["source"]["path"]] == shot
    assert profile_of(observation)["media"][picture["source"]["path"]]["width"] == 4


@pytest.mark.parametrize(("reasoning", "blocks", "content"), [
    ({"thinking_blocks": [{"type": "thinking", "thinking": "Sum it.", "signature": "sig-1"}],
      "reasoning_content": "Sum it."}, [{"chars": 7, "visibility": "raw", "signature": "sig-1"}], "Sum it."),
    ({"thinking_blocks": [{"type": "redacted_thinking", "data": "ciphertext"}]},
     [{"chars": 0, "visibility": "redacted", "data": "ciphertext"}], None),
    ({"responses_reasoning_item": {"id": "rs_1", "summary": ["Plan the sum."], "content": None,
                                   "encrypted_content": "gAAAA", "status": None}},
     [{"chars": 13, "visibility": "summary"}], "Plan the sum."),
    ({"responses_reasoning_item": {"id": "rs_1", "summary": [], "content": None, "encrypted_content": "gAAAA",
                                   "status": None}}, [{"chars": 0, "visibility": "redacted", "data": "gAAAA"}], None),
    ({"responses_reasoning_item": {"id": "rs_1", "summary": [], "content": None, "encrypted_content": None,
                                   "status": None}}, [{"chars": 0, "visibility": "omitted"}], None),
    ({"reasoning_content": "Plain reasoning."}, None, "Plain reasoning."),
], ids=["thinking", "redacted-thinking", "summary", "ciphertext", "omitted", "litellm-text"])
def test_reasoning_keeps_its_visibility_signature_and_ciphertext(reasoning, blocks, content):
    on_action, on_reply = Run(effort="high"), Run(effort="high")
    on_action.event(user("go")).respond("r1").event(action("r1", "call_1", **reasoning)).event(observed("call_1"))
    on_action.respond("r2").event(reply("r2", "5050"))
    on_reply.event(user("go")).respond("r1").event(reply("r1", "5050", **reasoning))

    for run in (on_action, on_reply):
        step = agent_steps(kit((run, "5050")).document)[0]
        assert step["reasoning_effort"] == "high"
        assert step.get("reasoning_content") == content
        assert profile_of(step).get("reasoning") == blocks


def test_a_call_the_sdk_made_again_is_a_step_with_its_error_before_the_call_that_answered():
    run = Run().event(user("go")).respond("r1").event(action("r1", "call_1")).event(observed("call_1"))
    run.retried(1, RuntimeError("litellm.ServiceUnavailableError: 503 Service Unavailable"))
    run.retried(2, RuntimeError("litellm.ServiceUnavailableError: 503 Service Unavailable"))
    run.respond("r2").event(reply("r2", "first second"))
    document = kit((run, "first second")).document
    steps = agent_steps(document)

    assert [step["llm_call_count"] for step in steps] == [1, 0, 0, 1]
    assert profile_of(steps[1])["error"] == {"kind": "infra_error", "code": "RuntimeError",
                                             "message": "litellm.ServiceUnavailableError: 503 Service Unavailable"}
    assert "metrics" not in steps[1] and steps[1]["model_name"] == MODEL
    assert steps[-1]["message"] == "first second"
    assert_sums_to(document, run)


class LLMNoResponseError(Exception):
    """The SDK's error for a response with no choices, raised inside its retry boundary after telemetry records it."""


def test_a_response_the_sdk_refused_and_made_again_is_its_calls_step_holding_the_error():
    """Gemini's empty choices: the response is in the metrics, tokens and cost and all, before the SDK raises and its
    retry listener hears of that attempt, so the attempt is one step, with the call's usage, time and error."""
    error = "Response choices is less than 1. Response: ModelResponse(id='r1', choices=[])"
    run = Run().event(user("go")).respond("r1", 900, 0, cost=0.0009).retried(1, LLMNoResponseError(error))
    run.respond("r2").event(reply("r2", "ok"))
    document = kit((run, "ok")).document
    refused, answered = agent_steps(document)
    (fed,) = [fed.record for fed in run.records if fed.aux == "llm_call" and fed.record.get("attempt")]

    assert [step["llm_call_count"] for step in (refused, answered)] == [1, 1]
    assert refused["metrics"]["prompt_tokens"] == 900 and refused["metrics"]["cost_usd"] == 0.0009
    assert profile_of(refused)["error"] == {"kind": "infra_error", "message": error, "code": "LLMNoResponseError"}
    assert fed["response_id"] == "r1" and refused["timestamp"] == fed["timestamp"] < answered["timestamp"]
    assert "error" not in profile_of(answered) and answered["message"] == "ok"
    assert_sums_to(document, run)


@pytest.mark.parametrize(("code", "kind", "error_kind"), [
    ("LLMAuthenticationError", "auth", "infra_error"), ("MaxIterationsReached", "agent_action", "agent_error")])
def test_the_runs_failure_is_a_step_with_no_model_call_holding_it(code, kind, error_kind):
    run = Run().event(user("go")).event(conversation_error(code, "it failed", kind))
    document = kit((run, "OpenHands failed: it failed", "failed")).document
    error_step, final = agent_steps(document)

    assert error_step["llm_call_count"] == 0 and "metrics" not in error_step
    assert profile_of(error_step)["error"] == {"kind": error_kind, "message": "it failed", "code": code}
    assert final["message"] == "OpenHands failed: it failed" and profile_of(final)["state"] == "failed"


def test_an_event_time_is_read_in_the_zone_the_clock_record_names():
    run = Run(offset_s=3600).event(user("go"))
    run.event(conversation_error("LLMBadRequestError", "refused", "config", at="2026-11-27T11:30:00.123456"))
    (error_step, _) = agent_steps(kit((run, "OpenHands failed: refused", "failed")).document)

    assert error_step["timestamp"] == "2026-11-27T10:30:00.123456Z"


def test_a_bad_call_gets_the_sdks_error_as_its_result():
    bad = action("r1", "call_1", "nonexistent", {"x": 1})
    bad["action"] = None
    run = Run().event(user("go")).respond("r1").event(bad)
    run.event(agent_error("call_1", "nonexistent", "Tool 'nonexistent' not found."))
    run.respond("r2").event(reply("r2", "sorry"))
    (result,) = results(agent_steps(kit((run, "sorry")).document)[0])

    assert result["content"] == "Tool 'nonexistent' not found." and profile_of(result) == {"status": "error"}


def test_arguments_that_arent_json_are_kept_whole():
    run = Run().event(user("go")).respond("r1").event(action("r1", "call_1", raw="{not json"))
    run.event(agent_error("call_1", "terminal", "Error validating args"))
    run.respond("r2").event(reply("r2", "ok"))
    (call,) = agent_steps(kit((run, "ok")).document)[0]["tool_calls"]

    assert call["arguments"] == {"input": "{not json"} and call["extra"] == {"raw_arguments": "{not json"}


def test_a_condensation_is_the_compaction_step_of_the_condensers_call():
    run = Run().event(user("go")).respond("r1").event(action("r1", "call_1")).event(observed("call_1"))
    run.event(event("CondensationRequest", "environment"))
    run.respond("cond-1", 4000, 300, cost=0.002)
    run.event(event("Condensation", "environment", forgotten_event_ids=["e1", "e2"],
                    summary="The user asked for hello.py; it prints 5050.", summary_offset=1,
                    llm_response_id="cond-1"))
    run.respond("r2").event(reply("r2", "5050"))
    document = kit((run, "5050")).document
    steps = agent_steps(document)

    assert [step["llm_call_count"] for step in steps] == [1, 0, 1, 1]
    assert profile_of(steps[1])["error"]["code"] == "CondensationRequest"
    assert steps[2]["message"] == "The user asked for hello.py; it prints 5050."
    assert profile_of(steps[2])["compaction"] == {"written_by": "assistant", "trigger": "auto"}
    assert steps[2]["metrics"]["prompt_tokens"] == 4000
    assert_sums_to(document, run)


def test_the_harness_messages_to_the_model_are_system_steps_and_the_prompt_is_the_users():
    run = Run().event(user("go", extended=("<skill>lint</skill>",))).respond("r1").event(reply("r1", ""))
    run.event(message("environment", [text("Your last response did not include a function call or a message.")]))
    run.respond("r2").event(reply("r2", "done"))
    document = kit((run, "done")).document

    assert [(step["source"], step["message"]) for step in document["steps"]] == [
        ("user", "do it"), ("system", "<skill>lint</skill>"), ("agent", ""),
        ("system", "Your last response did not include a function call or a message."), ("agent", "done")]


def test_a_reply_the_sdk_gives_as_the_agent_is_said_once_on_a_step_with_no_model_call():
    """A model that can't see the prompt's images is never called: the SDK replies as the agent, and that's the
    task's reply."""
    said = "The model anthropic/claude-haiku-4-5 can't read images."
    run = Run().event(user("look")).event(message("agent", [text(said)], role="assistant"))
    document = kit((run, said)).document

    assert [(step["source"], step["message"], step.get("llm_call_count")) for step in document["steps"]] == [
        ("user", "do it", None), ("agent", said, 0)]


def test_a_step_names_the_effort_its_llm_called_with_the_sdks_default_when_the_task_sets_none():
    """The SDK's LLM calls at its own ``reasoning_effort``, which is ``high`` unless agent-config's ``effort`` sets
    another, so a task that sets none still made its calls at one."""
    default, unset = Run(effort=SDK_EFFORT), Run(effort=None)
    for run in (default, unset):
        run.event(user("go")).respond("r1").event(reply("r1", "ok"))

    assert [agent_steps(kit((run, "ok")).document)[0].get("reasoning_effort") for run in (default, unset)] == [
        "high", None]


def test_a_finish_call_says_the_reply_once():
    plain = Run().event(user("go")).respond("r1").event(finish("r1", "call_f", "5050")).event(finished("call_f", "5050"))
    said = Run().event(user("go")).respond("r1").event(finish("r1", "call_f", "5050", thought="All done."))
    said.event(finished("call_f", "5050"))
    plain_steps, said_steps = (agent_steps(kit((run, "5050")).document) for run in (plain, said))

    assert [step["message"] for step in plain_steps] == ["5050"]
    assert plain_steps[0]["tool_calls"][0]["arguments"] == {"message": "5050"}
    assert [(step["message"], step.get("llm_call_count")) for step in said_steps] == [("All done.", 1), ("5050", 0)]


def test_an_interrupted_tasks_open_calls_are_answered_canceled():
    run = Run().event(user("go")).respond("r1").event(action("r1", "call_1", arguments={"command": "sleep 60"}))
    run.event(event("InterruptEvent", "user"))
    result = kit((run, "OpenHands did not finish within 1s", "failed"))
    (answer,) = results(agent_steps(result.document)[0])

    assert answer == {"source_call_id": "call_1", "content": "", "extra": {"agentenv": {"status": "canceled"}}}
    assert "unmatched" not in profile_of(result.document)


def test_a_stop_as_the_run_starts_pauses_it_and_the_pause_is_no_unmatched_record():
    """``interrupt()`` that finds no run task yet falls back to ``pause()``, which keeps the run from starting."""
    run = Run().event(user("go")).event(event("PauseEvent", "user"))
    result = check_adapter(OpenHandsAtif, [RecordedTask(context=context(), records=run.ended(), final=[
        "OpenHands did not finish within 1s"], state="failed")])

    assert not result.findings
    assert [(step["source"], step.get("llm_call_count")) for step in result.document["steps"]] == [
        ("user", None), ("agent", 0)]


def test_a_result_for_a_call_this_copy_never_saw_stays_in_the_native_log():
    run = Run().event(user("go")).event(observed("call_from_before", "late"))
    run.respond("r1").event(reply("r1", "ok"))
    result = check_adapter(OpenHandsAtif, [RecordedTask(context=context(), records=run.ended(), final=["ok"])])

    assert not result.errors
    assert profile_of(result.document)["unmatched"] == {"ObservationEvent": 1}
    assert [finding.rule for finding in result.findings] == ["kit-unmatched"]


def test_a_response_with_no_usage_is_a_step_without_counts_and_the_totals_are_unknown():
    run = Run().event(user("go")).respond("r1", usage=False, cost=None).event(reply("r1", "ok"))
    document = kit((run, "ok")).document
    (step,) = agent_steps(document)

    assert step["model_name"] == MODEL and "metrics" not in step
    assert set(document["final_metrics"]) == {"total_steps"}
    assert run.usage().provider_details == {"reasoning_tokens": 0, "unreported_calls": 1}


def test_a_cost_goes_on_its_calls_step_only_when_every_response_read_with_it_has_one():
    run = Run().event(user("go")).respond("r1", cost=0.003).respond("r2", cost=None).event(reply("r2", "ok"))
    document = kit((run, "ok")).document

    assert ["cost_usd" in step["metrics"] for step in agent_steps(document)] == [False, False]
    assert document["final_metrics"]["total_cost_usd"] == 0.003
    assert profile_of(document["final_metrics"])["cost_source"] == "harness"


def test_a_resumed_conversations_next_task_has_its_own_steps_tools_and_usage():
    first = coding()
    second = Run(tools=(TERMINAL, FINISH, {"type": "function", "function": {"name": "echo", "parameters": {}}}))
    second.event(user("Now to 1000")).respond("chatcmpl-3", 9500, 30, read=8800)
    second.event(action("chatcmpl-3", "toolu_2", arguments={"command": "python3 hello.py"})).event(
        observed("toolu_2", "500500\n"))
    second.respond("chatcmpl-4", 9600, 4, read=9400).event(reply("chatcmpl-4", "500500"))
    document = kit((first, "5050"), (second, "500500")).document

    assert [step["source"] for step in document["steps"]] == ["user", "agent", "agent", "user", "agent", "agent"]
    assert [step["message"] for step in agent_steps(document)] == ["Running it.", "5050", "", "500500"]
    assert [tool["function"]["name"] for tool in document["agent"]["tool_definitions"]] == ["terminal", "finish",
                                                                                             "echo"]
    assert_sums_to(document, first, second)


def test_from_atif_gives_the_sdks_events_back():
    document = kit((coding(), "5050")).document
    log = OpenHandsAtif.from_atif(document).log

    assert [entry["kind"] for entry in log] == ["MessageEvent", "ActionEvent", "ObservationEvent", "MessageEvent"]
    assert json.loads(log[1]["tool_call"]["arguments"]) == {"command": "python3 hello.py", "security_risk": "LOW"}
    assert log[2]["observation"]["content"] == [{"type": "text", "text": "5050\n"}]


def test_the_feed_reads_a_shared_metrics_object_once_and_nothing_without_a_copy():
    shared = SimpleNamespace(response_latencies=[SimpleNamespace(model="m", latency=1.0, response_id="r1")],
                             costs=[], token_usages=[])
    llms = [SimpleNamespace(metrics=shared, reasoning_effort=None, uses_responses_api=lambda: False)] * 2
    events, records = [], []
    copying = CallFeed(events.append, lambda name, record: records.append((name, record)), model=MODEL)
    native = CallFeed(events.append, None, model=MODEL)
    copying.llms = native.llms = llms
    copying.event({"kind": "MessageEvent"})
    copying.end()
    native.event({"kind": "MessageEvent"})
    native.retried(1, 5, RuntimeError("503"))
    native.tools([TERMINAL])
    native.clock()

    assert [name for name, _ in records] == ["llm_call"] and records[0][1]["outcome"] == "unreported"
    assert events == [{"kind": "MessageEvent"}] * 2 and not native.builds


def test_the_clock_record_is_the_local_offset_the_sdk_stamps_events_in():
    records = []
    CallFeed(lambda _: None, lambda name, record: records.append(record), model=MODEL).clock()
    offset = datetime.now().astimezone().utcoffset()

    assert records == [{"type": "clock", "utc_offset_s": int(offset.total_seconds())}]


@pytest.mark.parametrize("seed", range(40))
def test_a_seeded_conversations_steps_sum_to_its_turns_usage(seed):
    """Tasks of responses with and without calls, some made again after a failed attempt, each with or without a
    cost, through Chat Completions or the Responses API."""
    rng = random.Random(seed)
    runs = []
    for task in range(rng.randint(1, 3)):
        run = Run(responses_api=rng.random() < 0.3)
        run.event(user("go"))
        for response in range(rng.randint(1, 5)):
            rid = f"r{task}-{response}"
            prompt = rng.randint(10, 9000)
            read, write = rng.randint(0, prompt), 0 if run.responses_api else rng.randint(0, 500)
            if rng.random() < 0.15:
                run.retried(1, TimeoutError("timed out"))
            run.respond(rid, prompt, rng.randint(0, 80), read, write, cost=rng.choice([None, 0.001, 0.002]))
            calls = [f"c{task}-{response}-{n}" for n in range(rng.choice([0, 1, 1, 2]))]
            for call in calls:
                run.event(action(rid, call))
            for call in calls:
                run.event(observed(call))
            if not calls:
                run.event(reply(rid, "thinking"))
        run.respond(f"r{task}-end").event(reply(f"r{task}-end", "done"))
        runs.append(run)
    document = kit(*((run, "done") for run in runs)).document

    assert_sums_to(document, *runs)
    assert all(step.get("timestamp") and step.get("model_name") for step in agent_steps(document)
               if step.get("llm_call_count") != 0)


def test_the_tested_sdk_is_the_one_the_image_pins():
    pinned = [line.split("==")[1] for line in (AGENT_DIR / "requirements.txt").read_text().splitlines()
              if line.startswith("openhands-sdk==")]
    assert openhands_atif.TESTED_WITH == tuple(pinned) == (openhands_agent.OPENHANDS_VERSION,)


class FeedingRuntime:
    """A runtime that runs one model call and feeds the task as ``SdkRuntime.run`` does, through a CallFeed over its
    metrics, recording each turn it ran."""

    def __init__(self) -> None:
        self.turns: list[Any] = []

    def stop(self) -> None:
        pass

    async def run(self, turn: Any) -> Any:
        self.turns.append(turn)
        metrics = SimpleNamespace(response_latencies=[], costs=[], token_usages=[])
        feed = CallFeed(turn.on_event, turn.on_aux, model=turn.config.model)
        feed.llms = [SimpleNamespace(metrics=metrics, reasoning_effort=turn.config.effort or SDK_EFFORT,
                                     uses_responses_api=lambda: False)]
        feed.clock()
        feed.event(user(turn.text))
        if feed.builds:
            feed.tools([TERMINAL])
        usage = SimpleNamespace(prompt_tokens=120, completion_tokens=8, cache_read_tokens=100, cache_write_tokens=0,
                                reasoning_tokens=0, response_id="chatcmpl-1")
        metrics.response_latencies.append(SimpleNamespace(model="m", latency=0.5, response_id="chatcmpl-1"))
        metrics.costs.append(SimpleNamespace(model="m", cost=0.002, timestamp=0.0))
        metrics.token_usages.append(usage)
        feed.event(reply("chatcmpl-1", "Done."))
        feed.end()
        return openhands_agent.Outcome("Done.", openhands_agent.turn_usage([(usage, False)], responses=1, tool_calls=0,
                                                                           cost=0.002))


@pytest.fixture
def feeding(tmp_path: Path):
    runtime = FeedingRuntime()
    agent = openhands_agent.OpenHandsAgent(runtime, endpoint=openhands_agent.Endpoint("http://litellm.test"),
                                           home=tmp_path / "home", workspace=tmp_path / "ws")
    with TestClient(agent.create_app(), client=("127.0.0.1", 50000)) as client:
        client.post("/ext/agent-config", json={"model": MODEL})
        yield client, runtime


def _send(client: TestClient, context_id: str) -> str:
    return client.post("/a2a", json={"jsonrpc": "2.0", "id": "1", "method": "message/send", "params": {
        "message": {"kind": "message", "messageId": uuid.uuid4().hex, "role": "user", "contextId": context_id,
                    "parts": [{"kind": "text", "text": "fix it"}]},
        "configuration": {"blocking": True}}}).json()["result"]["id"]


def test_a_task_that_builds_an_atif_copy_feeds_it_the_calls_tools_clock_and_cost(feeding):
    client, runtime = feeding
    assert client.post("/ext/trajectory/prescribe", json={"formats": ["atif/1"]}).status_code == 200
    task_id = _send(client, "ctx-atif")
    read = client.post("/ext/trajectory", json={"task_id": task_id, "format": "atif/1"}).json()
    native = client.post("/ext/trajectory", json={"task_id": task_id}).json()

    assert runtime.turns[0].on_aux is not None
    assert [entry["name"] for entry in read["aux"]] == ["clock", "tools", "llm_call", "turn"]
    document = read["trajectory"]
    (step,) = agent_steps(document)
    assert step["message"] == "Done." and step["metrics"]["prompt_tokens"] == 120
    assert document["final_metrics"]["total_cost_usd"] == 0.002
    assert document["agent"]["tool_definitions"] == [TERMINAL]
    assert [entry["kind"] for entry in native["trajectory"]] == ["MessageEvent", "MessageEvent"]


def test_a_task_that_builds_no_atif_copy_feeds_nothing_beside_its_events(feeding):
    client, runtime = feeding
    task_id = _send(client, "ctx-native")
    atif = client.post("/ext/trajectory", json={"task_id": task_id, "format": "atif/1"})
    native = client.post("/ext/trajectory", json={"task_id": task_id}).json()

    assert runtime.turns[0].on_aux is None
    assert atif.status_code == 409
    assert native["format"] == "openhands-events/v1"
    assert [entry["kind"] for entry in native["trajectory"]] == ["MessageEvent", "MessageEvent"]


def test_the_card_offers_atif_beside_the_sdk_events_which_stay_the_default(feeding):
    client, _ = feeding
    card = client.get("/.well-known/agent.json").json()
    (trajectory,) = [ext for ext in card["capabilities"]["extensions"] if ext["uri"] == "urn:agentenv:trajectory/v1"]

    assert trajectory["params"]["formats"] == ["openhands-events/v1", "atif/1"]
    assert trajectory["params"]["native"] == trajectory["params"]["default"] == "openhands-events/v1"
    assert "prescribe" in trajectory["params"]["methods"]
