"""Pydantic AI's ATIF adapter on the messages the runtime appends, shaped as Pydantic AI 2.61 dumps them
(``ModelMessagesTypeAdapter`` in JSON mode), and the aux records the agent feeds while a task builds an atif/1 copy.
Each case runs through the protocol's adapter kit, so the live rule holds after every record and the document validates
as a producer's; the agent's own turn usage, summed from the same responses, is what the steps add up to."""
from __future__ import annotations

import base64
import json
import random
import re
import struct
import uuid
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from agentenv_protocol.a2a_agent import Usage
from agentenv_protocol.atif import Fed, RecordedTask, TurnContext, check_adapter, errors_in, profile_of, validate
from starlette.testclient import TestClient

from tst.unit.agents.pydantic_ai.agent import AGENT_DIR, pydantic_ai_agent, pydantic_ai_atif

PydanticAIAtif = pydantic_ai_atif.PydanticAIAtif
MODEL = "claude-haiku-4-5-20251001"
ROOT = "01a220d8-9f10-7328-81d4-f1b749fbee03"
CONVERSATION = "01a220d8-9f10-7328-81d4-f1b895b9d401"
START = datetime(2026, 11, 28, 6, 0, tzinfo=timezone.utc)
CLAUDE_FIRST = {"cache_creation_input_tokens": 10363, "cache_read_input_tokens": 0}
CLAUDE_SECOND = {"cache_creation_input_tokens": 199, "cache_read_input_tokens": 10363}


def png(width: int = 4, height: int = 3, colour: bytes = b"\x10\x20\xfb") -> bytes:
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    rows = b"".join(b"\x00" + colour * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


_clock = iter(range(1, 10**9))


def at() -> str:
    return (START + timedelta(milliseconds=next(_clock))).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def binary(data: bytes, media_type: str = "image/png") -> dict[str, Any]:
    return {"data": base64.urlsafe_b64encode(data).decode(), "media_type": media_type, "vendor_metadata": None,
            "kind": "binary", "identifier": uuid.uuid4().hex[:6]}


def prompt(text: str, *images: bytes) -> dict[str, Any]:
    return {"content": [text, *(binary(image) for image in images)], "timestamp": at(), "part_kind": "user-prompt"}


def returned(call_id: str, content: Any, name: str = "shell", outcome: str = "success") -> dict[str, Any]:
    return {"tool_name": name, "content": content, "tool_call_id": call_id, "tool_kind": None, "metadata": None,
            "timestamp": at(), "outcome": outcome, "retry": None, "part_kind": "tool-return"}


def feedback(text: str) -> dict[str, Any]:
    return {"content": text, "retry": {"cause": "no_output", "attempt": None, "max_retries": None},
            "timestamp": at(), "part_kind": "retry-feedback"}


def request(*parts: dict[str, Any], run: str = ROOT) -> dict[str, Any]:
    return {"parts": list(parts), "timestamp": at(), "instructions": "You are a software engineering agent.",
            "kind": "request", "run_id": run, "conversation_id": CONVERSATION, "metadata": None, "state": "complete"}


def text(content: str) -> dict[str, Any]:
    return {"content": content, "id": None, "provider_name": None, "provider_details": None, "citations": None,
            "part_kind": "text"}


def thinking(content: str, signature: str | None = None) -> dict[str, Any]:
    return {"content": content, "id": "reasoning_content", "signature": signature, "provider_name": "openai",
            "provider_details": None, "part_kind": "thinking"}


def call(call_id: str, name: str = "shell", **arguments: Any) -> dict[str, Any]:
    return {"tool_name": name, "args": json.dumps(arguments), "tool_call_id": call_id, "tool_kind": None, "id": None,
            "provider_name": None, "provider_details": None, "part_kind": "tool-call"}


def counts(input_tokens: int = 100, output_tokens: int = 10, read: int = 0, write: int = 0,
           **details: int) -> dict[str, Any]:
    """A response's ``RequestUsage`` as dumped: the cache from OpenAI's prompt details in its fields, and the keys of a
    LiteLLM proxy's Claude answers in ``details``."""
    return {"input_tokens": input_tokens, "cache_write_tokens": write, "cache_write_1h_tokens": 0,
            "cache_read_tokens": read, "output_tokens": output_tokens, "output_reasoning_tokens": 0,
            "input_audio_tokens": 0, "cache_audio_read_tokens": 0, "output_audio_tokens": 0, "audio_seconds": 0.0,
            "details": details, "cost": None}


def response(*parts: dict[str, Any], usage: dict[str, Any] | None = None, run: str = ROOT,
             state: str = "complete") -> dict[str, Any]:
    calls = any(part["part_kind"] == "tool-call" for part in parts)
    return {"parts": list(parts), "usage": usage or counts(), "model_name": MODEL, "timestamp": at(),
            "kind": "response", "provider_name": "openai", "provider_url": "http://litellm.test",
            "provider_details": {"finish_reason": "tool_calls" if calls else "stop"},
            "provider_response_id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "finish_reason": "tool_call" if calls else "stop",
            "run_id": run, "conversation_id": CONVERSATION, "metadata": None,
            "workspace_ref": {"provider": "local", "id": "/workspace"}, "history_digest": None, "failed_attempts": None,
            "state": state}


class Feed:
    """One task's records in arrival order, as the agent feeds them: the turn's settings first, then its messages and
    its delegated runs' as the hooks see them. ``usages`` holds every response's usage as the turn's usage reads it."""

    def __init__(self, effort: str | None = None) -> None:
        self.records: list[Fed] = [Fed({"effort": effort}, aux="turn")]
        self.usages: list[SimpleNamespace] = []
        self.state, self.reply = "completed", None

    def _count(self, message: dict[str, Any]) -> dict[str, Any]:
        if message["kind"] == "response" and message["state"] != "interrupted":
            usage = message["usage"]
            self.usages.append(SimpleNamespace(**{key: usage[key] for key in (
                "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "details")}))
        return message

    def native(self, *messages: dict[str, Any]) -> Feed:
        self.records.extend(Fed(self._count(message)) for message in messages)
        return self

    def delegated(self, run: str, spawned_by: str, *messages: dict[str, Any], parent: str | None = None) -> Feed:
        self.records.extend(Fed({"run": run, "parent": parent, "spawned_by": spawned_by, "message": self._count(m)},
                                aux="delegated") for m in messages)
        return self

    def failed(self, message: str, code: str, *unsent: dict[str, Any], kind: str = "infra_error") -> Feed:
        self.records.append(Fed({"messages": list(unsent), "at": at(),
                                 "error": {"kind": kind, "code": code, "message": message}}, aux="failed"))
        self.state, self.reply = "failed", message
        return self

    def ending(self, reply: str | None, state: str = "completed") -> Feed:
        self.state, self.reply = state, reply
        return self

    def usage(self) -> Usage:
        return pydantic_ai_agent.turn_usage(self.usages, tool_calls=0, requests=len(self.usages))


def context(index: int = 0) -> TurnContext:
    return TurnContext(index=index, context_id="ctx", task_id=f"t{index}", message_id=f"m{index}", inbound=["do it"],
                       agent_name="Pydantic AI", agent_version="2.61.0", agent_model="claude-haiku-4-5",
                       started_at=START)


def run(*feeds: Feed):
    kit = check_adapter(PydanticAIAtif, [
        RecordedTask(context=context(i), records=feed.records, state=feed.state,
                     final=None if feed.reply is None else [feed.reply]) for i, feed in enumerate(feeds)])
    kit.raise_for_errors()
    assert not kit.findings, kit.findings
    return kit


def agent_steps(trajectory: dict) -> list[dict]:
    return [step for step in trajectory["steps"] if step["source"] == "agent"]


def subagents(document: dict) -> dict[str, dict]:
    return {sub["trajectory_id"]: sub for sub in document.get("subagent_trajectories") or ()}


def results(step: dict) -> list[dict]:
    return (step.get("observation") or {}).get("results") or []


def assert_sums_to_the_turns(document: dict, *feeds: Feed) -> None:
    """The copy's totals are the turns' usage, summed from the same responses."""
    reported = [feed.usage() for feed in feeds]
    final = document["final_metrics"]
    assert final["total_prompt_tokens"] == sum(usage.input_tokens or 0 for usage in reported)
    assert final["total_completion_tokens"] == sum(usage.output_tokens or 0 for usage in reported)
    assert final["total_cached_tokens"] == sum(usage.cache_read_tokens or 0 for usage in reported)
    assert profile_of(final)["total_cache_write_tokens"] == sum(usage.cache_write_tokens or 0 for usage in reported)


def coding(feed: Feed | None = None, task: str = "Create hello.py and run it.") -> Feed:
    """A coding turn as Claude answers it through a LiteLLM proxy: a shell call, then the number."""
    feed = feed or Feed(effort="high")
    feed.native(request(prompt(task)),
                response(thinking("I'll write it, then run it."), call("toolu_1", command="python3 hello.py"),
                         usage=counts(10, 129, **CLAUDE_FIRST)),
                request(returned("toolu_1", "5050\n")),
                response(text("5050"), usage=counts(13, 43, **CLAUDE_SECOND)))
    return feed.ending("5050")


def test_a_response_is_one_step_with_its_model_time_effort_and_usage_and_its_calls_results():
    feed = coding()
    first = feed.records[2].record
    kit = run(feed)
    steps = agent_steps(kit.document)

    assert len(steps) == 2
    assert steps[0]["timestamp"] == first["timestamp"]
    assert (steps[0]["model_name"], steps[0]["reasoning_effort"]) == (MODEL, "high")
    assert steps[0]["reasoning_content"] == "I'll write it, then run it."
    assert steps[0]["tool_calls"] == [{"tool_call_id": "toolu_1", "function_name": "shell",
                                       "arguments": {"command": "python3 hello.py"}}]
    assert results(steps[0]) == [{"source_call_id": "toolu_1", "content": "5050\n"}]
    assert steps[0]["metrics"] == {"prompt_tokens": 10373, "cached_tokens": 0, "completion_tokens": 129,
                                   "extra": {"agentenv": {"cache_write_tokens": 10363}}}
    assert steps[1]["message"] == "5050" and steps[1]["metrics"]["prompt_tokens"] == 10575
    assert feed.usage().input_tokens == 20948
    assert_sums_to_the_turns(kit.document, feed)


@pytest.mark.parametrize("usage", [
    counts(2169, 20),
    counts(2303, 2, read=2048),
    counts(13, 43, **CLAUDE_SECOND),
    counts(5000, 10, cache_read_input_tokens=4000),
], ids=["no-cache", "openai-prompt-details", "litellm-claude", "prompt-as-large-as-the-cache"])
def test_a_steps_counts_are_its_calls_as_the_turns_usage_counts_them(usage):
    feed = Feed().native(request(prompt("go")), response(text("ok"), usage=usage)).ending("ok")
    kit = run(feed)

    assert_sums_to_the_turns(kit.document, feed)


def test_a_call_that_reported_no_usage_has_a_step_without_counts_so_the_totals_are_unknown():
    feed = Feed().native(request(prompt("go")), response(call("call_1", command="ls"), usage=counts(0, 0)),
                         request(returned("call_1", "a.py")), response(text("a.py")))
    kit = run(feed.ending("a.py"))
    steps = agent_steps(kit.document)

    assert "metrics" not in steps[0] and steps[0]["llm_call_count"] == 1
    assert steps[1]["metrics"]["prompt_tokens"] == 100
    assert "total_prompt_tokens" not in kit.document["final_metrics"]
    assert feed.usage().provider_details["unreported_calls"] == 1


@pytest.mark.parametrize(("content", "signature", "expected"), [
    ("Plan the sum.", None, None),
    ("Plan the sum.", "sig-1", [{"chars": 13, "visibility": "raw", "signature": "sig-1"}]),
    ("", None, [{"chars": 0, "visibility": "omitted"}]),
])
def test_thinking_is_raw_reasoning_and_keeps_its_signature(content, signature, expected):
    feed = Feed().native(request(prompt("go")), response(thinking(content, signature), text("5050")))
    step = agent_steps(run(feed.ending("5050")).document)[0]

    assert step.get("reasoning_content") == (content or None)
    assert profile_of(step).get("reasoning") == expected


@pytest.mark.parametrize(("outcome", "status"), [
    ("success", None), ("failed", "error"), ("denied", "error"), ("retried", "error"), ("interrupted", "canceled")])
def test_a_tool_returns_outcome_is_its_results_status(outcome, status):
    feed = Feed().native(request(prompt("go")), response(call("call_1", command="ls")),
                         request(returned("call_1", "out", outcome=outcome)), response(text("done")))
    (result,) = results(agent_steps(run(feed.ending("done")).document)[0])

    assert profile_of(result).get("status") == status


def test_a_return_that_isnt_text_is_the_json_the_model_was_sent():
    refused = [{"type": "missing", "loc": ["command"], "msg": "Field required", "input": {"nope": 1}}]
    feed = Feed().native(request(prompt("go")), response(call("call_1", nope=1), call("call_2", "list_files")),
                         request(returned("call_1", refused, outcome="retried"),
                                 returned("call_2", ["a.py", "b.py"], name="list_files")),
                         response(text("done")))
    first, second = results(agent_steps(run(feed.ending("done")).document)[0])

    assert json.loads(first["content"]) == refused and profile_of(first)["status"] == "error"
    assert json.loads(second["content"]) == ["a.py", "b.py"]


def test_an_image_a_tool_returns_is_a_media_file_with_its_bytes_from_url_safe_base64():
    image = next(data for data in (png(colour=bytes([red, 0xff, 0xfe])) for red in range(256))
                 if set(base64.urlsafe_b64encode(data).decode()) & set("-_"))
    screenshot = ["### Page\n- Title: Example Domain", binary(image)]
    feed = Feed().native(request(prompt("look")), response(call("call_1", "browser_take_screenshot")),
                         request(returned("call_1", screenshot, name="browser_take_screenshot")),
                         response(text("Example Domain")))
    kit = run(feed.ending("Example Domain"))
    (result,) = results(agent_steps(kit.document)[0])
    words, picture = result["content"]

    assert words == {"type": "text", "text": "### Page\n- Title: Example Domain"}
    assert picture["type"] == "image" and picture["source"]["media_type"] == "image/png"
    assert kit.media[picture["source"]["path"]] == image
    assert profile_of(result)["media"][picture["source"]["path"]]["width"] == 4


def test_a_url_among_a_returns_files_is_its_url():
    feed = Feed().native(request(prompt("go")), response(call("call_1", "fetch")),
                         request(returned("call_1", ["see", {"url": "https://example.com/a.png", "kind": "image-url",
                                                             "media_type": None, "identifier": "ab12"}], name="fetch")),
                         response(text("seen")))
    (result,) = results(agent_steps(run(feed.ending("seen")).document)[0])

    assert result["content"] == "see\n\nhttps://example.com/a.png"


def test_what_the_harness_adds_to_a_later_request_is_a_system_step_and_the_prompt_is_the_users():
    feed = Feed().native(request(prompt("What is in it?", png())), response(text("")),
                         request(feedback("Please return text or call a tool.")),
                         response(call("call_1", "read_file", path="AGENTS.md")),
                         request(returned("call_1", "# Rules", name="read_file"),
                                 {"content": "Context from AGENTS.md", "timestamp": at(), "part_kind": "user-prompt"}),
                         response(text("A rule.")))
    kit = run(feed.ending("A rule."))
    steps = kit.document["steps"]

    assert [step["source"] for step in steps] == ["user", "agent", "system", "agent", "system", "agent"]
    assert steps[0]["message"] == "do it"
    assert [steps[2]["message"], steps[4]["message"]] == ["Please return text or call a tool.",
                                                          "Context from AGENTS.md"]
    assert results(steps[3])[0]["content"] == "# Rules"


def test_the_last_steps_message_is_the_reply_the_run_strips_so_no_final_step_is_added():
    feed = Feed().native(request(prompt("go")), response(text("  50"), text("50\n")))
    steps = agent_steps(run(feed.ending("5050")).document)

    assert [step["message"] for step in steps] == ["5050"]


def test_a_run_that_failed_answers_its_calls_from_its_unsent_request_and_ends_on_its_error():
    reply = "The model endpoint answered 400: {'message': 'refused'}"
    feed = Feed().native(request(prompt("go")), response(call("call_1", command="echo a")))
    feed.failed(reply, "pydantic_ai.model_http_400", request(returned("call_1", "a\n")),
                response(usage=counts(0, 0), state="interrupted"))
    kit = run(feed)
    first, last = agent_steps(kit.document)

    assert results(first) == [{"source_call_id": "call_1", "content": "a\n"}]
    assert last["llm_call_count"] == 0 and last["message"] == reply and "metrics" not in last
    assert profile_of(last)["error"] == {"kind": "infra_error", "message": reply, "code": "pydantic_ai.model_http_400"}
    assert profile_of(last)["state"] == "failed"
    assert_sums_to_the_turns(kit.document, feed)


def test_a_run_refused_before_any_answer_is_its_error_alone():
    reply = "The model endpoint answered 400: {'message': 'nope'}"
    feed = Feed().failed(reply, "pydantic_ai.model_http_400", request(prompt("go")),
                         response(usage=counts(0, 0), state="interrupted"))
    (step,) = agent_steps(run(feed).document)

    assert step["message"] == reply and profile_of(step)["error"]["code"] == "pydantic_ai.model_http_400"


def test_a_call_a_cancelled_run_never_answered_is_answered_canceled():
    feed = Feed().native(request(prompt("go")), response(call("call_1", command="sleep 600"))).ending(None, "canceled")
    (step,) = agent_steps(run(feed).document)

    assert results(step) == [{"source_call_id": "call_1", "content": "",
                              "extra": {"agentenv": {"status": "canceled"}}}]
    assert profile_of(step)["state"] == "canceled"


SUB = "01a220d1-e9d6-746d-95fb-f2eb68fb7db9"


def delegation(feed: Feed, spawn: str = "call_r0") -> Feed:
    """The turn's run delegating a command to itself, whose run answers with its output."""
    feed.native(request(prompt("Delegate `wc -l /etc/passwd` to a sub-agent.")),
                response(call(spawn, "delegate_task", agent_name="self", task="Run `wc -l /etc/passwd`."),
                         usage=counts(30, 2)))
    feed.delegated(SUB, spawn, request(prompt("Run `wc -l /etc/passwd`."), run=SUB),
                   response(call("call_c1", command="wc -l /etc/passwd"), usage=counts(40, 3), run=SUB),
                   request(returned("call_c1", "19 /etc/passwd\n"), run=SUB),
                   response(text("19"), usage=counts(50, 4), run=SUB))
    return feed.native(request(returned(spawn, "19", name="delegate_task")), response(text("19"), usage=counts(70, 6)))


def test_a_delegated_run_is_a_sub_agent_under_the_call_that_started_it():
    feed = delegation(Feed()).ending("19")
    kit = run(feed)
    spawn = results(agent_steps(kit.document)[0])[0]
    sub = subagents(kit.document)[SUB]

    assert spawn["content"] == "19" and spawn["subagent_trajectory_ref"] == [{"trajectory_id": SUB}]
    assert sub["agent"] == {"name": "Pydantic AI", "version": "2.61.0"}
    assert [step["message"] for step in agent_steps(sub)] == ["", "19"]
    assert results(agent_steps(sub)[0]) == [{"source_call_id": "call_c1", "content": "19 /etc/passwd\n"}]
    assert agent_steps(sub)[0]["model_name"] == MODEL
    assert_sums_to_the_turns(kit.document, feed)


def test_a_sub_agent_another_than_self_keeps_its_name():
    feed = Feed().native(request(prompt("go")), response(call("call_r0", "delegate_task", agent_name="explorer",
                                                              task="look")))
    feed.delegated(SUB, "call_r0", request(prompt("look"), run=SUB), response(text("found"), run=SUB))
    feed.native(request(returned("call_r0", "found", name="delegate_task")), response(text("found")))

    assert subagents(run(feed.ending("found")).document)[SUB]["agent"]["name"] == "explorer"


def test_nested_and_parallel_delegations_keep_their_places_and_reused_ids_are_numbered():
    """vLLM's ids, ``functions.<name>:<index>``, repeat across runs and turns."""
    first, second = "functions.delegate_task:0", "functions.delegate_task:1"

    def turn(feed: Feed, tag: str) -> Feed:
        a, b, leaf = f"run-a-{tag}", f"run-b-{tag}", f"run-leaf-{tag}"
        feed.native(request(prompt("go")), response(call(first, "delegate_task", agent_name="self", task="one"),
                                                    call(second, "delegate_task", agent_name="self", task="two")))
        feed.delegated(a, first, request(prompt("one"), run=a),
                       response(call(first, "delegate_task", agent_name="self", task="leaf"), run=a))
        feed.delegated(b, second, request(prompt("two"), run=b), response(text("two done"), run=b))
        feed.delegated(leaf, first, request(prompt("leaf"), run=leaf),
                       response(call("functions.shell:0", command="echo leaf"), run=leaf),
                       request(returned("functions.shell:0", "leaf\n"), run=leaf),
                       response(text("leaf done"), run=leaf), parent=a)
        feed.delegated(a, first, request(returned(first, "leaf done", name="delegate_task"), run=a),
                       response(text("one done"), run=a))
        feed.native(request(returned(first, "one done", name="delegate_task"),
                            returned(second, "two done", name="delegate_task")), response(text("both done")))
        return feed.ending("both done")

    feeds = (turn(Feed(), "1"), turn(Feed(), "2"))
    kit = run(*feeds)
    subs = subagents(kit.document)

    assert set(subs) == {f"run-{name}-{tag}" for name in ("a", "b", "leaf") for tag in "12"}
    assert results(agent_steps(subs["run-a-1"])[0])[0]["subagent_trajectory_ref"] == [{"trajectory_id": "run-leaf-1"}]
    calls = [c["tool_call_id"] for t in (kit.document, *subs.values()) for s in agent_steps(t)
             for c in s.get("tool_calls") or ()]
    assert len(calls) == len(set(calls)) == 8
    assert [c["tool_call_id"] for c in agent_steps(kit.document)[2]["tool_calls"]] == [f"{first}#3", f"{second}#2"]
    assert_sums_to_the_turns(kit.document, *feeds)


def test_a_delegated_run_that_ended_with_calls_open_has_them_answered_canceled_when_its_own_call_is():
    feed = Feed().native(request(prompt("go")), response(call("call_r0", "delegate_task", agent_name="self",
                                                              task="go")))
    feed.delegated(SUB, "call_r0", request(prompt("go"), run=SUB), response(call("call_c1", command="ls"), run=SUB))
    feed.native(request(returned("call_r0", "Sub-agent 'self' failed: Exceeded maximum output retries (1)",
                                 name="delegate_task", outcome="retried")),
                response(call("call_2", command="ls")), request(returned("call_2", "a.py")), response(text("a.py")))
    kit = run(feed.ending("a.py"))

    (open_call,) = results(agent_steps(subagents(kit.document)[SUB])[0])
    assert open_call == {"source_call_id": "call_c1", "content": "", "extra": {"agentenv": {"status": "canceled"}}}
    assert profile_of(results(agent_steps(kit.document)[0])[0])["status"] == "error"


def test_a_conversations_next_turn_continues_its_document_with_its_own_settings():
    first = coding(Feed(effort="high"))
    second = Feed().native(request(prompt("Now sum to 1000.")), response(call("toolu_2", command="python3 hello.py")),
                           request(returned("toolu_2", "500500\n")), response(text("500500"))).ending("500500")
    kit = run(first, second)
    steps = agent_steps(kit.document)

    assert [step["message"] for step in steps] == ["", "5050", "", "500500"]
    assert [step.get("reasoning_effort") for step in steps] == ["high", "high", None, None]
    assert [step["source"] for step in kit.document["steps"]].count("user") == 2
    assert_sums_to_the_turns(kit.document, first, second)


def test_from_atif_gives_the_messages_pydantic_ai_resumes_from():
    document = run(coding()).document
    native = PydanticAIAtif.from_atif(document)

    assert [message["kind"] for message in native.log] == ["request", "response", "request", "response"]
    assert native.log[1]["parts"][1] == {"part_kind": "tool-call", "tool_name": "shell",
                                         "args": {"command": "python3 hello.py"}, "tool_call_id": "toolu_1"}
    assert native.log[1]["usage"] == {"input_tokens": 10373, "cache_read_tokens": 0, "output_tokens": 129}
    assert native.log[2]["parts"] == [{"part_kind": "tool-return", "tool_name": "shell", "content": "5050\n",
                                       "tool_call_id": "toolu_1"}]
    assert json.loads(native.session["messages.json"]) == native.log


def seeded_conversation(seed: int) -> list[Feed]:
    """A random conversation of one to three turns, as the runtime and its hooks feed them: responses of thinking, text
    and calls with both cache shapes, results of text, JSON and images, retry feedback, delegated runs nested up to two
    deep with ids reused across runs, and turns that fail on a call."""
    rng = random.Random(seed)
    ids = iter(range(10**6))
    image = png()

    def usage() -> dict[str, Any]:
        if rng.random() < 0.5:
            tokens = rng.randint(1, 9000)
            return counts(tokens, rng.randint(1, 50), read=rng.randint(0, tokens))
        return counts(rng.randint(1, 40), rng.randint(1, 50), cache_creation_input_tokens=rng.randint(0, 900),
                      cache_read_input_tokens=rng.randint(0, 9000))

    def play(feed: Feed, run: str | None, parent: str | None, spawned_by: str | None, depth: int,
             may_fail: bool) -> str | None:
        """A run's calls and answer, from its prompt on; the turn's own run (``run`` None) may fail instead."""
        tag = {} if run is None else {"run": run}

        def emit(*messages: dict[str, Any]) -> None:
            if run is None:
                feed.native(*messages)
            else:
                feed.delegated(run, spawned_by, *messages, parent=parent)

        emit(request(prompt("go" if run is None else "sub"), **tag))
        for _ in range(rng.randint(0, 3)):
            parts, made = [], []
            if rng.random() < 0.4:
                parts.append(thinking(rng.choice(["Plan.", ""])))
            if rng.random() < 0.3:
                parts.append(text("Looking."))
            for _ in range(rng.choice([1, 1, 2])):
                call_id, delegates = f"call_{next(ids) % 7}", depth < 2 and rng.random() < 0.2
                parts.append(call(call_id, "delegate_task", agent_name="self", task="sub") if delegates
                             else call(call_id, rng.choice(["shell", "browser_take_screenshot"]), command="ls"))
                made.append((call_id, delegates))
            emit(response(*parts, usage=usage(), **tag))
            returns = []
            for call_id, delegates in made:
                if delegates:
                    answer = play(feed, f"run-{next(ids)}", run, call_id, depth + 1, False)
                    returns.append(returned(call_id, answer, name="delegate_task"))
                else:
                    content = rng.choice(["out", ["shot", binary(image)], {"ok": True}])
                    returns.append(returned(call_id, content, outcome=rng.choice(["success"] * 4 + ["retried"])))
            if rng.random() < 0.2:
                returns.append(feedback("Please return text or call a tool."))
            if may_fail and rng.random() < 0.3:
                feed.failed("The model endpoint failed: Connection error.", "pydantic_ai.model_api",
                            request(*returns), response(usage=counts(0, 0), state="interrupted"))
                return None
            emit(request(*returns, **tag))
        emit(response(text("done"), usage=usage(), **tag))
        return "done"

    feeds = []
    for _ in range(rng.randint(1, 3)):
        feed = Feed(effort=rng.choice([None, "high"]))
        answer = play(feed, None, None, None, 0, rng.random() < 0.3)
        feeds.append(feed if answer is None else feed.ending(answer))
    return feeds


@pytest.mark.parametrize("seed", range(120))
def test_a_seeded_conversations_copy_answers_every_call_and_sums_to_its_turns(seed):
    feeds = seeded_conversation(seed)
    kit = run(*feeds)
    trajectories = [kit.document, *subagents(kit.document).values()]
    steps = [step for trajectory in trajectories for step in agent_steps(trajectory)]

    calls = [c["tool_call_id"] for step in steps for c in step.get("tool_calls") or ()]
    answered = {r["source_call_id"] for step in steps for r in results(step)}
    assert len(calls) == len(set(calls)) and set(calls) <= answered
    assert sum(step.get("llm_call_count") == 1 for step in steps) == sum(len(feed.usages) for feed in feeds)
    assert_sums_to_the_turns(kit.document, *feeds)


def test_the_adapter_was_checked_against_the_pydantic_ai_the_image_pins():
    pinned = re.search(r"^pydantic-ai-slim\[[^]]*\]==(\S+)$", (AGENT_DIR / "requirements.txt").read_text(),
                       re.M).group(1)

    assert pydantic_ai_atif.TESTED_WITH == (pinned,) == (pydantic_ai_agent.PYDANTIC_AI_VERSION,)


def test_the_image_installs_a_protocol_with_the_atif_kit_the_adapter_uses():
    requirements = (AGENT_DIR / "requirements.txt").read_text().splitlines()

    assert "agentenv-framework-protocol[agent]>=0.1.670" in requirements


#: Pydantic AI's profile of Claude through an OpenAI-compatible endpoint, and of a model it sends ``thinking`` to.
CLAUDE_PROFILE = {"supports_thinking": False, "thinking_always_enabled": False}
THINKING_PROFILE = {"supports_thinking": True, "thinking_always_enabled": False}


@dataclass
class ScriptedRuntime:
    """Answers each turn with one response, with the usage Claude's have through a LiteLLM proxy, and keeps the turns it
    ran; it feeds the turn's settings as the real runtime does, for a model of ``profile``, by default one Pydantic AI
    sends ``thinking`` to."""

    profile: dict[str, Any] = field(default_factory=lambda: dict(THINKING_PROFILE))
    turns: list[Any] = field(default_factory=list)

    async def stop(self) -> None:
        pass

    def check_messages(self, data: bytes) -> None:
        pass

    async def run(self, turn: Any) -> Any:
        self.turns.append(turn)
        if turn.on_aux is not None:
            turn.on_aux(pydantic_ai_atif.TURN, {"effort": pydantic_ai_agent.effort_sent(turn.config.effort,
                                                                                        self.profile)})
        turn.on_event(request(prompt(turn.text)))
        turn.on_event(response(text("Done."), usage=counts(13, 43, **CLAUDE_SECOND)))
        usage = pydantic_ai_agent.turn_usage([SimpleNamespace(input_tokens=13, output_tokens=43, cache_read_tokens=0,
                                                              cache_write_tokens=0, details=CLAUDE_SECOND)],
                                             tool_calls=0, requests=1)
        return pydantic_ai_agent.Outcome("Done.", usage)


@contextmanager
def serving(tmp_path, prescribe: bool,
            runtime: ScriptedRuntime | None = None) -> Iterator[tuple[ScriptedRuntime, dict, TestClient]]:
    """The agent on the scripted runtime, having run one task at effort high, with atif/1 prescribed or not."""
    runtime = runtime or ScriptedRuntime()
    agent = pydantic_ai_agent.PydanticAIAgent(runtime, endpoint=pydantic_ai_agent.Endpoint("http://litellm.test"),
                                              home=tmp_path / "home", workspace=tmp_path / "ws")
    with TestClient(agent.create_app(), client=("127.0.0.1", 50000)) as client:
        client.post("/ext/agent-config", json={"model": "claude-haiku-4-5", "effort": "high"})
        if prescribe:
            assert client.post("/ext/trajectory/prescribe", json={"formats": ["atif/1"]}).status_code == 200
        task = client.post("/a2a", json={"jsonrpc": "2.0", "id": "1", "method": "message/send", "params": {
            "message": {"kind": "message", "messageId": uuid.uuid4().hex, "role": "user", "contextId": "ctx-1",
                        "parts": [{"kind": "text", "text": "go"}]},
            "configuration": {"blocking": True}}}).json()["result"]
        yield runtime, task, client


def test_the_card_offers_atif_beside_its_messages_which_stay_the_default(tmp_path):
    with serving(tmp_path, prescribe=False) as (_, _, client):
        card = client.get("/.well-known/agent.json").json()
    params = next(e for e in card["capabilities"]["extensions"] if e["uri"] == "urn:agentenv:trajectory/v1")["params"]

    assert params["formats"] == ["pydantic-ai-messages/v1", "atif/1"]
    assert params["native"] == params["default"] == "pydantic-ai-messages/v1"


def test_a_task_that_builds_atif_hands_the_runtime_its_aux_which_feeds_the_turns_settings(tmp_path):
    with serving(tmp_path, prescribe=True) as (runtime, task, client):
        read = client.post("/ext/trajectory", json={"task_id": task["id"], "format": "atif/1"})

    assert callable(runtime.turns[0].on_aux)
    assert read.status_code == 200
    body = read.json()
    assert body["aux"] == [{"name": "turn", "record": {"effort": "high"}, "after": 0}]
    document = body["trajectory"]
    assert not errors_in(validate(document))
    (step,) = agent_steps(document)
    assert step["message"] == "Done." and step["reasoning_effort"] == "high"
    assert document["final_metrics"]["total_prompt_tokens"] == 10575


def test_a_task_that_builds_no_atif_hands_the_runtime_no_aux_and_its_messages_stay_as_they_are(tmp_path):
    with serving(tmp_path, prescribe=False) as (runtime, task, client):
        native = client.post("/ext/trajectory", json={"task_id": task["id"]}).json()["trajectory"]
        atif = client.post("/ext/trajectory", json={"task_id": task["id"], "format": "atif/1"})

    assert runtime.turns[0].on_aux is None
    assert [event["kind"] for event in native] == ["request", "response"]
    assert atif.status_code >= 400


@pytest.mark.parametrize(("profile", "effort"), [
    (CLAUDE_PROFILE, None), (THINKING_PROFILE, "high"), ({"thinking_always_enabled": True}, "high"), ({}, None)])
def test_the_effort_sent_is_the_configured_one_only_for_a_model_whose_profile_takes_thinking(profile, effort):
    assert pydantic_ai_agent.effort_sent("high", profile) == effort
    assert pydantic_ai_agent.effort_sent(None, profile) is None


def test_an_effort_pydantic_ai_never_sent_the_model_is_not_claimed(tmp_path):
    """Claude through a LiteLLM proxy is a model whose profile takes no ``thinking``, which Pydantic AI then drops, so
    its steps ran at no effort whatever agent-config set."""
    with serving(tmp_path, prescribe=True, runtime=ScriptedRuntime(CLAUDE_PROFILE)) as (runtime, task, client):
        body = client.post("/ext/trajectory", json={"task_id": task["id"], "format": "atif/1"}).json()

    assert runtime.turns[0].config.effort == "high"
    assert body["aux"] == [{"name": "turn", "record": {"effort": None}, "after": 0}]
    (step,) = agent_steps(body["trajectory"])
    assert "reasoning_effort" not in step
