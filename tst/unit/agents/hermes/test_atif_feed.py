"""What feeds Hermes' atif/1 copy: the plugin's hook events, which keep what the adapter reads from Hermes' own hook
payloads (the assistant message whole, the request's start, auxiliary calls, children), and the server's ``turn``
records, fed only when a run prescribes atif/1."""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from typing import Any

import pytest
from agentenv_protocol.a2a_agent import TrajectoryLog

from tst.unit.agents.hermes.agent import a2a_server, agentenv_plugin
from tst.unit.agents.hermes.conftest import result

MODEL = "anthropic/claude-haiku-4-5"
SESSION, TURN = "20261127_101500_a1b2c3", "20261127_101500_a1b2c3:task:0a1b2c3d"


@dataclass
class ToolCall:
    """Hermes' normalized tool call (agent/transports/types.py)."""

    id: str | None
    name: str
    arguments: str
    provider_data: dict | None = None


@dataclass
class NormalizedResponse:
    """Hermes' normalized assistant message, whose reasoning a chat-completions route parks in ``provider_data``."""

    content: str | None
    tool_calls: list[ToolCall] | None
    finish_reason: str
    reasoning: str | None = None
    provider_data: dict[str, Any] | None = field(default=None, repr=False)

    reasoning_content = property(lambda self: (self.provider_data or {}).get("reasoning_content"))


@pytest.fixture
def hooks(monkeypatch):
    """The plugin's hooks as Hermes registers them, each writing to a pipe; ``hooks.read()`` the events written."""
    read_end, write_end = os.pipe()
    monkeypatch.setenv("AGENTENV_EVENTS_FD", str(write_end))

    class Context:
        def __init__(self):
            self.hooks = {}

        def register_cli_command(self, *args):
            pass

        def register_hook(self, name, callback):
            self.hooks[name] = callback

        def read(self):
            os.close(write_end)
            with os.fdopen(read_end, "rb") as events:
                return [json.loads(line) for line in events.read().splitlines()]

    context = Context()
    agentenv_plugin.register(context)
    return context


def counts(prompt, output):
    return {"input_tokens": prompt, "output_tokens": output, "cache_read_tokens": 0, "cache_write_tokens": 0,
            "reasoning_tokens": 0, "request_count": 1, "prompt_tokens": prompt, "total_tokens": prompt + output}


def test_a_model_calls_event_keeps_its_message_whole_its_start_and_its_ids(hooks):
    text = "x" * 9000
    message = NormalizedResponse(content=text, finish_reason="tool_calls", provider_data={"reasoning_content": "plan"},
                                 tool_calls=[ToolCall("toolu_1", "terminal", '{"command": "ls"}')])
    hooks.hooks["post_api_request"](
        task_id="t", turn_id=TURN, api_request_id=f"{TURN}:api:1", session_id=SESSION, platform="cli", model=MODEL,
        provider="litellm", base_url="http://litellm/v1", api_mode="chat_completions", api_call_count=1,
        api_duration=1.5, started_at=1795218900.25, ended_at=1795218901.75, first_chunk_at=None,
        finish_reason="tool_calls", message_count=3, response_model="claude-haiku-4-5",
        response={"model": "claude-haiku-4-5", "assistant_message": {"content": text[:8000] + "...[truncated]"}},
        usage=counts(100, 20), context_length=200000, assistant_message=message, assistant_content_chars=9000,
        assistant_tool_call_count=1, moa_references=None)
    (event,) = hooks.read()

    assert event["message"] == {"content": text, "reasoning": "plan",
                                "tool_calls": [{"id": "toolu_1", "name": "terminal", "arguments": '{"command": "ls"}'}]}
    assert {key: event[key] for key in ("api_request_id", "started_at", "response_model", "usage")} == {
        "api_request_id": f"{TURN}:api:1", "started_at": 1795218900.25, "response_model": "claude-haiku-4-5",
        "usage": counts(100, 20)}
    assert not {"assistant_message", "provider", "base_url", "context_length"} & set(event)


def test_an_auxiliary_calls_event_keeps_its_counts_and_its_answers_text(hooks):
    hooks.hooks["post_auxiliary_call"](
        session_id=SESSION, task_id="t", turn_id=TURN, platform="cli", aux_task="compression", api_request_id="",
        retry_count=0, api_call_count=1, model=MODEL, provider="litellm", base_url="http://litellm/v1",
        api_mode="chat_completions", streaming=False, started_at=1795218900.25, message_count=40,
        ended_at=1795218903.0, api_duration=2.75, error=None, error_type=None, finish_reason="stop",
        response_model="claude-haiku-4-5", usage=counts(40000, 700),
        response={"model": "claude-haiku-4-5", "assistant_message": {"role": "assistant", "content": "## Summary"}},
        assistant_content_chars=10, assistant_tool_call_count=0)
    (event,) = hooks.read()

    assert {key: event[key] for key in ("aux_task", "model", "started_at", "content")} == {
        "aux_task": "compression", "model": MODEL, "started_at": 1795218900.25, "content": "## Summary"}
    assert event["usage"]["prompt_tokens"] == 40000 and "response" not in event and "error" not in event


def test_a_tool_result_naming_an_image_in_hermes_media_cache_carries_its_bytes(hooks, monkeypatch, tmp_path):
    """Hermes hands a route that takes no images an MCP screenshot as ``MEDIA:<path>`` alone, the file in its cache; a
    path outside the cache, or to a file not an image, is left a path."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "cache" / "images").mkdir(parents=True)
    shot = tmp_path / "cache" / "images" / "img_7d5bbce47a8d.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\nscreenshot")
    (tmp_path / "cache" / "images" / "notes.txt").write_text("not an image")
    (tmp_path / "state.png").write_bytes(b"\x89PNG\r\n\x1a\nprofile")
    tags = "\n".join(f"MEDIA:{path}" for path in (shot, tmp_path / "cache/images/notes.txt", tmp_path / "state.png",
                                                    tmp_path / "cache/images/gone.png", shot))
    for output in (json.dumps({"result": f"### Result\n{tags}"}), '{"output": "5050", "exit_code": 0}'):
        hooks.hooks["post_tool_call"](
            tool_name="mcp__browser__browser_take_screenshot", args={"scale": "css"}, result=output, task_id="t",
            session_id=SESSION, tool_call_id="toolu_1", turn_id=TURN, api_request_id=f"{TURN}:api:1", duration_ms=51,
            status="ok", error_type=None, error_message=None, middleware_trace=[])
    screenshot, shell = hooks.read()

    assert screenshot["media"] == [{"path": str(shot), "media_type": "image/png",
                                    "data": base64.b64encode(b"\x89PNG\r\n\x1a\nscreenshot").decode()}]
    assert screenshot["result"].endswith(f"MEDIA:{shot}\"}}") and "media" not in shell


def test_a_childs_start_names_its_session_and_its_parents(hooks):
    hooks.hooks["subagent_start"](parent_session_id=SESSION, parent_turn_id=TURN, parent_subagent_id=None,
                                  child_session_id="20261127_101530_c0ffee", child_subagent_id="sa-0-1a2b3c4d",
                                  child_role="leaf", child_goal="count the lines")
    (event,) = hooks.read()

    assert {key: value for key, value in event.items() if key != "timestamp"} == {
        "type": "subagent_start", "parent_session_id": SESSION, "child_session_id": "20261127_101530_c0ffee",
        "child_subagent_id": "sa-0-1a2b3c4d", "child_role": "leaf", "child_goal": "count the lines"}


EVENTS = [
    {"type": "on_session_start", "timestamp": 1795218900000, "session_id": SESSION, "model": MODEL},
    {"type": "post_api_request", "timestamp": 1795218901000, "session_id": SESSION, "turn_id": TURN,
     "api_request_id": f"{TURN}:api:1", "api_call_count": 1, "model": MODEL, "response_model": "claude-haiku-4-5",
     "started_at": 1795218900.5, "finish_reason": "stop", "usage": counts(100, 20), "message": {"content": "42"}},
    {"type": "post_llm_call", "timestamp": 1795218901010, "session_id": SESSION, "turn_id": TURN,
     "assistant_response": "42"},
    {"type": "on_session_end", "timestamp": 1795218901020, "session_id": SESSION, "turn_id": TURN, "completed": True,
     "failed": False, "interrupted": False},
]
PRICED = {"model": MODEL, "input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "cache_read_tokens": 0,
          "cache_write_tokens": 0, "api_calls": 1, "estimated_cost_usd": 0.0271, "cost_status": "estimated"}


@pytest.fixture
def turns(monkeypatch):
    """run_hermes, as Hermes would answer: each event handed to the server as the plugin writes it."""
    async def run(**kwargs):
        for event in EVENTS:
            kwargs["event_callback"](event)
        return result(response="42", session_id=SESSION, usage=PRICED, trajectory=EVENTS)

    monkeypatch.setattr(a2a_server, "run_hermes", run)


@pytest.fixture
def fed(monkeypatch):
    """The aux records the server feeds, by name, whatever the run prescribed."""
    records: list[tuple[str, Any]] = []
    feed = TrajectoryLog.aux
    monkeypatch.setattr(TrajectoryLog, "aux", lambda self, name, record: (records.append((name, record)),
                                                                          feed(self, name, record))[1])
    return records


@pytest.mark.asyncio
async def test_a_task_that_builds_an_atif_copy_is_fed_the_turns_effort_and_hermes_cost(hermes, turns, fed):
    async with hermes.client() as client:
        (await client.post("/ext/agent-config", json={"effort": "high"})).raise_for_status()
        (await client.post("/ext/trajectory/prescribe", json={"formats": ["atif/1"]})).raise_for_status()
        task = await hermes.send(client, "what is 6*7?")
        read = (await client.post("/ext/trajectory", json={"task_id": task["id"], "format": "atif/1"})).json()

    assert task["status"]["state"] == "completed"
    assert fed == [("turn", {"type": "start", "effort": "high"}), ("turn", {"type": "end", "cost_usd": 0.0271})]
    assert [entry["record"] for entry in read["aux"]] == [record for _, record in fed]
    document = read["trajectory"]
    (step,) = [step for step in document["steps"] if step["source"] == "agent"]
    assert (step["message"], step["reasoning_effort"], step["metrics"]["prompt_tokens"]) == ("42", "high", 100)
    assert document["final_metrics"]["total_cost_usd"] == 0.0271


@pytest.mark.asyncio
async def test_a_task_that_builds_no_atif_copy_is_fed_nothing_beside_its_events(hermes, turns, fed):
    async with hermes.client() as client:
        (await client.post("/ext/trajectory/prescribe", json={"formats": []})).raise_for_status()
        task = await hermes.send(client, "what is 6*7?")
        native = (await client.post("/ext/trajectory", json={"task_id": task["id"]})).json()

    assert task["status"]["state"] == "completed"
    assert fed == []
    assert native == {"format": "hermes-hook-events/1", "trajectory": EVENTS}
