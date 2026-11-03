"""One turn as the bridge runs it, against a scripted `hermes` in a process of its own: the profile it writes for the
turn, the events it reads off the pipe as they come, the result it makes of them, and a turn that times out or is
cancelled killing Hermes and every tool it started."""

from __future__ import annotations

import asyncio
import json
import os
import time

import pytest

from tst.unit.agents.hermes.agent import hermes_bridge

_MCP = {"env": {"url": "https://mcp.example/mcp", "headers": {"X-Key": "k", "agentenv-role": "registered"}},
        "peer_agents": {"url": "http://127.0.0.1:8000/ext/peer-agents/mcp", "headers": None}}


def _gone(pid: int, within: float = 5.0) -> bool:
    deadline = time.monotonic() + within
    while True:
        try:
            with open(f"/proc/{pid}/stat") as stat:
                state = stat.read().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            return True
        if state == "Z":
            return True
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)


@pytest.mark.asyncio
async def test_a_turn_runs_in_its_contexts_profile_with_the_endpoint_and_servers_it_was_given(
        monkeypatch, fake_hermes, folders, tmp_path):
    monkeypatch.setenv("LITELLM_BASE_URL", "http://litellm.example")
    monkeypatch.setenv("LITELLM_EXTRA_HEADERS", json.dumps({"X-Team": "evals", "x-litellm-tags": "taskId:mine"}))
    events: list[dict] = []

    result = await hermes_bridge.run_hermes(
        prompt="what is 6*7?", context_id="ctx-a", model="provider/model", mcp_servers=_MCP, system_prompt="be brief",
        project_id="p-1", task_id="t-1", role="solver", effort="high", skills_dir=tmp_path / "skills",
        event_callback=events.append, workspace=str(folders))

    home = hermes_bridge.profile_dir("ctx-a")
    ran = json.loads(fake_hermes.read_text())
    assert (ran["home"], ran["cwd"], ran["prompt"]) == (str(home), str(folders), "what is 6*7?")
    assert ran["args"][:2] == ["agentenv", "turn"]
    assert "--resume" not in ran["args"]
    assert ran["args"][ran["args"].index("--model") + 1] == "provider/model"
    assert ran["args"][ran["args"].index("--provider") + 1] == "litellm"
    assert ran["args"][ran["args"].index("--reasoning") + 1] == "high"

    config = json.loads((home / "config.yaml").read_text())
    assert config["model"] == {"provider": "litellm", "default": "provider/model"}
    assert config["providers"]["litellm"] == {
        "api": "http://litellm.example/v1", "key_env": "LITELLM_API_KEY", "transport": "chat_completions",
        # [model.headers], then the run's attribution, its tags joining the configured ones they don't name.
        "extra_headers": {"x-team": "evals", "x-litellm-tags": "taskId:mine,projectId:p-1",
                          "x-litellm-customer-id": "p-1"}}
    assert config["mcp_servers"] == {
        "env": {"url": "https://mcp.example/mcp", "headers": {"X-Key": "k", "AgentEnv-Role": "solver"}},
        "peer_agents": {"url": "http://127.0.0.1:8000/ext/peer-agents/mcp", "headers": {"AgentEnv-Role": "solver"}}}
    assert config["skills"] == {"external_dirs": [str(tmp_path / "skills")]}
    assert config["plugins"] == {"enabled": ["agentenv"]}
    assert (home / "plugins" / "agentenv").resolve() == hermes_bridge.PLUGIN_DIR
    assert (home / "SOUL.md").read_text() == "be brief"

    assert [event["type"] for event in events] == ["on_session_start", "pre_tool_call", "post_tool_call",
                                                   "post_llm_call", "on_session_end"]
    assert result.trajectory == events
    assert (result.response, result.error, result.session_id, result.tool_call_count) == (
        "42", None, "20260101_000000_abcdef", 1)
    assert result.usage["total_tokens"] == 128


@pytest.mark.asyncio
async def test_a_later_turn_resumes_its_session_and_a_turn_without_a_system_prompt_gets_hermes_own_back(
        fake_hermes, folders):
    await hermes_bridge.run_hermes(prompt="one", context_id="ctx-b", system_prompt="be brief", workspace=str(folders))
    home = hermes_bridge.profile_dir("ctx-b")

    result = await hermes_bridge.run_hermes(prompt="two", context_id="ctx-b", session_id="s-earlier",
                                            workspace=str(folders))

    ran = json.loads(fake_hermes.read_text())
    assert ran["args"][ran["args"].index("--resume") + 1] == "s-earlier"
    assert result.session_id == "s-earlier"
    assert not (home / "SOUL.md").exists()
    (home / "SOUL.md").write_text("Hermes' own")
    await hermes_bridge.run_hermes(prompt="three", context_id="ctx-b", workspace=str(folders))
    assert (home / "SOUL.md").read_text() == "Hermes' own"


@pytest.mark.asyncio
async def test_a_failed_model_call_fails_the_turn_with_what_the_provider_said(monkeypatch, fake_hermes, folders):
    monkeypatch.setenv("FAKE_HERMES_MODE", "fail")

    result = await hermes_bridge.run_hermes(prompt="go", context_id="ctx-c", workspace=str(folders))

    assert (result.error, result.error_code) == ("HTTP 401: invalid key", "401")
    assert result.session_id == "20260101_000000_abcdef"


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["timeout", "cancel"])
async def test_a_turn_that_times_out_or_is_cancelled_kills_hermes_and_the_tools_it_started(
        monkeypatch, fake_hermes, folders, tmp_path, ending):
    monkeypatch.setenv("FAKE_HERMES_MODE", "hang")
    pids, events = tmp_path / "pids", []
    turn = asyncio.create_task(hermes_bridge.run_hermes(
        prompt="go", context_id="ctx-d", timeout_s=3 if ending == "timeout" else 600, event_callback=events.append,
        workspace=str(folders)))
    async with asyncio.timeout(10):
        while not (pids.exists() and len(pids.read_text().split()) == 2):
            await asyncio.sleep(0.02)
    hermes, tool = (int(pid) for pid in pids.read_text().split())

    if ending == "cancel":
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(turn, 10)
    else:
        result = await asyncio.wait_for(turn, 10)
        assert result.error == "hermes timed out after 3s"
        # The session and the events Hermes wrote before it stopped are kept.
        assert result.session_id == "20260101_000000_abcdef"
        assert [event["type"] for event in result.trajectory] == ["on_session_start", "pre_tool_call"]

    assert _gone(hermes) and _gone(tool)
    assert [event["type"] for event in events][:2] == ["on_session_start", "pre_tool_call"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("answer", "valid"), [('{"answer": 42}', True), ("42, I think", False)])
async def test_an_output_format_is_asked_for_and_checked(monkeypatch, fake_hermes, folders, answer, valid):
    monkeypatch.setenv("FAKE_HERMES_ANSWER", answer)
    schema = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}

    result = await hermes_bridge.run_hermes(prompt="go", context_id="ctx-e", workspace=str(folders),
                                            output_format={"type": "json_schema", "schema": schema})

    assert "Return only a JSON object matching this schema" in json.loads(fake_hermes.read_text())["prompt"]
    if valid:
        assert (result.error, result.structured_output) == (None, {"answer": 42})
    else:
        assert result.error_code == "invalid_output_format"


def test_the_event_pipe_is_not_left_open_in_the_server(fake_hermes, folders):
    before = len(os.listdir("/dev/fd"))
    asyncio.run(hermes_bridge.run_hermes(prompt="go", context_id="ctx-f", workspace=str(folders)))
    assert len(os.listdir("/dev/fd")) == before
