"""What the CLI is run with: the configured endpoint as Anthropic's variables, the model's aliases pinned to the run's
model, [model.headers] and the run's attribution on every call, the turn and thinking limits, and what a 404 from
the endpoint means."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tst.unit.agents.claude_code.agent import claude_bridge
from tst.unit.agents.claude_code.support import FakeCLI

GATEWAY = {"LITELLM_API_KEY": "gateway-key", "LITELLM_BASE_URL": "https://gateway.example.com"}
ALIASES = ("ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
           "ANTHROPIC_DEFAULT_FABLE_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL")


def test_the_configured_endpoint_is_the_clis_with_its_key_as_a_bearer_token():
    env = claude_bridge.cli_env({**GATEWAY, "ANTHROPIC_API_KEY": "stale"})

    assert (env["ANTHROPIC_AUTH_TOKEN"], env["ANTHROPIC_BASE_URL"], env["ANTHROPIC_API_KEY"]) == (
        "gateway-key", "https://gateway.example.com", "")
    assert env["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"


def test_a_script_the_model_runs_can_call_the_endpoint_with_the_litellm_variables():
    headers = json.dumps({"X-Bill-To": "team-a"})

    env = claude_bridge.cli_env({**GATEWAY, "LITELLM_EXTRA_HEADERS": headers})

    kept = {name: env[name] for name in (*GATEWAY, "LITELLM_EXTRA_HEADERS")}
    assert kept == {**GATEWAY, "LITELLM_EXTRA_HEADERS": headers}


def test_anthropic_variables_already_set_are_kept():
    env = claude_bridge.cli_env({**GATEWAY, "ANTHROPIC_AUTH_TOKEN": "own", "ANTHROPIC_BASE_URL": "https://own.example"})

    assert (env["ANTHROPIC_AUTH_TOKEN"], env["ANTHROPIC_BASE_URL"]) == ("own", "https://own.example")


def test_no_endpoint_leaves_the_cli_to_its_own_and_an_api_key_alone():
    env = claude_bridge.cli_env({"ANTHROPIC_API_KEY": "direct"}, model="claude-x")

    assert env["ANTHROPIC_API_KEY"] == "direct" and "ANTHROPIC_AUTH_TOKEN" not in env
    assert not any(alias in env for alias in ALIASES)


def test_the_models_aliases_are_pinned_on_the_endpoint_but_an_explicit_one_wins():
    env = claude_bridge.cli_env({**GATEWAY, "ANTHROPIC_DEFAULT_HAIKU_MODEL": "small"}, model="claude-x")

    assert {alias: env[alias] for alias in ALIASES} == {
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-x", "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-x",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "small", "ANTHROPIC_DEFAULT_FABLE_MODEL": "claude-x",
        "CLAUDE_CODE_SUBAGENT_MODEL": "claude-x",
    }


def test_model_headers_and_the_runs_attribution_reach_every_call():
    env = claude_bridge.cli_env(
        {**GATEWAY, "LITELLM_EXTRA_HEADERS": json.dumps({"X-Bill-To": "team-a", "X-LiteLLM-Tags": "useCase:evals"})},
        project_id="p-1", task_id="t-1")

    assert env["ANTHROPIC_CUSTOM_HEADERS"].splitlines() == [
        "X-Bill-To: team-a",
        "x-litellm-customer-id: p-1",
        "x-litellm-tags: useCase:evals,projectId:p-1,taskId:t-1",
    ]


def test_a_tag_key_model_headers_name_keeps_their_value():
    env = claude_bridge.cli_env(
        {**GATEWAY, "LITELLM_EXTRA_HEADERS": json.dumps({"x-litellm-tags": "taskId:own, useCase:evals"})},
        project_id="p-1", task_id="t-1")

    assert "x-litellm-tags: taskId:own,useCase:evals,projectId:p-1" in env["ANTHROPIC_CUSTOM_HEADERS"].splitlines()


@pytest.mark.parametrize("raw", ["not json", '["a list"]'])
def test_model_headers_that_arent_a_json_object_are_ignored(raw):
    assert "ANTHROPIC_CUSTOM_HEADERS" not in claude_bridge.cli_env({**GATEWAY, "LITELLM_EXTRA_HEADERS": raw})


def test_attribution_alone_is_sent_without_model_headers():
    env = claude_bridge.cli_env(GATEWAY, task_id="t-1")

    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "x-litellm-tags: taskId:t-1"


@pytest.fixture
def spawned(monkeypatch):
    seen: dict = {}

    async def _exec(*cmd, **kwargs):
        seen["cmd"], seen["env"], seen["cwd"] = list(cmd), kwargs["env"], kwargs.get("cwd")
        return FakeCLI(b'{"type": "result", "subtype": "success", "result": "ok", "session_id": "s"}\n')

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)
    return seen


def test_the_turn_and_thinking_limits_reach_the_cli(spawned, monkeypatch):
    for name, value in GATEWAY.items():
        monkeypatch.setenv(name, value)

    asyncio.run(claude_bridge.run_claude(prompt="p", session_id="s", resume=False, model="claude-x", max_turns=7,
                                         max_thinking_tokens=4096))

    assert spawned["cmd"][spawned["cmd"].index("--max-turns") + 1] == "7"
    assert spawned["env"]["MAX_THINKING_TOKENS"] == "4096"
    assert spawned["env"]["ANTHROPIC_AUTH_TOKEN"] == "gateway-key"


def test_no_limits_add_no_flags(spawned):
    asyncio.run(claude_bridge.run_claude(prompt="p", session_id="s", resume=False))

    assert "--max-turns" not in spawned["cmd"] and "MAX_THINKING_TOKENS" not in spawned["env"]


def test_a_404_from_the_endpoint_says_it_may_not_serve_anthropics_messages_api(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://router.example.com/v1")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    not_found = {"type": "result", "subtype": "success", "is_error": True, "session_id": "s",
                 "terminal_reason": "api_error", "api_error_status": 404, "result": "API Error: 404 Not Found"}

    async def _exec(*cmd, **kwargs):
        return FakeCLI((json.dumps(not_found) + "\n").encode(), returncode=1)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)

    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="s", resume=False))

    assert result.attempts == 1
    assert result.error == (
        "claude CLI exited with code 1: API Error: 404 Not Found; https://router.example.com/v1 answered 404: Claude "
        "Code needs an endpoint serving Anthropic's Messages API (<base>/v1/messages), such as a LiteLLM gateway, "
        "that knows the model")


@pytest.mark.parametrize("workspace, project", [
    ("/app", "-app"),
    ("/workspaces/my_ws.x-y/Sub Dir", "-workspaces-my-ws-x-y-Sub-Dir"),  # as the CLI names it, checked against 2.1.280
])
def test_a_sessions_transcripts_are_where_the_cli_keeps_them_for_its_workspace(workspace, project):
    assert claude_bridge.project_dir(Path(workspace)) == claude_bridge.CLAUDE_STATE_DIR / "projects" / project


def test_a_workspace_reached_through_a_link_keeps_its_transcripts_where_the_cli_does_for_its_real_path(tmp_path):
    # The CLI names the directory for its working directory, which the OS gives it resolved.
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)

    assert claude_bridge.project_dir(tmp_path / "link") == claude_bridge.project_dir(tmp_path / "real")
    assert "-link" not in claude_bridge.project_dir(tmp_path / "link").name


def test_the_workspace_is_the_configured_one_else_the_working_directory_but_never_the_root(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path / "set"))
    assert claude_bridge._workspace() == tmp_path / "set"
    monkeypatch.delenv("AGENT_WORKSPACE_DIR")
    monkeypatch.chdir(tmp_path)
    assert claude_bridge._workspace() == tmp_path
    monkeypatch.chdir("/")
    assert claude_bridge._workspace() == Path("/app")


def test_the_cli_runs_in_the_workspace(spawned, monkeypatch, tmp_path):
    monkeypatch.setattr(claude_bridge, "WORKSPACE_DIR", tmp_path / "repo")

    asyncio.run(claude_bridge.run_claude(prompt="p", session_id="s", resume=False))

    assert spawned["cwd"] == tmp_path / "repo" and (tmp_path / "repo").is_dir()
