"""The per-turn OPENCODE_CONFIG_CONTENT (_build_config) and what reaches the CLI.

* `additional_drop_params: ["tool_choice"]` must be present on EVERY model entry: some LiteLLM adapters
  (fireworks_ai, e.g. kimi-k3) hard-reject `tool_choice` (400 UnsupportedParamsError) instead of dropping it, and
  opencode always sends its tool set. Unknown snake_case option keys pass through @ai-sdk/openai-compatible into the
  request body verbatim, which is what makes this work.
* `reasoningEffort` (camelCase, the spelling the provider maps to the wire `reasoning_effort` param) merges alongside
  it when an effort is set.
"""
import asyncio
import json

import pytest

from tst.unit.agents.opencode.agent import opencode_bridge
from tst.unit.agents.opencode.fake_cli import FakeCLI

_build_config = opencode_bridge._build_config
MODEL = "fireworks_ai/kimi-k3"
_STOP = {"type": "step_finish", "sessionID": "ses_1", "part": {"type": "step-finish", "reason": "stop"}}


def _model_entry(config: dict, model: str = MODEL) -> dict:
    return config["provider"]["litellm"]["models"][model]


def test_tool_choice_always_dropped():
    config = _build_config(MODEL, None, {}, None)
    options = _model_entry(config)["options"]
    assert options["additional_drop_params"] == ["tool_choice"]


def test_effort_merges_with_drop_params():
    config = _build_config(MODEL, None, {}, None, effort="medium")
    options = _model_entry(config)["options"]
    assert options["additional_drop_params"] == ["tool_choice"]
    assert options["reasoningEffort"] == "medium"


def test_no_effort_omits_reasoning_key():
    config = _build_config(MODEL, None, {}, None)
    options = _model_entry(config)["options"]
    assert "reasoningEffort" not in options


def test_model_reference_and_provider_untouched():
    config = _build_config(MODEL, None, {"x-litellm-tags": "taskId:t1"}, None)
    assert config["model"] == f"litellm/{MODEL}"
    assert config["provider"]["litellm"]["options"]["headers"]["x-litellm-tags"] == "taskId:t1"
    # The headless permission contract must survive options changes.
    assert config["permission"] == {"*": "allow", "question": "deny"}


@pytest.mark.parametrize("base, expected", [
    ("https://gateway.example", "https://gateway.example/v1"),
    ("https://gateway.example/", "https://gateway.example/v1"),
    ("https://gateway.example/v1", "https://gateway.example/v1"),
])
def test_the_endpoint_is_the_configured_base_urls_v1(base, expected):
    config = _build_config(MODEL, None, {}, None, environ={"LITELLM_BASE_URL": base})

    assert config["provider"]["litellm"]["options"]["baseURL"] == expected
    assert config["provider"]["litellm"]["options"]["apiKey"] == "{env:LITELLM_API_KEY}"


def test_the_endpoints_model_headers_go_under_the_runs_attribution_in_any_case():
    environ = {"LITELLM_EXTRA_HEADERS": json.dumps(
        {"X-Route": "eu", "X-LiteLLM-Customer-Id": "spoofed", "X-LiteLLM-Tags": "useCase:evals,taskId:own"})}

    headers = opencode_bridge.provider_headers(
        environ, {"x-litellm-customer-id": "p-1", "x-litellm-tags": "projectId:p-1,taskId:t-1"})

    assert headers == {"x-route": "eu", "x-litellm-customer-id": "p-1",
                       "x-litellm-tags": "useCase:evals,taskId:own,projectId:p-1"}


@pytest.mark.parametrize("raw", ["not json", '["x-route", "eu"]'])
def test_model_headers_that_are_not_a_json_object_are_ignored(raw):
    assert opencode_bridge.provider_headers({"LITELLM_EXTRA_HEADERS": raw}, {}) == {}


def test_the_runs_role_replaces_the_role_a_server_is_registered_with_in_any_case():
    servers = {"env": {"url": "https://mcp.example/mcp", "headers": {"agentenv-role": "registered", "X-Key": "k"}},
               "peer_agents": {"url": "http://127.0.0.1:8000/ext/peer-agents/mcp", "headers": None}}

    with_role = _build_config(MODEL, servers, {}, "solver")["mcp"]
    without_role = _build_config(MODEL, servers, {}, None)["mcp"]

    assert with_role == {
        "env": {"type": "remote", "url": "https://mcp.example/mcp", "enabled": True,
                "headers": {"X-Key": "k", "AgentEnv-Role": "solver"}},
        "peer_agents": {"type": "remote", "url": "http://127.0.0.1:8000/ext/peer-agents/mcp", "enabled": True,
                        "headers": {"AgentEnv-Role": "solver"}},
    }
    assert without_role["env"]["headers"] == {"agentenv-role": "registered", "X-Key": "k"}
    assert "headers" not in without_role["peer_agents"]


@pytest.fixture
def spawned(monkeypatch):
    """Every spawn's argv and environment, each answered by a CLI that finishes its turn."""
    calls: list[tuple[list[str], dict]] = []

    async def _exec(*cmd, **kwargs):
        calls.append((list(cmd), kwargs["env"]))
        return FakeCLI(json.dumps(_STOP).encode() + b"\n")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)
    monkeypatch.setattr(opencode_bridge, "_context_xdg", lambda _context: {"XDG_DATA_HOME": "/ctx/share"})
    return calls


def test_a_turn_runs_in_its_workspace_with_tls_checks_on(monkeypatch, spawned):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://gateway.example")
    monkeypatch.setenv("LITELLM_EXTRA_HEADERS", json.dumps({"x-route": "eu"}))
    monkeypatch.delenv("NODE_TLS_REJECT_UNAUTHORIZED", raising=False)

    asyncio.run(opencode_bridge.run_opencode(
        prompt="go", context_id="ctx", session_id="ses_0", model="m", project_id="p-1", task_id="t-1",
        attachment_paths=["/w/a.png"], workspace="/w"))

    ((argv, env),) = spawned
    assert argv == ["opencode", "run", "--format", "json", "--model", "litellm/m", "--dir", "/w", "-s", "ses_0", "go",
                    "--file", "/w/a.png"]
    assert "NODE_TLS_REJECT_UNAUTHORIZED" not in env and env["XDG_DATA_HOME"] == "/ctx/share"
    options = json.loads(env["OPENCODE_CONFIG_CONTENT"])["provider"]["litellm"]["options"]
    assert options["baseURL"] == "https://gateway.example/v1"
    assert options["headers"] == {"x-route": "eu", "x-litellm-customer-id": "p-1",
                                  "x-litellm-tags": "projectId:p-1,taskId:t-1"}
    assert opencode_bridge.get_last_attribution_snapshot()["project_id"] == "p-1"


def test_a_turn_with_no_model_of_its_own_runs_the_default(monkeypatch, spawned):
    monkeypatch.setattr(opencode_bridge, "DEFAULT_MODEL", "provider/fallback")

    asyncio.run(opencode_bridge.run_opencode(prompt="go", context_id="ctx"))

    ((argv, env),) = spawned
    assert argv[argv.index("--model") + 1] == "litellm/provider/fallback"
    assert json.loads(env["OPENCODE_CONFIG_CONTENT"])["model"] == "litellm/provider/fallback"


def test_the_system_prompt_and_the_output_format_lead_and_follow_the_message(spawned):
    schema = {"type": "object"}

    asyncio.run(opencode_bridge.run_opencode(
        prompt="go", context_id="ctx", system_prompt="be brief", output_format={"type": "json_schema", "schema": schema}))

    ((argv, _env),) = spawned
    assert argv[-1] == ('be brief\n\ngo\n\nReturn only a JSON object matching this schema, with no Markdown or '
                        'surrounding text: {"type":"object"}')
