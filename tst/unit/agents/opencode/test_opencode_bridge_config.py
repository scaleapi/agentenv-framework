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
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from tst.unit.agents.opencode.agent import model_relay, opencode_bridge
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


class _Spawns(list):
    """Each spawn's argv and environment; ``clis`` has the CLI that answered each, with what it read on stdin."""

    def __init__(self) -> None:
        super().__init__()
        self.clis: list[FakeCLI] = []


@pytest.fixture
def spawned(monkeypatch):
    """Every spawn's argv and environment, each answered by a CLI that finishes its turn."""
    calls = _Spawns()

    async def _exec(*cmd, **kwargs):
        calls.append((list(cmd), kwargs["env"]))
        calls.clis.append(FakeCLI(json.dumps(_STOP).encode() + b"\n"))
        return calls.clis[-1]

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
    assert argv == ["opencode", "run", "--format", "json", "--model", "litellm/m", "--dir", "/w", "-s", "ses_0",
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


def test_the_system_prompt_reaches_the_model_as_instructions_and_the_output_format_follows_the_message(spawned):
    """Prepended to the message, a system prompt read as the user's own words, which a model can refuse as an
    injection."""
    schema = {"type": "object"}

    asyncio.run(opencode_bridge.run_opencode(
        prompt="go", context_id="ctx", system_prompt="be brief", output_format={"type": "json_schema", "schema": schema}))

    ((argv, env),) = spawned
    assert spawned.clis[0].stdin.data == (b'go\n\nReturn only a JSON object matching this schema, with no Markdown or '
                                          b'surrounding text: {"type":"object"}')
    (instructions,) = json.loads(env["OPENCODE_CONFIG_CONTENT"])["instructions"]
    assert Path(instructions) == opencode_bridge.context_data_dir("ctx") / "system-prompt.md"
    assert Path(instructions).read_text() == "be brief"


def test_a_turn_with_no_system_prompt_names_no_instructions(spawned):
    asyncio.run(opencode_bridge.run_opencode(prompt="go", context_id="ctx"))

    ((_argv, env),) = spawned
    assert "instructions" not in json.loads(env["OPENCODE_CONFIG_CONTENT"])


# opencode-baseline-claude on claude-haiku-4-5, recorded on 2026-11-21 through Scale's LiteLLM proxy: each streamed
# call's usage as the proxy answered it, the session title's first, then the turn's two steps.
CLAUDE_TURN = [
    {"prompt_tokens": 598, "completion_tokens": 6, "total_tokens": 604,
     "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
     "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    {"prompt_tokens": 9345, "completion_tokens": 95, "total_tokens": 9440,
     "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 9342},
     "cache_creation_input_tokens": 9342, "cache_read_input_tokens": 0},
    {"prompt_tokens": 9546, "completion_tokens": 4, "total_tokens": 9550,
     "prompt_tokens_details": {"cached_tokens": 9342, "cache_write_tokens": 197},
     "cache_creation_input_tokens": 197, "cache_read_input_tokens": 9342},
]


def test_a_turns_provider_calls_its_own_model_relay_and_its_usage_is_the_relays_count(monkeypatch, spawned):
    """The relay counts every call the CLI makes, the title's too: the recorded turn's equals the proxy's 19,489 in
    (9,342 read from the cache, 9,539 written) and 105 out."""
    relays = []

    @asynccontextmanager
    async def relay(upstream, calls, **kwargs):
        relays.append(upstream)
        for usage in CLAUDE_TURN:
            calls.add(200, model_relay.answer_usage(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode()))
        yield "http://127.0.0.1:41000/token"

    monkeypatch.setenv("LITELLM_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setattr(opencode_bridge, "model_relay", relay)

    result = asyncio.run(opencode_bridge.run_opencode(prompt="go", context_id="ctx", model="m"))

    ((_, env),) = spawned
    assert relays == ["https://gateway.example"]
    assert json.loads(env["OPENCODE_CONFIG_CONTENT"])["provider"]["litellm"]["options"]["baseURL"] == \
        "http://127.0.0.1:41000/token/v1"
    assert env["LITELLM_BASE_URL"] == "https://gateway.example/v1"
    assert result.usage.to_dict() == {
        "tool_call_count": 0, "input_tokens": 19489, "output_tokens": 105, "total_tokens": 19594,
        "cache_read_tokens": 9342, "cache_write_tokens": 9539, "input_includes_cache": True,
        "provider_details": {"model_calls": 3}}
