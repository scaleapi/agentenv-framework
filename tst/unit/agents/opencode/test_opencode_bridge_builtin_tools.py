"""A selection of the CLI's own tools, as the config's ``permission``: ``"*"`` and ``only`` deny every tool but those
named and the tools of agent-env's MCP servers, a list to turn off denies those alone, and ``question`` is denied in
every case; with none, the config and argv are what they were. The card's agent-config takes the field, so deploy_agent
accepts a selection for opencode, and a tool the CLI doesn't have is refused. The selection's type has the protocol's
schema, and its tool names are those of the CLI version package.json pins."""

import asyncio
import json
from typing import Any

import pytest
from agentenv_protocol.a2a_agent import BuiltinTools

from agent_env.a2a_agent import A2AAgent
from agent_env.task_step.task_steps.deploy_agent import _check_builtin_tools
from tst.unit.agents.opencode.agent import AGENT_DIR, a2a_server, opencode_bridge
from tst.unit.agents.opencode.conftest import result
from tst.unit.agents.opencode.fake_cli import FakeCLI

MODEL = "fireworks_ai/kimi-k3"
_STOP = {"type": "step_finish", "sessionID": "ses_1", "part": {"type": "step-finish", "reason": "stop"}}
# What a "*" deny would otherwise deny that isn't a tool: the check a file tool makes before it touches a path outside
# the run directory, and the one made when the model repeats a call.
_CHECKS = {"external_directory": "allow", "doom_loop": "allow"}
_SERVERS = {"env4821": {"url": "https://gw/mcp", "headers": None},
            "peer_agents": {"url": "http://127.0.0.1:8000/ext/peer-agents/mcp", "headers": None},
            "fake-env.x": {"url": "https://gw2/mcp", "headers": None}}


def _permission(selection: dict | None, servers: dict | None = None) -> dict:
    tools = a2a_server.OpenCodeConfig(builtin_tools=selection).builtin_tools
    return opencode_bridge._build_config(MODEL, servers, {}, None, builtin_tools=tools)["permission"]


@pytest.fixture
def spawned(monkeypatch):
    """Every spawn's argv and environment, each answered by a CLI that finishes its turn."""
    calls: list[tuple[list[str], dict, FakeCLI]] = []

    async def _exec(*cmd, **kwargs):
        cli = FakeCLI(json.dumps(_STOP).encode() + b"\n")
        calls.append((list(cmd), kwargs["env"], cli))
        return cli

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)
    monkeypatch.setattr(opencode_bridge, "_context_xdg", lambda _context: {"XDG_DATA_HOME": "/ctx/share"})
    return calls


def test_no_selection_leaves_the_config_and_argv_as_they_were(monkeypatch, spawned):
    """The whole config and argv of a turn without a selection, as origin/main builds them, except that the prompt
    goes on stdin rather than as the positional message."""
    monkeypatch.setenv("LITELLM_BASE_URL", "https://gateway.example")
    monkeypatch.delenv("LITELLM_EXTRA_HEADERS", raising=False)

    asyncio.run(opencode_bridge.run_opencode(
        prompt="go", context_id="ctx", session_id="ses_0", model="m", mcp_servers={"env": {"url": "https://gw/mcp"}},
        effort="high", attachment_paths=["/w/a.png"], workspace="/w"))

    ((argv, env, cli),) = spawned
    assert argv == ["opencode", "run", "--format", "json", "--model", "litellm/m", "--dir", "/w", "-s", "ses_0",
                    "--file", "/w/a.png"]
    assert cli.stdin.data == b"go"
    assert env["OPENCODE_CONFIG_CONTENT"] == json.dumps({
        "$schema": "https://opencode.ai/config.json",
        "provider": {"litellm": {
            "npm": "@ai-sdk/openai-compatible", "name": "LiteLLM",
            "options": {"baseURL": "https://gateway.example/v1", "apiKey": "{env:LITELLM_API_KEY}"},
            "models": {"m": {"name": "m", "modalities": {"input": ["text", "image"], "output": ["text"]},
                             "options": {"additional_drop_params": ["tool_choice"], "reasoningEffort": "high"}}},
        }},
        "model": "litellm/m",
        "permission": {"*": "allow", "question": "deny"},
        "autoupdate": False,
        "mcp": {"env": {"type": "remote", "url": "https://gw/mcp", "enabled": True}},
    })


@pytest.mark.parametrize("selection, permission", [
    ({"disable": "*"}, {"*": "deny", **_CHECKS, "env4821_*": "allow", "peer_agents_*": "allow", "fake-env_x_*": "allow",
                        "question": "deny"}),
    ({"only": ["read", "grep"]}, {"*": "deny", **_CHECKS, "read": "allow", "grep": "allow", "env4821_*": "allow",
                                  "peer_agents_*": "allow", "fake-env_x_*": "allow", "question": "deny"}),
    ({"only": []}, {"*": "deny", **_CHECKS, "env4821_*": "allow", "peer_agents_*": "allow", "fake-env_x_*": "allow",
                    "question": "deny"}),
    ({"disable": ["bash", "webfetch"]}, {"*": "allow", "bash": "deny", "webfetch": "deny", "question": "deny"}),
])
def test_a_selection_is_the_permission_and_agent_envs_mcp_tools_stay(selection, permission):
    result = _permission(selection, _SERVERS)

    assert result == permission
    assert list(result)[-1] == "question"  # the CLI applies the last rule that matches a tool


def test_without_mcp_servers_disable_all_denies_every_tool():
    assert _permission({"disable": "*"}) == {"*": "deny", **_CHECKS, "question": "deny"}


def test_the_selection_reaches_the_cli(spawned):
    tools = a2a_server.OpenCodeConfig(builtin_tools={"disable": "*"}).builtin_tools

    asyncio.run(opencode_bridge.run_opencode(prompt="go", context_id="ctx", builtin_tools=tools))

    ((argv, env, cli),) = spawned
    assert json.loads(env["OPENCODE_CONFIG_CONTENT"])["permission"] == {"*": "deny", **_CHECKS, "question": "deny"}
    assert "go" not in argv
    assert cli.stdin.data == b"go"


def test_turns_get_the_selection_the_agent_config_set(monkeypatch, oc):
    """The agent passes the stored selection to each turn's run_opencode: none before one is set, then the one set,
    whose permission keeps the tools of the MCP servers registered with the agent."""
    monkeypatch.setenv("A2A_PORT", "8000")
    permissions = []

    async def fake_run_opencode(**kwargs):
        config = opencode_bridge._build_config(MODEL, kwargs["mcp_servers"], {}, None,
                                               builtin_tools=kwargs.get("builtin_tools"))
        permissions.append(config["permission"])
        return result()

    monkeypatch.setattr(a2a_server, "run_opencode", fake_run_opencode)

    async def exercise():
        async with oc.client() as client:
            await client.post("/ext/mcp-config", json={"name": "env4821", "url": "https://gw/mcp"})
            await oc.send(client, "one")
            assert (await client.post("/ext/agent-config", json={"builtin_tools": {"disable": "*"}})).status_code == 200
            await oc.send(client, "two")

    asyncio.run(exercise())

    assert permissions == [
        {"*": "allow", "question": "deny"},
        {"*": "deny", **_CHECKS, "env4821_*": "allow", "peer_agents_*": "allow", "question": "deny"},
    ]


@pytest.mark.parametrize("selection, match", [
    ({"only": ["write"]}, "write"),
    ({"disable": ["question"]}, "question"),
    ({"disable": "*", "only": ["read"]}, "exactly one"),
    ({}, "exactly one"),
    ({"disable": "all"}, "disable"),
])
def test_an_invalid_selection_is_refused(selection, match):
    with pytest.raises(ValueError, match=match):
        a2a_server.OpenCodeConfig(builtin_tools=selection)


def test_the_card_takes_builtin_tools_so_deploy_agent_accepts_a_selection(oc):
    extension = A2AAgent.find_extension(oc.card, A2AAgent.EXT_AGENT_CONFIG)

    assert "builtin_tools" in extension["params"]["methods"]["set"]["request"]["supported"]
    _check_builtin_tools(extension, "opencode", {"disable": "*"})
    _check_builtin_tools(extension, "opencode", {"only": ["read", "edit"]})
    with pytest.raises(RuntimeError, match="can't take builtin_tools"):
        _check_builtin_tools(extension, "opencode", {"disable": ["Bash"]})


def test_the_selection_has_the_protocols_schema():
    def untitled(schema: Any) -> Any:
        if isinstance(schema, dict):
            return {k: untitled(v) for k, v in schema.items() if k not in ("title", "description")}
        return [untitled(v) for v in schema] if isinstance(schema, list) else schema

    assert untitled(opencode_bridge.OpenCodeTools.model_json_schema()) == untitled(
        BuiltinTools[opencode_bridge.OpenCodeTool].model_json_schema())


def test_the_tool_names_are_those_of_the_cli_version_package_json_pins():
    pinned = json.loads((AGENT_DIR / "package.json").read_text())["dependencies"]["opencode-ai"]

    assert opencode_bridge.TOOLS_CLI_VERSION == pinned, (
        "the CLI version changed: record the tools the new CLI offers in OpenCodeTool, then TOOLS_CLI_VERSION")
