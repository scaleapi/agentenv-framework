"""A selection of the CLI's own tools, as `-c` overrides on the turn's CLI: ``"*"`` turns off every setting that brings
one, the `multi_agent_v2` collaboration tools included, ``only`` every setting but those it keeps, and a list to turn off
those it names; config.toml and the MCP servers agent-env gives the agent stay as they are. With no selection, the CLI
gets the overrides it got before. The card's agent-config takes the field, so deploy_agent accepts a selection for
codex, and a tool the CLI doesn't have, or part of a setting's tools, is refused. The selection's type has the
protocol's schema, and its tool names are those of the CLI version package.json pins."""

import asyncio
import json
import os
import tomllib
from pathlib import Path
from typing import Any

import pytest
from agentenv_protocol.a2a_agent import BuiltinTools

from agent_env.a2a_agent import A2AAgent
from agent_env.task_step.task_steps.deploy_agent import _check_builtin_tools
from tst.unit.agents.codex.agent import AGENT_DIR, a2a_server, codex_bridge, codex_config

_ALL_OFF = [
    "features.shell_tool=false",
    "tools.update_plan.enabled=false",
    "tools.experimental_request_user_input.enabled=false",
    "features.view_image=false",
    "features.multi_agent_v2=false",
    "features.multi_agent=false",
    "features.goals=false",
    'web_search="disabled"',
]


def _overrides(selection: dict | None) -> list[str]:
    return codex_config.builtin_tools_overrides(a2a_server.CodexConfig(builtin_tools=selection).builtin_tools)


@pytest.mark.parametrize("selection, overrides", [
    (None, []),
    ({"disable": "*"}, _ALL_OFF),
    ({"only": []}, _ALL_OFF),
    ({"disable": []}, []),
    ({"disable": ["web_search", "view_image"]}, ["features.view_image=false", 'web_search="disabled"']),
    ({"disable": ["spawn_agent", "send_message", "wait_agent", "followup_task", "interrupt_agent", "list_agents"]},
     ["features.multi_agent_v2=false", "features.multi_agent=false"]),
    ({"only": ["exec_command", "write_stdin", "update_plan"]},
     [o for o in _ALL_OFF if o not in ("features.shell_tool=false", "tools.update_plan.enabled=false")]),
])
def test_a_selection_is_the_overrides_that_turn_tools_off(selection, overrides):
    assert _overrides(selection) == overrides


def test_turns_get_the_selections_overrides_after_their_role_and_config_toml_stays(codex, monkeypatch):
    """No selection, with no role, is no override at all, as before; a selection adds its overrides after the role's,
    and config.toml, whose MCP servers give the agent agent-env's tools, is the same either way."""
    codex_home = Path(os.environ["CODEX_HOME"])
    monkeypatch.setenv("A2A_PORT", "8000")
    turns = []

    async def fake_run_codex(**kwargs):
        turns.append((tomllib.loads((codex_home / "config.toml").read_text()), list(kwargs["config_overrides"])))
        return codex_bridge.CodexResult(response="ok", tool_call_count=0, session_id="t")

    monkeypatch.setattr(a2a_server, "run_codex", fake_run_codex)

    async def exercise():
        async with codex.app.router.lifespan_context(codex.app), codex.client() as client:
            await client.post("/ext/mcp-config", json={"url": "https://gw/mcp", "name": "env4821"})
            await codex.send(client, "one")
            assert (await client.post("/ext/agent-config", json={"builtin_tools": {"disable": "*"}})).status_code == 200
            await codex.send(client, "two")
            await client.post("/ext/agent-config", json={"role": "solver", "builtin_tools": {"disable": ["web_search"]}})
            await codex.send(client, "three")

    asyncio.run(exercise())

    (before, none), (after, all_off), (_, solver) = turns
    assert none == []
    assert all_off == _ALL_OFF
    assert solver == [*codex_config.role_overrides({"env4821": {}, "peer_agents": {}}, "solver"),
                      'web_search="disabled"']
    assert before == after and before["features"] == {"multi_agent_v2": True}
    assert set(before["mcp_servers"]) == {"env4821", "peer_agents"}


def test_the_overrides_reach_the_cli_argv(monkeypatch, tmp_path):
    """run_codex appends each override as its own `-c`, after the turn's own settings."""
    seen = []

    class _Stop(Exception):
        pass

    async def _exec(*cmd, **kwargs):
        seen.append(list(cmd))
        raise _Stop

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)
    with pytest.raises(_Stop):
        asyncio.run(codex_bridge._run_codex_once(prompt="go", thread_id=None, config_overrides=_overrides(
            {"disable": "*"})))

    (argv,) = seen
    assert argv[argv.index("--dangerously-bypass-approvals-and-sandbox") + 1:] == [
        arg for override in _ALL_OFF for arg in ("-c", override)]


@pytest.mark.parametrize("selection, match", [
    ({"only": ["apply_patch"]}, "apply_patch"),
    ({"disable": ["tool_search"]}, "tool_search"),
    ({"disable": ["exec_command"]}, "on and off together"),
    ({"only": ["get_goal"]}, "on and off together"),
    ({"disable": "*", "only": ["update_plan"]}, "exactly one"),
    ({}, "exactly one"),
])
def test_an_invalid_selection_is_refused(selection, match):
    with pytest.raises(ValueError, match=match):
        a2a_server.CodexConfig(builtin_tools=selection)


def test_the_card_takes_builtin_tools_so_deploy_agent_accepts_a_selection(codex):
    extension = A2AAgent.find_extension(codex.card, A2AAgent.EXT_AGENT_CONFIG)

    assert "builtin_tools" in extension["params"]["methods"]["set"]["request"]["supported"]
    _check_builtin_tools(extension, "codex", {"disable": "*"})
    _check_builtin_tools(extension, "codex", {"only": ["update_plan"]})
    with pytest.raises(RuntimeError, match="can't take builtin_tools"):
        _check_builtin_tools(extension, "codex", {"disable": ["shell"]})


def test_the_selection_has_the_protocols_schema():
    def untitled(schema: Any) -> Any:
        if isinstance(schema, dict):
            return {k: untitled(v) for k, v in schema.items() if k not in ("title", "description")}
        return [untitled(v) for v in schema] if isinstance(schema, list) else schema

    assert untitled(codex_config.CodexTools.model_json_schema()) == untitled(
        BuiltinTools[codex_config.CodexTool].model_json_schema())


def test_every_tool_is_turned_off_by_one_setting():
    switched = [tool for tools, _ in codex_config._TOOL_SWITCHES for tool in tools]

    assert sorted(switched) == sorted(codex_config.CodexTool.__args__)


def test_the_tool_names_are_those_of_the_cli_version_package_json_pins():
    pinned = json.loads((AGENT_DIR / "package.json").read_text())["dependencies"]["@openai/codex"]

    assert codex_config.TOOLS_CLI_VERSION == pinned, (
        "the CLI version changed: record the tools the new CLI offers in CodexTool, then TOOLS_CLI_VERSION")
