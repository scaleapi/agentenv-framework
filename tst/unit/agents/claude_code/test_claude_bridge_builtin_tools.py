"""A selection of the CLI's own tools: ``"*"`` and ``only`` name the tools the CLI may offer, a list to turn off leaves
the CLI's defaults but those, and with any selection its MCP servers are only agent-env's; with none, the CLI keeps
its own. A tool the CLI doesn't have is refused. The selection's type has the protocol's schema, and its tool names
are those of the CLI version the Dockerfile pins."""

import re
from typing import Any

import pytest
from agentenv_protocol.a2a_agent import BuiltinTools

from tst.unit.agents.claude_code.agent import AGENT_DIR, a2a_server, claude_bridge


def _command(selection: dict | None) -> list[str]:
    return claude_bridge._command(
        "sess-1", False, model=None, allowed_tools=None, mcp_servers=None, system_prompt=None, output_format=None,
        role=None, effort=None, max_turns=None,
        builtin_tools=a2a_server.ClaudeCodeConfig(builtin_tools=selection).builtin_tools)


def test_no_selection_leaves_the_clis_tools_and_mcp_servers_alone():
    command = _command(None)

    assert not {"--tools", "--disallowedTools", "--strict-mcp-config"} & set(command)


@pytest.mark.parametrize("selection, flags", [
    ({"disable": "*"}, ["--tools", ""]),
    ({"only": ["Read", "Agent"]}, ["--tools", "Read,Agent"]),
    ({"only": []}, ["--tools", ""]),
    ({"disable": ["WebFetch", "WebSearch"]}, ["--disallowedTools", "WebFetch,WebSearch"]),
])
def test_a_selection_is_the_clis_tool_flags_with_only_agent_envs_mcp_servers(selection, flags):
    command = _command(selection)
    at = command.index(flags[0])

    assert command[at:at + 2] == flags and "--strict-mcp-config" in command
    assert not ({"--tools", "--disallowedTools"} - {flags[0]}) & set(command)


def test_a_tool_the_cli_does_not_have_is_refused():
    with pytest.raises(ValueError, match="Glob"):
        a2a_server.ClaudeCodeConfig(builtin_tools={"only": ["Glob"]})


def test_the_selection_has_the_protocols_schema():
    def untitled(schema: Any) -> Any:
        if isinstance(schema, dict):
            return {k: untitled(v) for k, v in schema.items() if k not in ("title", "description")}
        return [untitled(v) for v in schema] if isinstance(schema, list) else schema

    assert untitled(claude_bridge.ClaudeCodeTools.model_json_schema()) == untitled(
        BuiltinTools[claude_bridge.ClaudeCodeTool].model_json_schema())


def test_the_tool_names_are_those_of_the_cli_version_the_dockerfile_pins():
    pinned = re.search(r"^ARG CLAUDE_CODE_VERSION=(\S+)$", (AGENT_DIR / "Dockerfile").read_text(), re.M).group(1)

    assert claude_bridge.TOOLS_CLI_VERSION == pinned, (
        "the CLI version changed: record the tools the new CLI offers in ClaudeCodeTool, then TOOLS_CLI_VERSION")
