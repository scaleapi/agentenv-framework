"""A selection of the CLI's own tools: ``"*"`` and ``only`` name the tools the CLI may offer, a list to turn off leaves
the CLI's defaults but those, and with any selection its MCP servers are only agent-env's; with none, the CLI keeps
its own. A tool the CLI doesn't have is refused."""

import pytest

from tst.unit.agents.claude_code.agent import a2a_server, claude_bridge


def _command(selection: dict | None) -> list[str]:
    config = a2a_server.ClaudeCodeConfig(builtin_tools=selection)
    return claude_bridge._command(
        "sess-1", False, model=None, allowed_tools=None, mcp_servers=None, system_prompt=None, output_format=None,
        role=None, effort=None, max_turns=None,
        builtin_tools=None if config.builtin_tools is None else config.builtin_tools.model_dump(exclude_none=True))


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
