"""A selection of the CLI's own tools: the CLI is offered only the tools it leaves on, by the names its card lists,
and only the MCP servers agent-env gives it; with none, the CLI keeps its own."""

from typing import get_args

import pytest

from tst.unit.agents.claude_code.agent import a2a_server, claude_bridge

TOOLS = get_args(a2a_server.ClaudeCodeTool)


def _command(selection: dict | None) -> list[str]:
    config = a2a_server.ClaudeCodeConfig(builtin_tools=selection)
    return claude_bridge._command(
        "sess-1", False, model=None, allowed_tools=None, mcp_servers=None, system_prompt=None, output_format=None,
        role=None, effort=None, max_turns=None,
        builtin_tools=None if config.builtin_tools is None else config.builtin_tools.enabled(TOOLS))


def test_no_selection_leaves_the_clis_tools_and_mcp_servers_alone():
    command = _command(None)

    assert "--tools" not in command and "--strict-mcp-config" not in command


@pytest.mark.parametrize("selection, offered", [
    ({"disable": "*"}, []),
    ({"disable": ["WebFetch", "WebSearch"]}, [tool for tool in TOOLS if tool not in ("WebFetch", "WebSearch")]),
    ({"only": ["Read", "Agent"]}, ["Agent", "Read"]),
])
def test_a_selection_is_the_clis_tool_list_with_only_agent_envs_mcp_servers(selection, offered):
    command = _command(selection)

    assert command[command.index("--tools") + 1] == ",".join(offered) and "--strict-mcp-config" in command


def test_a_tool_the_cli_does_not_have_is_refused():
    with pytest.raises(ValueError, match="Glob"):
        a2a_server.ClaudeCodeConfig(builtin_tools={"only": ["Glob"]})
