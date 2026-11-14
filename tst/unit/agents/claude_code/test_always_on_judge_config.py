"""The always-on process judge's settings are ones the Claude Code agent takes, and do what the judge needs: its card
accepts builtin_tools, max_turns and peer_tools; the CLI offers only Bash and Read of its own tools (and the structured
output it adds), with no MCP server but the task's."""

from typing import get_args

from agent_env.task_step.task_steps.verifiers.always_on_agent_verifier import JUDGE_TOOLS, AlwaysOnAgentVerifier

from tst.unit.agents.claude_code.agent import a2a_server, claude_bridge
from tst.unit.agents.claude_code.support import card


def _judge_config() -> a2a_server.ClaudeCodeConfig:
    verifier = AlwaysOnAgentVerifier("judge", 1, criteria=[{"id": "c1"}], prompt_id="ask")
    return a2a_server.ClaudeCodeConfig(**verifier.judge_agent_config(), max_turns=verifier.judge_max_turns)


def test_the_agent_takes_the_judge_s_settings(app):
    [config] = [e for e in card(app)["capabilities"]["extensions"] if e["uri"] == "urn:agentenv:agent-config/v1"]
    supported = config["params"]["methods"]["set"]["request"]["supported"]
    assert {"builtin_tools", "max_turns", "peer_tools"} <= set(supported)
    assert set(JUDGE_TOOLS) <= set(get_args(claude_bridge.ClaudeCodeTool))
    assert _judge_config().max_turns == 400


def test_the_judge_s_cli_offers_bash_and_read_and_no_mcp_server_of_its_own():
    cfg = _judge_config()
    command = claude_bridge._command("sess", False, model=None, allowed_tools=None, mcp_servers=None,
                                     system_prompt=None, output_format=None, role=None, effort=None,
                                     max_turns=cfg.max_turns, builtin_tools=cfg.builtin_tools)
    assert command[command.index("--tools") + 1] == "Bash,Read"
    assert "--strict-mcp-config" in command and "--mcp-config" not in command
    assert command[command.index("--max-turns") + 1] == "400"

