"""The model the CLI is given: a native-1M model passed without [1m] gets its 1M window."""

import pytest

from tst.unit.agents.claude_code.agent import claude_bridge


def _cli_model(model: str | None) -> str | None:
    cmd = claude_bridge._command("sess-1", False, model=model, allowed_tools=None, mcp_servers=None,
                                 system_prompt=None, output_format=None, role=None, effort=None, max_turns=None)
    return cmd[cmd.index("--model") + 1] if "--model" in cmd else None


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-opus-4-8",
                                   "claude-opus-4-7"])
def test_a_native_1m_model_gets_its_1m_window(model):
    assert _cli_model(model) == f"{model}[1m]"


@pytest.mark.parametrize("model", ["claude-opus-5-5[1m]", "claude-opus-4-8[1m]", "claude-sonnet-4-6", "gpt-5.6-sol",
                                   "gemini/gemini-3.1-pro-preview"])
def test_other_models_pass_through_unchanged(model):
    assert _cli_model(model) == model


def test_no_model_sends_no_model_flag():
    assert _cli_model(None) is None
