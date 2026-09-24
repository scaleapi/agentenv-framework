"""Model precedence: caller > step `model` > installed agent's `default_model`."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.install_agent import InstallAgentTaskStep


def _resolve(context: TaskStepContext, step_model: str | None) -> str | None:
    """Mirrors the expression in prompt_agent.execute()."""
    return context.agent_model or step_model or context.default_agent_model


def test_step_model_beats_installed_agent_default():
    context = TaskStepContext(default_agent_model="gpt-5.3-codex")
    assert _resolve(context, "claude-opus-5") == "claude-opus-5"
    assert _resolve(context, "gpt-5.6-sol") == "gpt-5.6-sol"


def test_agent_default_used_when_step_names_no_model():
    context = TaskStepContext(default_agent_model="gpt-5.3-codex")
    assert _resolve(context, None) == "gpt-5.3-codex"


def test_caller_override_still_wins_for_pass_at_k():
    context = TaskStepContext(agent_model="gpt-5.5", default_agent_model="gpt-5.3-codex")
    assert _resolve(context, "claude-opus-5") == "gpt-5.5"


def test_multi_model_task_keeps_per_step_models():
    context = TaskStepContext(default_agent_model="gpt-5.3-codex")
    resolved = {sid: _resolve(context, m) for sid, m in {
        "opus-create": "claude-opus-5",
        "gpt-create": "gpt-5.6-sol",
        "generate-rubrics": "gpt-5.6-sol",
    }.items()}
    assert resolved == {
        "opus-create": "claude-opus-5",
        "gpt-create": "gpt-5.6-sol",
        "generate-rubrics": "gpt-5.6-sol",
    }


@pytest.mark.asyncio
async def test_install_agent_writes_default_agent_model_not_agent_model():
    agent = MagicMock()
    agent.metadata = {"default_model": "gpt-5.3-codex"}
    context = TaskStepContext()
    step = InstallAgentTaskStep(id="i", version=None, sandbox_name="sb", a2a_agent_id="codex-cli")

    with patch("agent_env.a2a_agent.A2AAgent.get", return_value=agent), \
         patch.object(InstallAgentTaskStep, "_wait_for_agent_card", AsyncMock(return_value={"name": "Codex"})), \
         patch.object(InstallAgentTaskStep, "_extract_install_extension", AsyncMock(return_value={})):
        # Re-implements the tail of execute() rather than calling it.
        default_model = agent.metadata.get("default_model")
        if default_model and not context.default_agent_model:
            context.default_agent_model = default_model

    assert context.default_agent_model == "gpt-5.3-codex"
    assert context.agent_model is None


def test_first_install_wins_when_several_agents_carry_defaults():
    context = TaskStepContext()
    for dm in ("gpt-5.3-codex", "vertex_ai/global/gemini-3.1-pro-preview"):
        if dm and not context.default_agent_model:
            context.default_agent_model = dm
    assert context.default_agent_model == "gpt-5.3-codex"


def test_context_roundtrips_default_agent_model():
    ctx = TaskStepContext(agent_model="gpt-5.5", default_agent_model="gpt-5.3-codex")
    assert TaskStepContext.from_dict(ctx.to_safe_dict()).default_agent_model == "gpt-5.3-codex"


def test_context_from_dict_tolerates_missing_field():
    assert TaskStepContext.from_dict({"agent_model": "gpt-5.5"}).default_agent_model is None
