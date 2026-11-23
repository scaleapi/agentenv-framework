"""An `ssk`-prefixed key points the agent at the deployment's inference endpoint when the step
chose no endpoint of its own. Explicit choices (step field, baked env_vars) always win, the URL
comes only from the environment (`SMIG_OPENAI_BASE_URL`), and resolution is per-agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps import deploy_agent as da
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep, _key_scoped_base_url

EP = "https://endpoint.example/openai/v1"


def test_key_scoped_base_url(monkeypatch):
    monkeypatch.delenv("SMIG_OPENAI_BASE_URL", raising=False)
    assert _key_scoped_base_url("ssk_is_abc") is None  # unset env -> no fallback, default proxy
    monkeypatch.setenv("SMIG_OPENAI_BASE_URL", EP)
    assert _key_scoped_base_url("ssk_is_abc") == EP
    assert _key_scoped_base_url("  ssk_is_abc  ") == EP  # stripped before the prefix check
    assert _key_scoped_base_url("sk-abc") is None
    assert _key_scoped_base_url("") is None
    assert _key_scoped_base_url(None) is None


def _endpoint(step: DeployAgentTaskStep, key) -> dict:
    ctx = TaskStepContext(metadata={"user_overrides": {"litellm_api_key": key} if key else {}})
    cfg = MagicMock()
    cfg.resolve_reference.side_effect = lambda v: v  # a plain key resolves to itself
    with patch.object(da, "get_config", return_value=cfg):
        return step._model_endpoint(ctx)


def test_ssk_key_falls_back_to_env_endpoint(monkeypatch):
    monkeypatch.setenv("SMIG_OPENAI_BASE_URL", EP)
    step = DeployAgentTaskStep(id="a", version=None, env_ids=["e"])
    assert _endpoint(step, "ssk_is_abc")["LITELLM_BASE_URL"] == EP


def test_explicit_step_base_wins_over_fallback(monkeypatch):
    monkeypatch.setenv("SMIG_OPENAI_BASE_URL", EP)
    step = DeployAgentTaskStep(
        id="a", version=None, env_ids=["e"], litellm_base_url="https://chosen.example/anthropic"
    )
    assert _endpoint(step, "ssk_is_abc")["LITELLM_BASE_URL"] == "https://chosen.example/anthropic"


def test_env_vars_baked_base_wins_over_fallback(monkeypatch):
    monkeypatch.setenv("SMIG_OPENAI_BASE_URL", EP)
    step = DeployAgentTaskStep(
        id="a", version=None, env_ids=["e"], env_vars={"LITELLM_BASE_URL": "https://baked.example"}
    )
    assert _endpoint(step, "ssk_is_abc")["LITELLM_BASE_URL"] == "https://baked.example"


def test_non_ssk_key_keeps_default_proxy(monkeypatch):
    monkeypatch.setenv("SMIG_OPENAI_BASE_URL", EP)
    step = DeployAgentTaskStep(id="a", version=None, env_ids=["e"])
    assert "LITELLM_BASE_URL" not in _endpoint(step, "sk-platform")


def test_ssk_key_keeps_default_proxy_when_env_unset(monkeypatch):
    monkeypatch.delenv("SMIG_OPENAI_BASE_URL", raising=False)
    step = DeployAgentTaskStep(id="a", version=None, env_ids=["e"])
    assert "LITELLM_BASE_URL" not in _endpoint(step, "ssk_is_abc")
