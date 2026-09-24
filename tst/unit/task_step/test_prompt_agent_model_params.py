"""Unit tests for the prompt_agent step's model_params override field."""

import pytest

from agent_env.store import LocalSecretStore
from agent_env.config import Config
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep


def _step(model_params):
    return PromptAgentTaskStep.from_dict({"id": "s", "version": 1, "prompt": "hi", "model_params": model_params})


def _cfg(tmp_path, monkeypatch, *, secrets):
    """A fresh AWS-free Config with an injected secret store (mirrors test_config_model)."""
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    cfg = Config()
    cfg.set_secret_store(LocalSecretStore(values=secrets, use_env=False))
    return cfg


def test_model_params_round_trips():
    step = _step({"aws_region_name": "us-west-2"})
    assert step.model_params == {"aws_region_name": "us-west-2"}
    assert step.to_dict()["model_params"] == {"aws_region_name": "us-west-2"}


def test_model_params_defaults_to_none():
    step = PromptAgentTaskStep.from_dict({"id": "s", "version": 1, "prompt": "hi"})
    assert step.model_params is None


def test_model_params_rejects_reserved_keys():
    with pytest.raises(ValueError, match="reserved"):
        _step({"api_key": "x"})


def test_step_model_params_secret_ref_resolves_through_get_model_params(tmp_path, monkeypatch):
    """The step's model_params secret:/env: refs resolve at run time via the exact
    call prompt_agent makes: get_config().get_model_params(self.model_params)."""
    monkeypatch.setenv("E2E_REGION", "us-west-2")
    cfg = _cfg(tmp_path, monkeypatch, secrets={"aws_sk": "resolved-sk"})
    step = _step({"aws_region_name": "env:E2E_REGION", "aws_secret_access_key": "secret:aws_sk"})
    resolved = cfg.get_model_params(step.model_params)
    assert resolved == {"aws_region_name": "us-west-2", "aws_secret_access_key": "resolved-sk"}


def test_step_model_params_persists_raw_ref_not_resolved_secret(tmp_path, monkeypatch):
    """The Task doc stores the raw secret: ref, never the resolved value (no secret at rest)."""
    step = _step({"aws_secret_access_key": "secret:aws_sk"})
    assert step.to_dict()["model_params"] == {"aws_secret_access_key": "secret:aws_sk"}
