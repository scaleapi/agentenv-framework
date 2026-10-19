"""Unit tests for per-run ``step_overrides`` of the solver on DeployAgentTaskStep and PromptAgentTaskStep.

One task version can then run under a different A2A agent, model or effort per run while the other
agents in the task (user simulators, judges) keep their stored settings.
"""

from __future__ import annotations

import pytest

from agent_env.config.model import ModelConfig
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps import prompt_agent
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep


def _ctx(step_params: dict | None = None, run_overrides: dict | None = None, **fields) -> TaskStepContext:
    user_overrides = dict(run_overrides or {})
    if step_params is not None:
        user_overrides["step_params"] = step_params
    metadata = {"user_overrides": user_overrides} if user_overrides else {}
    return TaskStepContext(metadata=metadata, **fields)


def _deploy_step(**kw) -> DeployAgentTaskStep:
    kw.setdefault("id", "deploy-solver")
    kw.setdefault("version", None)
    kw.setdefault("agent_name", "solver")
    kw.setdefault("a2a_agent_id", "stored-agent")
    kw.setdefault("a2a_agent_version", 7)
    return DeployAgentTaskStep(**kw)


def _prompt_step(**kw) -> PromptAgentTaskStep:
    kw.setdefault("id", "prompt")
    kw.setdefault("version", None)
    kw.setdefault("agent_name", "solver")
    kw.setdefault("prompt", "go")
    kw.setdefault("model", "stored-model")
    kw.setdefault("effort", "stored-effort")
    return PromptAgentTaskStep(**kw)


class TestDeployAgentRef:
    def test_stored_values_without_overrides(self):
        assert _deploy_step()._resolve_agent_ref(_ctx()) == ("stored-agent", 7)

    def test_step_override_swaps_agent_and_drops_stored_pin(self):
        ctx = _ctx({"deploy-solver": {"a2a_agent_id": "other-agent"}})
        assert _deploy_step()._resolve_agent_ref(ctx) == ("other-agent", None)

    def test_step_override_with_version(self):
        ctx = _ctx({"deploy-solver": {"a2a_agent_id": "other-agent", "a2a_agent_version": 3}})
        assert _deploy_step()._resolve_agent_ref(ctx) == ("other-agent", 3)

    def test_step_override_version_only_keeps_agent(self):
        ctx = _ctx({"deploy-solver": {"a2a_agent_version": 9}})
        assert _deploy_step()._resolve_agent_ref(ctx) == ("stored-agent", 9)

    def test_step_override_wins_over_run_wide_override(self):
        ctx = _ctx({"deploy-solver": {"a2a_agent_id": "other-agent"}}, {"a2a_agent_id": "run-wide", "a2a_agent_version": 2})
        assert _deploy_step()._resolve_agent_ref(ctx) == ("other-agent", None)

    def test_run_wide_override_still_applies(self):
        ctx = _ctx(run_overrides={"a2a_agent_id": "run-wide", "a2a_agent_version": 2})
        assert _deploy_step()._resolve_agent_ref(ctx) == ("run-wide", 2)

    def test_other_steps_overrides_are_ignored(self):
        ctx = _ctx({"deploy-judge": {"a2a_agent_id": "other-agent"}})
        assert _deploy_step()._resolve_agent_ref(ctx) == ("stored-agent", 7)


class TestPromptModelAndEffort:
    def test_stored_values_without_overrides(self):
        assert _prompt_step()._resolve_model_and_effort(_ctx()) == ("stored-model", "stored-effort")

    def test_step_override_wins_over_run_wide_and_stored(self):
        ctx = _ctx({"prompt": {"model": "step-model", "effort": "xhigh"}}, {"agent_effort": "low"}, agent_model="run-model")
        assert _prompt_step()._resolve_model_and_effort(ctx) == ("step-model", "xhigh")

    def test_run_wide_overrides_still_apply(self):
        ctx = _ctx(run_overrides={"agent_effort": "low"}, agent_model="run-model")
        assert _prompt_step()._resolve_model_and_effort(ctx) == ("run-model", "low")

    def test_partial_step_override_leaves_the_other_field(self):
        ctx = _ctx({"prompt": {"model": "step-model"}}, agent_model="run-model")
        assert _prompt_step()._resolve_model_and_effort(ctx) == ("step-model", "stored-effort")

    def test_default_model_is_the_last_resort(self):
        ctx = _ctx(default_agent_model="fallback")
        assert _prompt_step(model=None)._resolve_model_and_effort(ctx) == ("fallback", "stored-effort")


class TestUserSimModel:
    """The user simulator's model: a step override, then the run's ``user_model``, the stored value, and the config's
    ``user`` role only when it names one."""

    @pytest.fixture(autouse=True)
    def config(self, monkeypatch):
        roles = ModelConfig(default="config-default", roles={})
        monkeypatch.setattr(prompt_agent, "get_config", lambda: type("C", (), {
            "get_model_for_role": staticmethod(roles.model_for_role)})())
        return roles

    def test_stored_value_without_overrides(self):
        assert _prompt_step(user_model="stored-user")._resolve_user_model(_ctx()) == "stored-user"

    def test_run_override_wins_over_stored(self):
        ctx = _ctx(run_overrides={"user_model": "run-user"})
        assert _prompt_step(user_model="stored-user")._resolve_user_model(ctx) == "run-user"

    def test_step_override_wins_over_run_override(self):
        ctx = _ctx({"prompt": {"user_model": "step-user"}}, {"user_model": "run-user"})
        assert _prompt_step(user_model="stored-user")._resolve_user_model(ctx) == "step-user"

    def test_the_config_user_role_fills_in_and_its_default_doesnt(self, config):
        assert _prompt_step()._resolve_user_model(_ctx()) is None
        config.roles["user"] = "config-user"
        assert _prompt_step()._resolve_user_model(_ctx()) == "config-user"

    def test_the_run_wide_solver_model_leaves_the_user_simulator_alone(self):
        assert _prompt_step(user_model="stored-user")._resolve_user_model(_ctx(agent_model="run-solver")) == "stored-user"


def test_each_step_records_the_model_it_asks_for_by_role():
    context = TaskStepContext()
    context.record_model("solver", "prompt", "m")
    context.record_model("judge", "judge", None)
    context.record_model("judge", "judge-b", "j")

    assert context.metadata["models"] == {"solver": {"prompt": "m"}, "judge": {"judge": None, "judge-b": "j"}}
