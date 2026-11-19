"""A rubrics judge resolves its model when it runs: the run's ``judge_model``, the model its author named, then the
config's ``judge`` role and default, and claude-sonnet-4-6. A verifier built in code, as the A2A validator builds its
own, judges like one read from a task, and a stored task keeps only its author's model, never its writer's config."""

import pytest

from agent_env.config import reset_config
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep

CRITERIA = [{"id": "c1", "description": "Response says hello"}]


@pytest.fixture
def model_config(tmp_path, monkeypatch):
    """Install a config file holding ``model`` as its whole [model] table, so no developer config is read."""

    def install(model: str) -> None:
        config = tmp_path / "config.toml"
        config.write_text(f"[model]\n{model}")
        monkeypatch.setenv("AGENT_ENV_CONFIG", str(config))
        reset_config()

    yield install
    reset_config()


@pytest.fixture
def judge_role(model_config):
    model_config('default = "agent-model"\n\n[model.roles]\njudge = "judge-model"\n')


def _built(**kwargs) -> RubricsVerifierTaskStep:
    return RubricsVerifierTaskStep(id="verify", version=None, criteria=CRITERIA, prompt_id="p1", **kwargs)


def _run(**overrides) -> TaskStepContext:
    return TaskStepContext(metadata={"user_overrides": overrides} if overrides else {})


def test_a_verifier_built_in_code_judges_with_the_configured_judge_model(judge_role):
    assert _built()._judge_model(_run()) == "judge-model"


def test_a_verifier_read_from_a_task_judges_with_the_configured_judge_model(judge_role):
    data = _built().to_dict()
    del data["default_model"]

    assert RubricsVerifierTaskStep.from_dict(data)._judge_model(_run()) == "judge-model"


def test_a_model_the_step_names_wins_over_the_judge_role(judge_role):
    assert _built(default_model="named-model")._judge_model(_run()) == "named-model"


def test_the_run_s_judge_model_wins_over_the_step_s(judge_role):
    assert _built(default_model="named-model")._judge_model(_run(judge_model="run-model")) == "run-model"


def test_without_a_judge_role_the_judge_is_the_configured_default_model(model_config):
    model_config('default = "agent-model"\n')

    assert _built()._judge_model(_run()) == "agent-model"


def test_with_no_model_configured_the_judge_is_claude_sonnet(model_config):
    model_config("")

    assert _built()._judge_model(_run()) == "claude-sonnet-4-6"


def test_a_stored_task_carries_no_model_from_its_writer_s_config(model_config, judge_role):
    stored = _built().to_dict()
    assert stored["default_model"] is None

    model_config('[model.roles]\njudge = "reader-judge"\n')
    assert RubricsVerifierTaskStep.from_dict(stored)._judge_model(_run()) == "reader-judge"


def test_a_model_frozen_into_a_task_stored_before_keeps_judging(judge_role):
    stored = {**_built().to_dict(), "default_model": "frozen-model"}

    assert RubricsVerifierTaskStep.from_dict(stored)._judge_model(_run()) == "frozen-model"
