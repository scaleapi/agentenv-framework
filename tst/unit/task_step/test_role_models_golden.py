"""Each role's model at each call site that resolves one, across the configs and overrides in use: the solver's
default a deploy sets, the solver's model a prompt step asks for, the user simulator's at prompt_agent,
deploy_user_sim_agent and always_on, and the judge's, a model stored in a task included."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from agent_env.config import get_config, reset_config
from agent_env.config.model import ModelRole
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps import always_on as ao
from agent_env.task_step.task_steps.agent.deploy_user_sim_agent import DeployUserSimAgentTaskStep
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep

PERSON = json.loads((Path(__file__).parents[2] / "data" / "user_sim" / "hannah_person.json").read_text())
HANNAH = {"display_name": PERSON["name"], "first_name": PERSON["first"], "description": PERSON["description"],
          "permissions": PERSON["permissions"], "keep_private": PERSON["keep_private"]}
SDK_PROD = ('base_url = "https://litellm-proxy.example.com"\napi_key = "secret:litellm_api_key"\n\n'
            '[model.headers]\nx-litellm-tags = "useCase:agent-env"\n')
CONFIGS = {
    "none": "",
    "sdk_prod": SDK_PROD,
    "default_only": 'default = "D"\n',
    "every_role": 'default = "D"\n\n[model.roles]\nagent = "A"\nuser = "U"\njudge = "J"\n',
    "agent_role": 'default = "D"\n\n[model.roles]\nagent = "A"\n',
    "user_role": '[model.roles]\nuser = "U"\n',
    "judge_role": '[model.roles]\njudge = "J"\n',
}
SOLVER_DEFAULT = {"none": None, "sdk_prod": None, "default_only": "D", "every_role": "A", "agent_role": "A",
                  "user_role": None, "judge_role": None}
USER_CONFIG = {"none": None, "sdk_prod": None, "default_only": None, "every_role": "U", "agent_role": None,
               "user_role": "U", "judge_role": None}
JUDGE = {"none": "claude-sonnet-5-5", "sdk_prod": "claude-sonnet-5-5", "default_only": "D", "every_role": "J",
         "agent_role": "D", "user_role": "claude-sonnet-5-5", "judge_role": "J"}


@pytest.fixture
def model_config(tmp_path, monkeypatch):
    """Install a config file whose whole [model] table is the given text, so no developer config is read."""

    def install(model: str) -> None:
        path = tmp_path / "config.toml"
        path.write_text(f"[model]\n{model}")
        monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))
        reset_config()

    yield install
    reset_config()


def _ctx(step_params: dict | None = None, run: dict | None = None, **fields) -> TaskStepContext:
    overrides = dict(run or {})
    if step_params:
        overrides["step_params"] = step_params
    return TaskStepContext(metadata={"user_overrides": overrides} if overrides else {}, **fields)


def _first(*layers):
    return next((value for value in layers if value), None)


def _subsets(names: tuple[str, ...]):
    return [dict(zip(names, chosen)) for chosen in itertools.product((False, True), repeat=len(names))]


def _deploy_default() -> str | None:
    """The solver's default a deploy_agent step sets on the context."""
    return get_config().resolve_model(ModelRole.SOLVER).model


def _judge_model(verifier: RubricsVerifierTaskStep, context: TaskStepContext) -> str | None:
    """The model a rubrics judge asks for when it runs."""
    return verifier._judge_model(context)


@pytest.mark.parametrize("name", CONFIGS)
def test_the_solver_default_a_deploy_sets(model_config, name):
    model_config(CONFIGS[name])
    assert _deploy_default() == SOLVER_DEFAULT[name]


@pytest.mark.parametrize("layers", _subsets(("step_params", "run", "step", "deployed")), ids=str)
def test_the_solver_model_a_prompt_step_asks_for(layers):
    step = PromptAgentTaskStep(id="prompt", version=None, agent_name="solver", prompt="go",
                               model="step-model" if layers["step"] else None)
    context = _ctx({"prompt": {"model": "step-params-model"}} if layers["step_params"] else None,
                   agent_model="run-model" if layers["run"] else None,
                   default_agent_model="deployed-model" if layers["deployed"] else None)
    expected = _first(layers["step_params"] and "step-params-model", layers["run"] and "run-model",
                      layers["step"] and "step-model", layers["deployed"] and "deployed-model")
    assert step._resolve_model_and_effort(context)[0] == expected


@pytest.mark.parametrize("name", CONFIGS)
@pytest.mark.parametrize("layers", _subsets(("step_params", "run", "step")), ids=str)
def test_the_user_simulator_model_a_prompt_step_asks_for(model_config, name, layers):
    model_config(CONFIGS[name])
    step = PromptAgentTaskStep(id="prompt", version=None, agent_name="solver", prompt="go",
                               user_model="step-user" if layers["step"] else None)
    context = _ctx({"prompt": {"user_model": "step-params-user"}} if layers["step_params"] else None,
                   {"user_model": "run-user", "agent_model": "run-solver"} if layers["run"] else None,
                   agent_model="run-solver")
    expected = _first(layers["step_params"] and "step-params-user", layers["run"] and "run-user",
                      layers["step"] and "step-user", USER_CONFIG[name])
    assert step._resolve_user_model(context) == expected


USER_SIM_CASES = [
    ({}, ("sim-model", "user_sim_step")),
    ({"run": {"user_model": "run-model"}}, ("run-model", "run_user_model")),
    ({"step_params": {"user": {"model": "own-model"}}}, ("own-model", "user_sim_step_params")),
    ({"run": {"user_model": "run-model"}, "step_params": {"user": {"model": "own-model"}}},
     ("own-model", "user_sim_step_params")),
]


@pytest.mark.parametrize("name", CONFIGS)
@pytest.mark.parametrize("overrides, expected", USER_SIM_CASES, ids=["step", "run", "own", "own_over_run"])
def test_the_user_simulator_model_its_deploy_sets_ignores_model_roles(model_config, name, overrides, expected):
    model_config(CONFIGS[name])
    step = DeployUserSimAgentTaskStep.from_dict({
        "id": "user", "type": "deploy_user_sim_agent", "a2a_agent_id": "sim-image", "agent_name": "hannah",
        "model": "sim-model", "multi_day": True, "opening_prompt": PERSON["opening"]["text"], "persona": HANNAH})
    assert step._user_model(_ctx(overrides.get("step_params"), overrides.get("run"))) == expected


@pytest.mark.parametrize("name", CONFIGS)
@pytest.mark.parametrize("overrides, expected", USER_SIM_CASES, ids=["step", "run", "own", "own_over_run"])
def test_the_user_simulator_model_always_on_sets_ignores_model_roles(model_config, name, overrides, expected):
    model_config(CONFIGS[name])
    run = object.__new__(ao._Run)
    run.ctx = _ctx(overrides.get("step_params"), overrides.get("run"))
    run.us = {"step_id": "user", "model": "sim-model"}
    assert run._user_model() == expected


def _judge(data_model: str | None = None) -> RubricsVerifierTaskStep:
    data = RubricsVerifierTaskStep(id="judge", version=None, criteria=[{"id": "c1", "description": "says hi"}],
                                   prompt_id="prompt").to_dict()
    data["default_model"] = data_model
    return RubricsVerifierTaskStep.from_dict(data)


@pytest.mark.parametrize("name", CONFIGS)
@pytest.mark.parametrize("layers", _subsets(("run", "step")), ids=str)
def test_the_judge_model(model_config, name, layers):
    model_config(CONFIGS[name])
    verifier = _judge("step-judge" if layers["step"] else None)
    context = _ctx(run={"judge_model": "run-judge"} if layers["run"] else None)
    expected = _first(layers["run"] and "run-judge", layers["step"] and "step-judge", JUDGE[name])
    assert _judge_model(verifier, context) == expected


def test_a_judge_model_stored_in_a_task_wins_over_the_reader_s_config(model_config):
    model_config(CONFIGS["judge_role"])
    stored = RubricsVerifierTaskStep(id="judge", version=None, criteria=[{"id": "c1", "description": "says hi"}],
                                     prompt_id="prompt", default_model="writer-judge").to_dict()
    model_config(CONFIGS["every_role"])
    assert _judge_model(RubricsVerifierTaskStep.from_dict(stored), _ctx()) == "writer-judge"
