"""The rubrics family's named grades beyond what their legacy entries say: the settings a run grades with, the gate's
threshold, and the base fields each takes or refuses."""

from __future__ import annotations

import json

import pytest

from agent_env.config import get_config
from agent_env.task_step.context import TaskStepContext, VerifierNotPassingError
from agent_env.task_step.task_steps.verifiers.always_on_agent_verifier import AlwaysOnAgentVerifier
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from agent_env.task_step.task_steps.verifiers.user_sim_judge import UserSimJudge
from tst.unit.task_step.verifiers.test_rubrics_verifier_golden import (
    CASES,
    _binary,
    _context,
    _criteria,
    _direct_judge,
    _step,
)

OVERRIDE = [{"id": "only", "description": "the run's own criterion"}]


async def _grade(step, context: TaskStepContext) -> dict:
    try:
        await step.execute(context)
    except VerifierNotPassingError:
        pass
    return context.metadata["grades"][step.id]


@pytest.mark.asyncio
async def test_the_settings_are_what_the_run_graded_with_and_say_what_its_overrides_changed():
    step = _direct_judge(_step(criteria=_criteria(2)), lambda ids, n: _binary(ids))
    context = _context({"judge_model": "judge-x", "judge_effort": "high", "judge_batch_size": 1,
                        "apply_trajectory_filter": False, "judge_litellm_api_key": "sk-run-key",
                        "litellm_base_url": "http://llm.example.test", "step_params": {"verify": {"criteria": OVERRIDE}}})

    grade = await _grade(step, context)

    settings = grade["verifier_settings"]
    assert (settings["criteria"], settings["judge_model"], settings["effort"], settings["judge_batch_size"]) == (
        OVERRIDE, "judge-x", "high", 1)
    assert settings["trajectory_filter"] is None and settings["default_model"] == "judge-model"
    assert settings["overridden"] == ["criteria", "effort", "judge_batch_size", "judge_model"]
    assert "sk-run-key" not in json.dumps(grade) and "llm.example.test" not in json.dumps(grade)


@pytest.mark.asyncio
async def test_a_run_without_overrides_records_the_stored_settings_and_the_model_it_resolved():
    step, context = CASES["binary"]()

    settings = (await _grade(step, context))["verifier_settings"]

    assert settings["overridden"] == [] and settings["judge_model"] == "judge-model"
    assert settings["criteria"] == step.criteria and settings["score_aggregator"] == "all_pass"


@pytest.mark.asyncio
async def test_a_run_naming_the_steps_own_judge_model_overrides_nothing():
    step = _direct_judge(_step(criteria=_criteria(1)), lambda ids, n: _binary(ids))

    settings = (await _grade(step, _context({"judge_model": "judge-model"})))["verifier_settings"]

    assert settings["overridden"] == [] and settings["judge_model"] == "judge-model"


@pytest.mark.asyncio
async def test_an_agent_judge_that_takes_no_model_is_recorded_as_choosing_its_own():
    step, context = CASES["an_agent_judge_timing_out_once"]()

    settings = (await _grade(step, context))["verifier_settings"]

    assert settings["judge_model"] is None and settings["agent_name"] == "judge"


def test_a_judge_the_step_deploys_is_recorded_by_the_agent_and_sandbox_it_runs_on():
    step = _step(use_agent_judge=True)

    settings = step.verifier_settings(_context({"agent_sandbox": "modal"}))

    assert settings["judge_a2a_agent_id"] == get_config().get_default_a2a_agent_id()
    assert settings["judge_sandbox_type"] == "modal" and settings["overridden"] == ["judge_sandbox_type"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("min_passing_score", "raises"), [(None, True), (0.5, False), (0.75, True)])
async def test_the_gate_fails_a_score_under_the_steps_min_passing_score(min_passing_score, raises):
    step = _direct_judge(_step(score_aggregator="weighted_average", fail_task_when_not_passing=True,
                               min_passing_score=min_passing_score), lambda ids, n: _binary(ids, failing={"c2"}))
    context = _context()

    if raises:
        with pytest.raises(VerifierNotPassingError, match=r"isn't passing \(score 0.5\); rows not met: \['party-size'\]"):
            await step.execute(context)
    else:
        await step.execute(context)

    grade = context.metadata["grades"]["verify"]
    assert (grade["score"], grade["passed"]) == (0.5, not raises)
    assert context.metadata["verifications"]["vid"]["score"] == 0.5


def test_a_min_passing_score_is_stored_only_when_set_and_round_trips():
    assert "min_passing_score" not in _step().to_dict()
    stored = _step(min_passing_score=0.5).to_dict()
    assert stored["min_passing_score"] == 0.5
    again = RubricsVerifierTaskStep.from_dict(stored)
    assert (again.min_passing_score, again.verifier_id, again.score_aggregator.value) == (0.5, "vid", "all_pass")
    assert AlwaysOnAgentVerifier.from_dict({**stored, "type": AlwaysOnAgentVerifier.type}).min_passing_score == 0.5


def test_a_stored_rubrics_verifier_without_a_verifier_id_still_gets_one_of_its_own():
    stored = {k: v for k, v in _step().to_dict().items() if k != "verifier_id"}

    assert RubricsVerifierTaskStep.from_dict(stored).verifier_id not in ("verify", None)


def _user_sim_judge(**kw) -> UserSimJudge:
    return UserSimJudge.from_dict({"id": "judge_user_sim", "type": UserSimJudge.type, "run_step_id": "talk",
                                   "judge_model": "fake/judge", "verifier_id": "usj", **kw})


def test_the_user_sim_judge_takes_its_pinned_aggregator_and_refuses_a_min_passing_score():
    assert _user_sim_judge(score_aggregator="all_pass").score_aggregator.value == "all_pass"
    with pytest.raises(ValueError, match="min_passing_score can't be set on the user sim judge"):
        _user_sim_judge(min_passing_score=0.5)


def test_the_user_sim_judges_settings_are_only_the_overrides_it_applies():
    context = TaskStepContext(metadata={"user_overrides": {
        "judge_model": "judge-x", "judge_effort": "high", "apply_trajectory_filter": True, "judge_batch_size": 2,
        "judge_batch_concurrency": 3, "step_params": {"judge_user_sim": {"criteria": OVERRIDE}}}})

    settings = _user_sim_judge().verifier_settings(context)

    assert settings["judge_model"] == "judge-x" and settings["judge_batch_size"] == 2
    assert not {"criteria", "trajectory_filter", "effort", "judge_batch_concurrency"} & set(settings)
    assert settings["overridden"] == ["judge_batch_size", "judge_model"]


@pytest.mark.asyncio
async def test_an_always_on_verifier_refusing_a_runs_overrides_fails_its_grade_and_writes_no_entry():
    step = AlwaysOnAgentVerifier(id="judge", version=1, criteria=_criteria(1), prompt_id="ask", use_agent_judge=False,
                                 verifier_id="rubric")
    context = TaskStepContext(metadata={"user_overrides": {"step_params": {"judge": {"grade_up_to_turn": 1}}}})

    with pytest.raises(RuntimeError, match="grade_up_to_turn can't cut an always_on step's prompt"):
        await step.execute(context)

    assert context.metadata["grades"]["judge"]["verifier_status"] == "failed"
    assert context.metadata["grades"]["judge"]["verifier_settings"]["overridden"] == ["grade_up_to_turn"]
    assert "verifications" not in context.metadata
