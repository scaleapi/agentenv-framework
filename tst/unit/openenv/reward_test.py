"""``reward_of``: which recorded score is a task's reward. A step whose class says its score can't be the reward (it
checks the run, as a user sim judge does) is never picked and is refused as ``reward=``."""

from typing import ClassVar

from agent_env.openenv.reward import reward_of
from agent_env.task import Task
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep


class _RunCheck(RubricsVerifierTaskStep):
    can_be_reward: ClassVar[bool] = False


def _rubric(cls: type[RubricsVerifierTaskStep] = RubricsVerifierTaskStep, *, id: str) -> RubricsVerifierTaskStep:
    return cls(id=id, version=None, criteria=[], prompt_id="p1", use_agent_judge=False, verifier_id=f"{id}-v")


def _task(*steps: RubricsVerifierTaskStep) -> Task:
    return Task(id="t", version=None, steps=list(steps))


def test_a_rubric_beside_a_run_check_is_the_reward_without_naming_it():
    of, why = reward_of(_task(_rubric(id="grade"), _rubric(_RunCheck, id="user-sim")), None)

    assert why is None
    assert of({"grade": 0.75, "user-sim": 0.0}) == 0.75


def test_a_task_whose_only_score_is_a_run_check_is_unscored():
    of, why = reward_of(_task(_rubric(_RunCheck, id="user-sim")), None)

    assert why is None
    assert of({"user-sim": 1.0}) is None


def test_a_run_check_named_as_the_reward_is_refused():
    of, why = reward_of(_task(_rubric(id="grade"), _rubric(_RunCheck, id="user-sim")), "user-sim")

    assert of is None
    assert why == ("task 't''s step 'user-sim' (rubrics_verifier) records a score that checks the run and can't be "
                   "its reward; the scores that can are at ['grade']")


def test_two_rubrics_still_need_one_named():
    task = _task(_rubric(id="grade"), _rubric(id="again"), _rubric(_RunCheck, id="user-sim"))

    assert reward_of(task, None) == (None, "task 't' records a score at each of grade, again; name the one that is "
                                           "its reward with reward=<step id>")
    of, why = reward_of(task, "again")
    assert why is None and of({"grade": 0.0, "again": 1.0, "user-sim": 0.0}) == 1.0


def test_a_step_that_records_no_score_is_refused_as_before():
    assert reward_of(_task(_rubric(id="grade")), "nowhere") == (
        None, "task 't' records no score at step 'nowhere'; it records them at ['grade']")
