"""The criteria a run's verifications recorded as failed, read from its context the way its scores are."""

from agent_env.task.task import FailedCriterion, failed_criteria
from agent_env.task_step.context import TaskStepContext


class _Step:
    def __init__(self, id, verifier_id):
        self.id, self.verifier_id = id, verifier_id


def _failed(verifications, steps=()):
    return failed_criteria(list(steps), TaskStepContext(metadata={"verifications": verifications}))


def test_a_failed_criterion_is_named_by_the_step_that_recorded_it_or_else_by_its_key():
    verifications = {"v": {"score": 0, "results": [{"criterion": "greets", "result": False}]},
                     "v-trajectory-mistakes": {"score": 0.5, "results": [{"id": "m1", "score": 0.5}]}}

    assert _failed(verifications, [_Step("check", "v")]) == [
        FailedCriterion("check", "greets", ""), FailedCriterion("v-trajectory-mistakes", "m1", "")]


def test_only_a_verification_scoring_below_1_has_failed_criteria():
    verifications = {"full": {"score": 1, "results": [{"id": "a", "result": False}]},
                     "unscored": {"results": [{"id": "b", "result": False}]},
                     "partial": {"score": 0.5, "results": [{"id": "c", "result": True}, {"id": "d", "result": False}]}}

    assert [criterion.criterion for criterion in _failed(verifications)] == ["d"]


def test_a_result_without_a_result_fails_by_its_score_and_a_skipped_one_never_fails():
    verifications = {"v": {"score": 0.4, "results": [
        {"id": "low", "score": 0.4}, {"id": "full", "score": 1.0}, {"id": "unsure"},
        {"id": "skipped", "result": False, "skipped": True}, "not a result"]}}

    assert [criterion.criterion for criterion in _failed(verifications)] == ["low"]


def test_the_reason_is_the_first_one_the_verifier_gave():
    verifications = {"v": {"score": 0, "results": [
        {"id": "a", "result": False, "justification": "why", "message": "what", "description": "about"},
        {"id": "b", "result": False, "message": "what", "description": "about"},
        {"id": "c", "result": False, "justification": "  ", "description": "about"},
        {"type": "response_contains", "id": " ", "result": False}]}}

    assert [(criterion.criterion, criterion.reason) for criterion in _failed(verifications)] == [
        ("a", "why"), ("b", "what"), ("c", "about"), ("response_contains #3", "")]
