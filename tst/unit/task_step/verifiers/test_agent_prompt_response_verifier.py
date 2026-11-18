"""agent_prompt_response_verifier on the VerifierTaskStep base: its legacy entry is byte for byte what it wrote before
the move (the strings below were recorded from main at 810e27760), and it now writes a named grade beside it."""
import asyncio
import json

import pytest

from agent_env.plugins import _requirements
from agent_env.task.task import recorded_scores
from agent_env.task_step.context import PromptResponse, StepContextError, TaskStepContext
from agent_env.task_step.task_steps.verifiers import AgentPromptResponseVerifierTaskStep

CRITERIA = [
    {"id": "says-done", "type": "response_contains", "needles": ["done", "ok"]},
    {"type": "response_regex_present", "pattern": "^Total: \\d+$", "flags": ["MULTILINE"], "weight": 2},
    {"id": "names-file", "type": "response_contains", "needles": ["report.csv"], "weight": 0.5},
]

ROWS = (r'''[{"id": "says-done", "type": "response_contains", "needles": ["done", "ok"], "criterion_index": 0, '''
        r'''"score": 1.0, "result": true, "justification": "all needles present"}, {"type": "response_regex_present", '''
        r'''"pattern": "^Total: \\d+$", "flags": ["MULTILINE"], "weight": 2, "criterion_index": 1, "score": 1.0, '''
        r'''"result": true, "justification": "pattern matched"}, {"id": "names-file", "type": "response_contains", '''
        r'''"needles": ["report.csv"], "weight": 0.5, "criterion_index": 2, "score": 0.0, "result": false, '''
        r'''"justification": "missing: ['report.csv']"}]''')
BEFORE = {
    "all_pass": '{"results": ' + ROWS + ', "score": 0.0}',
    "weighted_average": '{"results": ' + ROWS + ', "score": 0.8571428571428571}',
    "prompt_error": ('{"results": [{"id": "prompt_error", "score": 0, "result": false, "message": "Skipped: prompt '
                     'had error_type=agent_error"}], "score": 0}'),
}
RESPONSES = {"all_pass": "done, ok\nTotal: 12\n", "weighted_average": "done ok\nTotal: 3"}


def _step(**kw) -> AgentPromptResponseVerifierTaskStep:
    return AgentPromptResponseVerifierTaskStep(id="check", version=None, prompt_id="p1", criteria=CRITERIA,
                                               verifier_id="vid", **kw)


def _run(step, response="", error_type=None) -> TaskStepContext:
    context = TaskStepContext(prompt_responses=[PromptResponse(prompt_id="p1", response=response, error_type=error_type)])
    asyncio.run(step.execute(context))
    return context


@pytest.mark.parametrize("aggregator", ["all_pass", "weighted_average"])
def test_the_legacy_entry_is_unchanged_and_the_grade_carries_its_score(aggregator):
    step = _step(score_aggregator=aggregator)
    context = _run(step, RESPONSES[aggregator])

    assert json.dumps(context.metadata["verifications"]["vid"]) == BEFORE[aggregator]
    grade = context.metadata["grades"]["check"]
    assert grade["score"] == json.loads(BEFORE[aggregator])["score"]
    assert (grade["verifier_status"], grade["verifier_type"]) == ("graded", "agent_prompt_response_verifier")
    assert grade["verifier_version"] == _requirements.installed("agentenv-framework")
    assert [row["id"] for row in grade["verifier_criteria"]] == ["says-done", "1", "names-file"]
    assert recorded_scores([step], context) == {"check": json.loads(BEFORE[aggregator])["score"]}


def test_a_prompt_error_is_still_a_graded_zero():
    context = _run(_step(), error_type="agent_error")

    assert json.dumps(context.metadata["verifications"]["vid"]) == BEFORE["prompt_error"]
    grade = context.metadata["grades"]["check"]
    assert (grade["verifier_status"], grade["score"], grade["passed"]) == ("graded", 0.0, False)


def test_an_unknown_criterion_type_is_a_failed_grade_and_still_raises():
    step = AgentPromptResponseVerifierTaskStep(id="check", version=None, prompt_id="p1",
                                               criteria=[{"type": "response_sentiment"}], verifier_id="vid")
    context = TaskStepContext(prompt_responses=[PromptResponse(prompt_id="p1", response="hi")])

    with pytest.raises(ValueError, match="unknown type 'response_sentiment'"):
        asyncio.run(step.execute(context))

    assert "verifications" not in context.metadata
    assert context.metadata["grades"]["check"]["verifier_status"] == "failed"
    assert context.metadata["grades"]["check"]["verifier_error_reasoning"].startswith("ValueError: criterion #0")


def test_a_missing_prompt_response_is_a_failed_grade_and_still_raises():
    context = TaskStepContext()

    with pytest.raises(StepContextError):
        asyncio.run(_step().execute(context))

    assert context.metadata["grades"]["check"]["verifier_status"] == "failed"


def test_its_stored_form_is_unchanged():
    before = json.loads(r'''{"criteria": [{"id": "says-done", "needles": ["done", "ok"], "type": "response_contains"},
        {"flags": ["MULTILINE"], "pattern": "^Total: \\d+$", "type": "response_regex_present", "weight": 2},
        {"id": "names-file", "needles": ["report.csv"], "type": "response_contains", "weight": 0.5}],
        "fail_task_on_error": true, "id": "check", "prompt_id": "p1", "score_aggregator": "all_pass",
        "type": "agent_prompt_response_verifier", "verifier_id": "vid", "version": null}''')

    assert _step().to_dict() == before
    assert AgentPromptResponseVerifierTaskStep.from_dict(before).to_dict() == before


def test_its_settings_are_its_own_fields():
    settings = _run(_step(), RESPONSES["all_pass"]).metadata["grades"]["check"]["verifier_settings"]

    assert settings == {"scoring_rules": 1, "prompt_id": "p1", "criteria": CRITERIA, "score_aggregator": "all_pass"}


def test_a_min_passing_score_round_trips_and_sets_the_pass_rule():
    step = AgentPromptResponseVerifierTaskStep.from_dict({**_step().to_dict(), "score_aggregator": "weighted_average",
                                                          "min_passing_score": 0.8})

    grade = _run(step, RESPONSES["weighted_average"]).metadata["grades"]["check"]

    assert step.to_dict()["min_passing_score"] == 0.8
    assert (grade["min_passing_score"], grade["passed"]) == (0.8, True)
