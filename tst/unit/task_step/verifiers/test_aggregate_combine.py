"""aggregate_verifiers naming its sources by step id: pooled rows, a weighted mean and gates over the sources' named
grades, what a failed, missing or skipped source does, its config problems, and the older verifier_ids form kept."""

import asyncio
from typing import ClassVar

import pytest

from agent_env.task import Task
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.verifiers.aggregate_verifiers import AggregateVerifiersTaskStep
from agent_env.task_step.task_steps.verifiers.verifier import GRADES_KEY, Verification, VerifierTaskStep


class _Fixed(VerifierTaskStep):
    """A verifier whose verdict is given: rows, a score of its own, a skip or an exception."""

    type: ClassVar[str] = "fixed_aggregate_test"
    entity_refs = ()

    def __init__(self, id, rows=(), score=None, skipped=None, raises=None, min_passing_score=None):
        super().__init__(id, None, verifier_id=f"{id}-vid", min_passing_score=min_passing_score)
        self.rows, self.fixed_score, self.skipped, self.raises = list(rows), score, skipped, raises

    async def verify(self, context):
        if self.raises:
            raise RuntimeError(self.raises)
        return Verification(criteria=[dict(row) for row in self.rows], score=self.fixed_score, skipped=self.skipped)


def _graded(*sources: VerifierTaskStep) -> TaskStepContext:
    context = TaskStepContext()
    for source in sources:
        try:
            asyncio.run(source.execute(context))
        except RuntimeError:
            pass
    return context


def _aggregate(**fields) -> AggregateVerifiersTaskStep:
    return AggregateVerifiersTaskStep(id="final", version=None, verifier_id="final-vid", **fields)


def _run(aggregate, context):
    try:
        asyncio.run(aggregate.execute(context))
    except RuntimeError:
        pass
    return context.metadata[GRADES_KEY]["final"]


def _row(score, passed=None, weight=1.0):
    return {"score": score, "result": score >= 1.0 if passed is None else passed, "weight": weight}


TESTS = _Fixed("tests", rows=[_row(1.0)])
RUBRIC = _Fixed("rubric", score=0.8, min_passing_score=0.7)
OUTCOME = _Fixed("outcome", score=0.85, min_passing_score=0.5)


def test_the_grades_docs_worked_example_tests_gate_and_rubric_and_outcome_weigh_to_082_against_08():
    final = _aggregate(combine="weighted_mean", sources=["rubric", "outcome"],
                       weights={"rubric": 0.6, "outcome": 0.4}, gates=["tests"], min_passing_score=0.8)

    grade = _run(final, _graded(TESTS, RUBRIC, OUTCOME))

    assert (grade["verifier_status"], grade["score"], grade["passed"]) == ("graded", pytest.approx(0.82), True)
    assert grade["verifier_output"]["gates"] == {"tests": {"verifier_status": "graded", "score": 1.0, "passed": True}}
    assert grade["verifier_settings"].get("score_aggregator") is None


def test_a_failed_gate_scores_0_and_says_what_the_score_would_have_been():
    final = _aggregate(combine="weighted_mean", sources=["rubric", "outcome"], gates=["tests"])

    grade = _run(final, _graded(_Fixed("tests", rows=[_row(0.0)]), RUBRIC, OUTCOME))

    assert (grade["score"], grade["passed"]) == (0.0, False)
    assert grade["verifier_output"]["failed_gates"] == ["tests"]
    assert grade["verifier_output"]["ungated_score"] == pytest.approx(0.825)


def test_gates_alone_score_1_when_they_all_pass_and_0_when_one_fails():
    passing = _run(_aggregate(gates=["tests", "rubric"]), _graded(TESTS, RUBRIC))
    failing = _run(_aggregate(gates=["tests", "rubric"]), _graded(TESTS, _Fixed("rubric", score=0.5)))

    assert (passing["score"], passing["passed"]) == (1.0, True)
    assert (failing["score"], failing["verifier_output"]["failed_gates"]) == (0.0, ["rubric"])


def test_pooling_grades_counts_every_source_a_score_only_one_by_its_own_row_and_all_pass_reads_each_rows_pass():
    rows = _Fixed("rows", rows=[_row(1.0), _row(0.0)])
    pooled = _run(_aggregate(sources=["rows", "outcome"]), _graded(rows, OUTCOME))
    all_pass = _run(_aggregate(sources=["tests", "rows"], score_aggregator="all_pass"), _graded(TESTS, rows))
    all_passing = _run(_aggregate(sources=["tests", "outcome"], score_aggregator="all_pass"),
                       _graded(TESTS, _Fixed("outcome", score=1.0)))

    assert pooled["score"] == pytest.approx((1.0 + 0.0 + 0.85) / 3)
    assert (all_pass["score"], all_passing["score"]) == (0.0, 1.0)


@pytest.mark.parametrize(("source", "said"), [
    (_Fixed("rubric", raises="judge out of retries"), "source 'rubric' failed, so the aggregate can't combine it: "
                                                      "RuntimeError: judge out of retries"),
    (None, "source 'rubric' has no grade: its step didn't run before the aggregate"),
])
def test_a_failed_or_missing_source_fails_the_aggregate(source, said):
    grade = _run(_aggregate(combine="weighted_mean", sources=["rubric", "outcome"]),
                 _graded(*[step for step in (source, OUTCOME) if step]))

    assert (grade["verifier_status"], grade["score"], grade["passed"]) == ("failed", None, False)
    assert grade["verifier_error_reasoning"].startswith(f"RuntimeError: {said}")


def test_a_skipped_source_is_left_out_and_every_source_skipped_or_a_gate_skipped_skips_the_aggregate():
    skipped = _Fixed("rubric", skipped="the agent errored")
    weighted = _run(_aggregate(combine="weighted_mean", sources=["rubric", "outcome"],
                               weights={"rubric": 3, "outcome": 1}), _graded(skipped, OUTCOME))
    every = _run(_aggregate(sources=["rubric"]), _graded(skipped))
    gate = _run(_aggregate(sources=["outcome"], gates=["rubric"]), _graded(skipped, OUTCOME))

    assert (weighted["verifier_status"], weighted["score"]) == ("graded", pytest.approx(0.85))
    assert (every["verifier_status"], every["verifier_skip_reasoning"]) == ("skipped", "every source it weighs was "
                                                                                     "skipped")
    assert (gate["verifier_status"], gate["verifier_skip_reasoning"]) == (
        "skipped", "gate 'rubric' was skipped, so whether it passes isn't known")


@pytest.mark.parametrize(("fields", "problem"), [
    ({"verifier_ids": ["a"], "sources": ["rubric"]}, "verifier_ids is the older form and takes none of sources"),
    ({"verifier_ids": []}, "it combines nothing: name sources or gates"),
    ({"combine": "weighted_mean"}, "it combines nothing: name sources or gates"),
    ({"sources": "rubric"}, "sources must be a list of step ids, not 'rubric'"),
    ({"sources": ["rubric", "rubric"]}, "sources names ['rubric'] more than once"),
    ({"gates": ["final"]}, "gates names the aggregate itself"),
    ({"sources": ["rubric"], "combine": "median"}, "combine must be one of ['pooled', 'weighted_mean'], not 'median'"),
    ({"sources": ["rubric"], "combine": "weighted_mean", "score_aggregator": "all_pass"},
     "score_aggregator aggregates pooled rows, so a weighted_mean takes none"),
    ({"sources": ["rubric"], "weights": {"rubric": 1}}, "weights are for a weighted_mean; pooled rows carry their own"),
    ({"sources": ["rubric", "outcome"], "combine": "weighted_mean", "weights": {"rubric": 0.6}},
     "weights gives none for ['outcome']: give every source a weight, or none for an equal mean"),
    ({"sources": ["rubric"], "combine": "weighted_mean", "weights": {"rubric": 1, "tests": 1}},
     "weights names ['tests'], which aren't in sources"),
    ({"sources": ["rubric"], "combine": "weighted_mean", "weights": {"rubric": -1}},
     "weights gives 'rubric' -1; a weight is a number of at least 0"),
    ({"sources": ["rubric"], "combine": "weighted_mean", "weights": {"rubric": True}},
     "weights gives 'rubric' True; a weight is a number of at least 0"),
    ({"sources": ["rubric", "outcome"], "combine": "weighted_mean", "weights": {"rubric": 0, "outcome": 0}},
     "every weight is 0, so nothing counts"),
])
def test_an_aggregate_a_run_cant_combine_as_written_is_a_config_problem(fields, problem):
    assert problem in " | ".join(_aggregate(**fields).config_problems())


def test_a_well_formed_aggregate_has_no_config_problem():
    for fields in ({"sources": ["rubric"]}, {"gates": ["tests"]}, {"verifier_ids": ["a-vid"]},
                   {"sources": ["rubric", "outcome"], "combine": "weighted_mean", "weights": {"rubric": 2, "outcome": 0},
                    "gates": ["tests"]}):
        assert _aggregate(**fields).config_problems() == []


def test_the_newer_fields_round_trip_and_the_older_form_writes_what_it_always_has():
    newer = _aggregate(combine="weighted_mean", sources=["rubric"], weights={"rubric": 1}, gates=["tests"])
    older = _aggregate(verifier_ids=["rubric-vid"], score_aggregator="all_pass")

    assert AggregateVerifiersTaskStep.from_dict(newer.to_dict()).to_dict() == newer.to_dict()
    assert "verifier_ids" not in newer.to_dict()
    assert {key: value for key, value in older.to_dict().items() if key not in ("id", "version", "type")} == {
        "fail_task_on_error": True, "verifier_ids": ["rubric-vid"], "score_aggregator": "all_pass",
        "verifier_id": "final-vid"}


def test_its_source_steps_are_the_steps_it_names_or_those_whose_verifier_ids_it_names():
    rubric, tests = _Fixed("rubric"), _Fixed("tests")
    steps = [rubric, tests]

    assert _aggregate(sources=["rubric"], gates=["tests", "rubric"]).source_steps(steps) == ["rubric", "tests"]
    assert _aggregate(verifier_ids=["tests-vid", "rubric-vid-trajectory-mistakes", "rubric-vid", "gone"]).source_steps(
        steps) == ["tests", "rubric"]


def test_a_stored_aggregate_of_one_rubric_as_the_hubs_are_loads_and_resolves_its_source():
    """Every aggregate the hub's evals run is this shape: one rubric source by verifier_id, pooled."""
    task = Task.from_dict({"id": "t", "version": 3, "steps": [
        {"id": "judge", "type": "rubrics_verifier", "verifier_id": "judge-vid", "prompt_id": "ask",
         "criteria": [{"criterion": "says hi"}]},
        {"id": "final", "type": "aggregate_verifiers", "verifier_ids": ["judge-vid"], "score_aggregator": "all_pass",
         "verifier_id": "final-vid", "fail_task_on_error": True}]})
    final = task.steps[1]

    assert (final.names_steps, final.source_steps(task.steps), final.config_problems()) == (False, ["judge"], [])
