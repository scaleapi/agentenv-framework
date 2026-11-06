from __future__ import annotations

import json
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.verifiers.judge_utils.judge_output_format import (
    JudgeOutputFormat,
    get_judge_output_format_spec,
)
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator, aggregate_score

NON_FINITE = {
    "nan": "NaN",
    "inf": "Infinity",
    "-inf": "-Infinity",
    "float-overflow": "1e999",
    "quoted-nan": '"NaN"',
    "quoted-inf": '"inf"',
    "quoted-infinity": '"Infinity"',
    "int-overflow": "1" + "0" * 400,
}
RUBRIC_FORMATS = [JudgeOutputFormat.RUBRIC_BINARY, JudgeOutputFormat.RUBRIC_PARTIAL,
                  JudgeOutputFormat.RUBRIC_EVIDENCE, JudgeOutputFormat.RUBRIC_EVIDENCE_WITH_MISTAKES]
_EVIDENCE_FORMATS = (JudgeOutputFormat.RUBRIC_EVIDENCE, JudgeOutputFormat.RUBRIC_EVIDENCE_WITH_MISTAKES)


def _judge_response(fmt: JudgeOutputFormat, scores: list[str]) -> str:
    evidence = fmt in _EVIDENCE_FORMATS
    rows = ", ".join(
        f'{{"id": "c{i}", "score": {score}, "justification": "j"'
        + (', "frame": "F1", "evidence_status": "visible"' if evidence else "") + "}"
        for i, score in enumerate(scores)
    )
    extra = ', "reasoning": "r", "checks": []' if evidence else ""
    if fmt is JudgeOutputFormat.RUBRIC_EVIDENCE_WITH_MISTAKES:
        extra += ', "findings": [], "botched": false, "summary": "s"'
    return f'{{"results": [{rows}]{extra}}}'


@pytest.mark.parametrize("fmt", RUBRIC_FORMATS, ids=lambda f: f.value)
@pytest.mark.parametrize("score", NON_FINITE.values(), ids=NON_FINITE.keys())
def test_non_finite_score_is_invalid_output_in_every_rubric_format(fmt, score):
    criteria = [{"id": "c0"}, {"id": "c1"}]
    spec = get_judge_output_format_spec(fmt)

    rows, discrepancy = spec.diagnose_response(_judge_response(fmt, ["0", score]), criteria)

    assert rows is None
    assert discrepancy is not None
    assert str(discrepancy) == "Each `score` must be a finite number; criterion id(s) c1 were not"
    assert [r["score"] for r in spec.parse_response(_judge_response(fmt, ["1", score]), criteria)] == [0.0, 0.0]


def _verifier(output_format: str) -> RubricsVerifierTaskStep:
    return RubricsVerifierTaskStep(
        id="verify", version=1, criteria=[{"id": "a", "title": "A"}, {"id": "b", "title": "B"}],
        prompt_id="p1", use_agent_judge=False, use_trajectory=False, verifier_id="vt",
        output_format=output_format, score_aggregator="weighted_average",
    )


def _context() -> TaskStepContext:
    return TaskStepContext(prompt_responses=[PromptResponse(prompt_id="p1", response="done", prompt_text="do it")])


def _prompted_ids(prompt: str) -> list[str]:
    start = prompt.index("[")
    return [c["id"] for c in json.loads(prompt[start:prompt.index("]", start) + 1])]


def _fake_judge(verifier: RubricsVerifierTaskStep, second_scores: list[str]) -> list[str]:
    prompts: list[str] = []

    async def fake_prompt_llm_judge(prompt: str, *, model: str, context: TaskStepContext) -> dict:
        prompts.append(prompt)
        first, second = _prompted_ids(prompt)
        score = second_scores[min(len(prompts), len(second_scores)) - 1]
        response = (f'{{"results": [{{"id": "{first}", "score": 0, "justification": "not done"}}, '
                    f'{{"id": "{second}", "score": {score}, "justification": "not done"}}]}}')
        return {"response": response, "trajectory_object_url": None}

    verifier._prompt_llm_judge = fake_prompt_llm_judge
    return prompts


@pytest.mark.parametrize("fmt", RUBRIC_FORMATS, ids=lambda f: f.value)
def test_integer_past_the_digit_limit_is_invalid_json(fmt):
    criteria = [{"id": "c0"}, {"id": "c1"}]

    rows, discrepancy = get_judge_output_format_spec(fmt).diagnose_response(
        _judge_response(fmt, ["0", "9" * 5000]), criteria,
    )

    assert rows is None
    assert discrepancy is not None and discrepancy.parse_error


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["rubric_binary", "rubric_partial"])
@pytest.mark.parametrize("score", NON_FINITE.values(), ids=NON_FINITE.keys())
async def test_non_finite_score_is_retried_and_never_scored(output_format, score):
    verifier = _verifier(output_format)
    prompts = _fake_judge(verifier, [score, "0"])
    ctx = _context()

    await verifier.execute(ctx)

    entry = ctx.metadata["verifications"]["vt"]
    assert entry["score"] == 0.0
    assert [r["score"] for r in entry["results"]] == [0.0, 0.0]
    assert entry["judge_output_retries"] == 1
    assert entry["judge_output_discrepancies"][0]["non_finite_score_ids"] == [_prompted_ids(prompts[0])[1]]
    assert f"- Each `score` must be a finite number; criterion id(s) {_prompted_ids(prompts[0])[1]} were not." in prompts[1]
    assert "not valid JSON" not in prompts[1]


@pytest.mark.asyncio
async def test_judge_that_keeps_returning_nan_fails_the_step():
    verifier = _verifier("rubric_partial")
    prompts = _fake_judge(verifier, ['"NaN"'])
    ctx = _context()

    with pytest.raises(ValueError, match="must be a finite number"):
        await verifier.execute(ctx)

    assert len(prompts) == verifier.DEFAULT_MAX_RETRIES
    assert all("must be a finite number" in p for p in prompts[1:])
    assert "vt" not in ctx.metadata.get("verifications", {})


_NON_FINITE_LITERALS = st.one_of(
    st.sampled_from([float("nan"), float("inf"), float("-inf")]).map(json.dumps),
    st.integers(10**309, 10**400).map(str),
    st.integers(-(10**400), -(10**309)).map(str),
    st.sampled_from(["nan", "NaN", "inf", "-inf", "Infinity", "-Infinity", "1e999"]).map(json.dumps),
)


def _finite_literals(fmt: JudgeOutputFormat) -> st.SearchStrategy[str]:
    if fmt is JudgeOutputFormat.RUBRIC_BINARY:
        return st.one_of(st.floats(0, 1).map(json.dumps), st.integers(0, 1).map(str))
    return st.one_of(
        st.floats(allow_nan=False, allow_infinity=False).map(json.dumps),
        st.integers(-(10**300), 10**300).map(str),
        st.floats(allow_nan=False, allow_infinity=False).map(lambda f: json.dumps(repr(f))),
    )


def _rows(fmt: JudgeOutputFormat, criteria: list[dict], scores: list[str]) -> list[dict] | None:
    rows, discrepancy = get_judge_output_format_spec(fmt).diagnose_response(_judge_response(fmt, scores), criteria)
    if rows is None:
        assert discrepancy is not None and discrepancy.non_finite_score_ids
    return rows


@settings(max_examples=500, deadline=None)
@given(fmt=st.sampled_from(RUBRIC_FORMATS), data=st.data())
def test_only_finite_scores_reach_aggregation_and_raising_one_never_lowers_it(fmt, data):
    literal = st.one_of(_finite_literals(fmt).map(lambda s: (s, True)), _NON_FINITE_LITERALS.map(lambda s: (s, False)))
    drawn = data.draw(st.lists(st.tuples(st.floats(-3, 3), literal), min_size=1, max_size=4))
    criteria = [{"id": f"c{i}", "weight": weight} for i, (weight, _) in enumerate(drawn)]
    scores = [score for _, (score, _) in drawn]

    rows = _rows(fmt, criteria, scores)

    if not all(finite for _, (_, finite) in drawn):
        assert rows is None
        return
    assert rows is not None
    for row in rows:
        assert math.isfinite(row["score"])
        if fmt is not JudgeOutputFormat.RUBRIC_BINARY:
            assert 0.0 <= row["score"] <= 1.0
    positive = [i for i, (weight, _) in enumerate(drawn) if weight > 0]
    if not positive:
        return
    i = data.draw(st.sampled_from(positive))
    higher = data.draw(st.floats(min_value=float(json.loads(scores[i])), allow_nan=False, allow_infinity=False))
    raised = _rows(fmt, criteria, [*scores[:i], json.dumps(higher), *scores[i + 1:]])
    assert raised is not None
    assert (aggregate_score(raised, ScoreAggregator.WEIGHTED_AVERAGE)
            >= aggregate_score(rows, ScoreAggregator.WEIGHTED_AVERAGE))
