"""The user_sim_checks judge output format: it grades the user sim (the AI playing the user), not the assistant."""

import json
from unittest.mock import AsyncMock

import pytest

from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.verifiers.judge_utils.judge_output_format import (
    JUDGE_OUTPUT_FORMAT_REGISTRY,
    JudgeOutputFormat,
    diagnose_user_sim_checks_response,
    format_user_sim_checks_correction,
    get_judge_output_format_spec,
)
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator, aggregate_score

SPEC = get_judge_output_format_spec(JudgeOutputFormat.USER_SIM_CHECKS)

RUN = """# THE USER SIM'S INSTRUCTIONS
<instructions>
You are Hannah Reyes. K1 Dad is allergic to shellfish. M3 Mention the allergy only when asked.
</instructions>

# THE RUN

## Turn 1 · GRADED
### What it was sent
<sent>
It's Mon 9:12 AM. NEW FROM YOUR AI ASSISTANT: Booking Saturday dinner for 4 at Pescado. Any dietary needs?
</sent>
### Its answer
outcome: reply
message: "dad cant do shellfish"

## Turn 2 · GRADED
### What it was sent
<sent>
It's Mon 9:40 AM. NEW FROM YOUR AI ASSISTANT: Booked for 7 PM. Want me to add a birthday note?
</sent>
### Its answer
outcome: reply
message: "sure. also dad cant do shellfish, its serious"
"""


def _criteria(*ids: str, graded=(1, 2), **extra) -> list[dict]:
    return [{"id": cid, "question": f"question {cid}", "graded_turns": list(graded), **extra} for cid in ids]


def _row(cid="c1", *, verdict="pass", turns=(), could_change_grade=False, reason="Answered what was asked.",
         **extra) -> dict:
    return {"id": cid, "reason": reason, "turns": list(turns), "verdict": verdict,
            "could_change_grade": could_change_grade, **extra}


def _answer(*rows: dict) -> str:
    return json.dumps({"results": list(rows)})


def _fail(cid="c1", *, turns=(2,), could_change_grade=True,
          reason="At turn 2 she repeated the shellfish allergy unasked; M3 says only when asked.", **extra) -> dict:
    return _row(cid, verdict="fail", turns=turns, could_change_grade=could_change_grade, reason=reason, **extra)


def _render(template, criteria_json="[]") -> str:
    """A template filled the way rubrics_verifier fills it: every keyword it passes, used or not."""
    return template.substitute(agent_prompt="the assistant's prompt", agent_response="the assistant's answer",
                               criteria_json=criteria_json, trajectory_content=RUN,
                               trajectory_path="/tmp/trajectory.json", checks_instruction="[]")


# ---- templates ----------------------------------------------------------------------------------------------------
def test_it_is_registered_and_found_by_name():
    assert JudgeOutputFormat("user_sim_checks") is JudgeOutputFormat.USER_SIM_CHECKS
    assert get_judge_output_format_spec("user_sim_checks") is JUDGE_OUTPUT_FORMAT_REGISTRY[
        JudgeOutputFormat.USER_SIM_CHECKS]
    assert SPEC.stored_format is None and SPEC.cites_evidence is False


def test_the_judge_is_told_it_grades_the_user_sim_never_the_assistant():
    for template in (SPEC.prompt_template_inline, SPEC.prompt_template_path, SPEC.prompt_template_no_trajectory):
        prompt = _render(template)
        assert prompt.startswith('You check a user simulator ("the user sim")')
        assert "You grade the user sim, never the assistant." in prompt
        # Nothing of the assistant's own prompt or answer is put in front of the judge as the thing to grade.
        assert "the assistant's prompt" not in prompt and "the assistant's answer" not in prompt


def test_the_run_comes_first_and_the_criteria_last_so_calls_on_one_run_share_a_cached_head():
    one = _render(SPEC.prompt_template_inline, json.dumps(_criteria("c1")))
    other = _render(SPEC.prompt_template_inline, json.dumps(_criteria("c1", "c2")))
    assert one.index(RUN) < one.index("# THE CRITERIA") < one.index('"question": "question c1"')
    assert one.endswith("Return only the JSON object.")
    head = one[:one.index("# THE CRITERIA")]
    assert other.startswith(head) and RUN in head
    path = _render(SPEC.prompt_template_path)
    assert path.index("/tmp/trajectory.json") < path.index("# THE CRITERIA")


def test_without_a_trajectory_the_judge_is_told_the_run_follows_the_criteria():
    prompt = _render(SPEC.prompt_template_no_trajectory)
    assert RUN not in prompt
    assert prompt.endswith("The user sim's instructions and the run follow.")


def test_the_judge_reasons_first_and_the_verdict_follows_the_reason():
    prompt = _render(SPEC.prompt_template_inline)
    assert "Reason first and decide last." in prompt and "at most 40 words" in prompt
    assert 'a reason that finds no clear failure gives "pass" (or "n/a"), never "fail"' in prompt
    shape = prompt.split("Return only this JSON object")[1]
    assert (shape.index('"id"') < shape.index('"reason"') < shape.index('"turns"') < shape.index('"verdict"')
            < shape.index('"could_change_grade"'))


def test_the_judge_is_told_how_to_read_the_run_and_the_criteria_fields():
    prompt = _render(SPEC.prompt_template_inline)
    for told in ("Turns marked CONTEXT ONLY are fixed history", "Grade only the turns marked GRADED",
                 '"graded_turns"', "The assistant's own words are the assistant's",
                 'A "hard" item is an exact rule', 'A "general" item is a judgement',
                 '"na_when": "never" means it always applies',
                 '"could_change_grade" says whether the failure could change how the assistant is graded'):
        assert told in prompt, told


# ---- schema -------------------------------------------------------------------------------------------------------
def test_each_row_gives_its_reason_first_then_turns_verdict_and_could_change_grade():
    schema = SPEC.output_format
    assert schema["type"] == "json_schema" and schema["schema"]["required"] == ["results"]
    row = schema["schema"]["properties"]["results"]["items"]
    order = ["id", "reason", "turns", "verdict", "could_change_grade"]
    assert list(row["properties"]) == order == row["required"]
    assert row["properties"]["verdict"]["enum"] == ["pass", "fail", "n/a"]
    assert row["properties"]["turns"] == {"type": "array", "items": {"type": "integer"}}
    assert row["properties"]["could_change_grade"] == {"type": "boolean"}
    assert row["additionalProperties"] is False and schema["schema"]["additionalProperties"] is False


# ---- reading an answer --------------------------------------------------------------------------------------------
def test_an_answer_is_read_into_rows_with_turns_kept_to_the_graded_ones():
    criteria = _criteria("c1", "c2", graded=[2], family="G1")
    rows, discrepancy = diagnose_user_sim_checks_response(
        _answer(_fail("c1", turns=[1, 2, "x"]), _row("c2", could_change_grade=True)), criteria)
    assert discrepancy is None
    fail, ok = rows
    assert (fail["verdict"], fail["turns"], fail["turns_outside_graded"]) == ("fail", [2], [1])
    assert fail["could_change_grade"] is True and fail["justification"] == fail["reason"]
    assert fail["family"] == "G1" and fail["question"] == "question c1"  # the criterion rides on its row
    # A pass can't change the grade, whatever the judge wrote.
    assert ok["could_change_grade"] is False and "turns_outside_graded" not in ok


def test_the_parser_reads_the_keys_in_any_order_and_tolerates_a_fence_or_text_around_the_object():
    flipped = {"could_change_grade": True, "verdict": "FAIL ", "turns": [2], "reason": "Leaked the allergy.", "id": "c1"}
    for text in (json.dumps({"results": [flipped]}), "```json\n" + _answer(flipped) + "\n```",
                 "Here is my answer:\n" + _answer(flipped) + "\nDone."):
        rows, discrepancy = diagnose_user_sim_checks_response(text, _criteria("c1"))
        assert discrepancy is None and rows[0]["verdict"] == "fail", text


def test_an_answer_that_is_not_a_json_object_or_has_the_wrong_rows_is_asked_again():
    criteria = _criteria("c1", "c2")
    for text, problem in (("dad cant do shellfish", "the answer is not a JSON object"),
                          (json.dumps({"verdict": "pass"}), "unexpected response shape: dict"),
                          (json.dumps({"results": ["pass", "pass"]}), "every result must be a JSON object")):
        rows, discrepancy = diagnose_user_sim_checks_response(text, criteria)
        assert rows is None and discrepancy.parse_error == problem
    rows, discrepancy = diagnose_user_sim_checks_response(_answer(_row("c1"), _row("c1"), _row("c9")), criteria)
    assert rows is None
    assert (discrepancy.missing_ids, discrepancy.duplicate_ids, discrepancy.unknown_ids) == (["c2"], ["c1"], ["c9"])


def test_a_verdict_or_could_change_grade_out_of_range_is_asked_again():
    rows, discrepancy = diagnose_user_sim_checks_response(
        _answer(_row("c1", verdict="maybe"), _row("c2", could_change_grade="yes")), _criteria("c1", "c2"))
    assert rows is None
    assert discrepancy.row_problems == ["c1: verdict 'maybe' is not one of ['pass', 'fail', 'n/a']",
                                        "c2: could_change_grade must be true or false"]


def test_a_fail_that_cites_no_graded_turn_is_asked_again_and_a_pass_needs_none():
    for turns, cited in (([99], r"\[99\]"), ([], r"\[\]")):
        rows, discrepancy = diagnose_user_sim_checks_response(_answer(_fail(turns=turns)), _criteria("c1", graded=[1]))
        assert rows is None
        assert discrepancy.row_problems == [f"c1: a fail must cite at least one graded turn; cited {turns}, graded [1]"]
        with pytest.raises(ValueError, match=f"a fail must cite at least one graded turn; cited {cited}"):
            SPEC.parse_response(_answer(_fail(turns=turns)), _criteria("c1", graded=[1]))
    for ok in (_row(turns=[99]), _row(verdict="n/a", turns=[99])):
        rows, discrepancy = diagnose_user_sim_checks_response(_answer(ok), _criteria("c1", graded=[1]))
        assert discrepancy is None and rows[0]["turns"] == [] and rows[0]["turns_outside_graded"] == [99]


def test_without_graded_turns_on_the_criterion_a_fail_still_has_to_cite_a_turn():
    criteria = [{"id": "c1", "question": "question c1"}]
    rows, discrepancy = diagnose_user_sim_checks_response(_answer(_fail(turns=[])), criteria)
    assert discrepancy.row_problems == ["c1: a fail must cite at least one graded turn; cited []"]
    rows, discrepancy = diagnose_user_sim_checks_response(_answer(_fail(turns=[7])), criteria)
    assert discrepancy is None and rows[0]["turns"] == [7] and "turns_outside_graded" not in rows[0]


# The endings of the two contradictory reasons seen in real runs, and a plain one.
SAYS_NO_FAILURE = ("Hannah flagged it anyway which is fine. No clear failure found.",
                   "This is not a delegation failure by Linda. No clear delegation failures found.",
                   "Every answer fits her rules; nothing fails.")


def test_a_fail_whose_reason_concludes_nothing_failed_is_asked_again():
    for reason in SAYS_NO_FAILURE:
        rows, discrepancy = diagnose_user_sim_checks_response(_answer(_fail(reason=reason)), _criteria("c1"))
        assert rows is None and len(discrepancy.row_problems) == 1
        assert "c1: a fail whose reason says" in discrepancy.row_problems[0]
        assert "contradicts itself; the verdict must follow the reason" in discrepancy.row_problems[0]
        # A pass may say it.
        rows, discrepancy = diagnose_user_sim_checks_response(_answer(_row(reason=reason)), _criteria("c1"))
        assert discrepancy is None and rows[0]["reason"] == reason
    # Only the last sentence is the conclusion: a fail that clears one turn and fails another is kept.
    for reason in ("No clear failure at turn 1; at turn 2 she revealed the private password.",
                   "Turn 1 is fine, not a failure. At turn 2 she volunteered the shellfish allergy unasked; M3 says "
                   "only when asked."):
        rows, discrepancy = diagnose_user_sim_checks_response(_answer(_fail(reason=reason)), _criteria("c1"))
        assert discrepancy is None and rows[0]["verdict"] == "fail" and rows[0]["could_change_grade"] is True


def test_an_n_a_where_na_when_is_never_is_asked_again_and_otherwise_stands():
    na = _row(verdict="n/a", reason="Rae never came up.")
    rows, discrepancy = diagnose_user_sim_checks_response(_answer(na), _criteria("c1", na_when="never"))
    assert discrepancy.row_problems == ["c1 may not be n/a: it always applies, so answer pass or fail"]
    for criteria in (_criteria("c1"), _criteria("c1", na_when="the assistant never asks about Rae's order")):
        rows, discrepancy = diagnose_user_sim_checks_response(_answer(na), criteria)
        assert discrepancy is None and rows[0]["verdict"] == "n/a" and rows[0]["score"] == 1.0


def test_every_unusable_row_is_named_in_one_go():
    rows, discrepancy = diagnose_user_sim_checks_response(
        _answer(_fail("c1", turns=[]), _row("c2"), _fail("c3", reason=SAYS_NO_FAILURE[0])), _criteria("c1", "c2", "c3"))
    assert rows is None and [p.split(":")[0] for p in discrepancy.row_problems] == ["c1", "c3"]
    assert discrepancy.to_dict()["row_problems"] == discrepancy.row_problems
    assert str(discrepancy) == "; ".join(discrepancy.row_problems)


def test_assistant_asked_is_kept_when_given_and_never_changes_the_score():
    rows, discrepancy = diagnose_user_sim_checks_response(
        _answer(_fail("c1", assistant_asked=True), _row("c2", assistant_asked="yes"), _row("c3")),
        _criteria("c1", "c2", "c3"))
    assert discrepancy is None
    assert rows[0]["assistant_asked"] is True and rows[0]["score"] == 0.0
    assert "assistant_asked" not in rows[1] and "assistant_asked" not in rows[2]


def test_parse_response_never_turns_an_unusable_answer_into_fails():
    with pytest.raises(ValueError, match="user_sim_checks answer not usable: invalid JSON"):
        SPEC.parse_response("not json", _criteria("c1"))
    assert SPEC.parse_response(_answer(_row("c1")), _criteria("c1"))[0]["result"] is True


# ---- the correction prompt ----------------------------------------------------------------------------------------
def test_the_correction_prompt_names_each_problem_and_asks_for_the_row_shape_again():
    previous = _answer(_fail("c1", turns=[99]), _row("c2"))
    _, discrepancy = diagnose_user_sim_checks_response(previous, _criteria("c1", "c2", graded=[1]))
    prompt = SPEC.format_correction_prompt(eval_prompt="EVAL", previous_response=previous, discrepancy=discrepancy)
    assert prompt.startswith("EVAL\n\n## Correction Required")
    assert "- c1: a fail must cite at least one graded turn; cited [99], graded [1]" in prompt
    assert "exactly 2 objects" in prompt and "Required criterion ids (in any order): c1, c2" in prompt
    for field in ('"id"', '"reason": at most 40 words', '"turns"', '"verdict"', '"could_change_grade"'):
        assert f"- {field}" in prompt
    assert '"score"' not in prompt and '"justification"' not in prompt
    assert prompt.endswith("## Your Previous Response\n" + previous)


def test_the_correction_prompt_for_unreadable_json_or_a_wrong_row_set():
    _, unreadable = diagnose_user_sim_checks_response("no json here", _criteria("c1"))
    prompt = format_user_sim_checks_correction(eval_prompt="EVAL", previous_response="no json here",
                                               discrepancy=unreadable)
    assert "Your response could not be read: the answer is not a JSON object" in prompt
    _, wrong = diagnose_user_sim_checks_response(_answer(_row("c1"), _row("c9")), _criteria("c1", "c2"))
    prompt = format_user_sim_checks_correction(eval_prompt="EVAL", previous_response="x" * 9000, discrepancy=wrong)
    assert "Missing result(s) for criterion id(s): c2" in prompt and "Unknown criterion id(s) in your results: c9" in prompt
    assert prompt.endswith("... [truncated]")


# ---- the score ----------------------------------------------------------------------------------------------------
def _score(*rows: dict, criteria: list[dict], aggregator=ScoreAggregator.WEIGHTED_AVERAGE) -> float:
    got, discrepancy = diagnose_user_sim_checks_response(_answer(*rows), criteria)
    assert discrepancy is None
    return aggregate_score(got, aggregator)


def test_a_row_lowers_the_score_only_on_a_fail_that_could_change_the_grade():
    criteria = _criteria("c1", "c2", "c3")
    assert _score(_row("c1"), _row("c2", verdict="n/a"), _row("c3"), criteria=criteria) == 1.0
    assert _score(_fail("c1", could_change_grade=False), _row("c2"), _row("c3"), criteria=criteria) == 1.0
    assert _score(_fail("c1"), _row("c2"), _row("c3"), criteria=criteria) == pytest.approx(2 / 3)
    rows, _ = diagnose_user_sim_checks_response(_answer(_fail("c1", could_change_grade=False)), _criteria("c1"))
    assert (rows[0]["verdict"], rows[0]["score"], rows[0]["result"]) == ("fail", 1.0, True)


def test_a_weight_0_row_is_recorded_but_never_gates():
    criteria = [*_criteria("guidance"), *_criteria("style", weight=0)]
    rows, _ = diagnose_user_sim_checks_response(_answer(_row("guidance"), _fail("style")), criteria)
    style = rows[1]
    assert (style["verdict"], style["could_change_grade"], style["weight"]) == ("fail", True, 0)
    assert (style["score"], style["result"]) == (1.0, True)
    for aggregator in ScoreAggregator:
        assert _score(_row("guidance"), _fail("style"), criteria=criteria, aggregator=aggregator) == 1.0
    assert _score(_fail("guidance"), _fail("style"), criteria=criteria) == 0.0


# ---- in rubrics_verifier's retry loop -----------------------------------------------------------------------------
def _verifier() -> RubricsVerifierTaskStep:
    return RubricsVerifierTaskStep(id="judge-user-sim", version=1, criteria=_criteria("c1", "c2", graded=[1]),
                                   prompt_id="prompt-1", use_trajectory=False, verifier_id="user-sim",
                                   output_format=JudgeOutputFormat.USER_SIM_CHECKS)


async def _run_judge(verifier: RubricsVerifierTaskStep):
    return await verifier._run_judge_with_output_retries(
        eval_prompt="EVAL", criteria=verifier.criteria, context=TaskStepContext(), model="claude-sonnet-4-6",
        judge_a2a_url=None, judge_agent_card={}, judge_agent=None, judge_effort=None,
        judge_max_thinking_tokens=None, use_agent_judge=False)


@pytest.mark.asyncio
async def test_rubrics_verifier_asks_again_for_an_unusable_row_and_keeps_the_usable_answer(monkeypatch):
    verifier = _verifier()
    unusable = _answer(_fail("c1", turns=[99]), _row("c2"))
    usable = _answer(_fail("c1", turns=[1]), _row("c2"))
    judge = AsyncMock(side_effect=[{"response": unusable}, {"response": usable}])
    monkeypatch.setattr(verifier, "_prompt_llm_judge", judge)
    rows, retries, discrepancies, _ = await _run_judge(verifier)
    assert retries == 1 and [r["verdict"] for r in rows] == ["fail", "pass"]
    assert discrepancies[0]["row_problems"] == ["c1: a fail must cite at least one graded turn; cited [99], graded [1]"]
    correction = judge.await_args_list[1].args[0]
    assert correction.startswith("EVAL\n\n## Correction Required") and unusable in correction
    assert aggregate_score(rows, ScoreAggregator.WEIGHTED_AVERAGE) == 0.5


@pytest.mark.asyncio
async def test_rubrics_verifier_names_the_unusable_row_when_the_judge_never_fixes_it(monkeypatch):
    verifier = _verifier()
    monkeypatch.setattr(verifier, "_prompt_llm_judge", AsyncMock(
        return_value={"response": _answer(_fail("c1", turns=[1], reason=SAYS_NO_FAILURE[0]), _row("c2"))}))
    with pytest.raises(ValueError, match=r"Judge output invalid after 3 attempts: c1: a fail whose reason says "
                                         r"'No clear failure found' contradicts itself"):
        await _run_judge(verifier)
