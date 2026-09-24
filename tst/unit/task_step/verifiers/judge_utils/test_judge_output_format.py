"""Unit tests for the judge output format registry."""

import json

import pytest

from agent_env.task_step.task_steps.verifiers.judge_utils.judge_output_format import (
    PARTIAL_PASS_THRESHOLD,
    JUDGE_OUTPUT_FORMAT_REGISTRY,
    JudgeOutputFormat,
    diagnose_rubric_binary_response,
    diagnose_rubric_partial_response,
    format_partial_judge_correction_prompt,
    format_judge_correction_prompt,
    get_judge_output_format_spec,
)
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator, aggregate_score


class TestJudgeOutputFormatRegistry:
    def test_rubric_binary_registered(self):
        assert JudgeOutputFormat.RUBRIC_BINARY in JUDGE_OUTPUT_FORMAT_REGISTRY

    def test_rubric_binary_schema_shape(self):
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_BINARY)
        schema = spec.output_format
        assert schema["type"] == "json_schema"
        items = schema["schema"]["properties"]["results"]["items"]
        assert set(items["properties"]) == {"id", "score", "justification"}
        assert items["required"] == ["id", "score", "justification"]
        assert callable(spec.parse_response)
        assert callable(spec.diagnose_response)
        assert callable(spec.format_correction_prompt)

    def test_get_spec_accepts_string_key(self):
        spec = get_judge_output_format_spec("rubric_binary")
        assert spec is JUDGE_OUTPUT_FORMAT_REGISTRY[JudgeOutputFormat.RUBRIC_BINARY]

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown judge output format"):
            get_judge_output_format_spec("not_a_format")


def _criteria(ids: list[str]) -> list[dict]:
    return [{"id": cid, "title": f"Test {cid}"} for cid in ids]


class TestNegativeCriterionPromptHardening:
    """The `"negative": true` instruction must spell out the reasoning→score
    mapping as an explicit two-step, anti-flip procedure: the judge was
    producing correct reasoning but the inverted (wrong) binary score, and grading
    near-identical outputs inconsistently."""

    def _render(self, fmt: JudgeOutputFormat) -> str:
        spec = get_judge_output_format_spec(fmt)
        return spec.prompt_template_no_trajectory.substitute(
            agent_prompt="p", agent_response="r", criteria_json="[]",
        )

    def test_binary_negative_two_step_and_selfcheck(self):
        rendered = self._render(JudgeOutputFormat.RUBRIC_BINARY)
        # explicit present/absent verdict step + mechanical mapping
        assert "PRESENT or ABSENT" in rendered
        assert "ABSENT → 1.0" in rendered
        # anti-flip self-consistency + determinism
        assert "Your score MUST match the verdict" in rendered
        assert "grade near-identical outputs identically" in rendered

    def test_binary_behavior_and_response_share_negative_note(self):
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_BINARY)
        behavior = spec.prompt_template_path.substitute(
            agent_prompt="p", trajectory_path="t", agent_response="r", criteria_json="[]",
        )
        response = spec.prompt_template_no_trajectory.substitute(
            agent_prompt="p", agent_response="r", criteria_json="[]",
        )
        for rendered in (behavior, response):
            assert "ABSENT → 1.0" in rendered
            assert "Your score MUST match the verdict" in rendered

    def test_partial_negative_two_step_and_selfcheck(self):
        rendered = self._render(JudgeOutputFormat.RUBRIC_PARTIAL)
        assert "ABSENT → 1.0" in rendered
        assert "HIGHER score means MORE absent" in rendered
        assert "Your score MUST agree with that verdict" in rendered


class TestRubricBinaryParser:
    def test_all_criteria_returned(self):
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_BINARY)
        response = json.dumps({"results": [
            {"id": "c1", "score": 1.0, "justification": "ok"},
            {"id": "c2", "score": 0.0, "justification": "nope"},
        ]})
        results = spec.parse_response(response, _criteria(["c1", "c2"]))
        assert len(results) == 2
        assert results[0]["result"] is True
        assert results[1]["result"] is False

    def test_missing_criteria_raises(self):
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_BINARY)
        response = json.dumps({"results": [
            {"id": "c1", "score": 1.0, "justification": "ok"},
        ]})
        with pytest.raises(ValueError, match=r"expected \d+ criteria"):
            spec.parse_response(response, _criteria(["c1", "c2", "c3"]))

    def test_malformed_json_returns_all_failures(self):
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_BINARY)
        results = spec.parse_response("not json at all", _criteria(["c1", "c2"]))
        assert len(results) == 2
        assert all(r["result"] is False for r in results)
        assert all(r["score"] == 0.0 for r in results)


class TestDiagnoseRubricBinaryResponse:
    def test_valid_response_returns_rows(self):
        response = json.dumps({"results": [
            {"id": "c1", "score": 1.0, "justification": "ok"},
            {"id": "c2", "score": 0.0, "justification": "nope"},
        ]})
        results, discrepancy = diagnose_rubric_binary_response(response, _criteria(["c1", "c2"]))
        assert discrepancy is None
        assert results is not None
        assert len(results) == 2

    def test_missing_ids_reported(self):
        response = json.dumps({"results": [
            {"id": "c1", "score": 1.0, "justification": "ok"},
        ]})
        results, discrepancy = diagnose_rubric_binary_response(response, _criteria(["c1", "c2", "c3"]))
        assert results is None
        assert discrepancy is not None
        assert discrepancy.missing_ids == ["c2", "c3"]
        assert discrepancy.duplicate_ids == []
        assert discrepancy.unknown_ids == []

    def test_duplicate_ids_reported(self):
        response = json.dumps({"results": [
            {"id": "c1", "score": 1.0, "justification": "first"},
            {"id": "c1", "score": 0.0, "justification": "duplicate"},
            {"id": "c2", "score": 0.0, "justification": "ok"},
        ]})
        results, discrepancy = diagnose_rubric_binary_response(response, _criteria(["c1", "c2"]))
        assert results is None
        assert discrepancy is not None
        assert discrepancy.duplicate_ids == ["c1"]
        assert discrepancy.missing_ids == []
        assert discrepancy.unknown_ids == []

    def test_unknown_ids_reported(self):
        response = json.dumps({"results": [
            {"id": "c1", "score": 1.0, "justification": "ok"},
            {"id": "c_extra", "score": 0.0, "justification": "bonus"},
        ]})
        results, discrepancy = diagnose_rubric_binary_response(response, _criteria(["c1"]))
        assert results is None
        assert discrepancy is not None
        assert discrepancy.unknown_ids == ["c_extra"]

    def test_extra_non_string_ids_reported(self):
        response = json.dumps({"results": [
            {"id": "c1", "score": 1.0, "justification": "ok"},
            {"id": "c2", "score": 0.0, "justification": "ok"},
            {"id": None, "score": 0.0, "justification": "extra"},
        ]})
        results, discrepancy = diagnose_rubric_binary_response(response, _criteria(["c1", "c2"]))
        assert results is None
        assert discrepancy is not None
        assert discrepancy.returned_count == 3
        assert discrepancy.expected_count == 2

    def test_malformed_json_returns_parse_discrepancy(self):
        results, discrepancy = diagnose_rubric_binary_response("not json", _criteria(["c1", "c2"]))
        assert results is None
        assert discrepancy is not None
        assert discrepancy.parse_error is not None
        assert discrepancy.returned_count == 0

    def test_unexpected_shape_returns_parse_discrepancy(self):
        results, discrepancy = diagnose_rubric_binary_response(
            json.dumps({"not_results": []}), _criteria(["c1"]),
        )
        assert results is None
        assert discrepancy is not None
        assert "unexpected response shape" in (discrepancy.parse_error or "")

    def test_discrepancy_to_dict(self):
        response = json.dumps({"results": []})
        _, discrepancy = diagnose_rubric_binary_response(response, _criteria(["c1", "c2"]))
        assert discrepancy is not None
        assert discrepancy.to_dict() == {
            "expected_count": 2,
            "returned_count": 0,
            "expected_ids": ["c1", "c2"],
            "missing_ids": ["c1", "c2"],
            "duplicate_ids": [],
            "unknown_ids": [],
        }


class TestFormatJudgeCorrectionPrompt:
    def test_includes_eval_prompt_and_problem_ids(self):
        response = json.dumps({"results": [
            {"id": "c1", "score": 1.0, "justification": "ok"},
        ]})
        _, discrepancy = diagnose_rubric_binary_response(response, _criteria(["c1", "c2"]))
        assert discrepancy is not None

        prompt = format_judge_correction_prompt(
            eval_prompt="Evaluate this agent.",
            previous_response=response,
            discrepancy=discrepancy,
        )
        assert prompt.startswith("Evaluate this agent.")
        assert "Correction Required" in prompt
        assert "Missing result(s) for criterion id(s): c2" in prompt
        assert "Required criterion ids (in any order): c1, c2" in prompt
        assert response in prompt

    def test_parse_error_prompt(self):
        _, discrepancy = diagnose_rubric_binary_response("Here is my answer: not json", _criteria(["c1"]))
        assert discrepancy is not None
        prompt = format_judge_correction_prompt(
            eval_prompt="Evaluate this agent.",
            previous_response="Here is my answer: not json",
            discrepancy=discrepancy,
        )
        assert "not valid JSON" in prompt
        assert "no preamble" in prompt


class TestRubricPartialFormat:
    def test_partial_registered(self):
        assert JudgeOutputFormat.RUBRIC_PARTIAL in JUDGE_OUTPUT_FORMAT_REGISTRY

    def test_get_spec_accepts_string_key(self):
        spec = get_judge_output_format_spec("rubric_partial")
        assert spec is JUDGE_OUTPUT_FORMAT_REGISTRY[JudgeOutputFormat.RUBRIC_PARTIAL]

    def test_schema_matches_binary_shape(self):
        # Same wire shape as binary; the difference is prompt + result derivation.
        partial = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL).output_format
        binary = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_BINARY).output_format
        assert partial == binary

    def test_prompt_allows_partial_credit(self):
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL)
        rendered = spec.prompt_template_no_trajectory.substitute(
            agent_prompt="p", agent_response="r", criteria_json="[]",
        )
        assert "Partial credit is allowed" in rendered
        assert "0.0 to 1.0" in rendered

    def test_partial_scores_preserved(self):
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL)
        response = json.dumps({"results": [
            {"id": "c1", "score": 0.75, "justification": "minor gaps"},
            {"id": "c2", "score": 0.25, "justification": "mostly unmet"},
        ]})
        results = spec.parse_response(response, _criteria(["c1", "c2"]))
        assert results[0]["score"] == 0.75
        assert results[1]["score"] == 0.25

    def test_result_boolean_from_threshold(self):
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL)
        response = json.dumps({"results": [
            {"id": "hi", "score": PARTIAL_PASS_THRESHOLD, "justification": "at threshold"},
            {"id": "lo", "score": PARTIAL_PASS_THRESHOLD - 0.01, "justification": "below"},
        ]})
        results = spec.parse_response(response, _criteria(["hi", "lo"]))
        by_id = {r["id"]: r for r in results}
        assert by_id["hi"]["result"] is True
        assert by_id["lo"]["result"] is False

    def test_out_of_range_scores_clamped(self):
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL)
        response = json.dumps({"results": [
            {"id": "hi", "score": 1.5, "justification": "over"},
            {"id": "lo", "score": -0.5, "justification": "under"},
        ]})
        results = spec.parse_response(response, _criteria(["hi", "lo"]))
        by_id = {r["id"]: r for r in results}
        assert by_id["hi"]["score"] == 1.0
        assert by_id["lo"]["score"] == 0.0

    def test_labeled_outcomes_score_passes_through(self):
        # Discrete multi-outcome: criterion carries `outcomes`; judge returns the
        # chosen outcome's score. The parser preserves both the value and the
        # passthrough `outcomes`/`weight` fields (needed for weighted_average).
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL)
        criteria = [{
            "id": "quality", "title": "Report quality", "weight": 2.0,
            "outcomes": [
                {"label": "excellent", "score": 1.0},
                {"label": "good", "score": 0.67},
                {"label": "fair", "score": 0.33},
                {"label": "poor", "score": 0.0},
            ],
        }]
        response = json.dumps({"results": [
            {"id": "quality", "score": 0.67, "justification": "good, not excellent"},
        ]})
        results = spec.parse_response(response, criteria)
        assert results[0]["score"] == 0.67
        assert results[0]["weight"] == 2.0
        assert len(results[0]["outcomes"]) == 4

    def test_malformed_json_returns_all_failures(self):
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL)
        results = spec.parse_response("not json", _criteria(["c1", "c2"]))
        assert all(r["result"] is False and r["score"] == 0.0 for r in results)

    def test_prompt_documents_minmax_range(self):
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL)
        rendered = spec.prompt_template_no_trajectory.substitute(
            agent_prompt="p", agent_response="r", criteria_json="[]",
        )
        assert '"min" and "max"' in rendered
        assert "do NOT rescale" in rendered

    def test_continuous_range_normalized_to_unit(self):
        # RFP-style continuous scale: outcomes {min:-1 "ugly", max:1 "beautiful"}.
        # The judge's raw value on [-1,1] is min-max normalized into [0,1].
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL)
        crit = [{
            "id": "beauty", "title": "Beauty",
            "outcomes": {"min": {"value": -1.0, "label": "ugly"},
                         "max": {"value": 1.0, "label": "beautiful"}},
        }]
        for raw, expected in [(-1.0, 0.0), (0.0, 0.5), (1.0, 1.0), (0.5, 0.75)]:
            rows = spec.parse_response(
                json.dumps({"results": [{"id": "beauty", "score": raw, "justification": "j"}]}), crit,
            )
            assert rows[0]["score"] == pytest.approx(expected)
            assert rows[0]["raw_score"] == raw  # original value preserved for display

    def test_continuous_range_clamps_out_of_range(self):
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL)
        crit = [{"id": "b", "title": "B", "outcomes": {"min": {"value": -1.0}, "max": {"value": 1.0}}}]
        rows = spec.parse_response(
            json.dumps({"results": [{"id": "b", "score": -5.0, "justification": "j"}]}), crit,
        )
        assert rows[0]["score"] == 0.0
        assert rows[0]["raw_score"] == -5.0

    def test_flat_minmax_shape_tolerated(self):
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL)
        crit = [{"id": "b", "title": "B", "outcomes": {"min": -1.0, "max": 1.0}}]
        rows = spec.parse_response(
            json.dumps({"results": [{"id": "b", "score": 0.0, "justification": "j"}]}), crit,
        )
        assert rows[0]["score"] == pytest.approx(0.5)

    def test_degenerate_range_warns_and_falls_back_to_clamp(self, caplog):
        # max == min: no normalization possible; warn, keep raw_score, clamp raw.
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL)
        crit = [{"id": "b", "title": "B", "outcomes": {"min": {"value": 1.0}, "max": {"value": 1.0}}}]
        with caplog.at_level("WARNING"):
            rows = spec.parse_response(
                json.dumps({"results": [{"id": "b", "score": 0.3, "justification": "j"}]}), crit,
            )
        assert rows[0]["score"] == pytest.approx(0.3)
        assert rows[0]["raw_score"] == 0.3
        assert "unusable continuous range" in caplog.text

    def test_inverted_range_warns_and_falls_back(self, caplog):
        # min > max is unusable: don't silently misread the judge's on-scale value.
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL)
        crit = [{"id": "b", "title": "B",
                 "outcomes": {"min": {"value": 1.0}, "max": {"value": -1.0}}}]
        with caplog.at_level("WARNING"):
            rows = spec.parse_response(
                json.dumps({"results": [{"id": "b", "score": 0.5, "justification": "j"}]}), crit,
            )
        assert rows[0]["raw_score"] == 0.5          # on-scale value preserved
        assert rows[0]["score"] == pytest.approx(0.5)  # clamped fallback, not normalized
        assert "unusable continuous range" in caplog.text

    def test_discrete_score_snaps_to_nearest_outcome(self, caplog):
        # Enforce the discrete constraint: an off-list value snaps to the nearest.
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL)
        crit = [{"id": "q", "title": "Q", "outcomes": [
            {"label": "excellent", "score": 1.0}, {"label": "good", "score": 0.67},
            {"label": "fair", "score": 0.33}, {"label": "poor", "score": 0.0}]}]
        with caplog.at_level("WARNING"):
            rows = spec.parse_response(
                json.dumps({"results": [{"id": "q", "score": 0.6, "justification": "j"}]}), crit,
            )
        assert rows[0]["score"] == pytest.approx(0.67)   # nearest listed outcome
        assert "snapping to nearest" in caplog.text

    def test_discrete_exact_outcome_no_warning(self, caplog):
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL)
        crit = [{"id": "q", "title": "Q", "outcomes": [
            {"label": "good", "score": 0.67}, {"label": "poor", "score": 0.0}]}]
        with caplog.at_level("WARNING"):
            rows = spec.parse_response(
                json.dumps({"results": [{"id": "q", "score": 0.67, "justification": "j"}]}), crit,
            )
        assert rows[0]["score"] == 0.67
        assert "snapping" not in caplog.text

    def test_negative_criterion_partial_score(self):
        # negative: true is a prompt-level semantic; the parser just stores the
        # partial score (partials allowed) and clamps. Penalty is applied later by
        # weighted_average via the negative weight.
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL)
        crit = [{"id": "no_hallucination", "title": "Invents unsupported claims",
                 "weight": -1.0, "negative": True}]
        rows = spec.parse_response(
            json.dumps({"results": [{"id": "no_hallucination", "score": 0.3, "justification": "j"}]}), crit)
        assert rows[0]["score"] == pytest.approx(0.3)      # partial accepted
        assert rows[0]["result"] is False                  # 0.3 < 0.5 threshold
        # clamping still applies to a negative criterion
        over = spec.parse_response(
            json.dumps({"results": [{"id": "no_hallucination", "score": 1.4, "justification": "j"}]}), crit)
        assert over[0]["score"] == 1.0
        # penalty flows through aggregation: (1 - 0.3) * |−1| over empty positive set -> 0.0
        assert aggregate_score(rows, ScoreAggregator.WEIGHTED_AVERAGE) == 0.0

    def test_count_mismatch_raises(self):
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL)
        response = json.dumps({"results": [{"id": "c1", "score": 0.5, "justification": "x"}]})
        with pytest.raises(ValueError, match=r"expected \d+ criteria"):
            spec.parse_response(response, _criteria(["c1", "c2"]))

    def test_diagnose_reports_missing_ids(self):
        response = json.dumps({"results": [{"id": "c1", "score": 0.5, "justification": "x"}]})
        results, discrepancy = diagnose_rubric_partial_response(response, _criteria(["c1", "c2"]))
        assert results is None
        assert discrepancy is not None and discrepancy.missing_ids == ["c2"]

    def test_correction_prompt_allows_fractional(self):
        response = json.dumps({"results": [{"id": "c1", "score": 0.5, "justification": "x"}]})
        _, discrepancy = diagnose_rubric_partial_response(response, _criteria(["c1", "c2"]))
        assert discrepancy is not None
        prompt = format_partial_judge_correction_prompt(
            eval_prompt="Evaluate.", previous_response=response, discrepancy=discrepancy,
        )
        assert "0.0 to 1.0" in prompt
        assert "exactly 1.0 or 0.0" not in prompt


class TestPartialPassThreshold:
    """Per-criterion `pass_threshold` controls the boolean `result`."""

    def test_default_threshold_when_absent(self):
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL)
        rows = spec.parse_response(json.dumps({"results": [
            {"id": "a", "score": PARTIAL_PASS_THRESHOLD, "justification": "j"},
            {"id": "b", "score": PARTIAL_PASS_THRESHOLD - 0.01, "justification": "j"},
        ]}), _criteria(["a", "b"]))
        by_id = {r["id"]: r for r in rows}
        assert by_id["a"]["result"] is True
        assert by_id["b"]["result"] is False

    def test_unit_scale_custom_threshold(self):
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL)
        crit = [{"id": "strict", "title": "S", "pass_threshold": 0.8}]
        below = spec.parse_response(
            json.dumps({"results": [{"id": "strict", "score": 0.7, "justification": "j"}]}), crit)
        above = spec.parse_response(
            json.dumps({"results": [{"id": "strict", "score": 0.85, "justification": "j"}]}), crit)
        assert below[0]["result"] is False   # 0.7 < 0.8
        assert above[0]["result"] is True    # 0.85 >= 0.8

    def test_ranged_threshold_read_in_raw_units(self):
        # pass_threshold is on the criterion's own [-1,1] scale; normalized like the score.
        # threshold raw 0.0 -> normalized 0.5; score raw 0.5 -> normalized 0.75 (passes).
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL)
        crit = [{
            "id": "beauty", "title": "Beauty", "pass_threshold": 0.0,
            "outcomes": {"min": {"value": -1.0}, "max": {"value": 1.0}},
        }]
        passing = spec.parse_response(
            json.dumps({"results": [{"id": "beauty", "score": 0.5, "justification": "j"}]}), crit)
        failing = spec.parse_response(
            json.dumps({"results": [{"id": "beauty", "score": -0.5, "justification": "j"}]}), crit)
        assert passing[0]["result"] is True    # raw 0.5 (norm 0.75) >= raw 0.0 (norm 0.5)
        assert failing[0]["result"] is False   # raw -0.5 (norm 0.25) < norm 0.5

    def test_invalid_threshold_falls_back_to_default(self):
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL)
        crit = [{"id": "a", "title": "A", "pass_threshold": "not-a-number"}]
        rows = spec.parse_response(
            json.dumps({"results": [{"id": "a", "score": 0.6, "justification": "j"}]}), crit)
        assert rows[0]["result"] is True  # 0.6 >= default 0.5

    def test_threshold_gates_all_pass(self):
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL)
        crit = [
            {"id": "a", "title": "A", "pass_threshold": 0.9},
            {"id": "b", "title": "B"},  # default 0.5
        ]
        rows = spec.parse_response(json.dumps({"results": [
            {"id": "a", "score": 0.85, "justification": "j"},  # below its 0.9 bar
            {"id": "b", "score": 0.85, "justification": "j"},  # above default 0.5
        ]}), crit)
        assert aggregate_score(rows, ScoreAggregator.ALL_PASS) == 0.0  # 'a' fails its threshold


class TestPartialAggregation:
    """End-to-end: partial rows must survive into a fractional aggregate score."""

    def test_weighted_average_yields_fraction(self):
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL)
        criteria = [
            {"id": "a", "title": "A", "weight": 3.0},
            {"id": "b", "title": "B", "weight": 1.0},
        ]
        response = json.dumps({"results": [
            {"id": "a", "score": 1.0, "justification": "full"},
            {"id": "b", "score": 0.0, "justification": "none"},
        ]})
        rows = spec.parse_response(response, criteria)
        score = aggregate_score(rows, ScoreAggregator.WEIGHTED_AVERAGE)
        assert score == pytest.approx(0.75)  # (1.0*3 + 0.0*1) / 4

    def test_all_pass_is_threshold_based_and_never_fractional(self):
        # For partial rows `result` is threshold-based, so all_pass means "every
        # criterion cleared PARTIAL_PASS_THRESHOLD" — and it always yields 0.0/1.0,
        # never a fraction. (Use weighted_average to keep partial credit.)
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL)
        mixed = spec.parse_response(json.dumps({"results": [
            {"id": "a", "score": 0.9, "justification": "clears"},
            {"id": "b", "score": 0.4, "justification": "below threshold"},
        ]}), _criteria(["a", "b"]))
        assert aggregate_score(mixed, ScoreAggregator.ALL_PASS) == 0.0  # b fails threshold

        all_clear = spec.parse_response(json.dumps({"results": [
            {"id": "a", "score": 0.9, "justification": "clears"},
        ]}), _criteria(["a"]))
        assert aggregate_score(all_clear, ScoreAggregator.ALL_PASS) == 1.0  # boolean, not 0.9

    def test_range_criterion_aggregates_on_normalized_score(self):
        # A [-1,1] criterion rated 0.5 normalizes to 0.75, which is what feeds
        # weighted_average (not the raw 0.5).
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_PARTIAL)
        crit = [{"id": "b", "title": "B", "weight": 1.0,
                 "outcomes": {"min": {"value": -1.0}, "max": {"value": 1.0}}}]
        rows = spec.parse_response(
            json.dumps({"results": [{"id": "b", "score": 0.5, "justification": "j"}]}), crit,
        )
        assert aggregate_score(rows, ScoreAggregator.WEIGHTED_AVERAGE) == pytest.approx(0.75)
