"""Unit tests for score aggregation and judge response parsing."""

import json

import pytest

from agent_env.task_step.task_steps.verifiers.judge_utils.judge_output_format import JudgeOutputFormat, get_judge_output_format_spec
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator, aggregate_score
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep


def _criterion(id: str, result):
    return {"id": id, "description": f"Test {id}", "result": result, "depends_on": []}


class TestAllPassAggregator:
    def test_all_pass(self):
        results = [_criterion("a", True), _criterion("b", True)]
        assert aggregate_score(results, ScoreAggregator.ALL_PASS) == 1.0

    def test_one_failure(self):
        results = [_criterion("a", True), _criterion("b", False)]
        assert aggregate_score(results, ScoreAggregator.ALL_PASS) == 0.0

    def test_all_failures(self):
        results = [_criterion("a", False), _criterion("b", False)]
        assert aggregate_score(results, ScoreAggregator.ALL_PASS) == 0.0

    def test_empty_results(self):
        assert aggregate_score([], ScoreAggregator.ALL_PASS) == 0.0

    def test_none_result_treated_as_failure(self):
        results = [_criterion("a", True), _criterion("b", None)]
        assert aggregate_score(results, ScoreAggregator.ALL_PASS) == 0.0

    def test_string_result_treated_as_failure(self):
        results = [_criterion("a", "skipped")]
        assert aggregate_score(results, ScoreAggregator.ALL_PASS) == 0.0

    def test_truthy_int_treated_as_failure(self):
        results = [_criterion("a", 1)]
        assert aggregate_score(results, ScoreAggregator.ALL_PASS) == 0.0


def _criteria(ids: list[str]) -> list[dict]:
    return [{"id": cid, "title": f"Test {cid}"} for cid in ids]


class TestParseJudgeResponse:
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

    def test_empty_results_raises(self):
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_BINARY)
        response = json.dumps({"results": []})
        with pytest.raises(ValueError, match=r"expected \d+ criteria"):
            spec.parse_response(response, _criteria(["c1", "c2"]))

    def test_extra_criteria_raises(self):
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_BINARY)
        response = json.dumps({"results": [
            {"id": "c1", "score": 1.0, "justification": "ok"},
            {"id": "c_extra", "score": 0.0, "justification": "bonus"},
        ]})
        with pytest.raises(ValueError, match=r"expected \d+ criteria"):
            spec.parse_response(response, _criteria(["c1"]))

    def test_malformed_json_returns_all_failures(self):
        spec = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_BINARY)
        results = spec.parse_response("not json at all", _criteria(["c1", "c2"]))
        assert len(results) == 2
        assert all(r["result"] is False for r in results)
        assert all(r["score"] == 0.0 for r in results)


class TestRubricsVerifierSerialization:
    def test_agent_name_none_round_trip(self):
        verifier = RubricsVerifierTaskStep(
            id="test", version=None, criteria=[{"id": "c1"}], prompt_id="p1"
        )
        data = verifier.to_dict()
        assert data["agent_name"] is None
        restored = RubricsVerifierTaskStep.from_dict(data)
        assert restored.agent_name is None

    def test_agent_name_set_round_trip(self):
        verifier = RubricsVerifierTaskStep(
            id="test", version=None, criteria=[{"id": "c1"}], prompt_id="p1",
            agent_name="judge-agent"
        )
        data = verifier.to_dict()
        assert data["agent_name"] == "judge-agent"
        restored = RubricsVerifierTaskStep.from_dict(data)
        assert restored.agent_name == "judge-agent"

    def test_output_format_round_trip(self):
        verifier = RubricsVerifierTaskStep(
            id="test", version=None, criteria=[{"id": "c1"}], prompt_id="p1",
            output_format=JudgeOutputFormat.RUBRIC_BINARY,
        )
        data = verifier.to_dict()
        assert data["output_format"] == "rubric_binary"
        restored = RubricsVerifierTaskStep.from_dict(data)
        assert restored.output_format == JudgeOutputFormat.RUBRIC_BINARY

    def test_legacy_data_without_output_format(self):
        data = {
            "id": "test", "type": "rubrics_verifier", "version": 1,
            "criteria": [{"id": "c1"}], "prompt_id": "p1",
            "default_model": "claude-sonnet-4-6",
        }
        restored = RubricsVerifierTaskStep.from_dict(data)
        assert restored.output_format == JudgeOutputFormat.RUBRIC_BINARY

    def test_legacy_data_without_agent_name(self):
        data = {
            "id": "test", "type": "rubrics_verifier", "version": 1,
            "criteria": [{"id": "c1"}], "prompt_id": "p1",
            "default_model": "claude-sonnet-4-6",
        }
        restored = RubricsVerifierTaskStep.from_dict(data)
        assert restored.agent_name is None
