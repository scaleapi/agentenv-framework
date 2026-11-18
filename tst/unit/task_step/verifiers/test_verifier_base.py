"""VerifierTaskStep: the named grade it writes beside a verifier's legacy entries, for each way verify can end."""
import asyncio
import hashlib
import json
import threading
from types import SimpleNamespace
from typing import ClassVar

import pytest

from agent_env.plugins import _registration
from agent_env.task.step_journal import replay_context
from agent_env.task_step.context import TaskStepContext, VerifierNotPassingError
from agent_env.task_step.context_ops import build_context_update_ops
from agent_env.task_step.task_step import StepRole, TaskStep
from agent_env.task_step.task_steps.verifiers import (
    Grade, ScoreAggregator, Verification, VerifierTaskStep, aggregate_score,
)
from agent_env.task_step.task_steps.verifiers import verifier as verifier_module


class _Toy(VerifierTaskStep):
    type: ClassVar[str] = "toy_verifier"

    def __init__(self, id="check", version=None, outcome=None, note="", **base):
        super().__init__(id, version, **base)
        self.outcome = outcome
        self.note = note

    def to_dict(self):
        return {**super().to_dict(), "note": self.note}

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data), note=data.get("note", ""))

    async def verify(self, context):
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _run(step: VerifierTaskStep, context: TaskStepContext | None = None) -> TaskStepContext:
    context = context or TaskStepContext()
    asyncio.run(step.execute(context))
    return context


ROWS = [{"id": "a", "score": 1.0, "result": True, "why": "found it"}, {"id": "b", "score": 0.0, "result": False}]


def test_a_graded_verification_writes_its_named_grade_and_the_legacy_entry():
    metadata = _run(_Toy(outcome=Verification(criteria=ROWS), verifier_id="vid", note="n")).metadata

    assert metadata["verifications"] == {"vid": {"results": ROWS, "score": 0.0}}
    grade = metadata["grades"]["check"]
    assert grade == {
        "name": "check", "score": 0.0, "min_passing_score": 1.0, "passed": False, "verifier_status": "graded",
        "verifier_error_reasoning": None, "verifier_skip_reasoning": None, "verifier_output": None,
        "verifier_criteria": [
            {"id": "a", "score": 1.0, "why": "found it", "passed": True, "weight": 1.0, "skipped": False},
            {"id": "b", "score": 0.0, "passed": False, "weight": 1.0, "skipped": False},
        ],
        "verifier_type": "toy_verifier", "verifier_version": None,
        "verifier_settings": {"scoring_rules": 1, "note": "n", "score_aggregator": "all_pass"},
    }


def test_the_grade_passes_at_its_min_passing_score():
    rows = [{"id": "a", "score": 1.0}, {"id": "b", "score": 0.5}]
    grade = _run(_Toy(outcome=Verification(criteria=rows), score_aggregator="weighted_average",
                      min_passing_score=0.75)).metadata["grades"]["check"]

    assert (grade["score"], grade["min_passing_score"], grade["passed"]) == (0.75, 0.75, True)


def test_a_direct_score_wins_and_keeps_the_rows_as_reported():
    metadata = _run(_Toy(outcome=Verification(criteria=ROWS, score=0.4))).metadata

    assert metadata["grades"]["check"]["score"] == 0.4
    assert [row["id"] for row in metadata["grades"]["check"]["verifier_criteria"]] == ["a", "b"]
    assert metadata["verifications"]["check"]["score"] == 0.4


def test_a_verification_with_no_rows_gets_one_for_the_grade_itself():
    grade = _run(_Toy(outcome=Verification(score=1))).metadata["grades"]["check"]

    assert grade["score"] == 1.0 and grade["passed"] is True
    assert grade["verifier_criteria"] == [{"id": "check", "score": 1.0, "passed": True, "weight": 1.0, "skipped": False}]


def test_a_raising_verifier_writes_a_failed_grade_no_legacy_entry_and_raises_the_same_exception():
    boom = ValueError("criterion #2 has unknown type 'x'")
    context = TaskStepContext()

    with pytest.raises(ValueError) as raised:
        _run(_Toy(outcome=boom), context)

    assert raised.value is boom
    assert "verifications" not in context.metadata
    grade = context.metadata["grades"]["check"]
    assert (grade["verifier_status"], grade["score"], grade["passed"]) == ("failed", None, False)
    assert grade["verifier_error_reasoning"] == "ValueError: criterion #2 has unknown type 'x'"
    assert grade["verifier_criteria"] == [{"id": "check", "score": None, "passed": False, "weight": 1.0, "skipped": False}]


def test_raise_after_keeps_the_verdict_graded_then_fails_the_step():
    not_passing = VerifierNotPassingError("check: 1 criterion failed")
    context = TaskStepContext()

    with pytest.raises(VerifierNotPassingError) as raised:
        _run(_Toy(outcome=Verification(criteria=ROWS, raise_after=not_passing)), context)

    assert raised.value is not_passing
    assert context.metadata["grades"]["check"]["verifier_status"] == "graded"
    assert context.metadata["verifications"]["check"] == {"results": ROWS, "score": 0.0}


def test_a_skip_is_its_own_status_with_its_reason():
    metadata = _run(_Toy(outcome=Verification(skipped="no graded turns"))).metadata

    grade = metadata["grades"]["check"]
    assert (grade["verifier_status"], grade["score"], grade["passed"]) == ("skipped", None, False)
    assert grade["verifier_skip_reasoning"] == "no graded turns"
    assert grade["verifier_criteria"][0]["skipped"] is True
    assert metadata["verifications"]["check"] == {"results": [], "score": None, "skipped": "no graded turns"}


def test_a_dict_outputs_keys_join_the_legacy_entry_after_its_results_and_score():
    output = {"command": "pytest", "exit_code": 1, "stdout_head": "1 failed"}

    metadata = _run(_Toy(outcome=Verification(criteria=ROWS, output=output), verifier_id="vid")).metadata

    assert json.dumps(metadata["verifications"]) == json.dumps({"vid": {"results": ROWS, "score": 0.0, **output}})
    assert metadata["grades"]["check"]["verifier_output"] == output


def test_an_output_that_isnt_a_dict_stays_off_the_legacy_entry():
    metadata = _run(_Toy(outcome=Verification(score=1, output="3 passed"))).metadata

    assert metadata["verifications"]["check"] == {"results": [], "score": 1}
    assert metadata["grades"]["check"]["verifier_output"] == "3 passed"


@pytest.mark.parametrize("rows", [
    [{"result": True}, {"result": False, "weight": 3}],
    [{"score": 0.5}, {"score": 1.0, "weight": 2}],
    [{"score": 1.0, "result": None}, {"score": 0.0, "result": True}],
    [{"score": 1.0, "weight": 1}, {"score": 0.0, "weight": -1}],
    [{"result": "yes"}, {}],
])
def test_the_grade_rows_pass_and_score_as_scoring_read_the_rows_the_verifier_gave(rows):
    by_rule = {ScoreAggregator.ALL_PASS: all, ScoreAggregator.ANY_PASS: any}
    for aggregator in ScoreAggregator:
        metadata = _run(_Toy(outcome=Verification(criteria=rows), score_aggregator=aggregator)).metadata

        graded, legacy = metadata["grades"]["check"]["verifier_criteria"], metadata["verifications"]["check"]
        if aggregator in by_rule:
            assert legacy["score"] == (1.0 if by_rule[aggregator](row["passed"] for row in graded) else 0.0)
        else:
            assert aggregate_score(graded, aggregator) == legacy["score"]
        assert [row["id"] for row in graded] == [str(i) for i in range(len(rows))]
        assert "result" not in graded[0] and legacy["results"] == rows


@pytest.mark.parametrize("outcome", [Verification(criteria=ROWS), Verification(skipped="nothing to judge"),
                                     ValueError("broken"), Verification(score=1, output={"log": "ok"})])
def test_every_grade_holds_the_fields_its_type_names(outcome):
    context = TaskStepContext()
    try:
        _run(_Toy(outcome=outcome), context)
    except ValueError:
        pass

    keys = set(context.metadata["grades"]["check"])
    assert Grade.__required_keys__ <= keys <= Grade.__required_keys__ | Grade.__optional_keys__


def test_settings_are_the_steps_own_fields_with_large_ones_kept_by_digest():
    note = "x" * 2000
    step = _Toy(outcome=Verification(score=1), note=note, verifier_id="vid", min_passing_score=0.5)

    grade = _run(step).metadata["grades"]["check"]

    body = json.dumps(note, sort_keys=True, separators=(",", ":")).encode()
    assert grade["verifier_settings"] == {"scoring_rules": 1, "score_aggregator": "all_pass",
                                          "note": {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}}
    assert grade["min_passing_score"] == 0.5


def test_settings_name_the_aggregator_the_step_scores_with_even_at_its_default():
    for aggregator, written in ((None, "all_pass"), ("weighted_average", "weighted_average")):
        settings = _run(_Toy(outcome=Verification(score=1), score_aggregator=aggregator)).metadata["grades"][
            "check"]["verifier_settings"]

        assert settings["score_aggregator"] == written


def test_base_fields_are_serialized_only_when_not_default_and_round_trip():
    assert _Toy().to_dict() == {"id": "check", "type": "toy_verifier", "version": None, "fail_task_on_error": True,
                                "note": ""}
    step = _Toy(verifier_id="vid", score_aggregator="weighted_average", min_passing_score=0.5)
    data = step.to_dict()
    assert (data["verifier_id"], data["score_aggregator"], data["min_passing_score"]) == ("vid", "weighted_average", 0.5)
    again = _Toy.from_dict(data)
    assert (again.verifier_id, again.score_aggregator, again.min_passing_score) == (
        "vid", ScoreAggregator.WEIGHTED_AVERAGE, 0.5)
    assert _Toy.from_dict(_Toy().to_dict()).verifier_id == "check"


@pytest.mark.parametrize("bad", [0, -0.5, 1.5, float("nan")])
def test_a_min_passing_score_outside_zero_to_one_is_a_config_problem(bad):
    assert _Toy(min_passing_score=bad).config_problems() == [
        f"min_passing_score must be above 0 and at most 1, not {float(bad)}"]
    assert _Toy(min_passing_score=1).config_problems() == []


class _Store:
    def __init__(self, fail=False):
        self.fail, self.puts, self.threads = fail, {}, []

    def put(self, key, body, content_type=None):
        self.threads.append(threading.current_thread())
        if self.fail:
            raise ConnectionError("store unreachable")
        self.puts[key] = (body, content_type)
        return f"file:///objects/{key}"


def _with_store(monkeypatch, store):
    monkeypatch.setattr(verifier_module, "get_config", lambda: SimpleNamespace(
        get_artifact_key_prefix=lambda: "pfx/", get_object_store=lambda: store))


def test_a_small_output_stays_on_the_grade():
    grade = _run(_Toy(outcome=Verification(score=1, output={"log": "ok"}))).metadata["grades"]["check"]

    assert grade["verifier_output"] == {"log": "ok"} and "verifier_output_ref" not in grade


def test_a_large_output_goes_to_the_object_store_from_a_worker_thread(monkeypatch):
    store = _Store()
    _with_store(monkeypatch, store)
    output = {"log": "x" * (verifier_module.OUTPUT_INLINE_BYTES + 1)}

    context = _run(_Toy(outcome=Verification(score=1, output=output)), TaskStepContext(instance_id="i1"))
    grade = context.metadata["grades"]["check"]

    [(key, (body, content_type))] = store.puts.items()
    assert store.threads != [threading.main_thread()]
    assert key.startswith("pfx/verifier-outputs/instance_id=i1/check-") and content_type == "application/json"
    assert json.loads(body) == output
    assert grade["verifier_output"] is None
    assert grade["verifier_output_ref"] == {"object_url": f"file:///objects/{key}", "bytes": len(body),
                                            "sha256": hashlib.sha256(body).hexdigest()}
    assert context.metadata["verifications"]["check"]["log"] == output["log"]


def test_an_output_the_store_refuses_stays_inline_and_the_step_still_passes(monkeypatch):
    _with_store(monkeypatch, _Store(fail=True))
    output = {"log": "x" * (verifier_module.OUTPUT_INLINE_BYTES + 1)}

    grade = _run(_Toy(outcome=Verification(score=1, output=output))).metadata["grades"]["check"]

    assert grade["verifier_output"] == output and grade["passed"] is True
    assert grade["verifier_output_offload_error"] == "ConnectionError: store unreachable"


def test_the_grade_and_legacy_entry_survive_the_step_journal():
    post = _run(_Toy(outcome=Verification(criteria=ROWS, output={"log": "ok"}), verifier_id="vid"))

    ops = build_context_update_ops(None, post).to_journal_dict()
    stored = replay_context([{"step_id": "check", "seq": 1, "ops": ops}], [])

    assert stored["metadata"]["grades"] == post.metadata["grades"]
    assert stored["metadata"]["verifications"] == post.metadata["verifications"]


def test_a_verifier_on_the_base_is_a_grader():
    assert _Toy.step_role is StepRole.GRADER and "step_role" not in _Toy().to_dict()


def test_a_verifier_must_implement_verify():
    class _NoVerify(VerifierTaskStep):
        type: ClassVar[str] = "no_verify"

        @classmethod
        def from_dict(cls, data):
            return cls(**cls._base_from_dict(data))

    assert _registration.unimplemented(_NoVerify, TaskStep, ("from_dict",)).endswith("_NoVerify must implement verify")
