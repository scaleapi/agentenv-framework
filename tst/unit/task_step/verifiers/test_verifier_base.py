"""VerifierTaskStep: the named grade it writes beside a verifier's legacy entries, for each way verify can end."""
import asyncio
import copy
import hashlib
import json
import threading
from types import SimpleNamespace
from typing import ClassVar

import pytest

from agent_env.plugins import _registration
from agent_env.task.step_journal import _apply_forward
from agent_env.task_step.context import TaskStepContext, VerifierNotPassingError
from agent_env.task_step.context_ops import build_context_update_ops
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.verifiers import ScoreAggregator, Verification, VerifierTaskStep, aggregate_score
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
        "verifier_criteria": [{**ROWS[0], "weight": 1.0, "skipped": False}, {**ROWS[1], "weight": 1.0, "skipped": False}],
        "verifier_type": "toy_verifier", "verifier_version": None,
        "verifier_settings": {"scoring_rules": 1, "note": "n"},
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
    assert grade["verifier_criteria"] == [{"id": "check", "score": 1.0, "result": True, "weight": 1.0, "skipped": False}]


def test_a_raising_verifier_writes_a_failed_grade_no_legacy_entry_and_raises_the_same_exception():
    boom = ValueError("criterion #2 has unknown type 'x'")
    context = TaskStepContext()

    with pytest.raises(ValueError) as raised:
        _run(_Toy(outcome=boom), context)

    assert raised.value is boom
    assert "verifications" not in context.metadata
    grade = context.metadata["grades"]["check"]
    assert (grade["verifier_status"], grade["score"], grade["passed"]) == ("failed", 0.0, False)
    assert grade["verifier_error_reasoning"] == "ValueError: criterion #2 has unknown type 'x'"
    assert grade["verifier_criteria"] == [{"id": "check", "score": 0.0, "result": False, "weight": 1.0, "skipped": False}]


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
    assert (grade["verifier_status"], grade["score"], grade["passed"]) == ("skipped", 0.0, False)
    assert grade["verifier_skip_reasoning"] == "no graded turns"
    assert grade["verifier_criteria"][0]["skipped"] is True
    assert metadata["verifications"]["check"] == {"results": [], "score": None, "skipped": "no graded turns"}


def test_legacy_entries_are_written_exactly_as_the_verifier_gives_them():
    entries = {"vid": {"format": "f", "results": ROWS, "score": 0.0, "command": "pytest"},
               "vid-trajectory-mistakes": {"results": [], "score": 1.0}}

    metadata = _run(_Toy(outcome=Verification(criteria=ROWS, legacy_entries=entries), verifier_id="vid")).metadata

    assert json.dumps(metadata["verifications"]) == json.dumps(entries)
    assert list(metadata["grades"]) == ["check"]


@pytest.mark.parametrize("rows", [
    [{"result": True}, {"result": False, "weight": 3}],
    [{"score": 0.5}, {"score": 1.0, "weight": 2}],
    [{"score": 1.0, "result": None}, {"score": 0.0, "result": True}],
    [{"score": 1.0, "weight": 1}, {"score": 0.0, "weight": -1}],
    [{"result": "yes"}, {}],
])
def test_the_grade_rows_aggregate_as_the_rows_the_verifier_gave(rows):
    graded = [verifier_module._row(i, row) for i, row in enumerate(rows)]

    for aggregator in ScoreAggregator:
        assert aggregate_score(graded, aggregator) == aggregate_score(rows, aggregator)
    assert [row["id"] for row in graded] == [str(i) for i in range(len(rows))]


def test_settings_are_the_steps_own_fields_with_large_ones_kept_by_digest():
    note = "x" * 2000
    step = _Toy(outcome=Verification(score=1), note=note, verifier_id="vid", min_passing_score=0.5)

    settings = _run(step).metadata["grades"]["check"]["verifier_settings"]

    body = json.dumps(note, sort_keys=True, separators=(",", ":")).encode()
    assert settings == {"scoring_rules": 1, "min_passing_score": 0.5,
                        "note": {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}}


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

    grade = _run(_Toy(outcome=Verification(score=1, output=output)), TaskStepContext(instance_id="i1")).metadata[
        "grades"]["check"]

    [(key, (body, content_type))] = store.puts.items()
    assert store.threads != [threading.main_thread()]
    assert key.startswith("pfx/verifier-outputs/instance_id=i1/check-") and content_type == "application/json"
    assert json.loads(body) == output
    assert grade["verifier_output"] is None
    assert grade["verifier_output_ref"] == {"object_url": f"file:///objects/{key}", "bytes": len(body),
                                            "sha256": hashlib.sha256(body).hexdigest()}


def test_an_output_the_store_refuses_stays_inline_and_the_step_still_passes(monkeypatch):
    _with_store(monkeypatch, _Store(fail=True))
    output = {"log": "x" * (verifier_module.OUTPUT_INLINE_BYTES + 1)}

    grade = _run(_Toy(outcome=Verification(score=1, output=output))).metadata["grades"]["check"]

    assert grade["verifier_output"] == output and grade["passed"] is True
    assert grade["verifier_output_offload_error"] == "ConnectionError: store unreachable"


def test_the_grade_and_legacy_entry_survive_the_stored_diff():
    pre = TaskStepContext(metadata={"verifications": {"earlier": {"results": [], "score": 1.0}}})
    post = copy.deepcopy(pre)
    asyncio.run(_Toy(outcome=Verification(criteria=ROWS, output={"log": "ok"}), verifier_id="vid").execute(post))

    doc = {"context": pre.to_safe_dict()}
    _apply_forward(doc, build_context_update_ops(pre, post))

    assert doc["context"]["metadata"]["grades"] == post.metadata["grades"]
    assert doc["context"]["metadata"]["verifications"] == post.metadata["verifications"]


def test_a_verifier_must_implement_verify():
    class _NoVerify(VerifierTaskStep):
        type: ClassVar[str] = "no_verify"

        @classmethod
        def from_dict(cls, data):
            return cls(**cls._base_from_dict(data))

    assert _registration.unimplemented(_NoVerify, TaskStep, ("from_dict",)).endswith("_NoVerify must implement verify")
