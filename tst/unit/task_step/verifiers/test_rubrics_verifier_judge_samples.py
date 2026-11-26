"""judge_samples: a rubric_binary or rubric_partial judge grades each criterion that many times and keeps its median, the
lower middle score on a tie, so a tie fails, the row of the first sample that scored it with every sample's score,
verdict and justification beside it and its agreement; a split under min_judge_agreement (0.5 by default, which never
fails) fails the grade, its rows kept, then the step, and so the task unless the step tolerates it; each
call's record names its sample; the samples share the judge's concurrency; one that fails fails the grade; a judge the
verifier deploys gets one of its own per sample, while the task's own judge takes every sample; a run's count reaches
every judge that can take it, and one that can't grades once and records it as not sent."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import litellm
import pytest
from click.testing import CliRunner

from agent_env.a2a_agent import A2AAgent, protocol
from agent_env.bundle.run import _overrides, run_bundle
from agent_env.cli.run import run as run_command
from agent_env.config import get_config
from agent_env.providers.sandbox_providers import sandbox_provider
from agent_env.task.task import Task
from agent_env.task_step.context import DeployedAgent
from agent_env.task_step.task_steps.verifiers.judge_utils import judge_samples, judge_usage
from agent_env.task_step.task_steps.verifiers.judge_utils.judge_samples import JudgeDisagreementError
from agent_env.task_step.task_steps.verifiers import rubrics_verifier as rv
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from agent_env.task_step.task_steps.verifiers.user_sim_judge import UserSimJudge
from tst.unit.task_step.verifiers import test_rubrics_verifier_golden as G
from tst.unit.task_step.verifiers.test_rubrics_verifier_judge_settings import _grade
from tst.util.config import install_model_config

TWO = G.TWO


def _verdicts(scores: dict[str, float], sample: int) -> str:
    return json.dumps({"results": [{"id": i, "score": s, "justification": f"sample {sample} scored {i} {s}"}
                                   for i, s in scores.items()]})


def _sampled_llm(monkeypatch, by_sample, *, failing_sample=None, slow_s=0.0) -> dict:
    """LiteLLM faked through the configured endpoint, each call answered by its sample's verdicts
    (``by_sample(sample, ids)`` gives the scores), read from the call's own context; ``failing_sample`` is refused on
    every attempt; the calls in flight at once are counted."""
    install_model_config(monkeypatch, base_url="https://gw.test/v1", api_key="configured-key")
    seen = {"calls": [], "in_flight": 0, "most": 0}

    async def acompletion(**kwargs):
        sample = judge_usage._SAMPLE.get()
        prompt = kwargs["messages"][-1]["content"]
        ids = G._ids(prompt if isinstance(prompt, str) else "".join(b.get("text", "") for b in prompt))
        seen["calls"].append((judge_usage._BATCH.get(), sample))
        seen["in_flight"] += 1
        seen["most"] = max(seen["most"], seen["in_flight"])
        try:
            await asyncio.sleep(slow_s)
            if sample is not None and sample == failing_sample:
                raise litellm.RateLimitError("slow down", llm_provider="openai", model="gpt-4o-mini")
            content = _verdicts(by_sample(sample, ids), sample or 1)
        finally:
            seen["in_flight"] -= 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                               usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                               _hidden_params={})

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    return seen


def _two_to_one(sample, ids) -> dict[str, float]:
    """c1 passes on samples 1 and 3 and fails on 2; c2 fails on 1 and passes on 2 and 3."""
    return {"c1": 0.0 if sample == 2 else 1.0, "c2": 0.0 if sample == 1 else 1.0, **{i: 1.0 for i in ids[2:]}}


def test_a_count_is_any_positive_integer():
    assert [judge_samples.validated(n) for n in (None, 1, 2, 3, 4)] == [None, 1, 2, 3, 4]
    for bad in (0, -3, True, 3.0, "3"):
        with pytest.raises(ValueError, match="positive integer"):
            judge_samples.validated(bad)


def test_each_criterion_keeps_the_first_row_that_scored_its_median_and_every_samples_score_verdict_and_reason():
    sampled = [[{"id": "a", "score": s, "result": s >= 0.5, "justification": f"{n}:{s}", "outcome": f"level {s}"}
                for s in [score]] for n, score in enumerate([0.25, 0.75, 0.5], 1)]

    [row] = judge_samples.median_rows(sampled)

    assert (row["score"], row["justification"], row["outcome"], row["agreement"]) == (0.5, "3:0.5", "level 0.5", 0.67)
    assert row["samples"] == [{"score": 0.25, "passed": False, "justification": "1:0.25"},
                              {"score": 0.75, "passed": True, "justification": "2:0.75"},
                              {"score": 0.5, "passed": True, "justification": "3:0.5"}]


def test_an_agreement_bar_is_a_number_from_0_to_1():
    assert [judge_samples.validated_agreement(v) for v in (None, 0, 0.5, 0.67, 1)] == [None, 0.0, 0.5, 0.67, 1.0]
    for bad in (-0.1, 1.5, True, "1", float("nan")):
        with pytest.raises(ValueError, match="min_judge_agreement must be a number from 0 to 1"):
            judge_samples.validated_agreement(bad)


@pytest.mark.parametrize("scores, agreement", [([1.0, 1.0, 0.0], 0.67), ([1.0, 0.0], 0.5), ([0.75, 0.7], 1.0),
                                               ([0.4, 0.6], 0.5), ([1.0, 1.0, 1.0, 0.0, 1.0], 0.8),
                                               ([1.0] * 200 + [0.0], 0.99)])
def test_a_rows_agreement_is_the_share_of_samples_giving_its_verdict_not_its_score(scores, agreement):
    sampled = [[{"id": "a", "score": score, "result": score >= 0.5}] for score in scores]

    [row] = judge_samples.median_rows(sampled)

    assert row["agreement"] == agreement
    assert judge_samples.disagreeing([row], 1.0) == ([] if agreement == 1.0 else [row])
    assert judge_samples.disagreeing([row], agreement) == []


@pytest.mark.parametrize("scores, kept", [([1.0, 0.0], 0.0), ([0.0, 1.0], 0.0), ([1.0, 1.0, 0.0, 1.0], 1.0),
                                          ([1.0, 0.0, 0.0, 1.0], 0.0), ([0.75, 0.25], 0.25)])
def test_an_even_count_s_tie_keeps_the_lower_middle_score_and_the_first_sample_that_gave_it(scores, kept):
    sampled = [[{"id": "a", "score": score, "justification": f"sample {n}"}] for n, score in enumerate(scores, 1)]

    [row] = judge_samples.median_rows(sampled)

    assert row["score"] == kept and row["justification"] == f"sample {scores.index(kept) + 1}"
    assert [each["score"] for each in row["samples"]] == scores


def test_a_task_that_asks_no_samples_or_samples_a_format_without_one_score_a_row_fails_to_load():
    with pytest.raises(ValueError, match="judge_samples must be a positive integer, got 0"):
        RubricsVerifierTaskStep.from_dict({**G._step().to_dict(), "judge_samples": 0})
    with pytest.raises(ValueError, match="min_judge_agreement must be a number from 0 to 1, got 2"):
        RubricsVerifierTaskStep.from_dict({**G._step().to_dict(), "min_judge_agreement": 2})
    with pytest.raises(ValueError, match="min_judge_agreement does not apply to output_format=rubric_evidence"):
        G._evidence_step("rubric_evidence", min_judge_agreement=0.67)
    with pytest.raises(ValueError, match="min_judge_agreement"):
        UserSimJudge.from_dict({"id": "sim", "version": 1, "type": "user_sim_judge", "run_step_id": "week",
                                "min_judge_agreement": 0.5})
    shared = G._step(judge_samples=3, min_judge_agreement=0.67)
    assert RubricsVerifierTaskStep.from_dict(shared.to_dict()).min_judge_agreement == 0.67
    assert "min_judge_agreement" not in G._step(judge_samples=3).to_dict()
    with pytest.raises(ValueError, match="judge_samples does not apply to output_format=rubric_evidence"):
        G._evidence_step("rubric_evidence", judge_samples=3)
    with pytest.raises(ValueError, match="judge_samples"):
        UserSimJudge.from_dict({"id": "sim", "version": 1, "type": "user_sim_judge", "run_step_id": "week",
                                "judge_samples": 3})
    for count in (2, 3):
        assert RubricsVerifierTaskStep.from_dict(G._step(judge_samples=count).to_dict()).judge_samples == count
    assert "judge_samples" not in G._step().to_dict()


@pytest.mark.asyncio
async def test_a_direct_judge_grades_each_criterion_three_times_and_keeps_each_ones_median(monkeypatch):
    seen = _sampled_llm(monkeypatch, _two_to_one)
    context = G._context()

    grade = await _grade(G._step(default_model="gpt-4o-mini", judge_samples=3, min_judge_agreement=0.67), context)

    rows = {row["id"]: row for row in grade["verifier_criteria"]}
    assert (rows["table-reserved"]["score"], rows["party-size"]["score"]) == (1.0, 1.0)
    assert rows["table-reserved"]["agreement"] == rows["party-size"]["agreement"] == 0.67
    assert rows["table-reserved"]["justification"] == "sample 1 scored c1 1.0"
    assert rows["party-size"]["justification"] == "sample 2 scored c2 1.0"
    assert [s["score"] for s in rows["table-reserved"]["samples"]] == [1.0, 0.0, 1.0]
    assert sorted(sample for _, sample in seen["calls"]) == [1, 2, 3]
    assert sorted((c["sample"], c["outcome"]) for c in grade["judge_calls"]) == [(1, "answered"), (2, "answered"),
                                                                                (3, "answered")]
    assert grade["verifier_settings"]["judge_samples"] == 3
    assert grade["verifier_settings"]["min_judge_agreement"] == 0.67
    entry = context.metadata["verifications"]["vid"]
    assert entry["judge_samples"] == 3 and all(len(row["samples"]) == 3 for row in entry["results"])


@pytest.mark.asyncio
async def test_two_samples_that_split_fail_the_criterion_with_the_failing_samples_reason(monkeypatch):
    _sampled_llm(monkeypatch, _two_to_one)

    grade = await _grade(G._step(default_model="gpt-4o-mini", judge_samples=2, min_judge_agreement=0.5), G._context())

    rows = {row["id"]: row for row in grade["verifier_criteria"]}
    assert (rows["table-reserved"]["score"], rows["party-size"]["score"]) == (0.0, 0.0)
    assert rows["table-reserved"]["justification"] == "sample 2 scored c1 0.0"
    assert rows["party-size"]["justification"] == "sample 1 scored c2 0.0"
    assert grade["score"] == 0.0 and grade["verifier_settings"]["judge_samples"] == 2


@pytest.mark.asyncio
async def test_a_judge_of_one_sample_grades_as_before(monkeypatch):
    _sampled_llm(monkeypatch, lambda sample, ids: dict.fromkeys(ids, 1.0))
    context = G._context()

    grade = await _grade(G._step(default_model="gpt-4o-mini"), context)

    assert all("samples" not in row for row in grade["verifier_criteria"])
    assert [c["sample"] for c in grade["judge_calls"]] == [None]
    assert "judge_samples" not in context.metadata["verifications"]["vid"]


@pytest.mark.asyncio
async def test_batches_times_samples_share_the_judges_concurrency(monkeypatch):
    seen = _sampled_llm(monkeypatch, lambda sample, ids: dict.fromkeys(ids, 1.0), slow_s=0.05)
    step = G._step(default_model="gpt-4o-mini", judge_samples=3, judge_batch_size=1, judge_batch_concurrency=2)

    grade = await _grade(step, G._context())

    assert sorted(seen["calls"]) == [(b, s) for b in (1, 2) for s in (1, 2, 3)]
    assert seen["most"] == 2
    assert sorted((c["batch"], c["sample"]) for c in grade["judge_calls"]) == sorted(seen["calls"])
    assert all(len(row["samples"]) == 3 for row in grade["verifier_criteria"])


@pytest.mark.asyncio
async def test_a_sample_out_of_retries_fails_the_grade_with_every_call_recorded(monkeypatch):
    _sampled_llm(monkeypatch, lambda sample, ids: dict.fromkeys(ids, 1.0), failing_sample=2, slow_s=0.01)
    context = G._context()

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        await G._step(default_model="gpt-4o-mini", judge_samples=3).execute(context)

    grade = context.metadata["grades"]["verify"]
    assert grade["verifier_status"] == "failed"
    assert [c["outcome"] for c in grade["judge_calls"] if c["sample"] == 2] == ["refused"] * 3


@pytest.mark.asyncio
async def test_a_runs_count_reaches_a_judge_that_can_take_it_and_says_it_was_overridden(monkeypatch):
    seen = _sampled_llm(monkeypatch, _two_to_one)

    grade = await _grade(G._step(default_model="gpt-4o-mini"),
                         G._context({"judge_samples": 3, "min_judge_agreement": 0.67}))

    assert len(seen["calls"]) == 3 and grade["verifier_status"] == "graded"
    settings = grade["verifier_settings"]
    assert settings["judge_samples"] == 3 and settings["min_judge_agreement"] == 0.67
    assert {"judge_samples", "min_judge_agreement"} <= set(settings["overridden"])


@pytest.mark.asyncio
async def test_a_runs_count_a_format_cant_take_grades_once_and_is_recorded_as_not_sent():
    calls: list[str] = []
    step = G._direct_judge(G._evidence_step("rubric_evidence"),
                           lambda ids, n: calls.append("call") or G._evidence(ids))

    grade = await _grade(step, G._context({"judge_samples": 3, "min_judge_agreement": 0.67}))

    assert calls == ["call"] and grade["verifier_status"] == "graded"
    assert grade["verifier_settings"]["judge_samples"] == 1
    assert grade["verifier_settings"]["not_sent"] == {"judge_samples": 3, "min_judge_agreement": 0.67}
    assert "min_judge_agreement" not in grade["verifier_settings"]


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides, why", [({"judge_samples": 0}, "judge_samples must be a positive integer, got 0"),
                                           ({"judge_samples": 3, "min_judge_agreement": 1.5},
                                            "min_judge_agreement must be a number from 0 to 1, got 1.5")])
async def test_a_runs_bad_count_or_bar_fails_the_grade_saying_why_before_any_call(monkeypatch, overrides, why):
    seen = _sampled_llm(monkeypatch, lambda sample, ids: dict.fromkeys(ids, 1.0))
    context = G._context(overrides)

    with pytest.raises(ValueError, match=why):
        await G._step(default_model="gpt-4o-mini").execute(context)

    assert context.metadata["grades"]["verify"]["verifier_status"] == "failed" and seen["calls"] == []


@pytest.mark.asyncio
async def test_the_tasks_own_agent_judge_takes_every_sample_each_in_a_conversation_of_its_own(monkeypatch):
    sent: list[dict] = []

    async def send(url, **kwargs):
        sent.append({"url": url, "context_id": kwargs.get("context_id"), "sample": judge_usage._SAMPLE.get()})
        return f"task-{len(sent)}", None

    async def poll(url, task_id, **kwargs):
        text = _verdicts(_two_to_one(judge_usage._SAMPLE.get(), ["c1", "c2"]), judge_usage._SAMPLE.get())
        return {"status": {"state": "completed", "message": {"parts": [{"kind": "text", "text": text}]}}}

    monkeypatch.setattr(protocol, "send_a2a_message", send)
    monkeypatch.setattr(protocol, "poll_a2a_task", poll)
    context = G._context()
    context.deployed_agents.append(DeployedAgent(agent_name="judge", api_url="http://judge.test",
                                                 a2a_url="http://judge.test", sandbox_id="sb-judge"))
    step = G._step(use_agent_judge=True, agent_name="judge", judge_samples=3, min_judge_agreement=0.67)

    async def configured(**kwargs):
        return None

    step._configure_judge_a2a = configured
    grade = await _grade(step, context)

    assert sorted(s["sample"] for s in sent) == [1, 2, 3] and {s["context_id"] for s in sent} == {None}
    assert {s["url"] for s in sent} == {"http://judge.test"}
    assert [row["score"] for row in grade["verifier_criteria"]] == [1.0, 1.0]
    assert sorted(c["sample"] for c in grade["judge_calls"]) == [1, 2, 3]


def test_the_run_flags_set_the_runs_count_and_bar_and_refuse_bad_ones(tmp_path):
    assert _overrides(None, None, None, 2) == {"judge_samples": 2}
    assert _overrides(None, None, None, 3, 0.0) == {"judge_samples": 3, "min_judge_agreement": 0.0}
    assert _overrides(None, None, None, None) == {}
    assert judge_samples.overrides(None, 1.0) == {"min_judge_agreement": 1.0}
    assert judge_samples.overrides(5, None) == {"judge_samples": 5}
    for bad in (0, -1):
        with pytest.raises(ValueError, match="judge_samples must be a positive integer"):
            run_bundle(tmp_path, judge_samples=bad)
    with pytest.raises(ValueError, match="min_judge_agreement must be a number from 0 to 1"):
        run_bundle(tmp_path, min_judge_agreement=1.5)
    result = CliRunner().invoke(run_command, ["some-bundle", "--judge-samples", "0"])
    assert result.exit_code != 0 and "positive integer" in result.output
    result = CliRunner().invoke(run_command, ["some-bundle", "--min-judge-agreement", "1.5"])
    assert result.exit_code != 0 and "number from 0 to 1" in result.output


@pytest.mark.asyncio
async def test_by_default_the_median_decides_a_split_and_each_row_says_how_much_its_samples_agreed(monkeypatch):
    _sampled_llm(monkeypatch, _two_to_one)

    grade = await _grade(G._step(default_model="gpt-4o-mini", judge_samples=3), G._context())

    assert (grade["verifier_status"], grade["score"]) == ("graded", 1.0)
    assert [row["agreement"] for row in grade["verifier_criteria"]] == [0.67, 0.67]
    assert grade["verifier_settings"]["min_judge_agreement"] == 0.5
    assert judge_samples.splits(grade["verifier_criteria"]) == ["table-reserved (2 of 3 pass)", "party-size (2 of 3 pass)"]


@pytest.mark.asyncio
async def test_a_bar_of_1_fails_a_split_grade_keeping_each_row_and_its_samples_then_fails_the_step(monkeypatch):
    _sampled_llm(monkeypatch, _two_to_one)
    context = G._context()

    with pytest.raises(JudgeDisagreementError) as raised:
        await G._step(default_model="gpt-4o-mini", judge_samples=3, min_judge_agreement=1.0).execute(context)

    assert str(raised.value) == (
        "Verifier 'vid': its judge samples split on 2 criteria, under min_judge_agreement 1: table-reserved (2 of 3 "
        "samples pass it), party-size (2 of 3 samples pass it). Each sample's verdict and justification are on the "
        "grade's rows; a lower min_judge_agreement keeps the median instead.")
    grade = context.metadata["grades"]["verify"]
    assert (grade["verifier_status"], grade["score"], grade["passed"]) == ("failed", None, False)
    assert grade["verifier_error_reasoning"] == str(raised.value)
    assert grade["verifier_settings"]["min_judge_agreement"] == 1.0
    rows = {row["id"]: row for row in grade["verifier_criteria"]}
    assert [s["passed"] for s in rows["table-reserved"]["samples"]] == [True, False, True]
    assert rows["table-reserved"]["agreement"] == 0.67 and rows["table-reserved"]["score"] == 1.0
    entry = context.metadata["verifications"]["vid"]
    assert entry["judge_disagreement"] == {"min_judge_agreement": 1.0, "criteria": [
        {"id": "table-reserved", "agreement": 0.67}, {"id": "party-size", "agreement": 0.67}]}
    assert len(entry["results"]) == 2


@pytest.mark.asyncio
async def test_samples_that_agree_grade_as_before_with_the_default_bar_recorded(monkeypatch):
    _sampled_llm(monkeypatch, lambda sample, ids: dict.fromkeys(ids, 1.0))

    grade = await _grade(G._step(default_model="gpt-4o-mini", judge_samples=3), G._context())

    assert (grade["verifier_status"], grade["score"]) == ("graded", 1.0)
    assert all(row["agreement"] == 1.0 for row in grade["verifier_criteria"])
    assert grade["verifier_settings"]["min_judge_agreement"] == 0.5
    assert judge_samples.splits(grade["verifier_criteria"]) == []


@pytest.mark.asyncio
async def test_one_sample_records_no_bar_and_a_bar_without_samples_never_fails(monkeypatch):
    _sampled_llm(monkeypatch, lambda sample, ids: {**dict.fromkeys(ids, 1.0), "c1": 0.0})

    plain = await _grade(G._step(default_model="gpt-4o-mini"), G._context())
    barred = await _grade(G._step(default_model="gpt-4o-mini", min_judge_agreement=1.0), G._context())

    assert plain["verifier_status"] == barred["verifier_status"] == "graded"
    assert "min_judge_agreement" not in plain["verifier_settings"]
    assert all("agreement" not in row for row in plain["verifier_criteria"] + barred["verifier_criteria"])


@pytest.mark.asyncio
@pytest.mark.parametrize("bar, graded", [(0.67, True), (0.7, False)])
async def test_a_bar_two_of_three_meets_keeps_the_median_and_a_higher_one_fails(monkeypatch, bar, graded):
    _sampled_llm(monkeypatch, _two_to_one)
    context = G._context({"judge_samples": 3, "min_judge_agreement": bar})

    if graded:
        await G._step(default_model="gpt-4o-mini").execute(context)
    else:
        with pytest.raises(JudgeDisagreementError, match="under min_judge_agreement 0.7"):
            await G._step(default_model="gpt-4o-mini").execute(context)

    assert context.metadata["grades"]["verify"]["verifier_status"] == ("graded" if graded else "failed")


@pytest.mark.asyncio
async def test_partial_scores_agree_on_their_verdict_so_close_scores_pass_and_a_split_verdict_fails(monkeypatch):
    by_sample = {1: {"c1": 0.75, "c2": 0.9}, 2: {"c1": 0.7, "c2": 0.25}, 3: {"c1": 0.8, "c2": 0.9}}
    _sampled_llm(monkeypatch, lambda sample, ids: by_sample[sample])
    context = G._context()

    with pytest.raises(JudgeDisagreementError, match=r"split on a criterion, .*: party-size \(2 of 3 samples pass it\)"):
        await G._step(default_model="gpt-4o-mini", judge_samples=3, output_format="rubric_partial",
                      min_judge_agreement=1.0).execute(context)

    rows = {row["id"]: row for row in context.metadata["grades"]["verify"]["verifier_criteria"]}
    assert (rows["table-reserved"]["score"], rows["table-reserved"]["agreement"]) == (0.75, 1.0)
    assert rows["party-size"]["agreement"] == 0.67


@pytest.mark.asyncio
@pytest.mark.usefixtures("local_stores")
@pytest.mark.parametrize("fatal", [True, False])
async def test_a_split_fails_the_task_or_is_tolerated_as_the_steps_fail_task_on_error_says(monkeypatch, fatal):
    _sampled_llm(monkeypatch, _two_to_one)
    task = Task(id="t", version=1, steps=[G._step(default_model="gpt-4o-mini", judge_samples=3,
                                                  min_judge_agreement=1.0, fail_task_on_error=fatal)])
    context = G._context()

    if fatal:
        with pytest.raises(JudgeDisagreementError):
            await task.run(context=context)
    else:
        context = await task.run(context=context)

    [failed] = context.metadata["failed_steps"]
    assert (failed["step_id"], failed["error_type"], failed["is_fatal"]) == ("verify", "JudgeDisagreementError", fatal)
    assert context.metadata["grades"]["verify"]["verifier_status"] == "failed"


class _Fleet:
    """Judge agents the verifier deploys, each with a sandbox of its own: what each was sent, what was staged in each,
    and which were terminated; the ``failing`` deploy (1-based) raises."""

    def __init__(self, failing=None):
        self.failing, self.deployed, self.files, self.terminated, self.sent = failing, [], {}, [], []
        fleet = self

        class Sandbox:
            ON_THIS_MACHINE = False

            def __init__(self, sandbox_id):
                self.sandbox_id = sandbox_id

            async def write_file_from_object(self, object_url, path, **kwargs):
                fleet.files.setdefault(self.sandbox_id, []).append(path)

            async def write_file_from_text(self, text, path, **kwargs):
                fleet.files.setdefault(self.sandbox_id, []).append(path)

            async def terminate(self):
                fleet.terminated.append(self.sandbox_id)

        class Provider:
            async def get_sandbox(self, sandbox_id):
                return Sandbox(sandbox_id)

        class Agent:
            async def deploy(self, **kwargs):
                n = len(fleet.deployed) + 1
                fleet.deployed.append(f"sb-{n}")
                if n == fleet.failing:
                    fleet.deployed.pop()
                    raise ConnectionError(f"deploy {n} dropped")
                await asyncio.sleep(0)
                return SimpleNamespace(a2a_url=f"http://judge-{n}.test", sandbox_id=f"sb-{n}", agent_card={},
                                       sandbox_type=None, control_headers={}, model_endpoint_host=None)

        self.provider, self.agent = Provider(), Agent()

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(A2AAgent, "get", staticmethod(lambda agent_id: self.agent))
        monkeypatch.setattr(sandbox_provider, "get_agent_sandbox_provider", lambda: self.provider)

        async def send(url, **kwargs):
            self.sent.append((judge_usage._SAMPLE.get(), url))
            return f"task-{len(self.sent)}", None

        async def poll(url, task_id, **kwargs):
            sample = judge_usage._SAMPLE.get()
            text = _verdicts(_two_to_one(sample, ["c1", "c2"]), sample)
            return {"status": {"state": "completed", "message": {"parts": [{"kind": "text", "text": text}]}}}

        monkeypatch.setattr(protocol, "send_a2a_message", send)
        monkeypatch.setattr(protocol, "poll_a2a_task", poll)


def _deployed_judge_step(**kw) -> RubricsVerifierTaskStep:
    step = G._step(use_agent_judge=True, use_trajectory=True, judge_a2a_agent_id="judge-agent", judge_samples=3, **kw)

    async def configured(**kwargs):
        return None

    step._configure_judge_a2a = configured
    step._compact_trajectory = lambda *a, **k: rv._CompactedTrajectory.for_screenshot(
        None, [], container_object_url="mem://traj/raw.json")
    return step


@pytest.mark.asyncio
@pytest.mark.usefixtures("local_stores")
async def test_a_judge_the_verifier_deploys_gets_one_of_its_own_per_sample_each_staged_and_terminated(monkeypatch):
    install_model_config(monkeypatch, base_url="https://gw.test/v1", api_key="configured-key")
    fleet = _Fleet()
    fleet.install(monkeypatch)

    grade = await _grade(_deployed_judge_step(min_judge_agreement=0.67), G._context())

    assert fleet.deployed == ["sb-1", "sb-2", "sb-3"] and sorted(fleet.terminated) == fleet.deployed
    assert {sandbox: len(paths) for sandbox, paths in fleet.files.items()} == {"sb-1": 1, "sb-2": 1, "sb-3": 1}
    by_sample = {}
    for sample, url in fleet.sent:
        by_sample.setdefault(sample, set()).add(url)
    assert sorted(by_sample) == [1, 2, 3] and all(len(urls) == 1 for urls in by_sample.values())
    assert len(set().union(*by_sample.values())) == 3
    assert [row["score"] for row in grade["verifier_criteria"]] == [1.0, 1.0]
    store = get_config().get_object_store()
    for call in grade["judge_calls"]:
        assert len(json.loads(store.get(call["input"]["object_url"]))["files"]) == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures("local_stores")
async def test_a_judge_that_fails_to_deploy_for_one_sample_fails_the_grade_and_every_deployed_one_is_terminated(
        monkeypatch):
    install_model_config(monkeypatch, base_url="https://gw.test/v1", api_key="configured-key")
    fleet = _Fleet(failing=2)
    fleet.install(monkeypatch)
    context = G._context()

    with pytest.raises(ConnectionError, match="deploy 2 dropped"):
        await _deployed_judge_step().execute(context)

    assert sorted(fleet.terminated) == sorted(fleet.deployed) and fleet.sent == []
    assert context.metadata["grades"]["verify"]["verifier_status"] == "failed"
