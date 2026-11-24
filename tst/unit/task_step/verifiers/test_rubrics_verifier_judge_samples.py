"""judge_samples: a rubric_binary or rubric_partial judge grades each criterion that many times, an odd count, and keeps
its median, the row of the first sample that scored it with every sample's score and justification beside it; each
call's record names its sample; the samples share the judge's concurrency; one that fails fails the grade; a run's
count reaches every judge that can take it, and one that can't grades once and records it as not sent."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import litellm
import pytest
from click.testing import CliRunner

from agent_env.a2a_agent import protocol
from agent_env.bundle.run import _overrides, run_bundle
from agent_env.cli.run import run as run_command
from agent_env.task_step.context import DeployedAgent
from agent_env.task_step.task_steps.verifiers.judge_utils import judge_samples, judge_usage
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


def test_a_count_is_odd_and_positive_and_an_even_one_names_its_odd_neighbours():
    assert [judge_samples.validated(n) for n in (None, 1, 3, 7)] == [None, 1, 3, 7]
    for bad in (0, -3, True, 3.0, "3"):
        with pytest.raises(ValueError, match="positive odd integer"):
            judge_samples.validated(bad)
    with pytest.raises(ValueError, match="must be odd, so each criterion's median is a score a sample gave: use 3 or 5"):
        judge_samples.validated(4)


def test_each_criterion_keeps_the_first_row_that_scored_its_median_and_every_samples_score_and_reason():
    sampled = [[{"id": "a", "score": s, "justification": f"{n}:{s}", "outcome": f"level {s}"} for s in [score]]
               for n, score in enumerate([0.25, 0.75, 0.5], 1)]

    [row] = judge_samples.median_rows(sampled)

    assert (row["score"], row["justification"], row["outcome"]) == (0.5, "3:0.5", "level 0.5")
    assert row["samples"] == [{"score": 0.25, "justification": "1:0.25"}, {"score": 0.75, "justification": "2:0.75"},
                              {"score": 0.5, "justification": "3:0.5"}]


def test_a_task_that_asks_an_even_count_or_samples_a_format_without_one_score_a_row_fails_to_load():
    with pytest.raises(ValueError, match="use 1 or 3"):
        RubricsVerifierTaskStep.from_dict({**G._step().to_dict(), "judge_samples": 2})
    with pytest.raises(ValueError, match="judge_samples does not apply to output_format=rubric_evidence"):
        G._evidence_step("rubric_evidence", judge_samples=3)
    with pytest.raises(ValueError, match="judge_samples"):
        UserSimJudge.from_dict({"id": "sim", "version": 1, "type": "user_sim_judge", "run_step_id": "week",
                                "judge_samples": 3})
    step = G._step(judge_samples=3)
    assert RubricsVerifierTaskStep.from_dict(step.to_dict()).judge_samples == 3
    assert "judge_samples" not in G._step().to_dict()


@pytest.mark.asyncio
async def test_a_direct_judge_grades_each_criterion_three_times_and_keeps_each_ones_median(monkeypatch):
    seen = _sampled_llm(monkeypatch, _two_to_one)
    context = G._context()

    grade = await _grade(G._step(default_model="gpt-4o-mini", judge_samples=3), context)

    rows = {row["id"]: row for row in grade["verifier_criteria"]}
    assert (rows["table-reserved"]["score"], rows["party-size"]["score"]) == (1.0, 1.0)
    assert rows["table-reserved"]["justification"] == "sample 1 scored c1 1.0"
    assert rows["party-size"]["justification"] == "sample 2 scored c2 1.0"
    assert [s["score"] for s in rows["table-reserved"]["samples"]] == [1.0, 0.0, 1.0]
    assert sorted(sample for _, sample in seen["calls"]) == [1, 2, 3]
    assert sorted((c["sample"], c["outcome"]) for c in grade["judge_calls"]) == [(1, "answered"), (2, "answered"),
                                                                                (3, "answered")]
    assert grade["verifier_settings"]["judge_samples"] == 3
    entry = context.metadata["verifications"]["vid"]
    assert entry["judge_samples"] == 3 and all(len(row["samples"]) == 3 for row in entry["results"])


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

    grade = await _grade(G._step(default_model="gpt-4o-mini"), G._context({"judge_samples": 3}))

    assert len(seen["calls"]) == 3
    settings = grade["verifier_settings"]
    assert settings["judge_samples"] == 3 and "judge_samples" in settings["overridden"]


@pytest.mark.asyncio
async def test_a_runs_count_a_format_cant_take_grades_once_and_is_recorded_as_not_sent():
    calls: list[str] = []
    step = G._direct_judge(G._evidence_step("rubric_evidence"),
                           lambda ids, n: calls.append("call") or G._evidence(ids))

    grade = await _grade(step, G._context({"judge_samples": 3}))

    assert calls == ["call"] and grade["verifier_status"] == "graded"
    assert grade["verifier_settings"]["judge_samples"] == 1
    assert grade["verifier_settings"]["not_sent"] == {"judge_samples": 3}


@pytest.mark.asyncio
async def test_a_runs_even_count_fails_the_grade_saying_why(monkeypatch):
    _sampled_llm(monkeypatch, lambda sample, ids: dict.fromkeys(ids, 1.0))
    context = G._context({"judge_samples": 2})

    with pytest.raises(ValueError, match="use 1 or 3"):
        await G._step(default_model="gpt-4o-mini").execute(context)

    assert context.metadata["grades"]["verify"]["verifier_status"] == "failed"


@pytest.mark.asyncio
async def test_an_agent_judge_takes_each_sample_in_a_conversation_of_its_own(monkeypatch):
    sent: list[dict] = []

    async def send(url, **kwargs):
        sent.append({"context_id": kwargs.get("context_id"), "sample": judge_usage._SAMPLE.get()})
        return f"task-{len(sent)}", None

    async def poll(url, task_id, **kwargs):
        text = _verdicts(_two_to_one(judge_usage._SAMPLE.get(), ["c1", "c2"]), judge_usage._SAMPLE.get())
        return {"status": {"state": "completed", "message": {"parts": [{"kind": "text", "text": text}]}}}

    monkeypatch.setattr(protocol, "send_a2a_message", send)
    monkeypatch.setattr(protocol, "poll_a2a_task", poll)
    context = G._context()
    context.deployed_agents.append(DeployedAgent(agent_name="judge", api_url="http://judge.test",
                                                 a2a_url="http://judge.test", sandbox_id="sb-judge"))
    step = G._step(use_agent_judge=True, agent_name="judge", judge_samples=3)

    async def configured(**kwargs):
        return None

    step._configure_judge_a2a = configured
    grade = await _grade(step, context)

    assert sorted(s["sample"] for s in sent) == [1, 2, 3] and {s["context_id"] for s in sent} == {None}
    assert [row["score"] for row in grade["verifier_criteria"]] == [1.0, 1.0]
    assert sorted(c["sample"] for c in grade["judge_calls"]) == [1, 2, 3]


def test_the_run_flag_sets_the_runs_count_and_refuses_an_even_one_or_none_at_all(tmp_path):
    assert _overrides(None, None, None, 3) == {"judge_samples": 3}
    assert _overrides(None, None, None, None) == {}
    for bad in (0, 2):
        with pytest.raises(ValueError, match="judge_samples must be"):
            run_bundle(tmp_path, judge_samples=bad)
    result = CliRunner().invoke(run_command, ["some-bundle", "--judge-samples", "2"])
    assert result.exit_code != 0 and "use 1 or 3" in result.output
