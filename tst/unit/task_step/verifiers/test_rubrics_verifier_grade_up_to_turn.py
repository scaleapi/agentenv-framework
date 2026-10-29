"""grade_up_to_turn: grade a multi-turn run as it stood after turn k.

A per-turn env snapshot is only gradable if the judge also sees just turns 1..k: otherwise a
criterion about what the agent said is credited from a later turn. Same seams as
test_rubrics_verifier_multiturn.py: patch the trajectory read, capture the judge's prompt.
"""
from __future__ import annotations

import pytest

from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep

TURNS = {"s3://t/1.json": "TURN_ONE_EVENTS", "s3://t/2.json": "TURN_TWO_EVENTS", "s3://t/3.json": "TURN_THREE_EVENTS"}


def _verifier(**kw) -> RubricsVerifierTaskStep:
    params = dict(
        id="verify", version=1, criteria=[{"id": "c1", "description": "outcome"}],
        prompt_id="p1", use_agent_judge=False, use_trajectory=True, verifier_id="vt",
    )
    params.update(kw)
    return RubricsVerifierTaskStep(**params)


def _three_turn_context() -> TaskStepContext:
    return TaskStepContext(prompt_responses=[PromptResponse(
        prompt_id="p1", response="FINAL_REPLY", prompt_text="do it",
        target_agent_per_turn_trajectory_object_urls=list(TURNS),
        agent_trajectory_object_url="s3://t/3.json",
        source_agent_per_turn_prompt_parts=[None, [{"kind": "text", "text": "a"}], [{"kind": "text", "text": "b"}]],
        structured_output={"from": "the last turn"},
    )])


def _judge_prompt(monkeypatch, verifier) -> dict:
    captured: dict = {}

    async def fake_run(*, eval_prompt, **kwargs):
        captured["eval_prompt"] = eval_prompt
        return ([{**verifier.criteria[0], "score": 1.0, "result": True}], 0, [], None)

    monkeypatch.setattr(verifier, "_read_trajectory_text", lambda uri: TURNS[uri])
    monkeypatch.setattr(verifier, "_run_judge_with_output_retries", fake_run)
    return captured


@pytest.mark.asyncio
async def test_the_judge_sees_only_the_turns_up_to_k(monkeypatch):
    v = _verifier(grade_up_to_turn=2)
    captured = _judge_prompt(monkeypatch, v)
    ctx = _three_turn_context()

    await v.execute(ctx)

    prompt = captured["eval_prompt"]
    assert "TURN_ONE_EVENTS" in prompt and "TURN_TWO_EVENTS" in prompt
    assert "TURN_THREE_EVENTS" not in prompt
    assert "FINAL_REPLY" not in prompt
    assert "as of turn 2 of 3" in prompt
    assert ctx.metadata["verifications"]["vt"]["graded_up_to_turn"] == 2
    stored = ctx.prompt_responses[0]
    assert stored.response == "FINAL_REPLY"
    assert len(stored.target_agent_per_turn_trajectory_object_urls) == 3
    assert stored.structured_output == {"from": "the last turn"}


@pytest.mark.asyncio
async def test_turn_one_is_graded_from_its_own_trajectory(monkeypatch):
    v = _verifier(grade_up_to_turn=1)
    captured = _judge_prompt(monkeypatch, v)

    await v.execute(_three_turn_context())

    prompt = captured["eval_prompt"]
    assert "TURN_ONE_EVENTS" in prompt
    assert "TURN_TWO_EVENTS" not in prompt and "TURN_THREE_EVENTS" not in prompt
    assert "FINAL_REPLY" not in prompt


@pytest.mark.asyncio
async def test_a_per_run_override_sets_the_turn(monkeypatch):
    v = _verifier()
    captured = _judge_prompt(monkeypatch, v)
    ctx = _three_turn_context()
    ctx.metadata["user_overrides"] = {"step_params": {"verify": {"grade_up_to_turn": 1}}}

    await v.execute(ctx)

    assert "TURN_TWO_EVENTS" not in captured["eval_prompt"]
    assert ctx.metadata["verifications"]["vt"]["graded_up_to_turn"] == 1
    assert v.grade_up_to_turn is None


@pytest.mark.asyncio
async def test_the_last_turn_or_later_grades_the_whole_run(monkeypatch):
    v = _verifier(grade_up_to_turn=3)
    captured = _judge_prompt(monkeypatch, v)

    await v.execute(_three_turn_context())

    prompt = captured["eval_prompt"]
    assert "TURN_THREE_EVENTS" in prompt and "FINAL_REPLY" in prompt


@pytest.mark.asyncio
async def test_without_it_every_turn_is_graded(monkeypatch):
    v = _verifier()
    captured = _judge_prompt(monkeypatch, v)
    ctx = _three_turn_context()

    await v.execute(ctx)

    assert "TURN_THREE_EVENTS" in captured["eval_prompt"]
    assert "graded_up_to_turn" not in ctx.metadata["verifications"]["vt"]


@pytest.mark.parametrize("bad", [0, -1, True, "2", 1.5])
def test_rejects_a_turn_that_is_not_a_positive_integer(bad):
    with pytest.raises(ValueError, match="grade_up_to_turn"):
        _verifier(grade_up_to_turn=bad)


def test_round_trips_and_is_omitted_when_unset():
    v = _verifier(grade_up_to_turn=2)
    assert v.to_dict()["grade_up_to_turn"] == 2
    assert RubricsVerifierTaskStep.from_dict(v.to_dict()).grade_up_to_turn == 2
    assert "grade_up_to_turn" not in _verifier().to_dict()


@pytest.mark.asyncio
async def test_a_later_turns_failure_does_not_skip_an_earlier_turn(monkeypatch):
    v = _verifier(grade_up_to_turn=2)
    captured = _judge_prompt(monkeypatch, v)
    ctx = _three_turn_context()
    ctx.prompt_responses[0].error_type = "agent_failed"

    await v.execute(ctx)

    assert "TURN_TWO_EVENTS" in captured["eval_prompt"]
    assert ctx.metadata["verifications"]["vt"]["graded_up_to_turn"] == 2
    assert ctx.prompt_responses[0].error_type == "agent_failed"


@pytest.mark.asyncio
async def test_a_cut_without_the_trajectory_is_refused(monkeypatch):
    v = _verifier(grade_up_to_turn=2, use_trajectory=False)
    _judge_prompt(monkeypatch, v)

    with pytest.raises(RuntimeError, match="use_trajectory=True"):
        await v.execute(_three_turn_context())


@pytest.mark.asyncio
async def test_a_cut_without_per_turn_trajectories_is_refused(monkeypatch):
    v = _verifier(grade_up_to_turn=2)
    _judge_prompt(monkeypatch, v)
    ctx = _three_turn_context()
    ctx.prompt_responses[0].target_agent_per_turn_trajectory_object_urls = None

    with pytest.raises(RuntimeError, match="no stored trajectory for turn 2"):
        await v.execute(ctx)


@pytest.mark.asyncio
async def test_a_cut_at_a_turn_whose_trajectory_was_lost_is_refused(monkeypatch):
    v = _verifier(grade_up_to_turn=2)
    _judge_prompt(monkeypatch, v)
    ctx = _three_turn_context()
    ctx.prompt_responses[0].target_agent_per_turn_trajectory_object_urls = ["s3://t/1.json", None, "s3://t/3.json"]

    with pytest.raises(RuntimeError, match="no stored trajectory for turn 2"):
        await v.execute(ctx)
