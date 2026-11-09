"""rubrics_verifier on an always_on step's prompt: the judge grades from the user's chat (the response and the
conversation) and the run's context (the trigger log, the run wake by wake with its app changes), never every wake's
trajectory. Any other step's prompt is graded as before."""

from __future__ import annotations

import pytest

from agent_env.task_step.task_steps import always_on as ao
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from tst.unit.task_step.always_on_harness import Rig

pytestmark = pytest.mark.usefixtures("local_stores")


def _verifier(**kw) -> RubricsVerifierTaskStep:
    return RubricsVerifierTaskStep(id="judge", version=1, criteria=[{"id": "c1", "description": "asked before booking"}],
                                   prompt_id="ask", use_agent_judge=False, verifier_id="rubric", **kw)


def _capture(monkeypatch, verifier) -> dict:
    captured: dict = {}

    async def judge(*, eval_prompt, **kwargs):
        captured["prompt"] = eval_prompt
        return ([{**verifier.criteria[0], "score": 1.0, "result": True}], 0, [], None)

    def no_trajectories(uri):
        raise AssertionError(f"the judge was sent a wake's trajectory: {uri}")

    monkeypatch.setattr(verifier, "_run_judge_with_output_retries", judge)
    monkeypatch.setattr(verifier, "_read_trajectory_text", no_trajectories)
    return captured


async def _run(monkeypatch, tmp_path):
    def agent(n, texts, rig):
        if n == 1:
            rig.message("Casa Jacarandas is $855. Book it?")
            return "PRIVATE: I'll book it anyway"
        if n == 2:
            rig.app_write("bookings", "request 855 USD", tool="stayncook_request_booking")
            rig.message("requested!")
            return "PRIVATE done"
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent, answers=[{"outcome": "reply", "message": "yes please"}])
    await rig.arm()
    rig.keep_changelog()
    rig.triggers(rig.notify_at("sam", "2026-11-10T14:00:00Z", "Sam: shh", channel="noor 30"))
    ctx = await rig.run(rig.step())
    [response] = ctx.prompt_responses
    response.target_agent_per_turn_trajectory_object_urls = [f"s3://wakes/{n}.json" for n in range(1, 4)]
    response.agent_trajectory_object_url = "s3://wakes/3.json"
    return rig, ctx


@pytest.mark.asyncio
async def test_the_judge_grades_from_the_chat_and_the_run_wake_by_wake_not_the_trajectories(monkeypatch, tmp_path):
    rig, ctx = await _run(monkeypatch, tmp_path)
    verifier = _verifier()
    captured = _capture(monkeypatch, verifier)

    await verifier.execute(ctx)

    prompt = captured["prompt"]
    assert "[Mon Nov 9, 2:00 PM UTC] Casa Jacarandas is $855. Book it?" in prompt and "PRIVATE" not in prompt
    conversation = prompt[prompt.index("## Conversation"):prompt.index("## Trigger firings")]
    assert "Linda: yes please" in conversation and "[End of run" in conversation
    wakes = prompt[prompt.index("## The run, wake by wake"):]
    assert "- Wake 2 · Mon Nov 9, 2:00 PM UTC · woken by the user's reply · sent the user 1 message" in wakes
    assert "  - it changed: INSERT public.bookings row 1 (request 855 USD)" in wakes
    assert "- Wake 3 · Tue Nov 10, 2:00 PM UTC · woken by an alert (noor 30)" in wakes
    assert "## How the run ended\nNothing could wake the assistant again, and nothing was pending." in wakes
    assert ctx.metadata["verifications"]["rubric"]["score"] == 1
    assert verifier.use_trajectory is True and verifier.run_context_for_judge is None  # the shared step is untouched


@pytest.mark.asyncio
async def test_parts_of_the_run_context_a_verifier_names_are_kept(monkeypatch, tmp_path):
    rig, ctx = await _run(monkeypatch, tmp_path)
    verifier = _verifier(run_context_for_judge=["conversation"])
    captured = _capture(monkeypatch, verifier)

    await verifier.execute(ctx)

    assert "## Conversation" in captured["prompt"] and "## Trigger firings" not in captured["prompt"]


@pytest.mark.asyncio
async def test_grading_an_always_on_prompt_as_of_a_turn_is_refused(monkeypatch, tmp_path):
    rig, ctx = await _run(monkeypatch, tmp_path)
    verifier = _verifier(grade_up_to_turn=2)
    _capture(monkeypatch, verifier)

    with pytest.raises(RuntimeError, match="grade_up_to_turn can't cut an always_on"):
        await verifier.execute(ctx)


@pytest.mark.asyncio
async def test_a_failed_always_on_run_is_skipped_like_a_failed_prompt(monkeypatch, tmp_path):
    def fails(n, texts, rig):
        raise TimeoutError("no answer")

    rig = Rig(monkeypatch, tmp_path, agent=fails)
    await rig.arm()
    ctx = await rig.run(rig.step())
    verifier = _verifier()
    _capture(monkeypatch, verifier)

    await verifier.execute(ctx)

    [row] = ctx.metadata["verifications"]["rubric"]["results"]
    assert row["id"] == "prompt_error" and "agent_turn_timeout" in row["message"]


@pytest.mark.asyncio
async def test_a_failed_always_on_run_is_skipped_even_by_a_verifier_that_couldn_t_grade_it(monkeypatch, tmp_path):
    """grade_up_to_turn can't cut an always_on prompt, but a run that broke is skipped before that is checked."""
    def fails(n, texts, rig):
        raise TimeoutError("no answer")

    rig = Rig(monkeypatch, tmp_path, agent=fails)
    await rig.arm()
    ctx = await rig.run(rig.step())
    verifier = _verifier(grade_up_to_turn=2)
    _capture(monkeypatch, verifier)

    await verifier.execute(ctx)

    [row] = ctx.metadata["verifications"]["rubric"]["results"]
    assert row["id"] == "prompt_error" and "agent_turn_timeout" in row["message"]


def test_the_wake_by_wake_text_says_when_the_record_is_unreadable():
    text = ao.render_for_judge({"object_url": "file:///nowhere/at/all.json", "end_reason": "end_time"})
    assert text.startswith("## The run, wake by wake\n(The run's record could not be read:")
