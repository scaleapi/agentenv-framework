"""always_on_agent_verifier: an always_on step's prompt is graded from the user's chat (the response and the
conversation) and the run's context (the trigger log, the run wake by wake with its app changes), never every wake's
trajectory. What needs the trajectories is refused when the task loads, as is a plain rubrics_verifier on an always_on
step's prompt; a plain rubrics_verifier grades any other prompt as before."""

from __future__ import annotations

import pytest

from agent_env.task.task import Task
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_steps import always_on as ao
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from agent_env.task_step.task_steps.verifiers import is_rubrics_verifier
from agent_env.task_step.task_steps.verifiers.always_on_agent_verifier import AlwaysOnAgentVerifier
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RUN_CONTEXT_PARTS, RubricsVerifierTaskStep
from tst.unit.task_step.always_on_harness import Rig

pytestmark = pytest.mark.usefixtures("local_stores")

CRITERIA = [{"id": "c1", "description": "asked before booking"}]


def _verifier(cls=AlwaysOnAgentVerifier, **kw) -> RubricsVerifierTaskStep:
    return cls(id="judge", version=1, criteria=CRITERIA, prompt_id="ask", use_agent_judge=False, verifier_id="rubric",
               **kw)


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


# ---- grading ---------------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_judge_grades_from_the_chat_and_the_run_wake_by_wake_not_the_trajectories(monkeypatch, tmp_path):
    rig, ctx = await _run(monkeypatch, tmp_path)
    verifier = _verifier()
    stored = verifier.to_dict()
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
    assert verifier.to_dict() == stored  # the step, shared by concurrent rollouts, is untouched
    assert verifier.use_trajectory is False and verifier.run_context_for_judge == list(RUN_CONTEXT_PARTS)


@pytest.mark.asyncio
async def test_parts_of_the_run_context_a_verifier_names_are_kept(monkeypatch, tmp_path):
    rig, ctx = await _run(monkeypatch, tmp_path)
    verifier = _verifier(run_context_for_judge=["conversation"])
    captured = _capture(monkeypatch, verifier)

    await verifier.execute(ctx)

    assert "## Conversation" in captured["prompt"] and "## Trigger firings" not in captured["prompt"]
    assert "## The run, wake by wake" in captured["prompt"]


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
async def test_a_run_s_grade_up_to_turn_is_refused(monkeypatch, tmp_path):
    rig, ctx = await _run(monkeypatch, tmp_path)
    verifier = _verifier()
    _capture(monkeypatch, verifier)
    ctx.metadata["user_overrides"] = {"step_params": {"judge": {"grade_up_to_turn": 2}}}

    with pytest.raises(RuntimeError, match="grade_up_to_turn can't cut an always_on"):
        await verifier.execute(ctx)


@pytest.mark.asyncio
async def test_a_prompt_no_always_on_step_made_is_refused_when_it_is_graded(monkeypatch):
    ctx = TaskStepContext()
    ctx.prompt_responses.append(PromptResponse(prompt_id="ask", response="done", prompt_text="hi", step_id="prompt"))
    verifier = _verifier()
    _capture(monkeypatch, verifier)

    with pytest.raises(RuntimeError, match="is not an always_on step's prompt.*grade it with rubrics_verifier"):
        await verifier.execute(ctx)


def test_the_wake_by_wake_text_says_when_the_record_is_unreadable():
    text = ao.render_for_judge({"object_url": "file:///nowhere/at/all.json", "end_reason": "end_time"})
    assert text.startswith("## The run, wake by wake\n(The run's record could not be read:")


# ---- refused when the task loads -------------------------------------------------------------------------------------

@pytest.mark.parametrize("settings,why", [
    ({"output_format": "rubric_evidence"}, "output_format=rubric_evidence cites per-criterion screenshots"),
    ({"output_format": "rubric_evidence_with_mistakes"}, "cites per-criterion screenshots"),
    ({"grade_up_to_turn": 2}, "grade_up_to_turn can't cut an always_on step's prompt"),
    ({"use_trajectory": True}, "use_trajectory must be off"),
])
def test_what_needs_the_wakes_trajectories_is_refused_when_the_task_loads(settings, why):
    with pytest.raises(ValueError, match=why):
        _verifier(**settings)
    stored = {"id": "judge", "type": "always_on_agent_verifier", "prompt_id": "ask", "criteria": CRITERIA, **settings}
    with pytest.raises(ValueError, match=why):
        Task.from_dict({"id": "t", "version": 1, "steps": [stored]})


def test_a_stored_verifier_needs_no_use_trajectory_and_survives_a_round_trip():
    stored = {"id": "judge", "type": "always_on_agent_verifier", "prompt_id": "ask", "criteria": CRITERIA}
    [verifier] = Task.from_dict({"id": "t", "version": 1, "steps": [stored]}).steps
    assert isinstance(verifier, AlwaysOnAgentVerifier) and verifier.use_trajectory is False
    again = AlwaysOnAgentVerifier.from_dict(verifier.to_dict())
    assert again.to_dict() == verifier.to_dict() and again.type == "always_on_agent_verifier"


def _always_on() -> ao.AlwaysOnTaskStep:
    return ao.AlwaysOnTaskStep("week", None, user_sim="linda", prompt_id="ask",
                               agent_name="solver")


def test_a_plain_rubrics_verifier_on_an_always_on_prompt_is_refused_when_the_task_loads():
    with pytest.raises(ValueError, match="step 'judge' \\(rubrics_verifier\\) grades the prompt 'ask' of always_on "
                                         "step 'week'.*use type always_on_agent_verifier"):
        Task("t", 1, steps=[_always_on(), _verifier(RubricsVerifierTaskStep)])
    stored = Task("t", 1, steps=[_always_on(), _verifier()]).to_dict()
    stored["steps"][1]["type"] = "rubrics_verifier"
    with pytest.raises(ValueError, match="use type always_on_agent_verifier"):
        Task.from_dict(stored)


def test_an_always_on_agent_verifier_on_another_step_s_prompt_is_refused_when_the_task_loads():
    prompt = PromptAgentTaskStep("ask-step", None, prompt="hi", prompt_id="ask", agent_name="solver")
    with pytest.raises(ValueError, match="grades prompt 'ask' of prompt_agent step 'ask-step', which isn't an "
                                         "always_on step; grade it with rubrics_verifier"):
        Task("t", 1, steps=[prompt, _verifier()])


def test_each_verifier_on_the_prompt_it_grades_loads():
    Task("t", 1, steps=[_always_on(), _verifier()])
    prompt = PromptAgentTaskStep("ask-step", None, prompt="hi", prompt_id="ask", agent_name="solver")
    Task("t", 1, steps=[prompt, _verifier(RubricsVerifierTaskStep)])


# ---- a plain rubrics_verifier --------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_plain_rubrics_verifier_grades_another_prompt_as_before(monkeypatch):
    ctx = TaskStepContext()
    ctx.prompt_responses.append(PromptResponse(prompt_id="ask", response="booked it", prompt_text="book it",
                                               step_id="prompt"))
    verifier = _verifier(RubricsVerifierTaskStep, use_trajectory=False)
    captured = _capture(monkeypatch, verifier)

    await verifier.execute(ctx)

    assert verifier.to_dict()["type"] == "rubrics_verifier" and verifier.run_context_for_judge is None
    assert "booked it" in captured["prompt"] and "## The run, wake by wake" not in captured["prompt"]
    assert ctx.metadata["verifications"]["rubric"]["score"] == 1


# ---- is_rubrics_verifier -------------------------------------------------------------------------------------------

def test_is_rubrics_verifier_recognises_the_subclass_as_a_step_and_as_a_stored_dict():
    for verifier in (_verifier(), _verifier(RubricsVerifierTaskStep)):
        assert is_rubrics_verifier(verifier) and is_rubrics_verifier(verifier.to_dict())
    assert not is_rubrics_verifier(_always_on()) and not is_rubrics_verifier(_always_on().to_dict())
    assert not is_rubrics_verifier({"type": "no_such_step"}) and not is_rubrics_verifier({})
