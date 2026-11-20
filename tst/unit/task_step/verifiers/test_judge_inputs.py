"""Every built-in judge's input, pinned: what a model judge is sent (its prompt, images, system prompt and cached
prefix), what an agent judge is given (its prompt and every file staged in its sandbox) and what a scripted verify is
called with, for runs whose native trajectories are stored as a run stores them. Each is what its golden file holds,
recorded before prompt steps kept per-turn trajectory records, and stays so byte for byte when the run's prompt
responses carry those records, read back from the stored form. Re-record after a deliberate change with
``AGENTENV_TEST_RECORD_GOLDEN=1`` and review the file's diff. Several cases build their runs with other tests' setups
(the always_on rig, user_sim_judge's fakes, the rubrics golden test's and run_context's helpers), so a change to one of
those is re-recorded here too."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.config import get_config
from agent_env.env.env import DeployedEnv
from agent_env.providers.sandbox_providers import sandbox_provider
from agent_env.task_step.context import DeployedAgent, PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.verifiers import always_on_agent_verifier
from agent_env.task_step.task_steps.verifiers.env_outcome_verifier import EnvOutcomeVerifierTaskStep
from agent_env.task_step.task_steps.verifiers.judge_utils.trajectory_filter import TrajectoryFilter
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from tst.unit.task_step import test_always_on_agent_verifier as A
from tst.unit.task_step.verifiers import test_rubrics_verifier_golden as G
from tst.unit.task_step.verifiers import test_run_context as R
from tst.unit.task_step.verifiers import test_user_sim_judge as T
from tst.util import verifier_golden

pytestmark = pytest.mark.usefixtures("local_stores")

FILE = "judge_inputs.json"
RECORDED = Path(__file__).parent / "recorded"
_TURN = json.loads((RECORDED / "turn_claude_code.json").read_text())
#: Two turns of a Claude Code run: the recorded turn's first half, then all of it.
TURNS = [json.dumps(_TURN[: len(_TURN) // 2]), json.dumps(_TURN)]


#: How many lines a long tool output has: enough for the default trajectory filter to compact it.
_LONG_OUTPUT_LINES = 400


def _tool_span(n: int) -> dict:
    output = f"line {n}\n" * _LONG_OUTPUT_LINES if n % 2 else "ok"
    return {"name": f"tool_{n}", "attributes": {"gen_ai.operation.name": "execute_tool",
                                                "gen_ai.prompt": json.dumps({"tool": f"tool_{n}", "input": {"n": n}}),
                                                "gen_ai.completion": json.dumps({"output": output})}}


#: Two turns of OTel tool spans, every other one's output long enough for the default filter to compact.
SPAN_TURNS = [json.dumps([_tool_span(n) for n in range(3)]), json.dumps([_tool_span(n) for n in range(6)])]
#: The folders a judge's run stages its files under, each named afresh per run.
_RUN_FOLDERS = re.compile(
    rf"({re.escape(RubricsVerifierTaskStep.TRAJECTORY_CONTAINER_DIR)}|{re.escape(always_on_agent_verifier.STAGING_DIR)})"
    r"/[0-9a-f]{12}")


def _recorded_form(context: TaskStepContext) -> TaskStepContext:
    """``context`` as a run that kept per-turn trajectory records leaves it, read back from its stored form."""
    for response in context.prompt_responses:
        turns = response.target_agent_per_turn_trajectory_object_urls or [response.agent_trajectory_object_url]
        response.target_agent_per_turn_trajectories = [
            {"native_format": "claude-code-stream-json", "atif": None} for _ in turns]
    return TaskStepContext.from_dict(context.to_dict())


#: The wall-clock day and time a chat run's user sim was shown, which its prompt step reads off this machine's clock.
_WALL_CLOCK = re.compile(r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) "
                         r"\d{1,2}\b|\b\d{1,2}:\d{2} [AP]M\b")
#: How long a captured string may be in the golden file before it's kept as its sha256 and length.
_KEPT_CHARS = 500


def _normalised(value: Any, *, wall_clock: bool = False) -> Any:
    text = _RUN_FOLDERS.sub(r"\1/<run>", json.dumps(verifier_golden.normalised(value)))
    return json.loads(_WALL_CLOCK.sub("<clock>", text) if wall_clock else text)


def _digested(value: Any) -> Any:
    """``value`` with each long string as its sha256 and length, which pin it as surely and keep the file small."""
    if isinstance(value, str) and len(value) > _KEPT_CHARS:
        return {"sha256": hashlib.sha256(value.encode()).hexdigest(), "chars": len(value)}
    if isinstance(value, dict):
        return {key: _digested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_digested(item) for item in value]
    return value


async def _pinned(case: str, judged: Callable[[Callable[[TaskStepContext], TaskStepContext]], Awaitable[Any]], *,
                  wall_clock: bool = False) -> None:
    """``judged`` builds the case's run, hands its context through the given form and returns what its judge was
    given: that is the golden file's, as the run is and with the records. ``wall_clock``: the run shows this machine's
    clock, which the golden file holds as ``<clock>``."""
    got = _normalised(await judged(lambda context: context), wall_clock=wall_clock)
    verifier_golden.check(FILE, case, _digested(got))
    if not verifier_golden.RECORDING:
        assert _normalised(await judged(_recorded_form), wall_clock=wall_clock) == got


def _stored(name: str, text: str) -> str:
    return get_config().get_object_store().put(f"judge-inputs/{name}", text.encode(), content_type="application/json",
                                               allow_overwrite=True)


def _run(*, turns: int = 1, agent_url: bool = True, first_turn_missing: bool = False, texts: list[str] = TURNS
         ) -> TaskStepContext:
    """A prompt step's context: its turns' native trajectories stored, the step's own url the last one's."""
    urls: list[str | None] = [_stored(f"turn-{n}.json", text) for n, text in enumerate(texts[-turns:], 1)]
    if first_turn_missing:
        urls = [None, *urls]
    return TaskStepContext(instance_id="inst-1", prompt_responses=[PromptResponse(
        prompt_id="p1", response="The table for four is booked.", prompt_text="Book a table for four.",
        agent_trajectory_object_url=urls[-1] if agent_url else None,
        target_agent_per_turn_trajectory_object_urls=urls,
        source_agent_per_turn_prompt_parts=[None, *[[{"kind": "text", "text": "And for Friday?"}]] * (len(urls) - 1)])])


class _Judge:
    """Every judge call answered passing, and what each was given."""

    def __init__(self, answer: Callable[[list[str]], str] = G._binary) -> None:
        self.answer = answer
        self.calls: list[dict] = []
        self.files: dict[str, Any] = {}

    async def model(self, eval_prompt, *, model, context, image_blocks=None, system=None, cached_prefix=None):
        self.calls.append({"model": model, "prompt": eval_prompt, "images": image_blocks, "system": system,
                           "cached_prefix": cached_prefix})
        return {"response": self.answer(G._ids(eval_prompt)), "trajectory_object_url": None}

    async def agent(self, *, eval_prompt, judge_a2a_url, judge_agent_card, judge_agent):
        self.calls.append({"prompt": eval_prompt})
        return {"response": self.answer(G._ids(eval_prompt)), "trajectory_object_url": None}

    async def write_file_from_object(self, object_url, destination_path, **kwargs):
        self.files[destination_path] = {"text": get_config().get_object_store().get(object_url).decode(), **kwargs}

    async def write_file_from_text(self, content, destination_path, **kwargs):
        self.files[destination_path] = {"text": content, **kwargs}

    def given(self) -> dict:
        return {"calls": self.calls, "files": dict(sorted(self.files.items()))}


def _model_judged(step_kw: dict, run: Callable[[], TaskStepContext], answer=G._binary):
    async def judged(form):
        judge = _Judge(answer)
        step = G._step(**step_kw)
        step._prompt_llm_judge = judge.model
        await step.execute(form(run()))
        return judge.given()

    return judged


def _agent_judged(monkeypatch, step_kw: dict, run: Callable[[], TaskStepContext]):
    async def judged(form):
        judge = _Judge()

        class Provider:
            async def get_sandbox(self, sandbox_id):
                assert sandbox_id == "sb-judge"
                return judge

        monkeypatch.setattr(sandbox_provider, "get_agent_sandbox_provider", lambda: Provider())
        step = G._step(use_agent_judge=True, agent_name="judge", **step_kw)
        step._invoke_judge_a2a = judge.agent

        async def configured(**kwargs):
            return None

        step._configure_judge_a2a = configured
        context = run()
        context.deployed_agents.append(DeployedAgent(agent_name="judge", api_url="http://judge.test",
                                                     a2a_url="http://judge.test", sandbox_id="sb-judge"))
        await step.execute(form(context))
        return judge.given()

    return judged


_SCREENSHOTS = {"use_trajectory": True, "criteria": G.EVIDENCE, "output_format": "rubric_evidence",
                "trajectory_filter": G._evidence_step("rubric_evidence").trajectory_filter}
MODEL_CASES = {
    "model_without_the_trajectory": ({}, _run),
    "model_one_turn": ({"use_trajectory": True}, _run),
    "model_one_turn_from_the_per_turn_urls": ({"use_trajectory": True},
                                              lambda: _run(agent_url=False, first_turn_missing=True)),
    "model_one_turn_compacted": ({"use_trajectory": True, "trajectory_filter": TrajectoryFilter()},
                                 lambda: _run(texts=SPAN_TURNS)),
    "model_two_turns": ({"use_trajectory": True}, lambda: _run(turns=2)),
    "model_two_turns_compacted": ({"use_trajectory": True, "trajectory_filter": TrajectoryFilter()},
                                  lambda: _run(turns=2, texts=SPAN_TURNS)),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", MODEL_CASES)
async def test_a_model_judges_input_is_pinned(case):
    step_kw, run = MODEL_CASES[case]
    await _pinned(case, _model_judged(step_kw, run))


@pytest.mark.asyncio
async def test_a_model_judges_input_graded_up_to_a_turn_is_pinned():
    def run():
        context = _run(turns=2)
        context.metadata["user_overrides"] = {"step_params": {"verify": {"grade_up_to_turn": 1}}}
        return context

    await _pinned("model_graded_up_to_turn_one", _model_judged({"use_trajectory": True}, run))


@pytest.mark.asyncio
async def test_a_model_judges_screenshots_and_action_log_are_pinned():
    def run():
        context = _run()
        context.prompt_responses[0].agent_trajectory_object_url = _stored("shots.json", G.SHOTS)
        context.prompt_responses[0].target_agent_per_turn_trajectory_object_urls = [
            context.prompt_responses[0].agent_trajectory_object_url]
        return context

    await _pinned("model_screenshots_per_criterion", _model_judged(
        _SCREENSHOTS, run, answer=lambda ids: G._evidence(ids)))


@pytest.mark.asyncio
async def test_a_model_judges_run_context_for_a_prompt_another_follows_is_pinned():
    """The earlier of two prompts: what it's shown of the run is cut where the next one began, by position."""
    first_id, first = R._stored_prompt("prompt-1", "Book lunch with Dana.", "Booked for noon.")
    second_id, second = R._stored_prompt("prompt-2", "Now book dinner.", "Booked for seven.")
    events = [{"seq": 1, "ts": R._between(first[0]["ts"], first[1]["ts"]), "kind": "fired", "trigger_id": "lunch"},
              {"seq": 2, "ts": R._between(second[0]["ts"], second[1]["ts"]), "kind": "fired", "trigger_id": "dinner"}]
    log = _stored(f"env_trigger_state/{second_id}.json",
                  json.dumps({"env_id": "calendar", "state": {"triggers": [], "events": events}}))

    def run():
        context = TaskStepContext(metadata={
            "a2a_conversations": {"prompt-1": first_id, "prompt-2": second_id},
            "env_trigger_state": {"calendar": {"object_url": log, "capture_step_id": "prompt-2"}}})
        for prompt_id, step_id, text, reply in (("p1", "prompt-1", "Book lunch with Dana.", "Booked for noon."),
                                                ("p2", "prompt-2", "Now book dinner.", "Booked for seven.")):
            context.prompt_responses.append(PromptResponse(prompt_id=prompt_id, response=reply, prompt_text=text,
                                                           step_id=step_id))
        return context

    await _pinned("model_run_context_of_an_earlier_prompt", _model_judged(
        {"run_context_for_judge": ["conversation", "trigger_log"]}, run))


AGENT_CASES = {
    "agent_one_turn": ({"use_trajectory": True}, _run),
    "agent_one_turn_compacted": ({"use_trajectory": True, "trajectory_filter": TrajectoryFilter()},
                                 lambda: _run(texts=SPAN_TURNS)),
    "agent_two_turns": ({"use_trajectory": True}, lambda: _run(turns=2)),
    "agent_two_turns_compacted": ({"use_trajectory": True, "trajectory_filter": TrajectoryFilter()},
                                  lambda: _run(turns=2, texts=SPAN_TURNS)),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", AGENT_CASES)
async def test_an_agent_judges_prompt_and_staged_files_are_pinned(monkeypatch, case):
    step_kw, run = AGENT_CASES[case]
    await _pinned(case, _agent_judged(monkeypatch, step_kw, run))


@pytest.mark.asyncio
async def test_an_unstaged_always_on_judges_input_is_pinned(monkeypatch, tmp_path):
    rig, run = await A._run(monkeypatch, tmp_path)

    async def judged(form):
        verifier = A._verifier(default_model="judge-model")
        captured = A._capture(monkeypatch, verifier)
        await verifier.execute(form(copy.deepcopy(run)))
        return captured

    await _pinned("always_on_unstaged", judged)


@pytest.mark.asyncio
async def test_a_staged_always_on_judges_prompt_and_files_are_pinned(monkeypatch):
    run, sandbox, sources = A._staged_run(monkeypatch)

    async def judged(form):
        sandbox.files.clear()
        verifier = A._verifier(use_agent_judge=True, use_trajectory=True, agent_name="judge")
        captured = A._agent_judge(monkeypatch, verifier, sources)
        await verifier.execute(form(copy.deepcopy(run)))
        return {**captured, "files": dict(sorted(sandbox.files.items()))}

    await _pinned("always_on_staged", judged)


def _user_sim_calls(fake: T.FakeModel) -> list[dict]:
    """What each of the user-sim judge's checks was sent, in an order its concurrency can't change."""
    return sorted(({k: v for k, v in call.items() if k != "alongside"} for call in fake.calls),
                  key=lambda call: json.dumps(call, sort_keys=True))


@pytest.mark.asyncio
async def test_a_user_sim_judges_input_on_an_always_on_run_is_pinned(monkeypatch, tmp_path):
    rig, run, prompt = await T._always_on(monkeypatch, tmp_path)

    async def judged(form):
        fake = T.FakeModel()
        await T._step(fake, monkeypatch, run_step_id="week").execute(form(copy.deepcopy(run)))
        return _user_sim_calls(fake)

    await _pinned("user_sim", judged)


@pytest.mark.asyncio
async def test_a_user_sim_judges_input_on_a_chat_run_is_pinned(monkeypatch):
    """A prompt step's user sim, judged from each read it stored (``metadata["user_sim_reads"]``)."""
    run = await T._chat(monkeypatch)

    async def judged(form):
        fake = T.FakeModel()
        await T._step(fake, monkeypatch).execute(form(copy.deepcopy(run)))
        return _user_sim_calls(fake)

    await _pinned("user_sim_chat", judged, wall_clock=True)


@pytest.mark.asyncio
async def test_what_a_scripted_verify_is_called_with_is_pinned(monkeypatch):
    script = (b"import dataclasses, json\n"
              b"async def verify(url, trajectory, run):\n"
              b"    run = json.loads(json.dumps(dataclasses.asdict(run), default=str))\n"
              b"    return [{'id': 'c', 'score': 1.0, 'given': {'url': url, 'trajectory': trajectory, 'run': run}}]\n")
    monkeypatch.setattr(FileArtifact, "get", staticmethod(lambda id, version=None: SimpleNamespace(load=lambda: script)))

    async def judged(form):
        context = _run(turns=2, first_turn_missing=True)
        context.deployed_envs.append(DeployedEnv(env_id="e", env_version=1, mcp_url="http://mcp"))
        step = EnvOutcomeVerifierTaskStep(id="outcome", version=None, env_id="e", file_artifact_id="s", verifier_id="v")
        done = await step.execute(form(context))
        return done.metadata["verifications"]["v"]["results"][0]["given"]

    await _pinned("scripted_verify", judged)
