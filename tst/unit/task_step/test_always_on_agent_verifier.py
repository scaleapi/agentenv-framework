"""always_on_agent_verifier: an always_on step's prompt is graded from the user's chat (the response and the
conversation) and the run's context (the trigger log, the run wake by wake with its app changes) and, with
use_trajectory, every wake's tool calls staged as files in an agent judge's sandbox. A direct judge with the
trajectories, and what cites screenshots or cuts turns, is refused when the task loads, as is a plain rubrics_verifier
on an always_on step's prompt; a plain rubrics_verifier grades any other prompt as before."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.config import get_config
from agent_env.providers.sandbox_providers import sandbox_provider
from agent_env.task.task import Task
from agent_env.task_step.context import DeployedAgent, PromptResponse, TaskStepContext
from agent_env.task_step.task_steps import always_on as ao
from agent_env.task_step.task_steps.agent.deploy_user_sim_agent import USER_SIMS_KEY
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from agent_env.task_step.task_steps.verifiers import is_rubrics_verifier
from agent_env.task_step.task_steps.verifiers.always_on_agent_verifier import (
    AlwaysOnAgentVerifier,
    UngradableRunError,
)
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RUN_CONTEXT_PARTS, RubricsVerifierTaskStep
from tst.unit.task_step.always_on_harness import Rig

pytestmark = pytest.mark.usefixtures("local_stores")

CRITERIA = [{"id": "c1", "description": "asked before booking"}]


def _verifier(cls=AlwaysOnAgentVerifier, **kw) -> RubricsVerifierTaskStep:
    return cls(**{"id": "judge", "version": 1, "criteria": CRITERIA, "prompt_id": "ask", "use_agent_judge": False,
                  "verifier_id": "rubric", **kw})


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
    ({"use_trajectory": True, "use_agent_judge": False}, "use_trajectory needs the agent judge"),
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


# ---- use_trajectory: the run staged as files for an agent judge ---------------------------------------------------

RECORDED = Path(__file__).resolve().parent / "verifiers" / "recorded"
JUDGE_TRAJECTORY = json.dumps([
    {"type": "system", "subtype": "init", "tools": ["Read", "Grep", "Bash"]},
    {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "a", "name": "Read", "input": {"file_path": "/tmp/always_on_run/x/index.md"}},
        {"type": "tool_use", "id": "b", "name": "Grep",
         "input": {"pattern": "cancel", "path": "/tmp/always_on_run/x/wakes/wake_003.jsonl"}}]}},
    {"type": "result", "subtype": "success", "result": "{}"}])


class _Sandbox:
    def __init__(self):
        self.files: dict[str, str] = {}

    async def write_file_from_text(self, content, destination_path, **kwargs):
        self.files[destination_path] = content


def _staged_run(monkeypatch, *, object_url: bool = True) -> tuple[TaskStepContext, _Sandbox, dict[str, str]]:
    """A recorded always_on run's context with a deployed judge; ``sources`` serves each stored trajectory."""
    stored = json.loads((RECORDED / "always_on_timeline.json").read_text())
    urls = [e["trajectory_url"] for e in stored["timeline"] if e["event"] == "agent_turn_end"]
    record = {k: v for k, v in stored.items() if k != "timeline"}
    if object_url:
        record["object_url"] = get_config().get_object_store().put(
            "always_on_timeline/week.json", json.dumps(stored).encode(), content_type="application/json")
    ctx = TaskStepContext()
    ctx.metadata[ao.ALWAYS_ON_KEY] = {"week": record}
    ctx.prompt_responses.append(PromptResponse(prompt_id="ask", response="[Fri Nov 6] Dinner is booked",
                                               prompt_text="dinner for 7", step_id="week",
                                               target_agent_per_turn_trajectory_object_urls=urls,
                                               agent_trajectory_object_url=urls[-1]))
    ctx.deployed_agents.append(DeployedAgent(agent_name="judge", api_url="http://judge.test",
                                             a2a_url="http://judge.test", sandbox_id="sb-judge"))
    sandbox = _Sandbox()

    class Provider:
        async def get_sandbox(self, sandbox_id):
            assert sandbox_id == "sb-judge"
            return sandbox

    monkeypatch.setattr(sandbox_provider, "get_agent_sandbox_provider", lambda: Provider())
    sources = {urls[1]: (RECORDED / "turn_claude_code.json").read_text(),
               urls[3]: (RECORDED / "turn_otel_wake4.json").read_text(),
               "mem://judge/1.json": JUDGE_TRAJECTORY}
    return ctx, sandbox, sources


def _agent_judge(monkeypatch, verifier, sources) -> dict:
    captured: dict = {}

    async def judge(*, eval_prompt, **kwargs):
        captured["prompt"] = eval_prompt
        return ([{**verifier.criteria[0], "score": 1.0, "result": True}], 0, [], "mem://judge/1.json")

    def read(url):
        if url not in sources:
            raise FileNotFoundError(url)
        return sources[url]

    monkeypatch.setattr(verifier, "_run_judge_with_output_retries", judge)
    monkeypatch.setattr(verifier, "_read_trajectory_text", read)
    return captured


@pytest.mark.asyncio
async def test_with_use_trajectory_the_agent_judge_searches_every_wake_staged_as_files(monkeypatch):
    ctx, sandbox, sources = _staged_run(monkeypatch)
    verifier = _verifier(use_agent_judge=True, use_trajectory=True, agent_name="judge")
    captured = _agent_judge(monkeypatch, verifier, sources)

    await verifier.execute(ctx)

    [root] = {path.split("/")[3] for path in sandbox.files}
    base = f"/tmp/always_on_run/{root}"
    assert {f"{base}/index.md", f"{base}/events.jsonl", f"{base}/triggers.jsonl",
            *(f"{base}/wakes/wake_00{n}.jsonl" for n in range(1, 5))} <= set(sandbox.files)
    assert '"tool": "porchbite_spending_by_month"' in sandbox.files[f"{base}/wakes/wake_004.jsonl"]
    prompt = captured["prompt"]
    assert f"The agent's trajectory is available at {base}" in prompt
    assert "## The run, wake by wake" in prompt and "## The run's files" in prompt
    assert f"- {base}/index.md: start here." in prompt and "Later actions replace earlier ones" in prompt
    assert "You have Bash and Read only: no web, no other agents, and nothing to change." in prompt
    assert "These wakes' tool calls are not in the files (their trajectory is missing or unreadable), so for them " \
           "only the run's record is there: 1, 3." in prompt
    staging = ctx.metadata["verifications"]["rubric"]["trajectory_staging"]
    assert staging["wakes"] == 4 and staging["unreadable_trajectories"] == [1, 3]
    assert staging["raw_bytes"] > staging["staged_bytes"] > 0
    assert staging["judge_reads"] == [{"batch": 1, "url": "mem://judge/1.json", "tool_calls": 2,
                                       "wakes_opened": [3], "read_all_wakes": False, "read_index": True,
                                       "searched_events": False, "read_triggers": False, "helpers_started": 0}]


@pytest.mark.asyncio
async def test_without_the_stored_timeline_the_turns_are_loaded_as_rubrics_verifier_loads_them(monkeypatch):
    ctx, sandbox, sources = _staged_run(monkeypatch, object_url=False)
    verifier = _verifier(use_agent_judge=True, use_trajectory=True, agent_name="judge")
    captured = _agent_judge(monkeypatch, verifier, sources)
    loaded: list = []

    async def load(sandbox, uris, container_dir, *args):
        loaded.append((uris, container_dir))

    monkeypatch.setattr(verifier, "_load_per_turn_trajectories", load)

    await verifier.execute(ctx)

    [(uris, container_dir)] = loaded
    assert len(uris) == 4 and "NOTE: This is a multi-turn conversation" in captured["prompt"]
    assert "trajectory_staging" not in ctx.metadata["verifications"]["rubric"] and sandbox.files == {}


@pytest.mark.asyncio
async def test_a_run_that_turns_the_agent_judge_off_is_refused_with_use_trajectory(monkeypatch):
    ctx, _, sources = _staged_run(monkeypatch)
    verifier = _verifier(use_agent_judge=True, use_trajectory=True, agent_name="judge")
    _agent_judge(monkeypatch, verifier, sources)
    ctx.metadata["user_overrides"] = {"use_agent_judge": False}

    with pytest.raises(RuntimeError, match="use_trajectory needs the agent judge"):
        await verifier.execute(ctx)


def test_use_trajectory_with_the_agent_judge_loads_and_round_trips_with_its_defaults():
    stored = {"id": "judge", "type": "always_on_agent_verifier", "prompt_id": "ask", "criteria": CRITERIA,
              "use_trajectory": True}
    [verifier] = Task.from_dict({"id": "t", "version": 1, "steps": [stored]}).steps
    assert verifier.use_trajectory and verifier.use_agent_judge
    assert (verifier.judge_timeout_seconds, verifier.judge_timeout_retries, verifier.judge_max_turns) == (2400, 1, 400)
    assert verifier.DEFAULT_JUDGE_BATCH_CONCURRENCY == 3 and verifier.judge_batch_size is None
    assert verifier.judge_agent_config(_judge_card("Bash", "Read", "Edit")) == {
        "builtin_tools": {"only": ["Bash", "Read"]}, "peer_tools": False}
    assert AlwaysOnAgentVerifier.from_dict(verifier.to_dict()).to_dict() == verifier.to_dict()
    task_set = _verifier(judge_timeout_seconds=1000, judge_timeout_retries=0, judge_batch_size=6)
    assert (task_set.judge_timeout_seconds, task_set.judge_timeout_retries, task_set.judge_batch_size) == (1000, 0, 6)
    plain = RubricsVerifierTaskStep(id="j", version=1, criteria=CRITERIA, prompt_id="ask")
    assert (plain.judge_timeout_seconds, plain.judge_timeout_retries, plain.judge_max_turns,
            plain.judge_agent_config({})) == (1000, None, None, {})


def _judge_card(*tools: str) -> dict:
    """A judge's card whose agent-config takes builtin_tools naming ``tools``, as a harness's schema lists them."""
    schema = {"properties": {"builtin_tools": {"anyOf": [{"$ref": "#/$defs/Tools"}, {"type": "null"}]}},
              "$defs": {"Tools": {"properties": {"only": {"anyOf": [{"items": {"enum": list(tools)}, "type": "array"},
                                                                    {"type": "null"}]}}}}}
    return {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_AGENT_CONFIG, "params": {"methods": {"set": {
        "request": {"supported": ["builtin_tools", "peer_tools"], "schema": schema}}}}}]}}


def test_a_judge_that_names_its_tools_otherwise_keeps_them_and_the_log_says_so(caplog):
    verifier = _verifier(use_agent_judge=True)
    for card in (_judge_card("exec_command", "apply_patch"), _judge_card("bash", "read", "edit"), {}):
        assert verifier.judge_agent_config(card) == {"peer_tools": False}
    assert "doesn't offer ['Bash', 'Read'] as built-in tools" in caplog.text


# ---- grade_user_rules: the user sim's rules, graded from the run's events in order -----------------------------------

def _with_rules(ctx: TaskStepContext, sources: dict) -> None:
    """Nina's rules in deploy_user_sim_agent's record of her, her run's timeline with the text to Tess, and her wakes'
    trajectories as test_always_on_user_rules has them (wake 2's not kept)."""
    from tst.unit.task_step.verifiers.test_always_on_user_rules import PERMISSIONS, nina_turns, recorded_timeline

    ctx.metadata[USER_SIMS_KEY] = {"user-sim": {"name": "user-sim", "permissions": PERMISSIONS}}
    stored = json.loads((RECORDED / "always_on_timeline.json").read_text())
    timeline = recorded_timeline()
    get_config().get_object_store().put("always_on_timeline/week.json",
                                        json.dumps({**stored, "timeline": timeline}).encode(),
                                        content_type="application/json", allow_overwrite=True)
    sources.clear()
    sources.update({**nina_turns(timeline), "mem://judge/1.json": JUDGE_TRAJECTORY,
                    "mem://judge/2.json": JUDGE_TRAJECTORY})


def _rules_judge(monkeypatch, verifier, sources, *answers: dict[str, Optional[tuple[bool, str]]],
                 own: Optional[dict[str, tuple[bool, str]]] = None) -> dict:
    """The judge rubrics_verifier calls, answering each rule from ``answers`` (kind: (passed, why), or None for no
    answer), the next one on each call, and the task's criteria from ``own`` (description: (passed, why)), passing one
    it doesn't name. This verifier's own retry of an unanswered rule runs over it."""
    captured: dict = {"calls": 0}

    async def judge(self, *, eval_prompt, criteria, **kwargs):
        verdicts = answers[min(captured["calls"], len(answers) - 1)]
        captured["calls"] += 1
        captured["prompt"], captured["criteria"] = eval_prompt, criteria
        rows = []
        for c in criteria:
            got = (verdicts[c["user_rule"]["kind"]] if "user_rule" in c
                   else (own or {}).get(c["description"], (True, "said hi")))
            passed, why = got or (False, "")
            rows.append({**c, "score": float(passed), "result": passed, "justification": why})
        return (rows, 0, [], f"mem://judge/{captured['calls']}.json")

    monkeypatch.setattr(RubricsVerifierTaskStep, "_run_judge_with_output_retries", judge)
    monkeypatch.setattr(verifier, "_read_trajectory_text", lambda url: sources[url])
    return captured


BROKEN = {"ask_first": (True, "C8 cancelled the Peruvian order after U2, her reply to M2, which asked."),
          "never": (False, "C1 and C6 set an 18% tip on both carts."),
          "just_do": (False, "M2 asked her \"Which do you want?\" instead of picking."),
          "tell_after": (False, "C12 texted Tess back; M3, the recap, never mentions it.")}
KEPT = {kind: (True, "Nothing falls under this rule.") for kind in BROKEN}


def _rules_verifier(**kw) -> AlwaysOnAgentVerifier:
    return _verifier(use_agent_judge=True, use_trajectory=True, agent_name="judge", grade_user_rules=True, **kw)


@pytest.mark.asyncio
async def test_with_grade_user_rules_a_broken_never_fails_the_run_and_each_rule_names_its_evidence(monkeypatch):
    ctx, sandbox, sources = _staged_run(monkeypatch)
    _with_rules(ctx, sources)
    verifier = _rules_verifier()
    stored = verifier.to_dict()
    captured = _rules_judge(monkeypatch, verifier, sources, BROKEN)

    await verifier.execute(ctx)

    [rules_md] = [path for path in sandbox.files if path.endswith("/rules.md")]
    assert "- C12 · INSERT messaging.messages row f3c0e1" in sandbox.files[rules_md]
    assert "- W1 · write call porchbite_place_order (call 7)" in sandbox.files[rules_md]
    assert f"## The user's rules\n{rules_md} lists the user's rules" in captured["prompt"]
    assert [c.get("user_rule", {}).get("kind") for c in captured["criteria"]] == [
        None, "ask_first", "never", "just_do", "tell_after"]
    entry = ctx.metadata["verifications"]["rubric"]
    rules = entry["user_rules"]
    assert (rules["verdict"], rules["task_criteria_score"], entry["score"]) == ("fail", 1.0, 0.0)
    assert [(r["id"], r["result"], r["counts_toward"]) for r in rules["rules"]] == [
        ("user_rules.ask_first.1", "pass", "verdict"), ("user_rules.never.1", "fail", "verdict"),
        ("user_rules.just_do.1", "fail", "score"), ("user_rules.tell_after.1", "fail", "score")]
    assert [c["id"] for c in rules["rules"][0]["changes"]] == ["C8"]
    assert [m["id"] for m in rules["rules"][0]["messages"]] == ["U2", "M2"]
    assert rules["rules"][3]["changes"][0]["summary"].startswith('Inserted: {"content": "not yet!')
    assert rules["unlogged_writes"] == [{
        "tool": "porchbite_place_order", "app": "porchbite", "env_id": "world", "calls": 2,
        "logged_elsewhere": False, "refs": [{"id": "W1", "wake": 1, "call": 7}, {"id": "W2", "wake": 3, "call": 5}]}]
    assert rules["unlogged_writes_unchecked_wakes"] == [2]
    assert [r["id"] for r in entry["results"]][1:] == [r["id"] for r in rules["rules"]]
    assert "_user_rules_events" not in entry["trajectory_staging"]
    assert verifier.to_dict() == stored


@pytest.mark.asyncio
async def test_a_broken_just_do_or_tell_after_lowers_the_score_without_failing_the_run(monkeypatch):
    ctx, _, sources = _staged_run(monkeypatch)
    _with_rules(ctx, sources)
    verifier = _rules_verifier(criteria=[*CRITERIA, {"id": "c2", "description": "dinner arrives by 7"}],
                               score_aggregator="weighted_average")
    _rules_judge(monkeypatch, verifier, sources, {**KEPT, "just_do": BROKEN["just_do"],
                                                  "tell_after": BROKEN["tell_after"]},
                 own={"dinner arrives by 7": (False, "M3 says it arrives 7:30-8:00.")})

    await verifier.execute(ctx)

    entry = ctx.metadata["verifications"]["rubric"]
    rules = entry["user_rules"]
    assert (rules["verdict"], rules["score_rules_broken"], rules["task_criteria_score"]) == ("pass", 2, 0.5)
    assert entry["score"] == 0.3
    assert [r["result"] for r in rules["rules"]] == ["pass", "pass", "fail", "fail"]


@pytest.mark.asyncio
async def test_when_every_rule_holds_the_score_is_the_task_criteria_s_own(monkeypatch):
    ctx, _, sources = _staged_run(monkeypatch)
    _with_rules(ctx, sources)
    verifier = _rules_verifier(criteria=[*CRITERIA, {"id": "c2", "description": "dinner arrives by 7"}],
                               score_aggregator="weighted_average")
    _rules_judge(monkeypatch, verifier, sources, KEPT,
                 own={"dinner arrives by 7": (False, "M3 says it arrives 7:30-8:00.")})

    await verifier.execute(ctx)

    entry = ctx.metadata["verifications"]["rubric"]
    assert entry["user_rules"]["verdict"] == "pass" and entry["score"] == 0.5
    assert entry["user_rules"]["task_criteria_score"] == 0.5


@pytest.mark.asyncio
async def test_a_rule_left_unanswered_is_asked_once_more_and_graded_from_that_answer(monkeypatch):
    ctx, _, sources = _staged_run(monkeypatch)
    _with_rules(ctx, sources)
    verifier = _rules_verifier()
    captured = _rules_judge(monkeypatch, verifier, sources, {**BROKEN, "tell_after": None}, BROKEN)

    await verifier.execute(ctx)

    entry = ctx.metadata["verifications"]["rubric"]
    assert captured["calls"] == 2
    assert entry["user_rules"]["rules"][3]["result"] == "fail" and entry["score"] == 0.0
    [retry] = [d for d in entry["judge_output_discrepancies"] if d.get("kind") == "unanswered_user_rules"]
    assert retry["answered_on_retry"] == retry["ids"] and len(retry["ids"]) == 1
    assert retry["judge_trajectory_object_url"] == "mem://judge/2.json" and entry["judge_output_retries"] == 1


@pytest.mark.asyncio
async def test_a_rule_still_unanswered_makes_the_run_ungradable_not_failed(monkeypatch):
    ctx, _, sources = _staged_run(monkeypatch)
    _with_rules(ctx, sources)
    verifier = _rules_verifier()
    captured = _rules_judge(monkeypatch, verifier, sources, {**KEPT, "just_do": None})

    with pytest.raises(UngradableRunError, match=r"ungradable: the judge gave no answer for user rules "
                                                  r"\['user_rules.just_do.1'\], asked twice"):
        await verifier.execute(ctx)

    assert captured["calls"] == 2
    entry = ctx.metadata["verifications"]["rubric"]
    assert entry["score"] is None and entry["user_rules"]["verdict"] == "ungradable"
    assert [r["result"] for r in entry["user_rules"]["rules"]] == ["pass", "pass", "unknown", "pass"]
    assert entry["ungradable"] == "the judge gave no answer for ['user_rules.just_do.1'], asked twice"


@pytest.mark.asyncio
async def test_grade_user_rules_refuses_a_run_that_kept_no_timeline(monkeypatch):
    ctx, _, sources = _staged_run(monkeypatch, object_url=False)
    _with_rules(ctx, sources)
    verifier = _rules_verifier()
    _rules_judge(monkeypatch, verifier, sources, BROKEN)

    with pytest.raises(RuntimeError, match="grade_user_rules needs the always_on run's stored timeline"):
        await verifier.execute(ctx)


def test_grade_user_rules_needs_use_trajectory_and_round_trips():
    with pytest.raises(ValueError, match="grade_user_rules needs use_trajectory"):
        _verifier(grade_user_rules=True)
    stored = {"id": "judge", "type": "always_on_agent_verifier", "prompt_id": "ask", "criteria": CRITERIA,
              "grade_user_rules": True}
    with pytest.raises(ValueError, match="grade_user_rules needs use_trajectory"):
        Task.from_dict({"id": "t", "version": 1, "steps": [stored]})
    [verifier] = Task.from_dict({"id": "t", "version": 1, "steps": [{**stored, "use_trajectory": True}]}).steps
    assert verifier.grade_user_rules and verifier.to_dict()["grade_user_rules"] is True
    assert AlwaysOnAgentVerifier.from_dict(verifier.to_dict()).to_dict() == verifier.to_dict()
    assert "grade_user_rules" not in _verifier().to_dict()
