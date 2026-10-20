"""The run a verifier grades: the stored conversation, the firing logs, and the cut as of a turn."""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime

from types import SimpleNamespace

import pytest

from agent_env.a2a_agent import conversation_store
from agent_env.artifact import FileArtifact
from agent_env.config import get_config
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.verifiers.env_outcome_verifier import EnvOutcomeVerifierTaskStep
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep, render_run_context
from agent_env.task_step.task_steps.verifiers.run_context import RunContext, load_run_context

TURNS = [("Can you book lunch with Dana?", "Booked for noon."),
         ("Make it 12:30 instead.", "Moved to 12:30."),
         ("Thanks, that's all.", "Glad to help.")]


def _text(text: str) -> list[dict]:
    return [{"kind": "text", "text": text}]


def _between(earlier: str, later: str) -> str:
    a, b = datetime.fromisoformat(earlier), datetime.fromisoformat(later)
    return (a + (b - a) / 2).isoformat()


def recorded_run(turns=TURNS) -> tuple[TaskStepContext, PromptResponse, dict]:
    """A three-turn run as prompt_agent records it: conversation, trigger and user-sim marks, and both firing logs."""
    conversation_id = f"conv-{uuid.uuid4().hex[:8]}"
    conversation_store.create_conversation(conversation_id, task_instance_id="run-1", source_agent_name="dana_sim",
                                           target_agent_name="solver")
    for user, agent in turns:
        conversation_store.add_a2a_task(conversation_id, parts=_text(user), a2a_task_id=uuid.uuid4().hex, role="user")
        time.sleep(0.005)  # distinct message stamps, so events can fall between them
        conversation_store.complete_a2a_task(conversation_id, parts=_text(agent), role="agent")
        time.sleep(0.005)
    messages = conversation_store.get_conversation(conversation_id)["messages"]
    turn_starts = [m["ts"] for m in messages if m["role"] == "user"]
    after_reply_1 = _between(messages[1]["ts"], turn_starts[1])
    env_events = [
        {"seq": 1, "ts": after_reply_1, "kind": "detected", "trigger_id": "confirm"},
        {"seq": 2, "ts": after_reply_1, "kind": "fired", "trigger_id": "confirm", "fire_count": 1},
        {"seq": 3, "ts": _between(turn_starts[2], messages[5]["ts"]), "kind": "fired", "trigger_id": "late",
         "fire_count": 1},
    ]
    envelope = {"env_id": "calendar", "state": {"triggers": [], "events": env_events}}
    object_url = get_config().get_object_store().put(f"env_trigger_state/{conversation_id}.json",
                                                     json.dumps(envelope).encode(), content_type="application/json")
    agent_events = [{"seq": 1, "ts": after_reply_1, "kind": "fired", "trigger_id": "reschedule"}]
    context = TaskStepContext(metadata={
        "a2a_conversations": {"prompt": conversation_id},
        "agent_trigger_firings": {"prompt": [{"turn": 1, "fired": ["reschedule"]}, {"turn": 2, "fired": []}]},
        "usersim_turn_outputs": {"prompt": [{"turn": 2, "fields": {"mood": "satisfied"}}]},
        "env_trigger_state": {"calendar": {"object_url": object_url, "capture_step_id": "prompt"}},
        "agent_trigger_registrations": [{"step_id": "arm", "agent_name": "dana_sim", "added": ["reschedule"]}],
        "agent_trigger_state": {"prompt": agent_events},
    })
    prompt_response = PromptResponse(prompt_id="p1", response="Glad to help.", prompt_text=TURNS[0][0], step_id="prompt")
    context.prompt_responses.append(prompt_response)
    return context, prompt_response, {"env": env_events, "agent": agent_events, "turn_starts": turn_starts}


@pytest.mark.asyncio
async def test_the_run_context_has_every_turn_and_both_firing_logs():
    context, prompt_response, recorded = recorded_run()

    run = await load_run_context(context, prompt_response)

    assert run.turns == 3 and run.unavailable == {}
    assert [(m["turn"], m["role"], m.get("author"), m["text"]) for m in run.conversation] == [
        (1, "user", "dana_sim", TURNS[0][0]), (1, "agent", "solver", TURNS[0][1]),
        (2, "user", "dana_sim", TURNS[1][0]), (2, "agent", "solver", TURNS[1][1]),
        (3, "user", "dana_sim", TURNS[2][0]), (3, "agent", "solver", TURNS[2][1]),
    ]
    user = [m for m in run.conversation if m["role"] == "user"]
    assert "fired" not in user[0] and user[1]["fired"] == ["reschedule"] and "fired" not in user[2]
    assert user[2]["fields"] == {"mood": "satisfied"} and "fields" not in user[1]
    assert run.env_trigger_events == {"calendar": recorded["env"]}
    assert run.agent_trigger_events == recorded["agent"]
    assert json.loads(json.dumps(run.to_dict()))["conversation"][0]["parts"] == _text(TURNS[0][0])


@pytest.mark.asyncio
async def test_as_of_a_turn_the_run_stops_before_the_next_turn_begins():
    context, prompt_response, recorded = recorded_run()
    run = await load_run_context(context, prompt_response)

    first, second = run.as_of_turn(1), run.as_of_turn(2)

    assert [m["turn"] for m in first.conversation] == [1, 1]
    assert first.env_trigger_events == {"calendar": recorded["env"][:2]}
    assert first.agent_trigger_events == recorded["agent"]
    assert [m["turn"] for m in second.conversation] == [1, 1, 2, 2]
    assert second.env_trigger_events == {"calendar": recorded["env"][:2]}
    assert run.as_of_turn(3) is run and run.as_of_turn(7) is run


@pytest.mark.asyncio
async def test_a_prompt_with_no_stored_conversation_is_one_turn_of_its_prompt_and_reply():
    prompt_response = PromptResponse(prompt_id="p1", response="done", prompt_text="do it", step_id="turn")

    run = await load_run_context(TaskStepContext(prompt_responses=[prompt_response]), prompt_response)

    assert [(m["turn"], m["role"], m["text"]) for m in run.conversation] == [(1, "user", "do it"), (1, "agent", "done")]
    assert run.env_trigger_events == {} and run.agent_trigger_events == [] and run.unavailable == {}


@pytest.mark.asyncio
async def test_what_the_run_meant_to_record_but_cannot_be_read_is_named_not_left_empty():
    context, prompt_response, _ = recorded_run()
    context.metadata["a2a_conversations"]["prompt"] = "conv-gone"
    context.metadata["env_trigger_state"] = {
        "calendar": {"error": "TimeoutError: capture budget of 90s exceeded"},
        "mail": {"object_url": get_config().get_object_store().object_url("env_trigger_state/nothing-here.json")},
    }

    run = await load_run_context(context, prompt_response)

    assert run.conversation == [] and run.env_trigger_events == {}
    assert run.unavailable["conversation"] == "conversation 'conv-gone' is not in the store"
    assert run.unavailable["env_trigger_events.calendar"] == "TimeoutError: capture budget of 90s exceeded"
    assert "env_trigger_events.mail" in run.unavailable


@pytest.mark.asyncio
@pytest.mark.parametrize("dropped", ["agent_trigger_firings", "agent_trigger_registrations"])
async def test_a_user_agent_firing_log_that_was_not_read_back_is_named_not_left_empty(dropped):
    context, prompt_response, _ = recorded_run()
    for key in ("agent_trigger_state", dropped):
        del context.metadata[key]

    run = await load_run_context(context, prompt_response)

    assert run.agent_trigger_events == [] and "agent_trigger_events" in run.unavailable
    text = render_run_context(run, ["trigger_log"])
    assert f"(not available: agent_trigger_events: {run.unavailable['agent_trigger_events']})" in text
    assert "(no trigger firings were recorded)" not in text


@pytest.mark.asyncio
async def test_a_user_agent_with_no_triggers_of_its_own_has_no_firing_log_to_miss():
    context, prompt_response, _ = recorded_run()
    for key in ("agent_trigger_state", "agent_trigger_firings"):
        del context.metadata[key]
    context.metadata["agent_trigger_registrations"] = [{"step_id": "arm", "agent_name": "other_sim", "added": ["x"]}]

    run = await load_run_context(context, prompt_response)

    assert run.agent_trigger_events == [] and run.unavailable == {}


@pytest.mark.asyncio
async def test_without_the_conversation_a_turn_cut_withholds_the_firing_logs_it_cannot_cut():
    context, prompt_response, recorded = recorded_run()
    context.metadata["a2a_conversations"]["prompt"] = "conv-gone"
    run = await load_run_context(context, prompt_response)

    first = run.as_of_turn(1)

    assert run.env_trigger_events == {"calendar": recorded["env"]} and run.agent_trigger_events == recorded["agent"]
    assert first.conversation == [] and first.env_trigger_events == {} and first.agent_trigger_events == []
    assert first.unavailable["conversation"] == run.unavailable["conversation"]
    assert first.unavailable["env_trigger_events.calendar"] == first.unavailable["agent_trigger_events"]


@pytest.mark.asyncio
async def test_without_a_prompt_the_run_context_has_the_env_firing_logs_alone():
    context, _, recorded = recorded_run()

    run = await load_run_context(context, None)

    assert run.conversation == [] and run.agent_trigger_events == []
    assert run.env_trigger_events == {"calendar": recorded["env"]}


def test_an_empty_run_context_has_no_turns():
    assert RunContext().turns == 0 and RunContext().as_of_turn(1) == RunContext()


# ---------------------------------------------------------------- the judge sees it

TRAJECTORIES = {"s3://t/1.json": "TURN_ONE_EVENTS", "s3://t/2.json": "TURN_TWO_EVENTS", "s3://t/3.json": "TURN_THREE_EVENTS"}


def _judged_run() -> tuple[TaskStepContext, dict]:
    context, prompt_response, recorded = recorded_run()
    prompt_response.target_agent_per_turn_trajectory_object_urls = list(TRAJECTORIES)
    prompt_response.agent_trajectory_object_url = "s3://t/3.json"
    prompt_response.source_agent_per_turn_prompt_parts = [None, _text(TURNS[1][0]), _text(TURNS[2][0])]
    return context, recorded


def _verifier(**kw) -> RubricsVerifierTaskStep:
    return RubricsVerifierTaskStep(id="verify", version=1, criteria=[{"id": "c1", "description": "outcome"}],
                                   prompt_id="p1", use_agent_judge=False, use_trajectory=True, verifier_id="vt", **kw)


def _judge_prompt(monkeypatch, verifier) -> dict:
    captured: dict = {}

    async def fake_run(*, eval_prompt, **kwargs):
        captured["eval_prompt"] = eval_prompt
        return ([{**verifier.criteria[0], "score": 1.0, "result": True}], 0, [], None)

    monkeypatch.setattr(verifier, "_read_trajectory_text", lambda uri: TRAJECTORIES[uri])
    monkeypatch.setattr(verifier, "_run_judge_with_output_retries", fake_run)
    return captured


@pytest.mark.asyncio
async def test_the_judge_sees_the_conversation_and_the_firing_logs_it_asks_for(monkeypatch):
    v = _verifier(run_context_for_judge=["conversation", "trigger_log"])
    captured = _judge_prompt(monkeypatch, v)
    context, recorded = _judged_run()

    await v.execute(context)

    prompt = captured["eval_prompt"]
    conversation = prompt[prompt.index("## Conversation"):prompt.index("## Trigger firings")]
    assert all(user in conversation and agent in conversation for user, agent in TURNS)
    assert f"- user (dana_sim, written by triggers ['reschedule']): {TURNS[1][0]}" in conversation
    assert f"- agent (solver): {TURNS[2][1]}" in conversation and "### Turn 3" in conversation
    log = prompt[prompt.index("## Trigger firings"):]
    assert "### env calendar" in log and "### user agent" in log
    assert json.dumps({k: v for k, v in recorded["env"][2].items() if k != "seq"}) in log and '"seq"' not in log


@pytest.mark.asyncio
async def test_graded_as_of_a_turn_the_judge_sees_neither_later_turns_nor_later_firings(monkeypatch):
    v = _verifier(run_context_for_judge=["conversation", "trigger_log"], grade_up_to_turn=2)
    captured = _judge_prompt(monkeypatch, v)
    context, _ = _judged_run()

    await v.execute(context)

    prompt = captured["eval_prompt"]
    assert TURNS[1][1] in prompt and "### Turn 3" not in prompt and TURNS[2][0] not in prompt
    assert '"trigger_id": "confirm"' in prompt and '"trigger_id": "late"' not in prompt


@pytest.mark.asyncio
async def test_graded_as_of_a_turn_without_the_conversation_the_judge_sees_no_firings(monkeypatch):
    v = _verifier(run_context_for_judge=["trigger_log"], grade_up_to_turn=1)
    captured = _judge_prompt(monkeypatch, v)
    context, _ = _judged_run()
    context.metadata["a2a_conversations"]["prompt"] = "conv-gone"

    await v.execute(context)

    prompt = captured["eval_prompt"]
    assert '"trigger_id"' not in prompt
    assert "(not available: env_trigger_events.calendar: withheld" in prompt
    assert "(not available: agent_trigger_events: withheld" in prompt


@pytest.mark.asyncio
async def test_the_judge_sees_only_the_parts_it_asks_for(monkeypatch):
    for parts, shown, hidden in ((None, [], ["## Conversation", "## Trigger firings"]),
                                 (["conversation"], ["## Conversation"], ["## Trigger firings"]),
                                 (["trigger_log"], ["## Trigger firings"], ["## Conversation"])):
        v = _verifier(run_context_for_judge=parts)
        captured = _judge_prompt(monkeypatch, v)
        await v.execute(_judged_run()[0])
        assert all(s in captured["eval_prompt"] for s in shown) and not any(s in captured["eval_prompt"] for s in hidden)


def test_a_long_message_and_a_long_firing_log_are_bounded_and_a_missing_log_is_named():
    max_chars = RubricsVerifierTaskStep.RUN_CONTEXT_MESSAGE_MAX_CHARS
    long = "x" * (max_chars + 50)
    run = RunContext(
        conversation=[{"turn": 1, "role": "user", "text": long, "parts": _text(long) + [{"kind": "file"}], "ts": None}],
        env_trigger_events={"calendar": [{"seq": i, "kind": "fired", "trigger_id": f"t{i}"}
                                         for i in range(RubricsVerifierTaskStep.RUN_CONTEXT_EVENTS_MAX + 3)]},
        unavailable={"env_trigger_events.mail": "TimeoutError: capture budget of 90s exceeded"})

    text = render_run_context(run, ["conversation", "trigger_log"])

    assert "x" * max_chars + "… (+1 non-text part)" in text and long not in text
    assert "(3 earlier events not shown)" in text and '"t2"' not in text and '"t3"' in text
    assert "(not available: env_trigger_events.mail: TimeoutError: capture budget of 90s exceeded)" in text
    assert "(no trigger firings were recorded)" in render_run_context(RunContext(), ["trigger_log"])


@pytest.mark.parametrize("bad", [[], ["conversation", "screens"], "conversation", ["Conversation"]])
def test_run_context_for_judge_takes_only_known_parts(bad):
    with pytest.raises(ValueError, match="run_context_for_judge"):
        _verifier(run_context_for_judge=bad)


def test_run_context_for_judge_round_trips_and_is_omitted_when_unset():
    v = _verifier(run_context_for_judge=["trigger_log", "conversation", "trigger_log"])
    assert v.to_dict()["run_context_for_judge"] == ["trigger_log", "conversation"]
    assert RubricsVerifierTaskStep.from_dict(v.to_dict()).run_context_for_judge == ["trigger_log", "conversation"]
    assert "run_context_for_judge" not in _verifier().to_dict()


# ---------------------------------------------------------------- a verify(mcp_url, run) script gets it

RUN_SCRIPT = b"""
async def verify(mcp_url, run):
    rescheduled = [m for m in run.conversation if m.get("fired") == ["reschedule"]]
    return [{"id": "reschedule-reached-the-agent", "score": 1.0, "result": bool(rescheduled),
             "turns": run.turns, "env_fired": [e["trigger_id"] for e in run.env_trigger_events["calendar"]
                                               if e["kind"] == "fired"]}]
"""


def _outcome(script: bytes, monkeypatch, **kw) -> EnvOutcomeVerifierTaskStep:
    monkeypatch.setattr(FileArtifact, "get", staticmethod(lambda id, version=None: SimpleNamespace(load=lambda: script)))
    return EnvOutcomeVerifierTaskStep(id="outcome", version=None, env_id="calendar", file_artifact_id="script",
                                      verifier_id="v", **kw)


@pytest.mark.asyncio
async def test_a_verify_asking_for_run_is_handed_the_run_it_grades(monkeypatch):
    context, _, _ = recorded_run()
    context.deployed_envs.append(SimpleNamespace(env_id="calendar", mcp_url="http://mcp"))

    await _outcome(RUN_SCRIPT, monkeypatch, prompt_id="p1").execute(context)

    [row] = context.metadata["verifications"]["v"]["results"]
    assert (row["result"], row["turns"], row["env_fired"]) == (True, 3, ["confirm", "late"])


@pytest.mark.asyncio
async def test_without_a_prompt_id_the_runs_last_prompt_is_graded(monkeypatch):
    context, _, _ = recorded_run()
    context.prompt_responses.insert(0, PromptResponse(prompt_id="warmup", response="ok", prompt_text="hi", step_id="w"))
    context.deployed_envs.append(SimpleNamespace(env_id="calendar", mcp_url="http://mcp"))

    await _outcome(RUN_SCRIPT, monkeypatch).execute(context)

    assert context.metadata["verifications"]["v"]["results"][0]["turns"] == 3


@pytest.mark.asyncio
async def test_a_verify_with_an_unrelated_second_parameter_is_called_as_before(monkeypatch):
    context, _, _ = recorded_run()
    context.deployed_envs.append(SimpleNamespace(env_id="calendar", mcp_url="http://mcp"))
    script = b"async def verify(mcp_url, retries=3):\n    return [{'id': 'c', 'score': 1.0, 'retries': retries}]\n"

    await _outcome(script, monkeypatch, prompt_id="p1").execute(context)

    assert context.metadata["verifications"]["v"]["results"] == [{"id": "c", "score": 1.0, "retries": 3}]


@pytest.mark.asyncio
async def test_a_one_argument_verify_is_called_as_before(monkeypatch):
    context, _, _ = recorded_run()
    context.deployed_envs.append(SimpleNamespace(env_id="calendar", mcp_url="http://mcp"))
    script = b"async def verify(mcp_url):\n    return [{'id': 'c', 'score': 1.0, 'url': mcp_url}]\n"

    await _outcome(script, monkeypatch, prompt_id="p1").execute(context)

    assert context.metadata["verifications"]["v"]["results"] == [{"id": "c", "score": 1.0, "url": "http://mcp"}]


@pytest.mark.asyncio
async def test_an_unknown_prompt_id_fails_the_step(monkeypatch):
    context, _, _ = recorded_run()
    context.deployed_envs.append(SimpleNamespace(env_id="calendar", mcp_url="http://mcp"))

    with pytest.raises(RuntimeError, match="prompt_id='nope' not found"):
        await _outcome(RUN_SCRIPT, monkeypatch, prompt_id="nope").execute(context)


def test_prompt_id_round_trips_and_is_omitted_when_unset(monkeypatch):
    step = _outcome(RUN_SCRIPT, monkeypatch, prompt_id="p1")
    assert EnvOutcomeVerifierTaskStep.from_dict(step.to_dict()).prompt_id == "p1"
    assert "prompt_id" not in _outcome(RUN_SCRIPT, monkeypatch).to_dict()


def _stored_prompt(step_id: str, user: str, agent: str) -> tuple[str, list[dict]]:
    conversation_id = f"conv-{uuid.uuid4().hex[:8]}"
    conversation_store.create_conversation(conversation_id, task_instance_id="run-1", source_agent_name="dana_sim",
                                           target_agent_name="solver")
    conversation_store.add_a2a_task(conversation_id, parts=_text(user), a2a_task_id=uuid.uuid4().hex, role="user")
    time.sleep(0.005)
    conversation_store.complete_a2a_task(conversation_id, parts=_text(agent), role="agent")
    time.sleep(0.005)
    return conversation_id, conversation_store.get_conversation(conversation_id)["messages"]


@pytest.mark.asyncio
async def test_an_earlier_prompt_sees_only_the_env_events_from_before_the_next_prompt_began():
    first_id, first = _stored_prompt("prompt-1", "Book lunch with Dana.", "Booked for noon.")
    second_id, second = _stored_prompt("prompt-2", "Now book dinner.", "Booked for seven.")
    env_events = [  # the env's whole log, as the second prompt's end-of-prompt capture stored it
        {"seq": 1, "ts": _between(first[0]["ts"], first[1]["ts"]), "kind": "fired", "trigger_id": "lunch_confirm"},
        {"seq": 2, "ts": _between(second[0]["ts"], second[1]["ts"]), "kind": "fired", "trigger_id": "dinner_confirm"},
    ]
    envelope = {"env_id": "calendar", "state": {"triggers": [], "events": env_events}}
    object_url = get_config().get_object_store().put(f"env_trigger_state/{second_id}.json",
                                                     json.dumps(envelope).encode(), content_type="application/json")
    context = TaskStepContext(metadata={
        "a2a_conversations": {"prompt-1": first_id, "prompt-2": second_id},
        "env_trigger_state": {"calendar": {"object_url": object_url, "capture_step_id": "prompt-2"}},
    })
    for prompt_id, step_id, text, reply in (("p1", "prompt-1", "Book lunch with Dana.", "Booked for noon."),
                                            ("p2", "prompt-2", "Now book dinner.", "Booked for seven.")):
        context.prompt_responses.append(PromptResponse(prompt_id=prompt_id, response=reply, prompt_text=text,
                                                       step_id=step_id))

    graded_first = await load_run_context(context, context.prompt_responses[0])
    graded_last = await load_run_context(context, context.prompt_responses[1])

    assert [e["trigger_id"] for e in graded_first.env_trigger_events["calendar"]] == ["lunch_confirm"]
    assert [e["trigger_id"] for e in graded_last.env_trigger_events["calendar"]] == ["lunch_confirm", "dinner_confirm"]
