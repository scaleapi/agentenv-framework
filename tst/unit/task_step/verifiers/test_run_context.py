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
from agent_env.task_step.task_steps.verifiers import Run, RunRecordUnavailable
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from agent_env.task_step.task_steps.verifiers.run_context import (EnvClock, _fold, _message, _run, load_run_context,
                                                                  render_run_context)

TURNS = [("Can you book lunch with Dana?", "Booked for noon."),
         ("Make it 12:30 instead.", "Moved to 12:30."),
         ("Thanks, that's all.", "Glad to help.")]


def _text(text: str) -> list[dict]:
    return [{"kind": "text", "text": text}]


def _between(earlier: str, later: str) -> str:
    a, b = datetime.fromisoformat(earlier), datetime.fromisoformat(later)
    return (a + (b - a) / 2).isoformat()


def _render(run: Run, parts: list[str]) -> str:
    return render_run_context(run, parts, message_max_chars=RubricsVerifierTaskStep.RUN_CONTEXT_MESSAGE_MAX_CHARS,
                              firings_max=RubricsVerifierTaskStep.RUN_CONTEXT_FIRINGS_MAX)


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
        {"seq": 1, "ts": after_reply_1, "kind": "detected", "trigger_id": "confirm",
         "provoking": {"tool": "book", "role": "solver"}},
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
async def test_the_run_has_every_turn_and_one_firing_per_fire_each_in_its_turn():
    context, prompt_response, recorded = recorded_run()

    run = await load_run_context(context, prompt_response)

    assert len(run.turns) == 3 and run.unavailable == {}
    assert [(t.number, t.message.author, t.message.text, t.reply.author, t.reply.text) for t in run.turns] == [
        (n, "dana_sim", user, "solver", agent) for n, (user, agent) in enumerate(TURNS, 1)]
    assert [t.message.written_by_triggers for t in run.turns] == [(), ("reschedule",), ()]
    assert [t.message.user_sim for t in run.turns] == [None, None, {"mood": "satisfied"}]
    assert run.turn(2).message.sent_at == datetime.fromisoformat(recorded["turn_starts"][1])
    assert [(f.source, f.trigger_id, f.turn, f.ok) for f in run.firings()] == [
        ("env:calendar", "confirm", 1, True), ("user_agent", "reschedule", 1, True), ("env:calendar", "late", 3, True)]
    confirm = run.firing("confirm", env="calendar")
    assert confirm.provoked_by == {"tool": "book", "role": "solver"} and confirm.at == datetime.fromisoformat(
        recorded["env"][1]["ts"])
    assert run.firing("late").provoked_by is None and run.firing("never") is None
    assert [f.trigger_id for f in run.turn(1).firings] == ["confirm", "reschedule"] and run.turn(2).firings == ()
    assert run.firings(env="calendar") == run.firings("confirm") + run.firings("late")


def test_the_gateway_log_folds_into_one_firing_per_fire():
    ts = [f"2026-10-20T10:00:0{i}+00:00" for i in range(8)]
    events = [
        {"seq": 1, "ts": ts[0], "kind": "detected", "trigger_id": "mirror", "provoking": {"tool": "a"}},
        {"seq": 2, "ts": ts[1], "kind": "detected", "trigger_id": "mirror", "provoking": {"tool": "b"}, "queued": 1},
        {"seq": 3, "ts": ts[2], "kind": "action_ok", "trigger_id": "mirror", "action_index": 0},
        {"seq": 4, "ts": ts[3], "kind": "fired", "trigger_id": "mirror", "fire_count": 1},
        {"seq": 5, "ts": ts[4], "kind": "failed", "trigger_id": "mirror", "action_index": 0},
        {"seq": 6, "ts": ts[5], "kind": "detected", "trigger_id": "tick", "provoking": {"tool": "c"}, "due": 2,
         "virtual_time": "2026-12-03T07:00:00Z"},
        {"seq": 7, "ts": ts[6], "kind": "fired", "trigger_id": "tick", "virtual_time": "2026-12-03T07:00:00Z"},
        {"seq": 8, "ts": ts[7], "kind": "fired", "trigger_id": "tick", "virtual_time": "2026-12-03T08:00:00Z"},
    ]

    firings = _fold("env:calendar", events, [])

    assert [(f.trigger_id, f.ok, f.provoked_by, f.turn) for f in firings] == [
        ("mirror", True, {"tool": "a"}, 0), ("mirror", False, {"tool": "b"}, 0),
        ("tick", True, {"tool": "c"}, 0), ("tick", True, {"tool": "c"}, 0)]
    assert firings[3].virtual_time == datetime.fromisoformat("2026-12-03T08:00:00+00:00")


@pytest.mark.asyncio
async def test_as_of_a_turn_the_run_stops_before_the_next_turn_begins():
    context, prompt_response, _ = recorded_run()
    run = await load_run_context(context, prompt_response)

    first, second = run.as_of(1), run.as_of(2)

    assert [t.number for t in first.turns] == [1] and [t.number for t in second.turns] == [1, 2]
    assert [f.trigger_id for f in first.firings()] == ["confirm", "reschedule"] == [
        f.trigger_id for f in second.firings()]
    assert run.as_of(3) is run and run.as_of(7) is run


@pytest.mark.asyncio
async def test_a_prompt_with_no_stored_conversation_is_one_turn_of_its_prompt_and_reply():
    prompt_response = PromptResponse(prompt_id="p1", response="done", prompt_text="do it", step_id="turn")

    run = await load_run_context(TaskStepContext(prompt_responses=[prompt_response]), prompt_response)

    [turn] = run.turns
    assert (turn.message.role, turn.message.text, turn.reply.role, turn.reply.text) == ("user", "do it", "agent", "done")
    assert run.firings() == () and run.unavailable == {}


@pytest.mark.asyncio
async def test_what_the_run_meant_to_record_but_cannot_be_read_raises_rather_than_reading_as_empty():
    context, prompt_response, _ = recorded_run()
    context.metadata["a2a_conversations"]["prompt"] = "conv-gone"
    context.metadata["env_trigger_state"] = {
        "calendar": {"error": "TimeoutError: capture budget of 90s exceeded"},
        "mail": {"object_url": get_config().get_object_store().object_url("env_trigger_state/nothing-here.json")},
    }

    run = await load_run_context(context, prompt_response)

    assert run.unavailable["conversation"] == "conversation 'conv-gone' is not in the store"
    assert run.unavailable["env:calendar"] == "TimeoutError: capture budget of 90s exceeded"
    assert "env:mail" in run.unavailable
    with pytest.raises(RunRecordUnavailable, match="conv-gone") as raised:
        run.turns
    assert raised.value.source == "conversation"
    with pytest.raises(RunRecordUnavailable, match="capture budget") as raised:
        run.firing("confirm", env="calendar")
    assert raised.value.source == "env:calendar"
    with pytest.raises(RunRecordUnavailable):
        run.firings()


@pytest.mark.asyncio
async def test_an_env_with_no_firing_log_raises_and_a_missing_log_reaches_the_turn_too():
    context, prompt_response, _ = recorded_run()
    context.metadata["env_trigger_state"]["mail"] = {"error": "TimeoutError: capture budget of 90s exceeded"}
    run = await load_run_context(context, prompt_response)

    with pytest.raises(RunRecordUnavailable, match="no firing log was captured"):
        run.firings(env="drive")
    with pytest.raises(RunRecordUnavailable, match="capture budget"):
        run.turn(1).firings
    assert [f.trigger_id for f in run.firings(env="calendar")] == ["confirm", "late"]


def test_a_firing_log_that_cannot_be_parsed_is_named_not_left_empty():
    run = _run([], {"env:calendar": [{"seq": 1, "kind": "fired", "trigger_id": "confirm"}]}, {})

    assert run.unavailable["env:calendar"].startswith("unreadable firing log: KeyError")
    with pytest.raises(RunRecordUnavailable):
        run.firings(env="calendar")


@pytest.mark.asyncio
@pytest.mark.parametrize("dropped", ["agent_trigger_firings", "agent_trigger_registrations"])
async def test_a_user_agent_firing_log_that_was_not_read_back_is_named_not_left_empty(dropped):
    context, prompt_response, _ = recorded_run()
    for key in ("agent_trigger_state", dropped):
        del context.metadata[key]

    run = await load_run_context(context, prompt_response)

    with pytest.raises(RunRecordUnavailable, match="wasn't read") as raised:
        run.firing("reschedule")
    assert raised.value.source == "user_agent"
    assert [f.trigger_id for f in run.firings(env="calendar")] == ["confirm", "late"]
    text = _render(run, ["trigger_log"])
    assert f"(not available: user_agent: {run.unavailable['user_agent']})" in text
    assert "(no trigger firings were recorded)" not in text


@pytest.mark.asyncio
async def test_a_user_agent_with_no_triggers_of_its_own_has_no_firing_log_to_miss():
    context, prompt_response, _ = recorded_run()
    for key in ("agent_trigger_state", "agent_trigger_firings"):
        del context.metadata[key]
    context.metadata["agent_trigger_registrations"] = [{"step_id": "arm", "agent_name": "other_sim", "added": ["x"]}]

    run = await load_run_context(context, prompt_response)

    assert run.firing("reschedule") is None and run.unavailable == {}


@pytest.mark.asyncio
async def test_without_the_conversation_the_firings_that_cannot_be_placed_in_turns_are_withheld():
    context, prompt_response, _ = recorded_run()
    context.metadata["a2a_conversations"]["prompt"] = "conv-gone"

    run = await load_run_context(context, prompt_response)

    assert run.unavailable["env:calendar"] == run.unavailable["user_agent"]
    assert run.unavailable["env:calendar"].startswith("withheld:") and run.as_of(1) is run
    with pytest.raises(RunRecordUnavailable, match="withheld"):
        run.firings(env="calendar")


@pytest.mark.asyncio
async def test_without_a_prompt_the_run_has_the_env_firings_alone():
    context, _, _ = recorded_run()

    run = await load_run_context(context, None)

    assert run.turns == () and [(f.trigger_id, f.turn) for f in run.firings()] == [("confirm", 0), ("late", 0)]


def test_an_empty_run_has_no_turns():
    assert Run().turns == () and Run().as_of(1) == Run()
    with pytest.raises(LookupError, match="no turn 1"):
        Run().turn(1)


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
    context, _ = _judged_run()

    await v.execute(context)

    prompt = captured["eval_prompt"]
    conversation = prompt[prompt.index("## Conversation"):prompt.index("## Trigger firings")]
    assert all(user in conversation and agent in conversation for user, agent in TURNS)
    assert f"- user (dana_sim, written by triggers reschedule): {TURNS[1][0]}" in conversation
    assert f"- agent (solver): {TURNS[2][1]}" in conversation and "### Turn 3" in conversation
    log = prompt[prompt.index("## Trigger firings"):]
    assert '- env:calendar: confirm fired during turn 1, provoked by {"tool": "book", "role": "solver"}' in log
    assert "- user_agent: reschedule fired during turn 1\n- env:calendar: late fired during turn 3" in log


@pytest.mark.asyncio
async def test_graded_as_of_a_turn_the_judge_sees_neither_later_turns_nor_later_firings(monkeypatch):
    v = _verifier(run_context_for_judge=["conversation", "trigger_log"], grade_up_to_turn=2)
    captured = _judge_prompt(monkeypatch, v)
    context, _ = _judged_run()

    await v.execute(context)

    prompt = captured["eval_prompt"]
    assert TURNS[1][1] in prompt and "### Turn 3" not in prompt and TURNS[2][0] not in prompt
    assert "confirm fired during turn 1" in prompt and "late fired" not in prompt


@pytest.mark.asyncio
async def test_graded_as_of_a_turn_without_the_conversation_the_judge_sees_no_firings(monkeypatch):
    v = _verifier(run_context_for_judge=["trigger_log"], grade_up_to_turn=1)
    captured = _judge_prompt(monkeypatch, v)
    context, _ = _judged_run()
    context.metadata["a2a_conversations"]["prompt"] = "conv-gone"

    await v.execute(context)

    prompt = captured["eval_prompt"]
    assert " fired during " not in prompt
    assert "(not available: env:calendar: withheld" in prompt and "(not available: user_agent: withheld" in prompt


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
    events = [{"seq": i, "ts": f"2026-10-20T10:{i // 60:02d}:{i % 60:02d}+00:00", "kind": "fired", "trigger_id": f"t{i}"}
              for i in range(RubricsVerifierTaskStep.RUN_CONTEXT_FIRINGS_MAX + 3)]
    run = _run([_message(1, "user", _text(long) + [{"kind": "file"}])], {"env:calendar": events},
               {"env:mail": "TimeoutError: capture budget of 90s exceeded"})

    text = _render(run, ["conversation", "trigger_log"])

    assert "x" * max_chars + "… (+1 non-text part)" in text and long not in text
    assert "(3 earlier firings not shown)" in text and " t2 fired" not in text and " t3 fired during turn 1" in text
    assert "(not available: env:mail: TimeoutError: capture budget of 90s exceeded)" in text
    assert "(no trigger firings were recorded)" in _render(Run(), ["trigger_log"])


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
from agent_env.task_step.task_steps.verifiers import Run

async def verify(mcp_url: str, run: Run | None = None):
    late = run.firing("late", env="calendar")
    reply = run.turn(late.turn).reply if late else None
    return [{"id": "answered-after-late", "score": 1.0, "result": bool(reply and "help" in reply.text),
             "turns": len(run.turns), "env_fired": [f.trigger_id for f in run.firings(env="calendar") if f.ok]}]
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

    assert [f.trigger_id for f in graded_first.firings(env="calendar")] == ["lunch_confirm"]
    assert [f.trigger_id for f in graded_last.firings(env="calendar")] == ["lunch_confirm", "dinner_confirm"]


@pytest.mark.asyncio
async def test_an_earlier_prompt_sharing_the_next_prompts_conversation_has_its_env_firings_withheld():
    context, prompt_response, _ = recorded_run()
    context.prompt_responses.append(PromptResponse(prompt_id="p2", response="ok", prompt_text="again", step_id="again"))
    context.metadata["a2a_conversations"]["again"] = context.metadata["a2a_conversations"]["prompt"]

    run = await load_run_context(context, prompt_response)

    with pytest.raises(RunRecordUnavailable, match="when the next prompt began is unknown"):
        run.firings(env="calendar")


@pytest.mark.asyncio
async def test_a_message_carries_each_envs_virtual_time_when_the_conversation_recorded_it(monkeypatch):
    context, prompt_response, _ = recorded_run()
    original = conversation_store.get_conversation

    def with_clock(conversation_id):
        doc = original(conversation_id)
        doc["messages"][0]["env_clock"] = [
            {"env_id": "calendar", "env_step_id": "deploy_calendar", "virtual_time": "2026-10-17T09:00:00Z"},
            {"env_id": "calendar", "env_step_id": "deploy_calendar_2", "virtual_time": "2026-10-18T09:00:00Z"}]
        return doc

    monkeypatch.setattr(conversation_store, "get_conversation", with_clock)
    run = await load_run_context(context, prompt_response)

    first, reply = run.turn(1).message, run.turn(1).reply
    assert first.env_clock == (EnvClock("calendar", "deploy_calendar", datetime.fromisoformat("2026-10-17T09:00:00Z")),
                               EnvClock("calendar", "deploy_calendar_2", datetime.fromisoformat("2026-10-18T09:00:00Z")))
    assert reply.env_clock is None  # a message stored without a clock has none


def _shared_conversation(*prompts: list[tuple[str, str]]) -> str:
    conversation_id = f"conv-{uuid.uuid4().hex[:8]}"
    conversation_store.create_conversation(conversation_id, task_instance_id="run-1", source_agent_name="dana_sim",
                                           target_agent_name="solver")
    for user, agent in (turn for turns in prompts for turn in turns):
        conversation_store.add_a2a_task(conversation_id, parts=_text(user), a2a_task_id=uuid.uuid4().hex, role="user")
        conversation_store.complete_a2a_task(conversation_id, parts=_text(agent), role="agent")
    return conversation_id


@pytest.mark.asyncio
async def test_prompts_sharing_one_conversation_each_see_only_their_own_turns():
    lunch, dinner = TURNS[:2], [("Now book dinner.", "Booked for seven.")]
    conversation_id = _shared_conversation(lunch, dinner)
    context = TaskStepContext(metadata={"a2a_conversations": {"lunch": conversation_id, "dinner": conversation_id}})
    context.prompt_responses += [
        PromptResponse(prompt_id="p1", response=lunch[1][1], prompt_text=lunch[0][0], step_id="lunch",
                       source_agent_per_turn_prompt_parts=[None, _text(lunch[1][0])]),
        PromptResponse(prompt_id="p2", response=dinner[0][1], prompt_text=dinner[0][0], step_id="dinner")]

    first = await load_run_context(context, context.prompt_responses[0])
    last = await load_run_context(context, context.prompt_responses[1])

    assert [(t.number, t.message.text, t.reply.text) for t in first.turns] == [(1, *lunch[0]), (2, *lunch[1])]
    assert [(t.number, t.message.text, t.reply.text) for t in last.turns] == [(1, *dinner[0])]


@pytest.mark.asyncio
async def test_a_shared_conversation_the_next_prompt_cannot_be_found_in_is_withheld():
    context, prompt_response, _ = recorded_run()
    context.prompt_responses.append(PromptResponse(prompt_id="p2", response="ok", prompt_text="again", step_id="again"))
    context.metadata["a2a_conversations"]["again"] = context.metadata["a2a_conversations"]["prompt"]

    run = await load_run_context(context, prompt_response)

    with pytest.raises(RunRecordUnavailable, match="withheld") as raised:
        run.turns
    assert raised.value.source == "conversation"


@pytest.mark.asyncio
async def test_an_env_firing_log_the_gateway_dropped_events_from_raises_rather_than_reading_as_never_fired():
    context, prompt_response, recorded = recorded_run()
    envelope = {"env_id": "calendar", "state": {"triggers": [], "events": recorded["env"], "events_dropped": 4}}
    context.metadata["env_trigger_state"]["calendar"]["object_url"] = get_config().get_object_store().put(
        f"env_trigger_state/dropped-{uuid.uuid4().hex[:8]}.json", json.dumps(envelope).encode(),
        content_type="application/json")

    run = await load_run_context(context, prompt_response)

    with pytest.raises(RunRecordUnavailable, match="dropped 4") as raised:
        run.firing("never", env="calendar")
    assert raised.value.source == "env:calendar"
    assert f"(not available: env:calendar: {run.unavailable['env:calendar']})" in _render(run, ["trigger_log"])


@pytest.mark.asyncio
async def test_a_user_agent_firing_log_that_lost_its_oldest_entries_raises_rather_than_reading_as_never_fired():
    context, prompt_response, recorded = recorded_run()
    context.metadata["agent_trigger_state"]["prompt"] = [{**recorded["agent"][0], "seq": 5}]

    run = await load_run_context(context, prompt_response)

    with pytest.raises(RunRecordUnavailable, match="oldest") as raised:
        run.firing("never")
    assert raised.value.source == "user_agent"
    assert [f.trigger_id for f in run.firings(env="calendar")] == ["confirm", "late"]


@pytest.mark.asyncio
async def test_a_later_prompt_graded_as_of_a_turn_shows_the_judge_its_own_turn_not_the_shared_conversations_first(
        monkeypatch):
    lunch = [TURNS[0]]
    dinner = [("Now book dinner.", "Booked for seven."), ("Make it eight.", "Moved to eight.")]
    conversation_id = _shared_conversation(lunch, dinner)
    context = TaskStepContext(metadata={"a2a_conversations": {"lunch": conversation_id, "dinner": conversation_id}})
    context.prompt_responses += [
        PromptResponse(prompt_id="p0", response=lunch[0][1], prompt_text=lunch[0][0], step_id="lunch"),
        PromptResponse(prompt_id="p1", response=dinner[1][1], prompt_text=dinner[0][0], step_id="dinner",
                       source_agent_per_turn_prompt_parts=[None, _text(dinner[1][0])],
                       target_agent_per_turn_trajectory_object_urls=["s3://t/1.json", "s3://t/2.json"],
                       agent_trajectory_object_url="s3://t/2.json")]
    v = _verifier(run_context_for_judge=["conversation"], grade_up_to_turn=1)
    captured = _judge_prompt(monkeypatch, v)

    await v.execute(context)

    conversation = captured["eval_prompt"][captured["eval_prompt"].index("## Conversation"):]
    assert dinner[0][0] in conversation and dinner[0][1] in conversation
    assert lunch[0][0] not in conversation and dinner[1][0] not in conversation
