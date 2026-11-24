"""The user-sim gate's own checks: the patterns, the judge's output parsing and the history it is shown."""

import asyncio

import pytest

from agent_env.config import model_call
from agent_env.config.model_pricing import Cost, CostStatus
from agent_env.task_step.task_steps.user_sim_gate import (
    TALKS_LIKE_THE_ASSISTANT, GateVerdict, UserSimGate, judge_call_config, parse_judge, pattern_reasons, render_history, rewrite_request)

_FILLER = " The details are in the shared folder and the numbers come from last month's records." * 6


def test_the_patterns_pass_what_users_write():
    for message in [
        "and the numbers she gave the board, do those hold up?",
        "ok. who actually worked on it, and what does each of them cost per month?",
        "Here's the email my manager sent, pasted so you have it:\n\n> Please hold the order until Friday." + _FILLER,
        "No, go ahead and send it. Use the second draft.",
        "I checked my inbox and there's nothing from them yet.",
        "I don't have access to that folder.",
        "Should I send you the spreadsheet?",
        "I'll pull the numbers from my bank tonight.",
        "I can send it to you after lunch.",
        "I apologize for the confusion, I meant March.",
        "The invoice has been sent to me already.",
        "My manager said the page has been updated.",
        "ok, send me the results when you have them",
    ]:
        assert pattern_reasons(message) == [], message


def test_the_patterns_catch_the_assistants_side():
    cases = {
        "Done. Here's what I found when I went through the files." + _FILLER: "it opens like a report of work done",
        "Here's where things stand after the changes." + _FILLER: "it opens like a report of work done",
        "Understood. Standing by.": "it talks like the assistant",
        "Want me to draft the reply to them now?": "it talks like the assistant",
        "I'm ready to help! What would you like to work on today?": "it talks like the assistant",
        "The page is now live, with the corrected totals." : "it says something is now live",
        "I apologize for my previous answer, here is the corrected one.": "it talks like the assistant",
        "Thanks for those corrections, incorporated. The cleaner version:" + _FILLER: "it reports something delivered",
        "| Item | Was | Is |\n|---|---|---|\n| A | 1 | 2 |\n" + _FILLER * 2: "it is a formatted deliverable",
        "I've reviewed the contract and I've compared it with the invoice." + _FILLER: "it reports checking or building",
    }
    for message, expected in cases.items():
        reasons = pattern_reasons(message)
        assert any(r.startswith(expected) for r in reasons), (message[:60], reasons)


def test_the_judge_output_is_read_through_surrounding_text_and_rejected_without_a_boolean():
    assert parse_judge('{"in_role": false, "reason": "reports work"}') == {"in_role": False, "reason": "reports work"}
    assert parse_judge('Sure.\n{"in_role": true, "reason": "asks"} trailing') == {"in_role": True, "reason": "asks"}
    assert parse_judge('{"in_role": "no"}') is None
    assert parse_judge("no json here") is None


def test_the_history_shows_the_last_turns_each_cut_to_size():
    history = [("user", "first"), ("assistant", "x" * 50), ("user", "second"), ("assistant", "third")]
    shown = render_history(history, turns=3, max_chars=10)
    assert shown.startswith("ASSISTANT:\nxxxxxxxxxx\n[… 40 more characters]")
    assert "first" not in shown and shown.endswith("ASSISTANT:\nthird")
    assert render_history([], turns=4, max_chars=10) == "(nothing yet)"
    assert render_history(history, turns=0, max_chars=10) == "(nothing yet)"   # context_turns 0: no history


def test_the_rewrite_request_names_why_and_says_it_is_not_from_the_assistant():
    note = rewrite_request(GateVerdict(passed=False, pattern_reasons=["it opens like a report of work done"],
                                       judge_in_role=False, judge_reason="it answers its own question"))
    assert note.startswith("[Conversation system note, not from the assistant.]")
    assert "rather than yours: it answers its own question." in note   # the judge's reason; patterns are records


def test_only_the_judge_holds_a_message_back_at_any_length_and_patterns_are_recorded():
    phrase = "Want me to send it to you?"
    shape = "Done. Here's what I found when I went through the files." + _FILLER
    short = "Contracted physicians are required in two states."

    assert asyncio.run(_judged(False).check(short, [])).passed is False      # no length floor
    for text in (phrase, shape):
        cleared = asyncio.run(_judged(True).check(text, []))
        assert cleared.passed is True and cleared.pattern_reasons             # a match alone holds nothing back
    held = asyncio.run(_judged(False).check(shape, []))
    assert (held.passed, held.reasons()) == (False, ["judged"])


def test_without_a_judge_the_gate_only_records_pattern_matches():
    verdict = asyncio.run(UserSimGate().check("I checked my bank app and I verified the charge is there.", []))
    assert verdict.passed is True and verdict.pattern_reasons


# ---- run_gate, for any conversation loop ------------------------------------------------------------------------------

def _scripted_gate(verdicts, **settings):
    from agent_env.task_step.task_steps.user_sim_gate import GateVerdict, UserSimGate

    class Gate(UserSimGate):
        async def check(self, candidate, history, *, call_config=None, **_):
            return GateVerdict(passed=verdicts.pop(0), pattern_reasons=["it opens like a report of work done"])

    return Gate(**settings)


def test_run_gate_rewrites_until_a_message_passes():
    import asyncio
    from agent_env.task_step.task_steps.user_sim_gate import run_gate

    asked, seen = [], []

    async def rewrite(note):
        asked.append(note)
        return f"rewrite {len(asked)}"

    outcome = asyncio.run(run_gate(_scripted_gate([False, False, True]), "draft", [], rewrite,
                                   on_verdict=lambda attempt, text, v: seen.append((attempt, text, v.passed))))
    assert (outcome.text, outcome.invalid, outcome.stopped) == ("rewrite 2", False, False)
    assert seen == [(0, "draft", False), (1, "rewrite 1", False), (2, "rewrite 2", True)]
    assert asked[0].startswith("[Conversation system note, not from the assistant.]")


def test_run_gate_marks_a_message_that_never_passes_and_honors_send():
    import asyncio
    from agent_env.task_step.task_steps.user_sim_gate import run_gate

    async def rewrite(note):
        return "still the assistant"

    failed = asyncio.run(run_gate(_scripted_gate([False, False], max_rewrites=1), "draft", [], rewrite))
    sent = asyncio.run(run_gate(_scripted_gate([False, False], max_rewrites=1, on_fail="send"), "draft", [], rewrite))
    assert (failed.invalid, len(failed.verdicts)) == (True, 2)
    assert (sent.invalid, sent.text) == (False, "still the assistant")


def test_run_gate_stops_when_the_user_side_ends_instead_of_rewriting():
    import asyncio
    from agent_env.task_step.task_steps.user_sim_gate import run_gate

    async def rewrite(note):
        return None

    outcome = asyncio.run(run_gate(_scripted_gate([False]), "draft", [], rewrite))
    assert (outcome.stopped, outcome.invalid, len(outcome.verdicts)) == (True, False, 1)


# ---- step types that run their own conversation ------------------------------------------------------------------

def test_a_step_type_that_replaces_the_conversation_refuses_the_gate_unless_it_applies_it():
    import pytest
    from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep

    class OwnLoop(PromptAgentTaskStep):
        async def _execute(self, context):
            return context

    class OwnLoopWithGate(OwnLoop):
        USER_SIM_GATE_SUPPORTED = True

    with pytest.raises(ValueError, match="OwnLoop 'talk': user_sim_gate is applied by prompt_agent's own conversation"):
        OwnLoop(id="talk", version=None, prompt="hi", user_sim_gate={})
    assert OwnLoop(id="talk", version=None, prompt="hi").user_sim_gate is None
    assert OwnLoopWithGate(id="talk", version=None, prompt="hi", user_sim_gate={}).user_sim_gate is not None
    assert PromptAgentTaskStep.applies_user_sim_gate() and not OwnLoop.applies_user_sim_gate()


def test_an_always_on_step_keeps_its_gate_through_the_step_dict():
    from agent_env.task_step.task_steps.always_on import AlwaysOnTaskStep

    step = AlwaysOnTaskStep("week", None, user_sim="user-sim", prompt_id="ask", agent_name="solver",
                            user_sim_gate={"max_rewrites": 1, "on_fail": "end"})
    again = AlwaysOnTaskStep.from_dict(step.to_dict())

    assert again.user_sim_gate == step.user_sim_gate and again.user_sim_gate.on_fail == "end"
    plain = AlwaysOnTaskStep("week", None, user_sim="user-sim", prompt_id="ask", agent_name="solver")
    assert "user_sim_gate" not in plain.to_dict()
    assert AlwaysOnTaskStep.from_dict(plain.to_dict()).user_sim_gate is None


def test_a_judge_that_cant_be_set_up_fails_the_step_since_nothing_else_holds_a_message_back(monkeypatch):
    def refuse(model, **kw):
        raise KeyError("no key")

    monkeypatch.setattr(model_call, "resolve_route", refuse)
    with pytest.raises(RuntimeError, match="user_sim_gate: judge setup: KeyError"):
        judge_call_config(UserSimGate(judge_model="judge-model"), {})


def test_a_judge_call_that_errors_sends_the_message_with_the_error_recorded():
    class Failing(UserSimGate):
        async def _judge(self, candidate, history, call_config, **_):
            raise TimeoutError("judge timed out")

    verdict = asyncio.run(Failing(judge_model="judge-model").check("Want me to send it to you?", []))
    assert (verdict.passed, verdict.judge_error) == (True, "TimeoutError: judge timed out")


def test_the_judge_calls_through_the_one_model_call_path_on_the_runs_route(monkeypatch):
    """The gate's judge goes through config/model_call.py (route and complete), so its routing (a Claude model drops
    an endpoint's trailing /v1, an unplaceable name goes as openai/<name>) and its cost record are core's."""
    import asyncio

    from agent_env.config import model_call
    from agent_env.task_step.task_steps.user_sim_gate import UserSimGate, judge_call_config

    resolved, sent = [], []

    def resolve_route(model, *, api_key, api_base, who, key_how):
        resolved.append((model, api_key, api_base, who))
        return "route"

    async def complete(route, messages, *, timeout, response_format=None):
        sent.append((route, timeout, response_format["json_schema"]["name"]))
        return model_call.ModelReply('{"in_role": true, "reason": "ok"}', "m", None, None, None)

    monkeypatch.setattr(model_call, "resolve_route", resolve_route)
    monkeypatch.setattr(model_call, "complete", complete)
    gate = UserSimGate(judge_model="claude-sonnet-5-5", judge_timeout_seconds=45)
    route = judge_call_config(gate, {"judge_litellm_api_key": "secret:k", "litellm_base_url": "https://llm.example/v1"})

    judged = asyncio.run(gate._judge("thanks, looks good", [], route))

    assert resolved == [("claude-sonnet-5-5", "secret:k", "https://llm.example/v1", "user_sim_gate")]
    assert sent == [("route", 45, "user_sim_gate")] and judged["in_role"] is True



def _judged(in_role):
    from agent_env.task_step.task_steps.user_sim_gate import UserSimGate

    class Gate(UserSimGate):
        async def _judge(self, candidate, history, call_config, **_):
            return {"in_role": in_role, "reason": "judged"}

    return Gate(judge_model="judge-model")




@pytest.mark.parametrize("settings, field", [
    ({"max_rewrites": 1.5}, "max_rewrites"), ({"max_rewrites": True}, "max_rewrites"),
    ({"max_rewrites": -1}, "max_rewrites"),
    ({"patterns": "false"}, "patterns"), ({"context_turns": "4"}, "context_turns"),
    ({"context_chars": 0}, "context_chars"), ({"judge_timeout_seconds": 0}, "judge_timeout_seconds"),
    ({"judge_model": ""}, "judge_model"), ({"on_fail": "skip"}, "on_fail"),
], ids=lambda v: str(v))
def test_a_setting_of_the_wrong_type_or_range_is_refused_when_the_task_loads(settings, field):
    from agent_env.task_step.task_steps.user_sim_gate import UserSimGate

    with pytest.raises(ValueError, match=f"user_sim_gate.{field} must be"):
        UserSimGate.from_dict(settings)


def test_good_settings_load_and_round_trip():
    from agent_env.task_step.task_steps.user_sim_gate import UserSimGate

    gate = UserSimGate.from_dict({"patterns": False, "judge_model": "j", "max_rewrites": 0, "judge_timeout_seconds": 30})
    assert UserSimGate.from_dict(gate.to_dict()) == gate



def test_the_judge_is_shown_what_the_user_found_when_a_hook_told_it(monkeypatch):
    import asyncio

    from agent_env.config import model_call
    from agent_env.task_step.task_steps.user_sim_gate import UserSimGate

    prompts = []

    async def complete(route, messages, *, timeout, response_format=None):
        prompts.append(messages[0]["content"])
        return model_call.ModelReply('{"in_role": true, "reason": "ok"}', "m", None, None, None)

    monkeypatch.setattr(model_call, "complete", complete)
    gate = UserSimGate(judge_model="judge-model")
    asyncio.run(gate._judge("i checked and the sheet isn't there", [], "route", user_note="no sheet in the folder"))
    asyncio.run(gate._judge("i checked and the sheet isn't there", [], "route"))

    assert "WHAT THE USER FOUND" in prompts[0] and "no sheet in the folder" in prompts[0]
    assert "WHAT THE USER FOUND" not in prompts[1]


def test_the_judge_is_shown_the_user_sims_instructions(monkeypatch):
    prompts = []

    async def complete(route, messages, *, timeout, response_format=None):
        prompts.append(messages[0]["content"])
        return model_call.ModelReply('{"in_role": true, "reason": "ok"}', "m", None, None, None)

    monkeypatch.setattr(model_call, "complete", complete)
    gate = UserSimGate(judge_model="judge-model")
    asyncio.run(gate._judge("i checked my bank app", [], "route", instructions="You are Ana. You check your bank app."))
    asyncio.run(gate._judge("i checked my bank app", [], "route"))

    assert "THE USER'S INSTRUCTIONS" in prompts[0] and "You check your bank app." in prompts[0]
    assert "THE USER'S INSTRUCTIONS" not in prompts[1]


@pytest.mark.parametrize("text", ["I\u2019m an AI, so I can\u2019t open that.",
                                  "Let me know if you\u2019d like me to resend it."])
def test_curly_apostrophes_read_as_straight_ones(text):
    assert TALKS_LIKE_THE_ASSISTANT in pattern_reasons(text)


def test_each_judge_call_is_handed_to_on_call_for_the_steps_usage(monkeypatch):
    reply = model_call.ModelReply('{"in_role": true, "reason": "asks"}', "judge-m", None, None,
                                  Cost(0.001, CostStatus.REPORTED, "provider", 0.001))

    async def complete(route, messages, **kw):
        return reply

    monkeypatch.setattr(model_call, "complete", complete)
    seen = []
    verdict = asyncio.run(UserSimGate(judge_model="judge-m").check("can you check the totals?", [], call_config=object(),
                                                                   on_call=seen.append))

    assert verdict.passed and seen == [reply]


@pytest.mark.parametrize("error, recorded", [
    (lambda: __import__("litellm").Timeout("judge timed out", model="judge-m", llm_provider="openai"), True),
    (lambda: ValueError("bad route"), False),
], ids=["timed-out", "never-sent"])
def test_a_judge_call_cut_off_by_its_timeout_is_still_handed_to_on_call_as_one_of_unknown_usage(
        monkeypatch, error, recorded):
    from types import SimpleNamespace

    from agent_env.config.model import ModelRole
    from agent_env.task_step.context import TaskStepContext
    from agent_env.task_step.usage import StepUsage

    async def complete(route, messages, **kw):
        raise error()

    monkeypatch.setattr(model_call, "complete", complete)
    context = TaskStepContext()
    usage = StepUsage(context, "talk")
    seen = []
    route = SimpleNamespace(model="judge-m", endpoint_host="gw:443")
    verdict = asyncio.run(UserSimGate(judge_model="judge-m").check(
        "can you check the totals?", [], call_config=route,
        on_call=lambda reply: (seen.append(reply), usage.reply(ModelRole.JUDGE, reply))))

    assert verdict.passed and verdict.judge_error
    if not recorded:
        assert seen == [] and "talk" not in context.metadata.get("step_usage", {})
        return
    [reply] = seen
    assert (reply.model, reply.endpoint_host, reply.usage, reply.cost.usd) == ("judge-m", "gw:443", None, None)
    [entry] = context.metadata["step_usage"]["talk"]
    assert entry["role"] == "judge" and entry["partial"] and entry["cost_usd"] is None
