"""The user-sim gate's own checks: the patterns, the judge's output parsing and the history it is shown."""

import pytest

from agent_env.task_step.task_steps.user_sim_gate import (
    GateVerdict, parse_judge, pattern_reasons, render_history, rewrite_request)

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


def test_the_rewrite_request_names_why_and_says_it_is_not_from_the_assistant():
    note = rewrite_request(GateVerdict(passed=False, pattern_reasons=["it opens like a report of work done"],
                                       judge_in_role=False, judge_reason="it answers its own question"))
    assert note.startswith("[Conversation system note, not from the assistant.]")
    assert "it opens like a report of work done; it answers its own question" in note


def test_the_judge_alone_fails_only_a_long_message():
    import asyncio
    from agent_env.task_step.task_steps.user_sim_gate import UserSimGate

    class Gate(UserSimGate):
        async def _judge(self, candidate, history, call_config):
            return {"in_role": False, "reason": "reads as analysis"}

    gate = Gate(judge_model="judge-model")
    short = asyncio.run(gate.check("Contracted physicians are required in two states.", []))
    long = asyncio.run(gate.check("The two states that need contracted physicians are these." + _FILLER, []))
    assert (short.passed, short.judge_in_role) == (True, False)     # recorded, not failed
    assert long.passed is False


# ---- run_gate, for any conversation loop ------------------------------------------------------------------------------

def _scripted_gate(verdicts, **settings):
    from agent_env.task_step.task_steps.user_sim_gate import GateVerdict, UserSimGate

    class Gate(UserSimGate):
        async def check(self, candidate, history, *, call_config=None, judge_unavailable=None):
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


def test_a_judge_that_couldnt_be_set_up_isnt_called_and_the_patterns_verdict_stands():
    import asyncio
    from agent_env.task_step.task_steps.user_sim_gate import UserSimGate

    class NeverCalled(UserSimGate):
        async def _judge(self, candidate, history, call_config):
            raise AssertionError("the judge was called")

    for patterns in (True, False):
        gate = NeverCalled(patterns=patterns, judge_model="judge-model")
        verdict = asyncio.run(gate.check("shorter, please", [], judge_unavailable="judge setup: KeyError: x"))
        assert (verdict.passed, verdict.judge_error) == (True, "judge setup: KeyError: x")


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
        async def _judge(self, candidate, history, call_config):
            return {"in_role": in_role, "reason": "judged"}

    return Gate(judge_model="judge-model")


def test_a_judge_reading_it_as_the_user_clears_a_phrase_but_not_a_reports_shape():
    import asyncio

    phrase = "Understood. Standing by."
    shape = "Done. Here's what I found when I went through the files." + _FILLER
    cleared = asyncio.run(_judged(True).check(phrase, []))
    assert (cleared.passed, cleared.cleared_by_judge) == (True, True) and cleared.pattern_reasons
    assert asyncio.run(_judged(True).check(shape, [])).passed is False  # shape + length: a failure whatever the judge
    assert asyncio.run(_judged(False).check(phrase, [])).passed is False



@pytest.mark.parametrize("settings, field", [
    ({"max_rewrites": 1.5}, "max_rewrites"), ({"max_rewrites": True}, "max_rewrites"),
    ({"max_rewrites": -1}, "max_rewrites"), ({"judge_min_chars": None}, "judge_min_chars"),
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
