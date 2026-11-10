"""The user-sim gate's own checks: the patterns, the judge's output parsing and the history it is shown."""

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
    ]:
        assert pattern_reasons(message) == [], message


def test_the_patterns_catch_the_assistants_side():
    cases = {
        "Done. Here's what I found when I went through the files." + _FILLER: "it opens like a report of work done",
        "Here's where things stand after the changes." + _FILLER: "it opens like a report of work done",
        "Understood. Standing by.": "it talks like the assistant",
        "Want me to draft the reply to them now?": "it talks like the assistant",
        "I'm ready to help! What would you like to work on today?": "it talks like the assistant",
        "The page is now live, with the corrected totals." : "it says something is now live or updated",
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
