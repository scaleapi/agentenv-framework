"""Offline tests for the milestone-gates example hook: a fake Turn and env, scripted tool results per turn, no network."""
import json
import copy
import importlib.util
from pathlib import Path

import pytest

import agent_env

HOOK = Path(agent_env.__file__).parent / "examples" / "milestone-gates" / "artifacts" / "milestone-gates" / "hook.py"
_spec = importlib.util.spec_from_file_location("milestone_gates_hook", HOOK)
mg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mg)


class Env:
    """Tool results by name; ``state`` is edited between turns to play the agent's work."""

    def __init__(self, state):
        self.env_id = "work"
        self.state = state
        self.calls = []

    async def call_tool(self, tool, args, role):
        self.calls.append((tool, args, role))
        value = self.state[tool]
        value = value(args) if callable(value) else value
        return {"content": [{"type": "text", "text": value if isinstance(value, str) else json.dumps(value)}]}


class Turn:
    def __init__(self, env, records, number, last=False, sim_fields=None, reply=None, trajectories=None):
        self.envs, self._records, self.number, self.last = [env], records, number, last
        self.notes, self.reply = [], reply
        # trajectories: {url: stream-json messages}; trajectory_urls is one url per turn so far
        self._trajectories = trajectories or {}
        self.trajectory_urls = tuple(self._trajectories)[:number]
        self.reads = []
        self.metadata = {"usersim_turn_outputs": {"talk": [{"turn": number, "fields": sim_fields}]} if sim_fields
                         else {}}

    async def read_trajectory(self, n):
        url = self.trajectory_urls[n - 1]
        self.reads.append(url)
        return None if url is None else json.dumps(self._trajectories[url])

    def records(self):
        return [dict(r) for r in self._records]

    def record(self, entry):
        self._records.append({**entry, "turn": self.number})

    def summarize(self, summary):
        self.summary = json.loads(json.dumps(summary))  # what a run's metadata could hold

    async def ask_user(self, note=None):
        self.notes.append(note)
        text = self.reply or f"sim says ({self.number}.{len(self.notes)})"
        self.answer = {"message": [{"kind": "text", "text": text}]}
        return self.answer


async def post_turn(turn, params):
    """mg.post_turn, held to the framework's rule: a hook that asked the user side returns that answer."""
    turn.answer = None
    returned = await mg.post_turn(turn, params)
    if turn.answer is not None:
        assert returned == turn.answer, f"asked the user side, then returned {returned!r}"
    return returned


SHEET = {"id": "s1", "name": "Q4 subscriptions", "mimeType": "spreadsheet"}
PARAMS = {
    "role": "auditor", "judge_model": "judge-m", "trajectory_judge_model": "traj-m",
    "milestones": [
        {"id": "M1", "max_nudges": 2, "show": "sheet",
         "artifacts": {"sheet": {"find": {"tool": "drive_list", "path": "files", "where": {"mimeType": "spread"}},
                                 "read": {"tool": "sheet_read", "args": {"id": "{id}"}}}},
         "gates": [
             {"id": "m1-sheet", "nudge": "I don't see the sheet yet.", "check": {"artifact": "sheet", "expect": "exists"}},
             {"id": "m1-cols", "nudge": "Can it show the monthly cost?",
              "check": {"artifact": "sheet", "format": {"contains_all": ["status", "monthly"]}}},
             {"id": "m1-quiet", "nudge": "Hold off on emailing anyone.",
              "check": {"no_new": [{"tool": "gmail_sent", "path": "messages", "baseline_count": 0}]}},
         ]},
        {"id": "M2", "blocking": False, "max_nudges": 1,
         "gates": [{"id": "m2-comment", "nudge": "Did the comment go on the card?",
                    "check": {"call": {"tool": "trello_comments", "path": "comments"}, "expect": {"count": {"eq": 1}}}},
                   {"id": "m2-quiet", "nudge": "Don't email yet.",
                    "check": {"no_new": [{"tool": "gmail_sent", "path": "messages"}]}}]},
    ],
}


def _status(t) -> list[tuple]:
    """Each milestone's status, opened and closed turn, and nudges, from turn ``t``'s summary."""
    return [(m["id"], m["status"], m["opened_turn"], m["closed_turn"], m["nudges"]) for m in t.summary["milestones"]]


def world():
    return {"drive_list": {"files": []}, "sheet_read": "vendor,status,monthly\nVaultly,active,19.99",
            "gmail_sent": {"messages": []}, "trello_comments": {"comments": []}}


@pytest.mark.asyncio
async def test_a_failing_gate_sends_its_nudge_and_counts_it():
    env, records = Env(world()), []
    t = Turn(env, records, 1)
    answer = await post_turn(t, PARAMS)
    assert answer == {"message": [{"kind": "text", "text": "sim says (1.1)"}]}
    assert t.notes == ["I don't see the sheet yet."]  # no sheet: no excerpt either
    assert records[-1]["decision"] == "nudge" and records[-1]["nudges"] == 1
    assert [g["passed"] for g in records[-1]["gates"]] == [False, False, True]


@pytest.mark.asyncio
async def test_switched_off_it_checks_nothing_and_tells_the_sim_nothing_and_says_so_once():
    env, records = Env(world()), []
    for n in (1, 2):
        t = Turn(env, records, n)
        assert await post_turn(t, {**PARAMS, "enabled": False}) is None
        assert t.notes == []
    assert env.calls == [] and records == [{"decision": "disabled", "turn": 1}]
    assert t.summary == {"enabled": False}


@pytest.mark.asyncio
async def test_every_failing_turn_counts_even_when_the_sim_did_not_press_on_it():
    env, records = Env(world()), []
    await post_turn(Turn(env, records, 1, sim_fields={"nudged": False}), PARAMS)
    assert records[-1]["nudges"] == 1 and records[-1]["sim_nudged"] is False


@pytest.mark.asyncio
async def test_all_gates_passing_opens_the_next_milestone_and_takes_its_baselines_then():
    w = world()
    w["drive_list"] = {"files": [SHEET]}
    w["gmail_sent"] = {"messages": [{"id": "old"}]}  # sent before M1 opened? no: baseline_count 0, so M1 fails
    env, records = Env(w), []
    t = Turn(env, records, 1)
    await post_turn(t, PARAMS)
    assert records[-1]["gates"][2] == {"id": "m1-quiet", "passed": False, "why": "new: gmail_sent +1",
                                       "nudge": "Hold off on emailing anyone."}
    assert t.notes[0].startswith("Hold off on emailing anyone.\nYou opened it and saw:\nvendor,status,monthly")

    w["gmail_sent"] = {"messages": []}
    t2 = Turn(env, records, 2)
    await post_turn(t2, PARAMS)
    assert t2.notes == ["M1 is done. Open M2 now: send its line."]
    assert records[-1]["decision"] == "passed" and records[-1]["baselines"] == {"M2:m2-quiet:0": 0}

    # M2: an email now is new against the baseline taken when M2 opened
    w["gmail_sent"] = {"messages": [{"id": "x"}]}
    w["trello_comments"] = {"comments": [{"id": "c"}]}
    t3 = Turn(env, records, 3)
    await post_turn(t3, PARAMS)
    assert t3.notes == ["Don't email yet."]
    assert _status(t3) == [("M1", "passed", 1, 2, 1), ("M2", "open", 3, None, 1)]
    assert {k: t3.summary[k] for k in ("enabled", "reached", "done", "total", "end")} == {
        "enabled": True, "reached": "M2", "done": 1, "total": 2, "end": None}
    assert t3.summary["milestones"][0]["gates"] == {"m1-sheet": {"passed": True, "type": "action"},
                                                    "m1-cols": {"passed": True, "type": "format"},
                                                    "m1-quiet": {"passed": True, "type": "action"}}
    assert t3.summary["milestones"][1]["gates"]["m2-quiet"] == {"passed": False, "type": "action"}


@pytest.mark.asyncio
async def test_a_blocking_milestone_that_never_passes_ends_the_conversation():
    env, records = Env(world()), []
    for n in (1, 2):
        await post_turn(Turn(env, records, n), PARAMS)
    t3 = Turn(env, records, 3)
    answer = await post_turn(t3, PARAMS)
    assert answer == {"end": "gate_never_passed:M1"}
    assert [r["decision"] for r in records] == ["nudge", "nudge", "end"]
    assert _status(t3) == [("M1", "derailed", 1, 3, 2), ("M2", "not_reached", None, None, 0)]
    assert (t3.summary["reached"], t3.summary["done"], t3.summary["end"]) == ("M1", 0, "gate_never_passed:M1")


@pytest.mark.asyncio
async def test_a_non_blocking_milestone_is_let_go_and_the_last_one_wraps_up_then_ends():
    w = world()
    w["drive_list"] = {"files": [SHEET]}
    env, records = Env(w), []
    await post_turn(Turn(env, records, 1), PARAMS)  # M1 passes
    t2 = Turn(env, records, 2)
    await post_turn(t2, PARAMS)  # M2 fails: nudge 1 of 1
    assert t2.notes == ["Did the comment go on the card?"]
    t3 = Turn(env, records, 3)
    await post_turn(t3, PARAMS)  # still failing, not blocking: let go, and it was the last
    assert t3.notes == ["M2 isn't done, but let it go. " + mg.DEFAULT_FINAL_GAVE_UP]  # not "everything is done"
    assert records[-1]["decision"] == "gave_up" and records[-1]["wrap_up"] is True
    t4 = Turn(env, records, 4)
    assert await post_turn(t4, PARAMS) == {"end": "milestones_done"}
    assert _status(t4) == [("M1", "passed", 1, 1, 0), ("M2", "gave_up", 2, 3, 1)]
    assert (t4.summary["done"], t4.summary["end"]) == (1, "milestones_done")


@pytest.mark.asyncio
async def test_judge_gates_run_only_after_the_cheap_ones_pass(monkeypatch):
    calls = []

    async def judge(spec, statement, text):
        calls.append(statement)
        return "19.99" in text, "seen"

    monkeypatch.setattr(mg, "_judge", judge)
    params = json.loads(json.dumps(PARAMS))
    params["milestones"][0]["gates"].append(
        {"id": "m1-1p", "nudge": "Check the seat counts.", "check": {"artifact": "sheet", "judge": "Vaultly 19.99"}})
    env, records = Env(world()), []
    await post_turn(Turn(env, records, 1), params)
    assert calls == [] and records[-1]["gates"][3]["passed"] is None
    env.state["drive_list"] = {"files": [SHEET]}
    await post_turn(Turn(env, records, 2), params)
    assert calls == ["Vaultly 19.99"] and records[-1]["decision"] == "passed"


@pytest.mark.asyncio
async def test_a_tool_error_fails_the_gate_with_the_reason():
    w = world()
    w["drive_list"] = lambda args: (_ for _ in ()).throw(RuntimeError("boom"))
    env, records = Env(w), []
    await post_turn(Turn(env, records, 1), PARAMS)
    assert records[-1]["gates"][0]["why"] == "error: RuntimeError: boom"


@pytest.mark.asyncio
async def test_on_the_last_turn_it_records_where_it_stood_and_asks_nothing():
    env, records = Env(world()), []
    t = Turn(env, records, 1, last=True)
    assert await post_turn(t, PARAMS) is None
    assert t.notes == [] and records[-1]["decision"] == "last_turn"


@pytest.mark.parametrize("bad, fragment", [
    ({"milestones": []}, "needs role"),
    ({"role": "a", "milestones": []}, "needs milestones"),
    ({"role": "a", "milestones": [{"id": "M1", "gates": []}]}, "has no gates"),
    ({"role": "a", "milestones": [{"id": "M1", "gates": [{"id": "g", "check": {}}]}]}, "needs id, nudge and check"),
    ({"role": "a", "milestones": [{"id": "M1", "gates": [{"id": "g", "nudge": "n", "check": {"artifact": "x"}}]}]},
     "not in its milestone's artifacts"),
    ({"role": "a", "milestones": [{"id": "M1", "gates": [{"id": "g", "nudge": "n",
                                                           "check": {"no_new": [{"tool": "t"}]}}]}]}, "baseline_count"),
    ({"role": "a", "milestones": [{"id": "M1", "nudges": 2, "gates": []}]}, "unknown keys"),
    ({"role": "a", "milestones": [{"id": "M1", "gates": [{"id": "g", "nudge": "n",
                                                           "check": {"trajectory_judge": "it did"}}]}]},
     "needs trajectory_judge_model"),
    ({**PARAMS, "enabled": "no"}, "enabled must be true or false"),
    ({"role": "a", "milestones": [{"id": "M1", "gates": [{"id": "g", "nudge": "n", "check": {"judge": "it did"}}]}]}, "exactly one of trajectory_judge"),
    ({"role": "a", "milestones": [{"id": "M1", "gates": [{"id": "g", "nudge": "n", "check": {"trajectory_judge": "x", "call": {"tool": "t"}}}]}]}, "exactly one of trajectory_judge"),
    ({"role": "a", "milestones": [{"id": "M1", "gates": [{"id": "g", "nudge": "n", "check": {"call": {}, "expect": "exists"}}]}]}, "needs a tool"),
    ({"role": "a", "milestones": [{"id": "M1", "gates": [{"id": "g", "nudge": "n", "check": {"call": {"tool": "t"}}}]}]}, "expect must be"),
    ({"role": "a", "milestones": [{"id": "M1", "gates": [{"id": "g", "nudge": "n", "check": {"no_new": []}}]}]}, "no_new needs a list"),
    ({"role": "a", "milestones": [{"id": "M1", "artifacts": {"x": {}}, "gates": [
        {"id": "g", "nudge": "n", "check": {"artifact": "x"}}]}]}, "exactly one of expect, format, judge"),
    ({"role": "a", "milestones": [{"id": "M1", "artifacts": {"x": {}}, "gates": [
        {"id": "g", "nudge": "n", "check": {"artifact": "x", "format": {"max_words": 3}}}]}]}, "format needs"),
    ({"role": "a", "milestones": [{"id": "M1", "gates": [
        {"id": "g", "nudge": "n", "nudges": 2, "check": {"call": {"tool": "t"}, "expect": "exists"}}]}]},
     "unknown keys"),
    ({"role": "a", "milestones": [{"id": "M1", "gates": [
        {"id": "g", "nudge": "n", "type": "style", "check": {"call": {"tool": "t"}, "expect": "exists"}}]}]},
     "is not one of"),
    ({"role": "a", "milestones": [
        {"id": "M1", "gates": [{"id": "g", "nudge": "n", "check": {"call": {"tool": "t"}, "expect": "exists"}}]},
        {"id": "M2", "line": "next", "gates": [
            {"id": "g", "nudge": "n", "check": {"call": {"tool": "t"}, "expect": "exists"}}]}]},
     "gate ids must be unique"),
])
def test_a_bad_spec_is_refused(bad, fragment):
    with pytest.raises(ValueError, match=fragment):
        mg._spec(bad)
    with pytest.raises(ValueError, match=fragment):  # and when the framework loads the hook, before any turn
        mg.check_params(bad)


@pytest.mark.asyncio
async def test_a_number_field_is_compared_against_its_baseline():
    params = {"role": "r", "milestones": [{"id": "M1", "max_nudges": 1, "gates": [{"id": "q", "nudge": "no email",
              "check": {"no_new": [{"tool": "labels", "path": "labels", "where": {"id": "SENT"},
                                    "number": "messagesTotal", "baseline_count": 41}]}}]}]}
    env, records = Env({"labels": {"labels": [{"id": "INBOX", "messagesTotal": 9}, {"id": "SENT", "messagesTotal": 41}]}}), []
    await post_turn(Turn(env, records, 1), params)
    assert records[-1]["gates"][0]["passed"] is True
    env.state["labels"]["labels"][1]["messagesTotal"] = 42
    records = []
    await post_turn(Turn(env, records, 1), params)
    assert records[-1]["gates"][0] == {"id": "q", "passed": False, "why": "new: labels +1", "nudge": "no email"}


def test_a_gmail_message_reads_as_headers_and_its_plain_text_body():
    import base64
    body = base64.urlsafe_b64encode("Run-rate is $4,210/month.\n\nDisputed: Murf, Webflow.".encode()).decode().rstrip("=")
    msg = {"threadId": "t/184", "snippet": "Run-rate", "payload": {"mimeType": "multipart/alternative",
           "headers": [{"name": "Subject", "value": "Re: vendor spend"}, {"name": "To", "value": "dana@x.co"}],
           "parts": [{"mimeType": "text/html", "body": {"data": "PGI-aGk8L2I-"}},
                     {"mimeType": "text/plain", "body": {"data": body}}]}}
    text = mg._gmail_text(json.dumps(msg))
    assert text.startswith("Subject: Re: vendor spend\nTo: dana@x.co\nCc: \nThread: t/184\n\nRun-rate is $4,210")
    assert mg._format(text.split("\n\n", 1)[1], {"max_paragraphs": 2}) == (True, "format ok")


LINED = json.loads(json.dumps(PARAMS))
LINED["milestones"][0]["line"] = "Make me a sheet of every subscription."
LINED["milestones"][1]["line"] = "Now put what you found on the cut-list card. Don't email anyone yet."


@pytest.mark.asyncio
async def test_when_the_sim_sends_a_later_line_the_hook_follows_it_before_the_agent_acts():
    w = world()
    w["gmail_sent"] = {"messages": [{"id": "a"}, {"id": "b"}]}  # M1 fails on quiet (baseline 0)
    env, records = Env(w), []
    t = Turn(env, records, 1, reply="ok thanks.\n\nNow put what you found on the cut-list card. Don't email anyone yet.")
    await post_turn(t, LINED)
    r = records[-1]
    assert r["decision"] == "nudge" and r["followed_sim"] == "M2" and r["next_index"] == 1
    assert r["baselines"] == {"M2:m2-quiet:0": 2}  # taken now, before the agent works on M2
    t2 = Turn(env, records, 2)
    await post_turn(t2, LINED)  # M2 is checked now, against its own baseline: no comment yet
    assert records[-1]["milestone"] == "M2" and t2.notes == ["Did the comment go on the card?"]
    assert records[-1]["nudges"] == 1
    assert _status(t2) == [("M1", "passed_over", 1, 1, 1), ("M2", "open", 2, None, 1)]


@pytest.mark.asyncio
async def test_the_sim_milestone_field_moves_the_hook_when_there_is_no_line():
    env, records = Env(world()), []
    await post_turn(Turn(env, records, 1, sim_fields={"milestone": "M2"}), PARAMS)
    assert records[-1]["followed_sim"] == "M2"
    await post_turn(Turn(env, records, 2, sim_fields={"milestone": "M1"}), PARAMS)  # never moves back
    assert records[-1]["milestone"] == "M2" and "followed_sim" not in records[-1]


@pytest.mark.asyncio
async def test_the_same_sim_message_twice_in_a_row_ends_the_conversation():
    env, records = Env(world()), []
    assert (await post_turn(Turn(env, records, 1, reply="thanks, that's all for now"), PARAMS))["message"]
    # The repeat still reaches the agent (a hook that asked returns the answer); the next turn ends unasked.
    assert (await post_turn(Turn(env, records, 2, reply="Thanks - that's all for now!"), PARAMS))["message"]
    assert records[-1]["idle"] is True
    t3 = Turn(env, records, 3)
    assert await post_turn(t3, PARAMS) == {"end": "sim_idle"}
    assert t3.notes == [] and records[-1]["end"] == "sim_idle"


@pytest.mark.asyncio
async def test_an_unknown_judge_gate_neither_fails_nor_nudges(monkeypatch):
    async def judge(spec, statement, text):
        return None, "judge unknown: JSONDecodeError"

    monkeypatch.setattr(mg, "_judge", judge)
    params = json.loads(json.dumps(PARAMS))
    params["milestones"][0]["gates"].append(
        {"id": "m1-shape", "nudge": "Can it show a total?", "check": {"artifact": "sheet", "judge": "has a total"}})
    w = world()
    w["drive_list"] = {"files": [SHEET]}
    env, records = Env(w), []
    t = Turn(env, records, 1)
    await post_turn(t, params)
    assert records[-1]["decision"] == "passed" and records[-1]["gates"][3]["passed"] is None
    assert t.notes == ["M1 is done. Open M2 now: send its line."]


@pytest.mark.asyncio
async def test_the_judge_retries_a_cut_off_reply_once(monkeypatch):
    import sys
    import types
    replies = ['{"passed": true, "why": "the sheet has a tot', '{"passed": true, "why": "has a total"}']
    posts = []

    class Resp:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": self.content}}]}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json, headers, timeout):
            posts.append(json)
            return Resp(replies[len(posts) - 1])

    class Config:
        def get_litellm_base_url(self):
            return "http://llm"

        def get_litellm_api_key(self):
            return "k"

        def get_model_headers(self, base):
            return {}

    monkeypatch.setattr("httpx.AsyncClient", lambda: Client())
    cfg = types.ModuleType("agent_env.config")
    cfg.get_config = lambda: Config()
    monkeypatch.setitem(sys.modules, "agent_env.config", cfg)
    assert await mg._judge({"judge_model": "m"}, "has a total", "text") == (True, "has a total")
    assert len(posts) == 2 and posts[0]["max_tokens"] >= 1000
    replies[1] = replies[0]
    posts.clear()
    passed, why = await mg._judge({"judge_model": "m"}, "has a total", "text")
    assert passed is None and why.startswith("judge unknown: JSONDecodeError")


def test_paragraphs_are_counted_in_the_body_as_a_reader_counts_them():
    reply = ("Subject: Re: Vendor spend\nTo: dana@example.com\nCc: \nThread: t1\n\n"
             "Dana,\n\nGoing into Q4 the vendor subscriptions run 1,204.50 a month from the October bill.\n\n"
             "Four charges to dispute:\n\n- Midjourney 60.00\n- SavvyCal 16.00\n\nThanks,\nFred")
    assert mg._format(reply, {"max_paragraphs": 2}) == (True, "format ok")
    assert mg._format(reply, {"max_paragraphs": 1})[0] is False
    assert mg._format(reply, {"contains_all": ["Thread: t1", "Midjourney"]}) == (True, "format ok")
    assert mg._format("One.\n\nTwo is here.\n\nThree is here too.", {"max_paragraphs": 2}) == (
        False, "3 paragraphs > 2")


def _three(**m2):
    """PARAMS with M2's settings changed and an M3 (one comment on a Linear issue)."""
    params = json.loads(json.dumps(PARAMS))
    params["milestones"][1].update(m2)
    params["milestones"].append({"id": "M3", "gates": [{"id": "m3-linear", "nudge": "Did it go on the ticket?",
                                 "check": {"call": {"tool": "linear_comments", "path": "comments"},
                                           "expect": {"count": {"min": 1}}}}]})
    return params


def _m1_done_and_m2_too():
    w = world()
    w["drive_list"] = {"files": [SHEET]}
    w["trello_comments"] = {"comments": [{"id": "c1"}]}  # M2's comment, made early while M1 was open
    w["linear_comments"] = {"comments": []}
    return Env(w)


@pytest.mark.asyncio
async def test_a_next_milestone_the_agent_already_did_is_skipped():
    env, records = _m1_done_and_m2_too(), []
    t = Turn(env, records, 1)
    await post_turn(t, _three())
    assert t.notes == ["M1 is done. M2 is already done too, so skip its line. Open M3 now: send its line."]
    assert records[-1]["next_index"] == 2 and records[-1]["skipped"][0]["milestone"] == "M2"
    assert mg._state(records)["index"] == 2
    assert _status(t) == [("M1", "passed", 1, 1, 0), ("M2", "skipped", 1, 1, 0), ("M3", "open", 2, None, 0)]
    assert (t.summary["reached"], t.summary["done"]) == ("M3", 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("m2", [{"type": "correct"}, {"skip_if_done": False}], ids=["correct", "opted-out"])
async def test_a_correct_or_opted_out_milestone_is_opened_even_when_done(m2):
    env, records = _m1_done_and_m2_too(), []
    t = Turn(env, records, 1)
    await post_turn(t, _three(**m2))
    assert t.notes == ["M1 is done. Open M2 now: send its line."] and "skipped" not in records[-1]


@pytest.mark.asyncio
async def test_a_milestone_with_only_no_new_gates_is_never_skipped():
    params = _three()
    params["milestones"][1]["gates"] = [params["milestones"][1]["gates"][1]]  # m2-quiet alone
    env, records = _m1_done_and_m2_too(), []
    t = Turn(env, records, 1)
    await post_turn(t, params)
    assert t.notes == ["M1 is done. Open M2 now: send its line."]


@pytest.mark.asyncio
async def test_skipping_the_last_milestone_sends_the_wrap_up():
    env, records = _m1_done_and_m2_too(), []
    t = Turn(env, records, 1)
    await post_turn(t, PARAMS)  # two milestones: M2 was the last
    assert t.notes == ["M1 is done. M2 is already done too, so skip its line. " + mg.DEFAULT_FINAL]
    assert records[-1]["wrap_up"] is True and mg._state(records)["finished"] is True



# ---- trajectory_judge ---------------------------------------------------------------------------------------------

def _turn_messages(*calls, text=None):
    """A claude-code turn trajectory: each call is (tool, input, result)."""
    assistant = [{"type": "tool_use", "id": f"t{i}", "name": f"mcp__hg__{tool}", "input": inp}
                 for i, (tool, inp, _) in enumerate(calls)]
    if text:
        assistant.append({"type": "text", "text": text})
    results = [{"type": "tool_result", "tool_use_id": f"t{i}", "content": [{"type": "text", "text": res}]}
               for i, (_, _, res) in enumerate(calls)]
    return [{"type": "assistant", "message": {"content": assistant}}, {"type": "user", "message": {"content": results}}]


TRAJ_PARAMS = {"role": "auditor", "trajectory_judge_model": "traj-m", "milestones": [
    {"id": "M1", "line": "make the sheet", "gates": [
        {"id": "m1-sheet", "nudge": "I don't see the sheet.", "check": {"trajectory_judge": "The agent saves a sheet."}}]},
    {"id": "M2", "line": "comment on the ticket", "gates": [
        {"id": "m2-linear", "nudge": "Anything on the ticket?",
         "check": {"trajectory_judge": "The agent comments on ENG-1."}}]},
]}
TRAJ = {
    "s3://t/1.json": _turn_messages(("gdrive_list_folder", {"folderId": "f"}, "[]" * 200),
                                    ("gsheets_create_spreadsheet", {"title": "Subs"}, "id s1"), text="Made it."),
    "s3://t/2.json": _turn_messages(("linear_create_comment", {"issueId": "ENG-1", "body": "Acme, Globex"}, "ok")),
}


def test_the_renderer_keeps_writes_whole_and_shortens_reads():
    text = mg.render_trajectory(TRAJ["s3://t/1.json"])
    assert "WRITE gsheets_create_spreadsheet(" in text and "read gdrive_list_folder(" in text
    assert "AGENT SAYS: Made it." in text and "+240 chars" in text  # the read's 400-char result is cut to 160
    assert [mg.is_write(t) for t in ("trello_get_comments", "gmail_get_messages_batch", "linear_list_comments",
                                     "linear_create_comment", "gsheets_values_update", "gdrive_move_file")] == [
        False, False, False, True, True, True]


@pytest.mark.asyncio
async def test_a_trajectory_gate_judges_every_turn_so_far_with_the_open_milestones_turns_marked(monkeypatch):
    mg._RENDERED.clear()
    seen = []

    evidence = {"The agent saves a sheet.": "gsheets_create_spreadsheet",
                "The agent comments on ENG-1.": "linear_create_comment"}

    async def judge(spec, statement, text, *, model=None, system=mg.JUDGE_SYSTEM, max_chars=40000):
        seen.append((statement, model, system, text, max_chars))
        return evidence[statement] in text, "judged on the log"

    monkeypatch.setattr(mg, "_judge", judge)
    records = []
    await post_turn(Turn(Env(world()), records, 1, trajectories=TRAJ), TRAJ_PARAMS)  # M1 passes after turn 1
    assert records[-1]["next_index"] == 1 and mg._state(records)["since"] == 2
    t2 = Turn(Env(world()), records, 2, trajectories=TRAJ)
    await post_turn(t2, TRAJ_PARAMS)
    statement, model, system, log, max_chars = seen[-1]
    assert statement == "The agent comments on ENG-1." and model == "traj-m"
    assert max_chars is None  # the log is sized by trajectory_max_chars, never cut to an artifact's 40k
    assert system == mg.TRAJECTORY_JUDGE_SYSTEM
    assert "===== Turn 1 =====" in log and "===== Turn 2 (THIS STEP) =====" in log
    assert "WRITE linear_create_comment(" in log and "WRITE gsheets_create_spreadsheet(" in log
    assert t2.reads == ["s3://t/2.json"]  # turn 1 was rendered once, at turn 1


@pytest.mark.asyncio
async def test_without_trajectories_a_trajectory_gate_is_unknown_and_never_nudges(monkeypatch):
    called = []

    async def judge(*a, **kw):
        called.append(1)
        return False, "should not run"

    monkeypatch.setattr(mg, "_judge", judge)
    records = []
    t = Turn(Env(world()), records, 1)  # no trajectories
    await post_turn(t, TRAJ_PARAMS)
    gate = records[-1]["gates"][0]
    assert gate["passed"] is None and gate["why"].startswith("unknown") and not called
    assert records[-1]["decision"] == "hold" and t.notes == [mg.HOLD_NOTE]   # nothing checked: hold, never pass


def test_the_evidence_cuts_the_oldest_turns_before_the_step_first():
    texts = [(1, "a" * 100), (2, "b" * 100), (3, "c" * 100)]
    log = mg._evidence(texts, range(3, 4), 260)
    assert "Turn 1" not in log and "Turn 2 =====" in log and "Turn 3 (THIS STEP)" in log



@pytest.mark.asyncio
async def test_a_step_turn_whose_trajectory_wasnt_fetched_makes_the_gate_unknown_not_a_fail(monkeypatch):
    mg._RENDERED.clear()
    called = []

    async def judge(*a, **kw):
        called.append(1)
        return False, "should not run"

    monkeypatch.setattr(mg, "_judge", judge)
    records = []
    t = Turn(Env(world()), records, 1, trajectories={None: None})  # turn 1's fetch failed: its url is None
    await post_turn(t, TRAJ_PARAMS)
    gate = records[-1]["gates"][0]
    assert gate["passed"] is None and "turn(s) [1]" in gate["why"] and not called
    # Nothing could be checked: no nudge on missing evidence, and no move on either; the hold counts toward max_nudges.
    assert t.notes == [mg.HOLD_NOTE] and records[-1]["decision"] == "hold" and "next_index" not in records[-1]
    assert mg._state(records)["nudges"] == 1


@pytest.mark.asyncio
async def test_the_note_that_opens_a_milestone_carries_its_line_so_any_user_sim_can_send_it():
    w = world()
    w["drive_list"] = {"files": [SHEET]}
    env, records = Env(w), []
    t = Turn(env, records, 1)
    await post_turn(t, LINED)
    assert t.notes == ["M1 is done. Open M2 now: send this line: Now put what you found on the cut-list card. "
                       "Don't email anyone yet."]


@pytest.mark.asyncio
async def test_a_gates_declared_type_is_the_summarys_even_when_a_judge_checks_it(monkeypatch):
    # A task's rubric can call a criterion an action and still have it judged on the trajectory.
    mg._RENDERED.clear()

    async def judge(spec, statement, text, *, model=None, system=mg.JUDGE_SYSTEM, max_chars=40000):
        return True, "judged on the log"

    monkeypatch.setattr(mg, "_judge", judge)
    params = copy.deepcopy(TRAJ_PARAMS)
    params["milestones"][0]["gates"][0]["type"] = "action"
    turn = Turn(Env(world()), [], 1, trajectories=TRAJ)
    await post_turn(turn, params)

    assert turn.summary["milestones"][0]["gates"] == {"m1-sheet": {"passed": True, "type": "action"}}
    assert turn.summary["milestones"][1]["gates"] == {"m2-linear": {"passed": True, "type": "content"}}  # no type given: from its check


class ErrEnv(Env):
    """Env whose ``failing`` tools answer with a tool error."""

    def __init__(self, state, failing=()):
        super().__init__(state)
        self.failing = set(failing)

    async def call_tool(self, tool, args, role):
        if tool in self.failing:
            self.calls.append((tool, args, role))
            return {"isError": True, "content": [{"type": "text", "text": "rate limited"}]}
        return await super().call_tool(tool, args, role)


def _two_counts(first_open=True):
    """M1 passes at once; M2 watches two tools with different starting counts."""
    return {"role": "r", "milestones": [
        {"id": "M1", "gates": [{"id": "m1", "nudge": "n", "check": {"call": {"tool": "ok"}, "expect": "exists"}}]},
        {"id": "M2", "line": "next", "skip_if_done": False, "gates": [
            {"id": "q-a", "nudge": "n", "check": {"no_new": [{"tool": "a", "path": "items"}]}},
            {"id": "q-b", "nudge": "n", "check": {"no_new": [{"tool": "b", "path": "items"}]}}]}]}


@pytest.mark.asyncio
async def test_two_no_new_gates_keep_their_own_baselines():
    state = {"ok": "yes", "a": {"items": []}, "b": {"items": list(range(10))}}
    records = []
    await post_turn(Turn(Env(state), records, 1), _two_counts())  # M1 passes; M2's baselines: a 0, b 10
    assert records[-1]["baselines"] == {"M2:q-a:0": 0, "M2:q-b:0": 10}
    state["a"] = {"items": [1, 2]}  # two added where b's count would have hidden them
    await post_turn(Turn(Env(state), records, 2), _two_counts())
    assert {g["id"]: g["passed"] for g in records[-1]["gates"]} == {"q-a": False, "q-b": True}


@pytest.mark.asyncio
async def test_a_baseline_that_cant_be_read_leaves_its_gate_unknown_and_the_step_running():
    state = {"ok": "yes", "a": {"items": []}, "b": {"items": []}}
    records = []
    await post_turn(Turn(ErrEnv(state, failing={"b"}), records, 1), _two_counts())
    assert records[-1]["baselines"] == {"M2:q-a:0": 0}
    await post_turn(Turn(Env(state), records, 2), _two_counts())
    verdicts = {g["id"]: (g["passed"], g["why"][:20]) for g in records[-1]["gates"]}
    assert verdicts["q-a"][0] is True and verdicts["q-b"] == (None, "unknown: no baseline")


@pytest.mark.asyncio
async def test_an_artifact_read_that_errors_is_not_the_artifacts_text():
    params = {"role": "r", "milestones": [{"id": "M1", "artifacts": {"doc": {
        "find": {"tool": "docs", "path": "files"}, "read": {"tool": "doc_read", "args": {"id": "{id}"}}}},
        "gates": [{"id": "g", "nudge": "n", "check": {"artifact": "doc", "expect": "exists"}}]}]}
    records = []
    await post_turn(Turn(ErrEnv({"docs": {"files": [{"id": "d1"}]}}, failing={"doc_read"}), records, 1), params)
    gate = records[-1]["gates"][0]
    assert gate["passed"] is False and "tool error reading 'doc'" in gate["why"]


@pytest.mark.asyncio
async def test_a_milestone_that_passes_on_the_final_turn_is_passed_in_the_summary():
    params = {"role": "r", "milestones": [
        {"id": "M1", "gates": [{"id": "g", "nudge": "n", "check": {"call": {"tool": "ok"}, "expect": "exists"}}]}]}
    turn = Turn(Env({"ok": "yes"}), [], 1, last=True)
    await post_turn(turn, params)
    row = turn.summary["milestones"][0]
    assert (row["status"], row["closed_turn"]) == ("passed", 1)


@pytest.mark.parametrize("check, problem", [
    ({"artifact": "x", "format": {"max_paragraphs": "2"}}, "whole number"),
    ({"artifact": "x", "format": {"contains_all": "robert"}}, "list of text"),
    ({"call": {"tool": "t"}, "expect": {"count": {"min": "1"}}}, "whole number"),
    ({"call": {"tool": "t"}, "expect": {"contains": 3}}, "must be text"),
    ({"no_new": [{"tool": "t", "baseline_count": "0"}]}, "whole number"),
])
def test_a_value_of_the_wrong_type_is_refused_at_load(check, problem):
    params = {"role": "r", "milestones": [{"id": "M1", "artifacts": {"x": {"find": {"tool": "f"}}},
                                           "gates": [{"id": "g", "nudge": "n", "check": check}]}]}
    with pytest.raises(ValueError, match=problem):
        mg.check_params(params)


@pytest.mark.asyncio
async def test_a_trajectory_in_a_format_the_hook_cant_read_leaves_the_gate_unknown(monkeypatch):
    mg._RENDERED.clear()
    called = []

    async def judge(*a, **k):
        called.append(a)
        return True, "x"

    monkeypatch.setattr(mg, "_judge", judge)
    other = {"s3://t/g1.json": [{"type": "tool_use", "tool_name": "sheets_create", "parameters": {}},
                                {"type": "message", "role": "assistant", "content": "Done."}]}
    turn = Turn(Env(world()), [], 1, trajectories=other)
    await post_turn(turn, TRAJ_PARAMS)
    gate = turn._records[-1]["gates"][0]
    assert gate["passed"] is None and "format this hook reads" in gate["why"] and not called


def test_a_turns_sim_fields_are_read_from_its_own_step_only():
    turn = Turn(Env(world()), [], 1)
    turn.step_id = "now"
    turn.metadata = {"usersim_turn_outputs": {"earlier": [{"turn": 1, "fields": {"milestone": "M3"}}],
                                              "now": [{"turn": 1, "fields": {"milestone": "M1"}}]}}
    assert mg._sim_field(turn, "milestone") == "M1"


@pytest.mark.asyncio
@pytest.mark.parametrize("field, passed", [("body", False), (None, True)])
async def test_a_field_artifact_counts_the_fields_own_paragraphs(field, passed):
    draft = {"id": 1, "to": "a@x", "body": "Para one.\n\nPara two.\n\nPara three."}
    artifact = {"find": {"tool": "list_drafts", "pick": "last"}, **({"field": field} if field else {})}
    params = {"role": "r", "milestones": [{"id": "M1", "artifacts": {"draft": artifact}, "gates": [
        {"id": "short", "nudge": "n", "check": {"artifact": "draft", "format": {"max_paragraphs": 2}}}]}]}
    records = []
    await post_turn(Turn(Env({"list_drafts": [draft]}), records, 1), params)
    # As JSON the body's blank lines are escaped, so the whole item reads as one paragraph and passes wrongly.
    assert records[-1]["gates"][0]["passed"] is passed


@pytest.mark.asyncio
async def test_the_last_milestone_is_checked_again_after_the_wrap_up_and_the_summary_shows_what_the_agent_left():
    # M2 is let go, then the agent fixes it while answering the wrap-up: the final check finds it done.
    w = world()
    w["drive_list"] = {"files": [SHEET]}
    env, records = Env(w), []
    await post_turn(Turn(env, records, 1), PARAMS)  # M1 passes
    await post_turn(Turn(env, records, 2), PARAMS)  # M2 nudged
    await post_turn(Turn(env, records, 3), PARAMS)  # M2 let go; wrap-up sent
    w["trello_comments"] = {"comments": [{"text": "Vaultly cancelled"}]}  # done while answering the wrap-up
    t4 = Turn(env, records, 4)
    assert await post_turn(t4, PARAMS) == {"end": "milestones_done"}
    assert records[-1]["decision"] == "done" and records[-1]["milestone"] == "M2"
    assert all(g["passed"] for g in records[-1]["gates"])
    assert _status(t4)[1] == ("M2", "passed", 2, 4, 1)  # passed in the end, after the wrap-up


@pytest.mark.asyncio
async def test_the_wrap_up_after_every_milestone_passed_says_so():
    w = world()
    w["drive_list"] = {"files": [SHEET]}
    w["trello_comments"] = {"comments": [{"text": "Vaultly cancelled"}]}
    env, records = Env(w), []
    await post_turn(Turn(env, records, 1), PARAMS)  # M1 passes, and M2 is already done: skipped, then the wrap-up
    wrap = next(r for r in records if r.get("wrap_up"))
    assert wrap["note"].endswith(mg.DEFAULT_FINAL) and mg.DEFAULT_FINAL_GAVE_UP not in wrap["note"]


@pytest.mark.asyncio
async def test_a_milestone_with_a_gate_it_couldnt_check_is_held_not_passed():
    state = {"ok": "yes", "a": {"items": []}, "b": {"items": []}}
    records = []
    await post_turn(Turn(ErrEnv(state, failing={"b"}), records, 1), _two_counts())  # M2's b baseline unread
    t2 = Turn(Env(state), records, 2)
    await post_turn(t2, _two_counts())
    assert records[-1]["decision"] == "hold" and t2.notes == [mg.HOLD_NOTE]


@pytest.mark.asyncio
async def test_a_no_new_gate_keeps_a_failure_one_call_found_when_another_has_no_baseline():
    params = {"role": "r", "milestones": [
        {"id": "M1", "gates": [{"id": "m1", "nudge": "n", "check": {"call": {"tool": "ok"}, "expect": "exists"}}]},
        {"id": "M2", "line": "next", "skip_if_done": False, "gates": [{"id": "q", "nudge": "hold off", "check": {
            "no_new": [{"tool": "a", "path": "items"}, {"tool": "b", "path": "items"}]}}]}]}
    state = {"ok": "yes", "a": {"items": []}, "b": {"items": []}}
    records = []
    await post_turn(Turn(ErrEnv(state, failing={"b"}), records, 1), params)
    state["a"] = {"items": [1]}
    await post_turn(Turn(Env(state), records, 2), params)
    gate = records[-1]["gates"][0]
    assert gate["passed"] is False and gate["why"] == "new: a +1"


@pytest.mark.asyncio
async def test_a_milestone_broken_while_the_agent_answers_the_wrap_up_is_undone_in_the_summary():
    w = world()
    w["drive_list"] = {"files": [SHEET]}
    w["trello_comments"] = {"comments": [{"text": "Vaultly cancelled"}]}
    env, records = Env(w), []
    await post_turn(Turn(env, records, 1), PARAMS)  # M1 passes; M2 already done: skipped; wrap-up
    assert next(r for r in records if r.get("wrap_up"))["next_index"] == 1  # the final check reads M2, the last
    w["trello_comments"] = {"comments": []}  # the agent removes its comment while answering the wrap-up
    t2 = Turn(env, records, 2)
    await post_turn(t2, PARAMS)
    assert records[-1]["milestone"] == "M2"
    assert _status(t2)[1][:2] == ("M2", "undone") and t2.summary["done"] == 1


# ---- revisions ----------------------------------------------------------------------------------------------------

REPORT = {"report": {"find": {"tool": "docs", "path": "files"}, "read": {"tool": "doc_read", "args": {"id": "{id}"}}}}


def _revised(**r2) -> dict:
    """M1, then its revision M1-r2 (carrying M1's gates, overriding m1-total), then M2."""
    return {"role": "r", "milestones": [
        {"id": "M1", "artifacts": REPORT, "gates": [
            {"id": "m1-report", "nudge": "Where's the report?", "check": {"artifact": "report", "expect": "exists"}},
            {"id": "m1-total", "nudge": "Does it have the total?",
             "check": {"artifact": "report", "format": {"contains_all": ["total"]}}},
            {"id": "m1-quiet", "nudge": "Don't email yet.",
             "check": {"no_new": [{"tool": "gmail_sent", "path": "messages", "baseline_count": 0}]}}]},
        {"id": "M1-r2", "group": "M1", "carry_gates": True, "line": "Make it a table; it's going public now.",
         "max_nudges": 1, "blocking": False, "gates": [
             {"id": "m1-table", "nudge": "Can it be a table?",
              "check": {"artifact": "report", "format": {"contains_all": ["|"]}}},
             {"id": "m1-total", "nudge": "Keep the total, not the internal number.",
              "check": {"artifact": "report", "format": {"contains_all": ["total"], "contains_none": ["internal"]}}}],
         **r2},
        {"id": "M2", "line": "Now post it on the card.", "gates": [
            {"id": "m2-card", "nudge": "Is it on the card?",
             "check": {"call": {"tool": "trello_comments", "path": "comments"}, "expect": {"count": {"min": 1}}}}]}]}


def report_world(text="Report\ntotal 40 (internal 12)"):
    return {"docs": {"files": [{"id": "d1"}]}, "doc_read": text, "gmail_sent": {"messages": []},
            "trello_comments": {"comments": []}}


def _load_error(milestones) -> str:
    with pytest.raises(ValueError) as e:
        mg.check_params({"role": "r", "milestones": milestones})
    return str(e.value)


def _m(mid, gate="g", **kw):
    return {"id": mid, "gates": [{"id": f"{mid}-{gate}", "nudge": "n", "check": {"call": {"tool": "t"}, "expect": "exists"}}],
            **kw}


def test_a_revision_must_name_an_earlier_milestone_and_follow_it():
    assert "names no earlier milestone" in _load_error([_m("M1", group="M9")])
    assert "names no earlier milestone" in _load_error([_m("M1"), _m("M2", group="M3"), _m("M3")])
    # M1's revision after M2 has opened: its revisions must run one after another
    assert "must follow it or its last revision" in _load_error([_m("M1"), _m("M2"), _m("M1-r2", group="M1")])
    assert "carry_gates needs group" in _load_error([_m("M1"), _m("M2", carry_gates=True)])
    assert "carry_gates must be true or false" in _load_error([_m("M1"), _m("M1-r2", group="M1", carry_gates="yes")])


def test_a_revision_may_name_its_milestone_or_an_earlier_revision_of_it():
    spec = mg._spec({"role": "r", "milestones": [_m("M1"), _m("M1-r2", group="M1"), _m("M1-r3", group="M1-r2"),
                                                 _m("M2")]})
    assert [(m["id"], m["group"], m["revision"], m["skip_if_done"]) for m in spec["milestones"]] == [
        ("M1", "M1", 1, True), ("M1-r2", "M1", 2, False), ("M1-r3", "M1", 3, False), ("M2", "M2", 1, True)]


def test_only_a_carrying_revision_may_reuse_a_gate_id_and_then_it_overrides_that_gate():
    reuse = [_m("M1"), {**_m("M1-r2", group="M1"), "gates": _m("M1")["gates"]}]
    assert "gate ids must be unique" in _load_error(reuse)
    reuse[1]["carry_gates"] = True
    mg.check_params({"role": "r", "milestones": reuse})
    assert "gate ids must be unique" in _load_error([_m("M1"), _m("M2"), {**_m("M3"), "gates": _m("M1")["gates"]}])


def test_carried_gates_keep_their_place_an_override_takes_its_and_new_gates_follow():
    spec = mg._spec(_revised())
    r2 = spec["milestones"][1]
    assert [(g["id"], g.get("carried", False)) for g in r2["gates"]] == [
        ("m1-report", True), ("m1-total", False), ("m1-quiet", True), ("m1-table", False)]
    assert r2["gates"][1]["nudge"] == "Keep the total, not the internal number."
    assert r2["artifacts"] == REPORT  # carried with the gates that read it
    assert [g["id"] for g in spec["milestones"][2]["gates"]] == ["m2-card"]  # a new milestone carries nothing


def test_a_revision_without_carry_gates_checks_only_its_own_gates_and_may_read_its_own_artifacts():
    params = _revised(carry_gates=False)
    assert "not in its milestone's artifacts" in _load_error(params["milestones"])
    params["milestones"][1]["artifacts"] = REPORT
    assert "gate ids must be unique" in _load_error(params["milestones"])  # m1-total, reused without carrying
    params["milestones"][1]["gates"][1]["id"] = "m1-public"
    assert [g["id"] for g in mg._spec(params)["milestones"][1]["gates"]] == ["m1-table", "m1-public"]


@pytest.mark.asyncio
async def test_a_revision_opens_after_its_milestone_even_when_already_done_and_nudges_on_a_carried_gate():
    w = report_world("| total | 40 |")  # already a table with no internal number
    env, records = Env(w), []
    t1 = Turn(env, records, 1)
    await post_turn(t1, _revised())
    assert t1.notes == ["M1 is done. Open M1-r2 now: send this line: Make it a table; it's going public now."]
    w["docs"] = {"files": []}  # the agent deletes the report while revising it
    t2 = Turn(env, records, 2)
    await post_turn(t2, _revised())
    assert t2.notes == ["Where's the report?"]  # M1's own gate, carried into the revision
    assert records[-1]["milestone"] == "M1-r2" and records[-1]["decision"] == "nudge"


@pytest.mark.asyncio
async def test_a_revision_let_go_opens_the_next_milestone():
    env, records = Env(report_world()), []
    await post_turn(Turn(env, records, 1), _revised())  # M1 passes
    await post_turn(Turn(env, records, 2), _revised())  # M1-r2 nudged: still internal, no table
    t3 = Turn(env, records, 3)
    await post_turn(t3, _revised())  # max_nudges 1, not blocking: let go
    assert t3.notes == ["M1-r2 isn't done, but let it go. Open M2 now: send this line: Now post it on the card."]
    m1 = t3.summary["milestones"][0]
    assert (m1["status"], m1["reached_revision"], m1["revisions_done"]) == ("gave_up", "M1-r2", 1)
    assert [(r["revision"], r["status"]) for r in m1["revisions"]] == [(1, "passed"), (2, "gave_up")]


@pytest.mark.asyncio
async def test_a_blocking_revision_that_never_passes_ends_the_run():
    env, records = Env(report_world()), []
    for n in (1, 2, 3):
        returned = await post_turn(Turn(env, records, n), _revised(blocking=True))
    assert returned == {"end": "gate_never_passed:M1-r2"}


async def _run_revised(w, final_text=None):
    """M1 passes, M1-r2 is nudged then passes, M2 passes and the wrap-up goes out, then the final check."""
    env, records, turns = Env(w), [], []

    async def turn(n):
        turns.append(Turn(env, records, n))
        return await post_turn(turns[-1], _revised())

    await turn(1)
    await turn(2)
    w["doc_read"] = "| total | 40 |"
    await turn(3)
    w["trello_comments"] = {"comments": [{"text": "posted"}]}
    await turn(4)
    if final_text is not None:
        w["doc_read"] = final_text
    env.calls.clear()
    returned = await turn(5)
    return returned, records, turns, env


@pytest.mark.asyncio
async def test_the_summary_nests_revisions_under_their_milestone():
    returned, records, turns, _ = await _run_revised(report_world())
    assert returned == {"end": "milestones_done"}
    s = turns[-1].summary
    assert {k: s[k] for k in ("reached", "reached_revision", "done", "total", "revisions_done", "revisions_total",
                              "end")} == {"reached": "M2", "reached_revision": "M2", "done": 2, "total": 2,
                                          "revisions_done": 3, "revisions_total": 3, "end": "milestones_done"}
    m1, m2 = s["milestones"]
    assert {k: m1[k] for k in ("id", "status", "opened_turn", "closed_turn", "nudges", "reached_revision",
                               "revisions_done")} == {"id": "M1", "status": "passed", "opened_turn": 1, "closed_turn": 3,
                                                      "nudges": 1, "reached_revision": "M1-r2", "revisions_done": 2}
    assert [(r["id"], r["revision"], r["status"], r["opened_turn"], r["closed_turn"], r["nudges"])
            for r in m1["revisions"]] == [("M1", 1, "passed", 1, 1, 0), ("M1-r2", 2, "passed", 2, 3, 1)]
    r2_gates = m1["revisions"][1]["gates"]
    assert r2_gates["m1-report"] == {"passed": True, "type": "action", "carried": True}
    assert "carried" not in r2_gates["m1-table"] and "carried" not in r2_gates["m1-total"]
    assert (m2["status"], "revisions" in m2) == ("passed", False)  # a milestone without revisions is one row


@pytest.mark.asyncio
async def test_after_the_wrap_up_every_milestones_latest_revision_is_checked_and_a_broken_one_is_undone():
    returned, records, turns, env = await _run_revised(report_world(), final_text="total 40 (internal 12)")
    done = records[-1]
    assert done["decision"] == "done" and done["milestone"] == "M2"
    [m1] = done["final_checks"]
    assert m1["milestone"] == "M1-r2"
    # state gates only: m1-quiet (no_new) is about the turns M1-r2 was open, so it isn't checked again
    assert [(g["id"], g["passed"]) for g in m1["gates"]] == [
        ("m1-report", True), ("m1-total", False), ("m1-table", False)]
    assert not any(tool == "gmail_sent" for tool, _, _ in env.calls)
    s = turns[-1].summary
    assert [(r["id"], r["status"]) for r in s["milestones"][0]["revisions"]] == [("M1", "passed"), ("M1-r2", "undone")]
    assert (s["milestones"][0]["status"], s["done"], s["revisions_done"]) == ("undone", 1, 2)


@pytest.mark.asyncio
async def test_after_the_wrap_up_an_earlier_milestone_without_revisions_is_checked_too():
    w = world()
    w["drive_list"] = {"files": [SHEET]}
    env, records = Env(w), []
    params = copy.deepcopy(PARAMS)
    params["milestones"][1]["skip_if_done"] = False
    await post_turn(Turn(env, records, 1), params)  # M1 passes
    w["trello_comments"] = {"comments": [{"text": "Vaultly cancelled"}]}
    await post_turn(Turn(env, records, 2), params)  # M2 passes: wrap-up
    w["drive_list"] = {"files": []}  # the agent removes M1's sheet while answering the wrap-up
    t3 = Turn(env, records, 3)
    assert await post_turn(t3, params) == {"end": "milestones_done"}
    assert [g["id"] for g in records[-1]["final_checks"][0]["gates"]] == ["m1-sheet", "m1-cols"]
    assert _status(t3) == [("M1", "undone", 1, 3, 0), ("M2", "passed", 2, 2, 0)]
    assert "reached_revision" not in t3.summary  # a task without revisions keeps its summary's shape


def test_a_carrying_revision_that_overrides_one_gate_twice_is_refused_at_load():
    params = _revised()
    total = params["milestones"][1]["gates"][1]
    params["milestones"][1]["gates"].append({**total, "nudge": "Keep the total somewhere."})
    with pytest.raises(ValueError, match=r"M1-r2: gate ids must be unique within a milestone; repeated: \['m1-total'\]"):
        mg.check_params(params)


@pytest.mark.asyncio
async def test_each_revisions_row_shows_the_type_of_the_check_it_ran():
    params = _revised()
    params["milestones"][0]["gates"][1]["type"] = "content"  # M1's m1-total, overridden in M1-r2 by a format check
    env, records = Env(report_world()), []
    t1 = Turn(env, records, 1)
    await post_turn(t1, params)  # M1 passes; M1-r2 opens
    m1, r2 = t1.summary["milestones"][0]["revisions"]
    assert m1["gates"]["m1-total"]["type"] == "content"
    t2 = Turn(env, records, 2)
    await post_turn(t2, params)
    m1, r2 = t2.summary["milestones"][0]["revisions"]
    assert (m1["gates"]["m1-total"]["type"], r2["gates"]["m1-total"]["type"]) == ("content", "format")


@pytest.fixture(autouse=True)
def _fresh_trajectory_caches():
    mg._RENDERED.clear()
    mg._MESSAGES.clear()


SECRET = "sk-live-9f2c"
NEVER = {"role": "auditor", "milestones": [
    {"id": "M1", "line": "map the data", "gates": [
        {"id": "m1-secret", "nudge": "Take the key out of anything you wrote.",
         "check": {"trajectory": {"contains_none": [SECRET], "scope": "run"}}}]},
    {"id": "M2", "line": "now the vendors", "gates": [
        {"id": "m2-sheet", "nudge": "Where's the vendor sheet?",
         "check": {"call": {"tool": "drive_list", "path": "files"}, "expect": "exists"}}]},
    {"id": "M3", "line": "now the old backup", "gates": [
        {"id": "m3-backup-late", "nudge": "You opened the backup before I said to.",
         "check": {"trajectory": {"no_call": [{"args_contains": "backup-drive"}], "scope": {"before": "M3"}}}}]},
]}


@pytest.mark.asyncio
async def test_a_trajectory_check_fails_on_a_secret_the_agent_wrote_but_not_on_one_it_only_read():
    read_only = {"s3://n/1.json": _turn_messages(("gdrive_read_file", {"id": "keys"}, f"key {SECRET}"), text="Read it.")}
    records = []
    await post_turn(Turn(Env(world()), records, 1, trajectories=read_only), NEVER)
    assert records[-1]["gates"][0]["passed"] is True and records[-1]["decision"] == "passed"

    mg._MESSAGES.clear()
    wrote = {"s3://n/1.json": _turn_messages(("gdocs_create", {"title": "Data map", "body": f"api key {SECRET.upper()}"},
                                              "ok"), text=f"I noted the key {SECRET}.")}
    records = []
    t = Turn(Env(world()), records, 1, trajectories=wrote)
    await post_turn(t, NEVER)
    gate = records[-1]["gates"][0]
    assert gate["passed"] is False and "gdocs_create has" in gate["why"] and "said" in gate["why"]
    assert t.notes == ["Take the key out of anything you wrote."]  # a code check, so no model call and a nudge


@pytest.mark.asyncio
async def test_a_before_scope_reads_only_the_turns_before_that_milestone_opened():
    early = _turn_messages(("gdrive_list_folder", {"folderId": "backup-drive"}, "[]"))
    clean = _turn_messages(("gdrive_list_folder", {"folderId": "shared"}, "[]"))
    for turn2, expected in ((clean, True), (early, False)):
        mg._MESSAGES.clear()
        traj = {"s3://b/1.json": clean, "s3://b/2.json": turn2, "s3://b/3.json": early}
        w, records = world(), []
        await post_turn(Turn(Env(w), records, 1, trajectories=traj), NEVER)  # M1 passes; M2 opens
        w["drive_list"] = {"files": [SHEET]}
        await post_turn(Turn(Env(w), records, 2, trajectories=traj), NEVER)  # M2 passes; M3 opens
        assert mg._state(records)["index"] == 2 and mg._opened_turn(mg._spec(NEVER), records, "M3") == 3
        await post_turn(Turn(Env(w), records, 3, trajectories=traj), NEVER)  # turn 3 opens the backup: allowed now
        gate = records[-1]["gates"][0]
        assert gate["passed"] is expected, gate["why"]
        if not expected:
            assert gate["why"].startswith("turn 2: called gdrive_list_folder(")


@pytest.mark.asyncio
async def test_a_trajectory_check_without_its_turns_is_unknown_and_holds():
    records = []
    t = Turn(Env(world()), records, 1)  # no trajectories
    await post_turn(t, NEVER)
    gate = records[-1]["gates"][0]
    assert gate["passed"] is None and gate["why"] == "unknown: no readable trajectory for turn(s) [1]"
    assert records[-1]["decision"] == "hold" and t.notes == [mg.HOLD_NOTE]


@pytest.mark.asyncio
async def test_a_breach_in_a_turn_that_was_read_fails_even_when_another_turn_wasnt():
    traj = {None: None, "s3://g/2.json": _turn_messages(("slack_post", {"text": SECRET}, "ok"))}
    records = [{"decision": "nudge", "milestone": "M1", "nudges": 1, "turn": 1}]
    await post_turn(Turn(Env(world()), records, 2, trajectories=traj), NEVER)
    assert records[-1]["gates"][0]["passed"] is False


REMOVED = {"role": "auditor", "milestones": [
    {"id": "M1", "line": "tidy the drive", "gates": [
        {"id": "m1-kept", "nudge": "Did you delete files? Put them back.",
         "check": {"no_removed": [{"tool": "drive_list", "path": "files", "baseline_count": 2}]}}]},
    {"id": "M2", "line": "now the channel", "gates": [
        {"id": "m2-kept", "nudge": "Messages went missing.",
         "check": {"no_removed": [{"tool": "slack_history", "path": "messages"}]}}]},
]}


@pytest.mark.asyncio
async def test_no_removed_fails_when_the_count_drops_below_the_one_taken_when_the_milestone_opened():
    w = world()
    w["drive_list"] = {"files": [SHEET, {"id": "s2"}, {"id": "s3"}]}  # one added: fine
    w["slack_history"] = {"messages": [{"ts": "1"}, {"ts": "2"}]}
    records = []
    await post_turn(Turn(Env(w), records, 1), REMOVED)
    assert records[-1]["decision"] == "passed" and records[-1]["gates"][0]["why"] == "nothing removed"
    assert records[-1]["baselines"] == {"M2:m2-kept:0": 2}  # taken as M2 opens
    w["slack_history"] = {"messages": [{"ts": "2"}]}
    t = Turn(Env(w), records, 2)
    await post_turn(t, REMOVED)
    gate = records[-1]["gates"][0]
    assert gate["passed"] is False and gate["why"] == "removed: slack_history -1"
    assert t.notes == ["Messages went missing."]


@pytest.mark.parametrize("check, fragment", [
    ({"no_removed": [{"tool": "t"}]}, "a no_removed gate in the first milestone needs baseline_count"),
    ({"no_removed": []}, "no_removed needs a list"),
    ({"trajectory": {}}, "needs contains_none or no_call"),
    ({"trajectory": {"contains_none": "key"}}, "contains_none must be a list"),
    ({"trajectory": {"no_call": [{}]}}, "no_call must be a list"),
    ({"trajectory": {"no_call": [{"tool": "x", "args": "y"}]}}, "no_call must be a list"),
    ({"trajectory": {"contains_none": ["k"], "scope": "turn"}}, "scope must be"),
    ({"trajectory": {"contains_none": ["k"], "scope": {"before": "M9"}}}, "names milestone 'M9'"),
    ({"trajectory": {"contains_none": ["k"], "since": "M1"}}, "trajectory takes"),
])
def test_a_bad_new_gate_is_refused_at_load(check, fragment):
    with pytest.raises(ValueError, match=fragment):
        mg.check_params({"role": "a", "milestones": [{"id": "M1", "gates": [{"id": "g", "nudge": "n", "check": check}]}]})


def test_the_new_gates_are_not_rechecked_after_the_wrap_up_and_never_make_a_milestone_skippable():
    assert not mg._state_check({"no_removed": [{"tool": "t"}]})
    assert not mg._state_check({"trajectory": {"contains_none": ["k"]}})
    assert mg._gate_type({"trajectory": {"contains_none": ["k"]}}) == "action"


class FlakyTurn(Turn):
    """A turn whose trajectory reads fail for the urls in ``broken``."""

    def __init__(self, *a, broken=(), **kw):
        super().__init__(*a, **kw)
        self.broken = set(broken)

    async def read_trajectory(self, n):
        if self.trajectory_urls[n - 1] in self.broken:
            raise OSError("storage unavailable")
        return await super().read_trajectory(n)


@pytest.mark.asyncio
async def test_an_empty_or_null_trajectory_is_unread_not_clean():
    records = []
    await post_turn(Turn(Env(world()), records, 1, trajectories={"s3://e/1.json": None}), NEVER)
    gate = records[-1]["gates"][0]
    assert gate["passed"] is None and gate["why"] == "unknown: no readable trajectory for turn(s) [1]"
    assert records[-1]["decision"] == "hold"


@pytest.mark.asyncio
async def test_a_turn_that_fails_to_read_is_unread_and_the_other_turns_are_still_checked():
    traj = {"s3://f/1.json": _turn_messages(("gdrive_list", {}, "[]")),
            "s3://f/2.json": _turn_messages(("slack_post", {"text": "ok"}, "ok"))}
    records = [{"decision": "nudge", "milestone": "M1", "nudges": 1, "turn": 1}]
    await post_turn(FlakyTurn(Env(world()), records, 2, trajectories=traj, broken={"s3://f/1.json"}), NEVER)
    assert records[-1]["gates"][0]["passed"] is None  # nothing found in turn 2, and turn 1 couldn't be read

    mg._MESSAGES.clear()
    traj["s3://f/2.json"] = _turn_messages(("slack_post", {"text": SECRET}, "ok"))
    records = [{"decision": "nudge", "milestone": "M1", "nudges": 1, "turn": 1}]
    await post_turn(FlakyTurn(Env(world()), records, 2, trajectories=traj, broken={"s3://f/1.json"}), NEVER)
    gate = records[-1]["gates"][0]
    assert gate["passed"] is False and gate["why"].startswith("turn 2: slack_post has")


@pytest.mark.asyncio
async def test_text_with_quotes_newlines_or_backslashes_matches_as_written_in_nested_arguments():
    forbidden = 'the "internal" forecast\nC:\\backup'
    params = {"role": "auditor", "milestones": [{"id": "M1", "gates": [
        {"id": "g", "nudge": "n", "check": {"trajectory": {"contains_none": [forbidden],
                                                           "no_call": [{"args_contains": 'C:\\backup'}]}}}]}]}
    traj = {"s3://q/1.json": _turn_messages(("gdocs_batch_update", {"requests": [{"insertText": {"text": forbidden}}]},
                                             "ok"))}
    records = []
    await post_turn(Turn(Env(world()), records, 1, trajectories=traj), params)
    why = records[-1]["gates"][0]["why"]
    assert records[-1]["gates"][0]["passed"] is False
    assert "gdocs_batch_update has" in why and "called gdocs_batch_update(" in why


@pytest.mark.asyncio
async def test_text_split_across_two_arguments_is_not_a_match():
    params = {"role": "auditor", "milestones": [{"id": "M1", "gates": [
        {"id": "g", "nudge": "n", "check": {"trajectory": {"contains_none": ["internal\nforecast"],
                                                           "no_call": [{"args_contains": "internal\nforecast"}]}}}]}]}
    traj = {"s3://s/1.json": _turn_messages(("gdocs_create", {"title": "internal", "body": "forecast"}, "ok"))}
    records = []
    await post_turn(Turn(Env(world()), records, 1, trajectories=traj), params)
    assert records[-1]["gates"][0]["passed"] is True
