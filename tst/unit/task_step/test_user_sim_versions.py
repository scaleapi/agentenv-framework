"""The built-in user sim: byte for byte the system prompt and the reads the tested replay sent, then what each read
shows and how answers parse."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from agent_env.task_step.task_steps.agent import user_sim_versions as us

DATA = Path(__file__).parents[2] / "data" / "user_sim"
PERSON = json.loads((DATA / "hannah_person.json").read_text())
PROMPT = (DATA / "hannah_system_prompt.txt").read_text()
RUNS = json.loads((DATA / "hannah_reads.json").read_text())["runs"]
MOMENTS = [pytest.param(run, k, id=f"{run}-r{k:02d}") for run, data in RUNS.items() for k in range(len(data["reads"]))]
MESSAGE = {"at": "2026-11-17T17:00:00Z", "text": "Booked Thursday dinner at 7."}
NOTHING = {"outcome": "no_reply", "message": ""}


def _person(**changes) -> dict:
    return {**copy.deepcopy(PERSON), **changes}


def _read(time: str = "2026-11-17T17:05:00Z", **parts) -> dict:
    return {"time": time, "assistant_messages": [MESSAGE], "phone": [], "you_sent": [], "done_ends_run": False,
            **parts}


# ---- the system prompt ----------------------------------------------------------------------------------------------
def test_the_prompt_is_the_tested_one_byte_for_byte():
    assert us.setup("ours", PERSON) == {"system_prompt": PROMPT, "output_format": us.OUTPUT_FORMAT}


def test_the_opening_request_follows_the_first_who_you_are_line():
    prompt = us.system_prompt(_person(opening={"at": "2026-11-16T15:50:00Z", "text": "sort the weekend\nthanks"}))
    assert '  W1  Hannah Reyes.' in prompt.split("\n  W2")[0]
    assert '\n  W2  Earlier you texted the assistant:\n      "sort the weekend\n      thanks"\n  W3  It is' in prompt
    assert "Earlier you texted" not in us.system_prompt(_person(opening=None))


def test_when_done_ends_the_run_the_user_is_told_done_is_only_a_sign_off():
    prompt = us.system_prompt(_person(done_ends_run=True))
    assert us.DONE_ENDS_RUN in prompt and "the week goes on" not in prompt
    assert us.DONE_MULTI_DAY in us.system_prompt(_person(done_ends_run=False))


def test_a_multi_day_user_with_no_when_done_section_still_gets_the_sign_off_rule():
    person = _person()
    del person["description"]["when_done"]
    prompt = us.system_prompt(person)
    assert us.DONE_MULTI_DAY_NO_SECTION in prompt and "WHEN YOU'RE DONE" not in prompt


def test_rules_are_written_from_permissions_with_the_broken_rule_line_only_when_something_needs_asking():
    prompt = us.system_prompt(_person(permissions={
        "just_do": ["book her flights", "reply to hosts"],
        "tell_after": {"items": ["what it booked"], "then": "Ask once if it's missing."}}))
    assert "  A1  Just do: book her flights; reply to hosts.\n" in prompt
    assert "  A2  Tell you after: what it booked. Ask once if it's missing.\n" in prompt
    assert us.BROKEN_RULE_LINE not in prompt
    assert us.BROKEN_RULE_LINE in us.system_prompt(_person(permissions={"never": ["text her mom"]}))


def test_sections_with_nothing_in_them_are_left_out():
    prompt = us.system_prompt(_person(permissions={}, keep_private=[], opening=None,
                                      description={"who_you_are": ["Hannah Reyes."]}))
    assert "YOUR RULES FOR THE ASSISTANT" not in prompt and "PRIVATE" not in prompt
    assert prompt.endswith("\n\nWHO YOU ARE\n  W1  Hannah Reyes.\n")


def test_a_line_s_later_lines_hang_under_its_text():
    assert us.tag_line("W2", 'Earlier you texted:\n"hi"') == '  W2  Earlier you texted:\n      "hi"'
    assert us.tag_line("", "Answer short.\nNo emojis.") == "  Answer short.\n    No emojis."


# ---- the load check -------------------------------------------------------------------------------------------------
def test_the_tested_settings_have_no_problems():
    assert us.problems(PERSON) == []


@pytest.mark.parametrize("changes, problem", [
    ({"name": ""}, "name is missing"),
    ({"description": None}, "description must be an object of sections"),
    ({"description": "You are Hannah."}, "description must be an object of sections"),
    ({"description": {}}, "description is empty"),
    ({"description": {"who_you_are": []}}, "description is empty"),
    ({"description": {"who_you_are": ["Hannah."], "events": ["Tue: an email"]}}, "unknown sections ['events']"),
    ({"description": {"who_you_are": ["Hannah.", ""]}}, "description.who_you_are must be a list of non-empty lines"),
    ({"permissions": ["book flights"]}, "permissions must be an object"),
    ({"permissions": {"just_do": ["x"], "maybe": ["y"]}}, "permissions has unknown kinds ['maybe']"),
    ({"permissions": {"never": "text her mom"}}, "permissions.never must be a list of rules"),
    ({"permissions": {"never": {"items": ["x"], "unless": "y"}}}, "permissions.never must be a list of rules"),
    ({"keep_private": "the engagement"}, "keep_private must be a list of non-empty lines"),
    ({"time_zone": "Pacific"}, "is not an IANA time zone"),
])
def test_a_bad_setting_is_named(changes, problem):
    found = us.problems(_person(**changes))
    assert any(problem in p for p in found), found
    with pytest.raises(ValueError, match="user sim"):
        us.setup("ours", _person(**changes))


def test_the_built_in_user_sim_has_one_version():
    with pytest.raises(ValueError, match="one version"):
        us.setup("today", PERSON)


# ---- each read ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("run, k", MOMENTS)
def test_each_read_renders_as_the_replay_sent_it(run, k):
    data = RUNS[run]
    history = [{"read": read, "answer": us.parse(answer)}
               for read, answer in zip(data["reads"][:k], data["answers"][:k])]
    assert us.render(PERSON, history, data["reads"][k]) == data["expected"][k]


def test_the_assistant_s_pings_never_show_as_something_she_sent():
    read = _read(you_sent=[{"at": "2026-11-17T16:55:00Z", "text": "[Tue 8:55 AM · Messages] New text", "kind": "ping"},
                           {"at": "2026-11-17T16:56:00Z", "text": "keep it chill pls"}])
    text = us.render(PERSON, [], read)
    assert "New text" not in text
    assert "YOU SENT YOUR AI ASSISTANT:\n  [8:56 AM] keep it chill pls\n" in text
    assert "You last messaged your AI assistant at 8:56 AM." in text
    later = us.render(PERSON, [{"read": read, "answer": NOTHING}], _read("2026-11-17T18:05:00Z"))
    assert "You last messaged your AI assistant at 8:56 AM." in later


@pytest.mark.parametrize("kind", ["opening", "user_says", "user_says:noor-asks", "final", None])
def test_her_opening_request_scripted_lines_and_final_check_in_show(kind):
    line = {"at": "2026-11-17T16:56:00Z", "text": "send me the rundown", **({"kind": kind} if kind else {})}
    assert "  [8:56 AM] send me the rundown" in us.render(PERSON, [], _read(you_sent=[line]))


def test_last_messaged_counts_her_replies_by_when_she_read_but_not_a_no_reply():
    first = _read("2026-11-16T17:05:00Z")
    replied = us.render(PERSON, [{"read": first, "answer": {"outcome": "reply", "message": "ok!"}}], _read())
    assert "You last messaged your AI assistant on Mon at 9:05 AM." in replied
    for answer in (NOTHING, {"outcome": "reply", "message": "  "}):
        assert "You last messaged" not in us.render(PERSON, [{"read": first, "answer": answer}], _read())


def test_she_isn_t_asked_when_the_assistant_sent_nothing():
    assert us.render(PERSON, [], _read(assistant_messages=[{"at": "2026-11-17T17:00:00Z", "text": " "}])) is None


def test_everything_new_from_the_assistant_comes_in_one_read():
    later = {"at": "2026-11-17T17:02:00Z", "text": "And Saturday is Pell & Finch."}
    text = us.render(PERSON, [], _read(assistant_messages=[MESSAGE, later]))
    assert text.count("this is what you're answering") == 1
    assert "  [9:00 AM]\n    Booked Thursday dinner at 7.\n  [9:02 AM]\n    And Saturday is Pell & Finch." in text


def test_a_phone_item_shows_in_full_once_then_by_its_first_line():
    item = {"at": "2026-11-17T16:00:00Z", "channel": "Gmail · Carol Reyes (Mom)",
            "text": "Thursday ✈️\nWhat time do you land?"}
    first = _read(phone=[item])
    assert "YOUR NEW EMAILS, TEXTS AND NOTIFICATIONS (since you last checked):\n  [8:00 AM · Gmail · Carol Reyes (Mom)] Thursday ✈️\n" \
           "    What time do you land?" in us.render(PERSON, [], first)
    text = us.render(PERSON, [{"read": first, "answer": NOTHING}], _read("2026-11-18T17:05:00Z"))
    assert "YOUR EARLIER EMAILS, TEXTS AND NOTIFICATIONS (you've already seen these):\n  [Tue 8:00 AM · Gmail · Carol Reyes (Mom)] " \
           "Thursday ✈️\n\n" in text
    assert "YOUR NEW EMAILS, TEXTS AND NOTIFICATIONS" not in text


def test_every_phone_item_shows_once_at_its_time():
    items = [{"at": "2026-11-17T16:00:00Z", "channel": "Tablefern", "text": "Table for 4 confirmed",
              "times": ["2026-11-17T16:00:00Z", "2026-11-17T16:30:00Z"]},
             {"at": "2026-11-17T16:10:00Z", "channel": "Gmail · Priya Shah", "text": "Re: Friday"}]
    text = us.render(PERSON, [], _read(phone=items))
    assert "  [8:00 AM · Tablefern] Table for 4 confirmed\n  [8:10 AM · Gmail · Priya Shah] Re: Friday\n" in text
    assert text.count("Table for 4 confirmed") == 1


def test_the_read_after_the_final_check_in_renders_like_any_other_and_its_answer_parses():
    mom = {"at": "2026-11-17T16:00:00Z", "channel": "Gmail · Carol Reyes (Mom)", "text": "Thursday ✈️"}
    earlier = _read("2026-11-17T17:05:00Z", you_sent=[{"at": "2026-11-17T16:56:00Z", "text": "keep it chill pls"}],
                    phone=[mom])
    wrap_up = {"at": "2026-11-23T02:01:00Z", "text": "All set: flights, the flat and three dinners.\nNothing open."}
    last = {"time": "2026-11-23T02:05:00Z", "assistant_messages": [wrap_up], "phone": [], "done_ends_run": False,
            "you_sent": [{"at": "2026-11-23T02:00:00Z", "text": "send me the rundown"}], "last_read": True}
    text = us.render(PERSON, [{"read": earlier, "answer": {"outcome": "reply", "message": "ok!"}}], last)
    assert text == (
        "It's Sun, Nov 22, 6:05 PM.\n"
        "You last messaged your AI assistant at 6:00 PM.\n"
        "\nYOUR EARLIER EMAILS, TEXTS AND NOTIFICATIONS (you've already seen these):\n  [Tue 8:00 AM · Gmail · Carol Reyes (Mom)] "
        "Thursday ✈️\n"
        "\nYOU SENT YOUR AI ASSISTANT:\n  [6:00 PM] send me the rundown\n"
        "\nNEW FROM YOUR AI ASSISTANT (this is what you're answering):\n  [6:01 PM]\n"
        "    All set: flights, the flat and three dinners.\n    Nothing open.\n"
        "\nYour AI assistant sent you this. As Hannah, decide: reply, don't reply, or say you're done.\n"
        "Answer in JSON: " + us.ANSWER_LINE)
    got = us.parse({"outcome": "done", "message": "you're the best 🙌", "used": "none"})
    assert (got["outcome"], got["message"], got["schema_errors"]) == ("done", "you're the best 🙌", [])


def test_earlier_on_your_phone_keeps_only_the_last_week():
    old = {"at": "2026-11-09T16:00:00Z", "channel": "Gmail · Priya Shah", "text": "Re: the flat"}
    week = {"at": "2026-11-11T16:00:00Z", "channel": "Gmail · Carol Reyes (Mom)", "text": "Thursday ✈️"}
    history = [{"read": _read("2026-11-11T17:05:00Z", phone=[old, week]), "answer": NOTHING}]
    text = us.render(PERSON, history, _read())
    assert "YOUR EARLIER EMAILS, TEXTS AND NOTIFICATIONS (you've already seen these):\n" \
           "  [Wed Nov 11, 8:00 AM · Gmail · Carol Reyes (Mom)] Thursday ✈️\n" + us.EARLIER_CUT + "\n" in text
    assert "Re: the flat" not in text


def test_an_earlier_item_without_a_time_keeps_the_time_of_the_read_it_came_in():
    undated = {"channel": "Gmail · Priya Shah", "text": "Re: the flat"}
    history = [{"read": _read("2026-11-09T16:00:00Z", phone=[undated]), "answer": NOTHING},
               {"read": _read("2026-11-16T17:05:00Z", phone=[{**undated, "text": "Keys on Friday"}],
                              you_sent=[{"kind": "user_says", "text": "can you check the flat?"}]),
                "answer": NOTHING}]
    text = us.render(PERSON, history, _read())
    assert "  [Mon 9:05 AM · Gmail · Priya Shah] Keys on Friday\n" + us.EARLIER_CUT + "\n" in text
    assert "Re: the flat" not in text
    assert "You last messaged your AI assistant on Mon at 9:05 AM." in text


def test_in_a_long_run_each_read_stays_bounded():
    history, longest, sizes = [], 0, []
    for k in range(80):  # two reads a day for 40 days, two phone items each
        at = f"2026-11-{1 + k // 2:02d}T{14 + 6 * (k % 2)}:00:00Z" if k < 60 else \
            f"2026-12-{1 + (k - 60) // 2:02d}T{14 + 6 * (k % 2)}:00:00Z"
        phone = [{"at": at, "channel": "Gmail · Carol Reyes (Mom)", "text": f"Day {k}: the plan\nmore below"},
                 {"at": at, "channel": "Tablefern", "text": f"Table {k} confirmed"}]
        read = _read(at, phone=phone, assistant_messages=[{"at": at, "text": f"Update {k}."}])
        text = us.render(PERSON, history, read)
        section = text.split("YOUR EARLIER EMAILS, TEXTS AND NOTIFICATIONS (you've already seen these):\n")[1].split("\n\n")[0] if k else ""
        lines = [line for line in section.splitlines() if line != us.EARLIER_CUT]
        longest = max(longest, len(lines))
        assert len(lines) <= us.EARLIER_MAX
        assert (us.EARLIER_CUT in section) == (2 * k > us.EARLIER_MAX)
        sizes.append(len(text))
        history.append({"read": read, "answer": NOTHING})
    assert longest == us.EARLIER_MAX
    assert max(sizes[20:]) - min(sizes[20:]) < 100


def test_a_time_without_its_offset_is_refused():
    with pytest.raises(ValueError, match="no UTC offset"):
        us.render(PERSON, [], _read("2026-11-17 09:05"))


# ---- the answer -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("raw, outcome, message", [
    ({"outcome": "reply", "message": "yes book it", "used": "D3"}, "reply", "yes book it"),
    ({"outcome": "no_reply", "message": "", "used": "H2"}, "no_reply", ""),
    ({"outcome": "done", "message": "we're all set!! 🙌"}, "done", "we're all set!! 🙌"),
    ({"outcome": "reply", "message": "  "}, "no_reply", "  "),
    ('{"outcome": "no_reply", "message": ""}', "no_reply", ""),
    ('```json\n{"outcome": "reply", "message": "ok"}\n```', "reply", "ok"),
    ('Thinking it over.\n{"outcome": "reply", "message": "ok"}', "reply", "ok"),
])
def test_an_answer_parses_to_what_the_step_records(raw, outcome, message):
    got = us.parse(raw)
    assert (got["outcome"], got["message"], got["schema_errors"]) == (outcome, message, [])


def test_used_is_optional_and_a_score_is_kept():
    assert us.parse({"outcome": "reply", "message": "ok"})["used"] is None
    assert "used" not in us.ANSWER_SCHEMA["required"]
    assert us.parse({"outcome": "done", "message": "thanks!", "score": 4})["score"] == 4


@pytest.mark.parametrize("raw, error", [
    ("sure, book it", "not a JSON object"),
    (None, "not a JSON object"),
    ({"message": "ok"}, "missing outcome"),
    ({"outcome": "maybe", "message": "ok"}, "outcome 'maybe' is not one of"),
    ({"outcome": "reply", "message": "ok", "mood": "fine"}, "unknown field mood"),
    ({"outcome": "reply", "message": 5}, "message must be text"),
    ({"outcome": "done", "message": "ok", "score": "4"}, "score must be a whole number"),
])
def test_a_malformed_answer_sends_nothing_and_says_why(raw, error):
    got = us.parse(raw)
    assert (got["outcome"], got["message"]) == ("no_reply", "")
    assert any(error in e for e in got["schema_errors"]), got["schema_errors"]


# ---- the plug-in ----------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_each_read_is_asked_as_the_replay_asked_it():
    run = RUNS["pvF"]
    sent = []

    async def ask(text):
        sent.append(text)
        got = run["answers"][len(sent) - 1]
        return {"text": json.dumps(got), "structured": got, "trajectory_url": None}

    history = []
    for read in run["reads"][:-1]:
        got = await us.answer("ours", copy.deepcopy(PERSON), history, read, ask)
        history.append({"read": read, "answer": got})
    assert sent == run["expected"][:-1]
    assert [h["answer"]["outcome"] for h in history] == [a["outcome"] for a in run["answers"]]
    assert {"outcome", "message", "used", "input", "raw", "schema_errors"} <= set(history[0]["answer"])


def _asker(*answers):
    sent = []

    async def ask(text):
        sent.append(text)
        got = answers[len(sent) - 1]
        return {"text": got if isinstance(got, str) else json.dumps(got),
                "structured": got if isinstance(got, dict) else None, "trajectory_url": None}

    return ask, sent


@pytest.mark.asyncio
async def test_a_malformed_answer_gets_one_retry_and_the_valid_answer_is_used():
    ask, sent = _asker("sure, the blue one", {"outcome": "reply", "message": "the blue one", "used": "D1"})
    got = await us.answer("ours", PERSON, [], _read(), ask)
    assert sent[1] == us.RETRY and sent[1].endswith("only as JSON: " + us.ANSWER_LINE)
    assert (got["outcome"], got["message"], got["used"], got["schema_errors"]) == ("reply", "the blue one", "D1", [])
    assert got["retried"] is True and got["turns"] == 2 and got["input"] == sent[0]
    assert got["first_answer"] == {"raw": "sure, the blue one", "schema_errors": ["the answer is not a JSON object"]}


@pytest.mark.asyncio
async def test_malformed_twice_is_no_reply_with_both_errors():
    ask, sent = _asker("sure", {"outcome": "maybe", "message": "ok"})
    got = await us.answer("ours", PERSON, [], _read(), ask)
    assert len(sent) == 2 and (got["outcome"], got["message"], got["retried"]) == ("no_reply", "", True)
    assert got["schema_errors"][0] == "first answer: the answer is not a JSON object"
    assert got["schema_errors"][1].startswith("answer after the retry: outcome 'maybe' is not one of")


@pytest.mark.asyncio
async def test_an_empty_reply_and_a_valid_answer_get_no_retry():
    for answer in ({"outcome": "reply", "message": ""}, {"outcome": "reply", "message": "ok"}):
        ask, sent = _asker(answer)
        got = await us.answer("ours", PERSON, [], _read(), ask)
        assert len(sent) == 1 and got["retried"] is False and got["schema_errors"] == []
    assert got["outcome"] == "reply"


@pytest.mark.asyncio
async def test_she_isn_t_asked_when_nothing_is_new_from_the_assistant():
    async def ask(text):
        raise AssertionError("asked")

    got = await us.answer("ours", PERSON, [], _read(assistant_messages=[]), ask)
    assert (got["outcome"], got["message"], got["input"]) == ("no_reply", "", None)


@pytest.mark.asyncio
async def test_when_the_user_sim_can_t_be_reached_nothing_is_sent():
    async def ask(text):
        return {"text": None, "structured": None, "error": "timeout"}

    got = await us.answer("ours", PERSON, [], _read(), ask)
    assert got["outcome"] == "no_reply" and got["schema_errors"] and got["retried"] is False
