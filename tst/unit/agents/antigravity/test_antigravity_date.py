"""The env's time in Antigravity's model calls (antigravity_date): on an armed clock the turn's relay puts the env's
time, as a ``date`` started by the A2A server reads it, in the local-time line the harness, on real time, opens each
user message with, keeps the time each turn's line was given for the turns a resumed conversation sends again, and
refuses a call whose time it can't read; off the clock, or with no line to rewrite, a call goes as the harness wrote
it."""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from tst.unit.agents.antigravity.agent import antigravity_date, model_relay

REAL = "2026-11-25T10:00:00-08:00"
EARLIER = "2026-11-25T09:00:00-08:00"


def _line(stamp: str) -> str:
    return f"<user_information>\nThe current local time is: {stamp}. This is the latest source of truth.\n"


def _call(*users, parts: bool = False) -> dict:
    messages = [{"role": "system", "content": "You are Antigravity."}]
    for text in users:
        messages += [{"role": "user", "content": [{"type": "text", "text": text}] if parts else text},
                     {"role": "assistant", "content": "ok"}]
    return {"model": "m", "stream": True, "messages": messages[:-1]}


@pytest.fixture
def times(tmp_path) -> antigravity_date.Records:
    return antigravity_date.Records(tmp_path / "conversations" / "env-times.json")


@pytest.fixture
def env_clock(tmp_path, monkeypatch):
    """An armed clock whose time a stand-in ``date`` prints (``FAKE_DATE_OUT``; it fails without one), recording the
    environment it got and each time it ran."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "date").write_text('#!/bin/sh\nenv > "$FAKE_DATE_ENV"\necho run >> "$FAKE_DATE_RUNS"\n'
                                  '[ "$1" = "+%Y-%m-%dT%H:%M:%S%:z" ] || exit 2\n'
                                  '[ -n "$FAKE_DATE_SLEEP" ] && sleep "$FAKE_DATE_SLEEP"\n'
                                  '[ -n "$FAKE_DATE_OUT" ] || { echo no clock >&2; exit 1; }\necho "$FAKE_DATE_OUT"\n')
    (bin_dir / "date").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DATE_ENV", str(tmp_path / "date.env"))
    monkeypatch.setenv("FAKE_DATE_RUNS", str(tmp_path / "date.runs"))
    monkeypatch.setenv("AGENTENV_REAL_TIME", "1")
    monkeypatch.setattr(antigravity_date, "clock_armed", lambda: True)
    return tmp_path


def _rewrite(times, call: dict) -> dict:
    return json.loads(asyncio.run(antigravity_date.env_time_rewrite(times)(json.dumps(call).encode())))


def _runs(env_clock) -> int:
    runs = env_clock / "date.runs"
    return len(runs.read_text().splitlines()) if runs.exists() else 0


@pytest.mark.parametrize("parts", [False, True])
@pytest.mark.parametrize("date_out, shown", [
    ("2031-03-03T23:06:07-06:00", "2031-03-03T23:06:07-06:00"),
    ("2031-03-04T05:06:07+00:00", "2031-03-04T05:06:07Z"),
    ("1950-01-01T00:00:00+05:30", "1950-01-01T00:00:00+05:30"),
])
def test_on_an_armed_clock_the_model_is_told_the_env_time_a_command_reads(env_clock, monkeypatch, times, parts,
                                                                          date_out, shown):
    monkeypatch.setenv("FAKE_DATE_OUT", date_out)

    call = _rewrite(times, _call(_line(REAL) + "go", parts=parts))

    user = call["messages"][1]["content"]
    assert (user[0]["text"] if parts else user) == _line(shown) + "go"
    assert call["messages"][0]["content"] == "You are Antigravity."
    assert list(times.all().values()) == [shown]


def test_date_runs_on_the_envs_clock_not_the_servers(env_clock, monkeypatch, times):
    monkeypatch.setenv("FAKE_DATE_OUT", "2031-03-03T23:06:07-06:00")

    _rewrite(times, _call(_line(REAL) + "go"))

    assert "AGENTENV_REAL_TIME=" not in (env_clock / "date.env").read_text()


def test_a_turns_line_keeps_the_env_time_it_was_first_given_for_the_rest_of_the_turn(env_clock, monkeypatch, times):
    rewrite = antigravity_date.env_time_rewrite(times)
    call = json.dumps(_call(_line(REAL) + "go")).encode()
    monkeypatch.setenv("FAKE_DATE_OUT", "2031-03-03T23:06:07-06:00")
    first = asyncio.run(rewrite(call))
    monkeypatch.setenv("FAKE_DATE_OUT", "2031-03-03T23:09:00-06:00")

    again = asyncio.run(rewrite(call))

    assert again == first
    assert _runs(env_clock) == 1


def test_a_turn_whose_message_matches_an_earlier_ones_to_the_second_gets_the_time_now(env_clock, monkeypatch, times):
    monkeypatch.setenv("FAKE_DATE_OUT", "2031-03-03T23:06:07-06:00")
    _rewrite(times, _call(_line(REAL) + "go"))
    monkeypatch.setenv("FAKE_DATE_OUT", "2031-03-04T02:00:00-06:00")

    again = _rewrite(times, _call(_line(REAL) + "go"))

    assert again["messages"][1]["content"] == _line("2031-03-04T02:00:00-06:00") + "go"


def test_a_resumed_conversations_earlier_turns_keep_their_times_and_the_new_one_gets_now(env_clock, monkeypatch,
                                                                                        times):
    times.put(antigravity_date.line_key(_line(EARLIER) + "one", EARLIER), "2031-03-02T08:00:00-06:00")
    monkeypatch.setenv("FAKE_DATE_OUT", "2031-03-03T23:06:07-06:00")

    call = _rewrite(times, _call(_line(EARLIER) + "one", _line(REAL) + "two"))

    assert [call["messages"][i]["content"] for i in (1, 3)] == [_line("2031-03-02T08:00:00-06:00") + "one",
                                                                _line("2031-03-03T23:06:07-06:00") + "two"]


def test_two_turns_that_start_in_one_second_keep_their_own_env_times(env_clock, monkeypatch, times):
    monkeypatch.setenv("FAKE_DATE_OUT", "2031-03-03T23:06:07-06:00")
    _rewrite(times, _call(_line(REAL) + "one"))
    monkeypatch.setenv("FAKE_DATE_OUT", "2031-03-08T09:00:00-06:00")

    call = _rewrite(times, _call(_line(REAL) + "one", _line(REAL) + "two"))

    assert [call["messages"][i]["content"] for i in (1, 3)] == [_line("2031-03-03T23:06:07-06:00") + "one",
                                                                _line("2031-03-08T09:00:00-06:00") + "two"]


def _ran(created: str, completed: str, output: str = "42") -> str:
    return f"Created At: {created}\nCompleted At: {completed}\n\nOutput:\n{output}"


def _with_run(call: dict, result: str, after: int, call_id: str = "c1") -> dict:
    """``call`` with a command's call ``call_id`` and its ``result`` after its message ``after``."""
    messages = call["messages"]
    messages[after + 1:after + 1] = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": call_id, "type": "function", "function": {
            "name": "run_command", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": call_id, "content": result}]
    return call


def test_a_commands_result_is_stamped_at_the_envs_offset_keeping_its_duration(env_clock, monkeypatch, times):
    monkeypatch.setenv("FAKE_DATE_OUT", "2031-03-03T23:06:07-06:00")
    monkeypatch.setattr(antigravity_date, "_real_now", lambda: "2026-11-25T18:08:35+00:00")
    result = _ran("2026-11-25T18:08:29Z", "2026-11-25T18:08:34Z")

    call = _rewrite(times, _with_run(_call(_line(REAL) + "go"), result, after=1))

    assert call["messages"][-1]["content"] == _ran("2031-03-03T23:06:01-06:00", "2031-03-03T23:06:06-06:00")
    assert call["messages"][1]["content"] == _line("2031-03-03T23:06:07-06:00") + "go"
    assert _runs(env_clock) == 1


def test_a_commands_result_keeps_its_stamps_in_later_turns(env_clock, monkeypatch, times):
    monkeypatch.setenv("FAKE_DATE_OUT", "2031-03-03T23:06:07-06:00")
    monkeypatch.setattr(antigravity_date, "_real_now", lambda: "2026-11-25T18:08:35+00:00")
    result = _ran("2026-11-25T18:08:29Z", "2026-11-25T18:08:34Z")
    first = _rewrite(times, _with_run(_call(_line(REAL) + "go"), result, after=1))
    monkeypatch.setenv("FAKE_DATE_OUT", "2031-03-09T00:00:00+00:00")
    monkeypatch.setattr(antigravity_date, "_real_now", lambda: "2026-11-25T18:20:00+00:00")
    later = _with_run(_call(_line(REAL) + "go", _line("2026-11-25T10:12:00-08:00") + "again"), result, after=1)

    again = _rewrite(times, later)

    assert again["messages"][3]["content"] == first["messages"][-1]["content"]
    assert again["messages"][-1]["content"] == _line("2031-03-09T00:00:00Z") + "again"


def test_a_commands_result_from_a_turn_off_the_clock_is_left_as_written(env_clock, monkeypatch, times):
    monkeypatch.setenv("FAKE_DATE_OUT", "2031-03-03T23:06:07-06:00")
    result = _ran("2026-11-25T18:08:29Z", "2026-11-25T18:08:34Z")

    call = _rewrite(times, _with_run(_call(_line(EARLIER) + "one", _line(REAL) + "two"), result, after=1))

    assert call["messages"][3]["content"] == result


def _sent(rewrite, call: dict) -> dict:
    return json.loads(asyncio.run(rewrite(json.dumps(call).encode())))


def test_a_command_result_on_the_date_the_model_was_told_gets_no_note(env_clock, monkeypatch, times):
    rewrite = antigravity_date.env_time_rewrite(times)
    monkeypatch.setenv("FAKE_DATE_OUT", "2031-03-03T23:06:07-06:00")
    _sent(rewrite, _call(_line(REAL) + "go"))
    monkeypatch.setenv("FAKE_DATE_OUT", "2031-03-03T23:59:00-06:00")

    ran = _ran("2026-11-25T18:08:29Z", "2026-11-25T18:08:34Z")

    call = _sent(rewrite, _with_run(_call(_line(REAL) + "go"), ran, after=1))

    assert "The date has changed" not in json.dumps(call)


def test_a_date_change_during_a_turn_is_noted_on_the_result_after_it_and_kept_there(env_clock, monkeypatch, times):
    rewrite = antigravity_date.env_time_rewrite(times)
    monkeypatch.setenv("FAKE_DATE_OUT", "2031-03-03T23:06:07-06:00")
    _sent(rewrite, _call(_line(REAL) + "go"))
    monkeypatch.setenv("FAKE_DATE_OUT", "2031-03-05T09:00:00-06:00")
    first = _ran("2026-11-25T18:08:29Z", "2026-11-25T18:08:34Z")
    second = _ran("2026-11-25T18:09:00Z", "2026-11-25T18:09:01Z", "7")
    note = antigravity_date.DATE_CHANGED.format("2031-03-05T09:00:00-06:00")

    noted = _sent(rewrite, _with_run(_call(_line(REAL) + "go"), first, after=1))
    monkeypatch.setenv("FAKE_DATE_OUT", "2031-03-05T09:01:00-06:00")
    then = _sent(rewrite, _with_run(_with_run(_call(_line(REAL) + "go"), first, after=1), second, after=3,
                                    call_id="c2"))
    monkeypatch.setenv("FAKE_DATE_OUT", "2031-03-06T08:00:00-06:00")
    later = _sent(antigravity_date.env_time_rewrite(times),
                  _with_run(_call(_line(REAL) + "go", _line("2026-11-25T10:20:00-08:00") + "again"), first, after=1))

    assert noted["messages"][-1]["content"].endswith(note)
    assert then["messages"][3]["content"].endswith(note)
    assert "The date has changed" not in then["messages"][5]["content"]
    assert later["messages"][3]["content"].endswith(note)
    assert later["messages"][3]["content"].count("date has changed") == 1
    assert later["messages"][-1]["content"] == _line("2031-03-06T08:00:00-06:00") + "again"


def test_a_result_with_an_earlier_ones_text_keeps_its_own_note_or_none(env_clock, monkeypatch, times):
    rewrite = antigravity_date.env_time_rewrite(times)
    same = _ran("2026-11-25T18:08:29Z", "2026-11-25T18:08:34Z")

    def results(count: int) -> dict:
        call = _call(_line(REAL) + "go")
        for n in range(count):
            _with_run(call, same, after=1 + 2 * n, call_id=f"c{n + 1}")
        return call

    monkeypatch.setenv("FAKE_DATE_OUT", "2031-03-03T23:06:07-06:00")
    _sent(rewrite, _call(_line(REAL) + "go"))
    monkeypatch.setenv("FAKE_DATE_OUT", "2031-03-05T09:00:00-06:00")
    _sent(rewrite, results(1))
    two = _sent(rewrite, results(2))
    monkeypatch.setenv("FAKE_DATE_OUT", "2031-03-07T10:00:00-06:00")
    three = _sent(rewrite, results(3))

    notes = [[message["content"].count("The date has changed") for message in call["messages"]
              if message["role"] == "tool"] for call in (two, three)]
    assert notes == [[1, 0], [1, 0, 1]]
    assert three["messages"][3]["content"].endswith(antigravity_date.DATE_CHANGED.format("2031-03-05T09:00:00-06:00"))
    assert three["messages"][7]["content"].endswith(antigravity_date.DATE_CHANGED.format("2031-03-07T10:00:00-06:00"))


def test_off_the_clock_no_note_is_added(monkeypatch, times):
    monkeypatch.setattr(antigravity_date, "clock_armed", lambda: False)
    call = _with_run(_call(_line(REAL) + "go"), _ran("2026-11-25T18:08:29Z", "2026-11-25T18:08:34Z"), after=1)

    assert _sent(antigravity_date.env_time_rewrite(times), call) == call


def test_an_earlier_turn_the_agent_followed_no_clock_for_is_left_as_written(env_clock, monkeypatch, times):
    monkeypatch.setenv("FAKE_DATE_OUT", "2031-03-03T23:06:07-06:00")

    call = _rewrite(times, _call(_line(EARLIER) + "one", _line(REAL) + "two"))

    assert call["messages"][1]["content"] == _line(EARLIER) + "one"


@pytest.mark.parametrize("environ, reason", [
    ({}, "date exited 1: no clock"),
    ({"FAKE_DATE_OUT": "Wed Mar  3 23:06:07 CST 2031"}, "date printed 'Wed Mar  3 23:06:07 CST 2031'"),
])
def test_a_call_whose_env_time_cant_be_read_is_refused(env_clock, monkeypatch, times, environ, reason):
    for name, value in environ.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(model_relay.Refusal, match=reason):
        _rewrite(times, _call(_line(REAL) + "go"))
    assert times.all() == {}


def test_a_date_that_hangs_is_killed_and_the_call_refused(env_clock, monkeypatch, times):
    monkeypatch.setenv("FAKE_DATE_OUT", "2031-03-03T23:06:07-06:00")
    monkeypatch.setenv("FAKE_DATE_SLEEP", "30")
    monkeypatch.setattr(antigravity_date, "_DATE_TIMEOUT_S", 0.3)

    with pytest.raises(model_relay.Refusal, match="no answer within"):
        _rewrite(times, _call(_line(REAL) + "go"))


def test_a_call_that_isnt_json_is_refused_on_an_armed_clock(env_clock, times):
    with pytest.raises(model_relay.Refusal, match="isn't JSON"):
        asyncio.run(antigravity_date.env_time_rewrite(times)(b"not json"))


@pytest.mark.parametrize("call", [
    _call("no local time here"),
    {"model": "m", "messages": [{"role": "system", "content": _line(REAL)}]},
    [1, 2],
])
def test_a_call_with_no_users_local_time_line_goes_as_written(env_clock, times, call):
    body = json.dumps(call, indent=1).encode()

    assert asyncio.run(antigravity_date.env_time_rewrite(times)(body)) == body
    assert _runs(env_clock) == 0


def test_off_the_clock_a_call_goes_as_the_harness_wrote_it(monkeypatch, times):
    monkeypatch.setattr(antigravity_date, "clock_armed", lambda: False)
    body = b"not even json " + _line(REAL).encode()

    assert asyncio.run(antigravity_date.env_time_rewrite(times)(body)) == body


def test_a_snapshots_times_join_the_agents_own_which_win(times):
    times.put(REAL, "own")

    times.merge({REAL: "snapshot's", EARLIER: "restored"})

    assert times.all() == {REAL: "own", EARLIER: "restored"}


def test_unreadable_times_read_as_none(times):
    times.path.parent.mkdir(parents=True)
    times.path.write_text("[1, 2]")
    assert times.all() == {}
    times.path.write_text("{torn")
    assert times.get(REAL) is None
