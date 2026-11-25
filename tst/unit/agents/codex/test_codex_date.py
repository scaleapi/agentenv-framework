"""The env's date in Codex's model calls (codex_date): on an armed clock each CLI run's relay puts the env's date, as a
``date`` started by the A2A server reads it, where the CLI, a static binary on real time, put its own, in every
environment context of the call, and the env's time in its time reminders; it refuses a call whose date it can't read
or place. Off the clock, or with nothing to rewrite, a call goes as the CLI wrote it."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time

import httpx
import pytest
from aiohttp import web

from tst.unit.agents.codex.agent import codex_date, model_relay
from tst.util import fake_date

# The environment context as Codex 0.162.1 writes it (codex-rs core/src/context/world_state/environment.rs).
CONTEXT = ("<environment_context>\n  <cwd>/app</cwd>\n  <shell>bash</shell>\n  <current_date>2026-11-25</current_date>\n"
           "  <timezone>Etc/UTC</timezone>\n</environment_context>")
LATER = "<environment_context>\n  <current_date>2026-11-26</current_date>\n</environment_context>"
REMINDER = "<current_time_reminder>It is 2026-11-25 17:34:15 UTC.</current_time_reminder>"
PROMPT = "What's the date? The notes say <current_date>as written</current_date>."
OUTPUT = "<environment_context><current_date>2026-11-25</current_date></environment_context>"


def _message(role: str, *texts: str, parts: bool = True) -> dict:
    content = [{"type": "input_text", "text": text} for text in texts] if parts else texts[0]
    return {"type": "message", "role": role, "content": content}


def _call(*, parts: bool = True, later: bool = True, reminder: bool = True) -> dict:
    items = [_message("developer", "<permissions instructions>sandbox</permissions instructions>"),
             _message("user", "# AGENTS.md instructions for /app", CONTEXT, parts=parts) if parts
             else _message("user", CONTEXT, parts=False),
             _message("user", PROMPT),
             {"type": "function_call", "name": "exec_command", "arguments": json.dumps({"cmd": "date"}),
              "call_id": "c1"},
             {"type": "function_call_output", "call_id": "c1", "output": OUTPUT}]
    if later:
        items.append(_message("user", LATER))
    if reminder:
        items.append(_message("developer", REMINDER))
    return {"model": "gpt-x", "instructions": "You are Codex.", "input": items, "tools": [], "stream": True}


@pytest.fixture
def env_clock(tmp_path, monkeypatch):
    """An armed clock, in UTC, whose time the stand-in ``date`` prints (tst/util/fake_date.py)."""
    recorded = fake_date.install(tmp_path, monkeypatch)
    monkeypatch.setattr(codex_date, "clock_armed", lambda: True)
    yield recorded
    monkeypatch.undo()
    time.tzset()


def _rewrite(call: dict) -> dict:
    return json.loads(asyncio.run(codex_date.env_date_rewrite(json.dumps(call).encode())))


def _texts(call: dict) -> list[str]:
    return [part["text"] if isinstance(part, dict) else part for item in call["input"] if item["type"] == "message"
            for part in (item["content"] if isinstance(item["content"], list) else [item["content"]])]


@pytest.mark.parametrize("date_out, today, now", [
    ("1801742400 000000000", "2027-02-04", "2027-02-04 12:00:00"),
    ("7258161600 250000000", "2200-01-01", "2200-01-01 12:00:00"),
    ("-631108800 000000000", "1950-01-01", "1950-01-01 12:00:00"),
    # Half a second before 1950: date prints the second below and a positive fraction.
    ("-631152001 500000000", "1949-12-31", "1949-12-31 23:59:59"),
])
@pytest.mark.parametrize("parts", [True, False])
def test_on_an_armed_clock_the_model_is_told_the_env_date_and_time_a_command_reads(env_clock, monkeypatch, date_out,
                                                                                  today, now, parts):
    monkeypatch.setenv("FAKE_DATE_OUT", date_out)
    sent = _call(parts=parts)

    call = _rewrite(sent)

    context = CONTEXT.replace("2026-11-25", today)
    assert _texts(call) == [t.replace(CONTEXT, context).replace(LATER, LATER.replace("2026-11-26", today))
                            .replace(REMINDER, f"<current_time_reminder>It is {now} UTC.</current_time_reminder>")
                            for t in _texts(sent)]
    assert call["input"][3:5] == sent["input"][3:5]  # the model's call and what the command printed go as they were
    assert {k: v for k, v in call.items() if k != "input"} == {k: v for k, v in sent.items() if k != "input"}
    assert "AGENTENV_REAL_TIME" not in env_clock.read_text() and "TZ=UTC" in env_clock.read_text()


def test_a_date_outside_an_environment_context_goes_as_written(env_clock, monkeypatch):
    """A prompt's own words, which name a date element of their own, aren't the CLI's date."""
    monkeypatch.setenv("FAKE_DATE_OUT", "1801742400 000000000")

    assert PROMPT in _texts(_rewrite(_call()))


@pytest.mark.parametrize("quoted", [
    f"Explain this: {CONTEXT}",
    "The log said <current_time_reminder>Now: 17:34</current_time_reminder>, which is odd.",
    f"{REMINDER} was in the transcript.",
])
def test_a_prompt_quoting_the_clis_date_or_reminder_goes_as_written_and_isnt_refused(env_clock, monkeypatch, quoted):
    """The CLI writes its environment context and reminder each as a text of its own, between its markers."""
    monkeypatch.setenv("FAKE_DATE_OUT", "1801742400 000000000")
    call = _call()
    call["input"].append(_message("user", quoted))

    assert _texts(_rewrite(call))[-1] == quoted


def test_a_date_the_cli_marked_unavailable_stays_so(env_clock, monkeypatch):
    monkeypatch.setenv("FAKE_DATE_OUT", "1801742400 000000000")
    gone = '<environment_context>\n  <current_date status="unavailable" />\n</environment_context>'
    call = _call()
    call["input"].append(_message("user", gone))

    assert gone in _texts(_rewrite(call))


def test_the_date_is_written_in_the_zone_the_cli_writes_its_own_in(env_clock, monkeypatch):
    monkeypatch.setenv("FAKE_DATE_OUT", "1801785600 000000000")  # 2027-02-05T00:00Z, still the 4th in New York
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()

    texts = _texts(_rewrite(_call()))

    assert "<current_date>2027-02-04</current_date>" in texts[2]
    assert "<current_time_reminder>It is 2027-02-05 00:00:00 UTC.</current_time_reminder>" in texts
    assert "<timezone>Etc/UTC</timezone>" in texts[2]  # the zone the CLI named, which isn't time's to change


def test_a_lone_surrogate_in_a_tool_an_mcp_server_sent_goes_as_it_came(env_clock, monkeypatch):
    """The CLI sends it escaped, as a server's tool list had it; UTF-8 can't encode it unescaped."""
    monkeypatch.setenv("FAKE_DATE_OUT", "1801742400 000000000")
    call = _call()
    call["tools"].append({"type": "function", "name": "env_cut", "description": "cut \ud83d"})

    assert _rewrite(call)["tools"][0]["description"] == "cut \ud83d"


@pytest.mark.usefixtures("socket_enabled")
def test_the_relay_sends_the_endpoint_the_call_with_the_env_date(env_clock, monkeypatch):
    monkeypatch.setenv("FAKE_DATE_OUT", "1801742400 000000000")
    received = []

    async def endpoint(request):
        received.append((request.headers["Content-Length"], await request.read()))
        return web.json_response({"output": [], "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}})

    async def run():
        app = web.Application()
        app.router.add_post("/openai/v1/responses", endpoint)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        upstream = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        try:
            async with model_relay.model_relay(upstream, model_relay.Calls(),
                                               rewrite=codex_date.env_date_rewrite) as root:
                async with httpx.AsyncClient(timeout=10) as client:
                    return (await client.post(f"{root}/openai/v1/responses", content=json.dumps(_call()))).status_code
        finally:
            await runner.cleanup()

    assert asyncio.run(run()) == 200
    ((length, body),) = received
    assert int(length) == len(body) and json.loads(body)["stream"] is True
    assert CONTEXT.replace("2026-11-25", "2027-02-04") in _texts(json.loads(body))


@pytest.mark.parametrize("armed", [False, True])
def test_a_call_with_nothing_to_rewrite_or_off_the_clock_goes_as_written_and_reads_no_date(env_clock, monkeypatch,
                                                                                          armed):
    monkeypatch.setenv("FAKE_DATE_OUT", "1801742400 000000000")
    monkeypatch.setattr(codex_date, "clock_armed", lambda: armed)
    calls = [json.dumps(_call(), indent=1).encode(), b"not json"] if not armed else []
    nothing = {"model": "gpt-x", "input": [_message("user", "hi"),
                                           {"type": "function_call_output", "call_id": "c", "output": OUTPUT}]}
    calls += [json.dumps(nothing, indent=1).encode()]

    for body in calls:
        assert asyncio.run(codex_date.env_date_rewrite(body)) == body
    assert not env_clock.exists()


def test_on_an_armed_clock_a_call_that_isnt_json_is_refused_rather_than_sent_with_the_real_date(env_clock):
    with pytest.raises(codex_date.Refusal, match="isn't JSON"):
        asyncio.run(codex_date.env_date_rewrite(json.dumps(_call()).encode()[:-1]))


@pytest.mark.parametrize("text, what", [
    ('<environment_context>\n  <current_date tz="UTC">2026-11-25</current_date>\n</environment_context>', "a date"),
    ("<environment_context>\n  <current_date>2026-11-25\n</environment_context>", "a date"),
    ("<current_time_reminder>It is 2026-11-25T17:34:15Z.</current_time_reminder>", "a time"),
    ("<current_time_reminder>Now: 17:34</current_time_reminder>", "a time"),
])
def test_a_date_or_time_written_in_a_form_this_doesnt_know_refuses_the_call(env_clock, monkeypatch, text, what):
    monkeypatch.setenv("FAKE_DATE_OUT", "1801742400 000000000")
    call = _call()
    call["input"].append(_message("user", text))

    with pytest.raises(codex_date.Refusal, match=f"has {what} this agent doesn't know how to place"):
        _rewrite(call)
    assert not env_clock.exists()


@pytest.mark.parametrize("environ, reason", [
    ({}, "date exited 1: no clock"),
    ({"FAKE_DATE_OUT": "1801742400 000000000", "FAKE_DATE_EXIT": "3"}, "date exited 3"),
    ({"FAKE_DATE_OUT": "soon"}, "date printed 'soon\\n'"),
    ({"FAKE_DATE_OUT": "1 0", "FAKE_DATE_SLEEP": "10"}, "date gave no answer within 2 s"),
])
def test_a_date_that_cant_be_read_refuses_the_call_rather_than_tell_the_model_real_time(env_clock, monkeypatch,
                                                                                         environ, reason):
    """A date that hangs is killed with the sleep it started, which holds its output open, so the refusal comes in
    its time."""
    monkeypatch.setattr(codex_date, "_DATE_TIMEOUT_S", 2)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    started = time.monotonic()

    with pytest.raises(codex_date.Refusal) as refused:
        _rewrite(_call())

    assert str(refused.value).startswith(f"the env's date couldn't be read for the model: {reason}")
    assert time.monotonic() - started < 6


def test_a_date_that_exits_leaving_a_process_on_its_output_is_refused_and_that_process_killed(env_clock, monkeypatch,
                                                                                              tmp_path):
    """The date has exited, so only its session's group still reaches what it left holding the pipes. The timeout
    leaves a loaded machine time to start the stand-in and its child."""
    monkeypatch.setattr(codex_date, "_DATE_TIMEOUT_S", 3)
    monkeypatch.setenv("FAKE_DATE_OUT", "1801742400 000000000")
    monkeypatch.setenv("FAKE_DATE_KID", str(tmp_path / "kid"))

    with pytest.raises(codex_date.Refusal, match="date gave no answer within 3 s"):
        _rewrite(_call())

    kid, deadline = (tmp_path / "kid").read_text().strip(), time.monotonic() + 5
    while (state := subprocess.run(["ps", "-o", "stat=", "-p", kid], capture_output=True, text=True).stdout.strip()) \
            and not state.startswith("Z") and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not state or state.startswith("Z"), state
