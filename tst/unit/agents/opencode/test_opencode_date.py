"""The env's date in OpenCode's model calls (opencode_date): on an armed clock the turn's relay puts the env's date, as
a ``date`` started by the A2A server reads it, where the CLI, on real time, put its own, and refuses a call whose date
it can't read or place; off the clock, or with nothing to rewrite, a call goes as the CLI wrote it."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time

import httpx
import pytest
from aiohttp import web

from tst.unit.agents.opencode.agent import model_relay, opencode_date
from tst.util import fake_date

# OpenCode's system prompt as its SystemPrompt.environment writes it (opencode-ai 1.17.18, src/session/system.ts).
ENV_BLOCK = ("Here is some useful information about the environment you are running in:\n<env>\n"
             "  Working directory: /app\n  Workspace root folder: /app\n  Is directory a git repo: no\n"
             "  Platform: linux\n  Today's date: Sat Nov 21 2026\n</env>")
SYSTEM = (f"You are OpenCode, the best coding agent on the planet.\n{ENV_BLOCK}\n"
          "The task's notes say: Today's date: as written")
SEARCH = "Search the web.\n\nThe current year is 2026. You MUST use this year when searching for recent information."


def _call(system=SYSTEM, *, parts=False, tools=True, user=SYSTEM) -> dict:
    content = [{"type": "text", "text": "header"}, {"type": "text", "text": system}] if parts else system
    call = {"model": "m", "stream": True, "messages": [{"role": "system", "content": content},
                                                       {"role": "user", "content": user}]}
    if tools:
        call["tools"] = [{"type": "function", "function": {"name": "websearch", "description": SEARCH}},
                         {"type": "function", "function": {"name": "bash", "description": "The current year is 2026."}}]
    return call


@pytest.fixture
def env_clock(tmp_path, monkeypatch):
    """An armed clock, in UTC, whose time the stand-in ``date`` prints (tst/util/fake_date.py)."""
    recorded = fake_date.install(tmp_path, monkeypatch)
    monkeypatch.setattr(opencode_date, "clock_armed", lambda: True)
    yield recorded
    monkeypatch.undo()
    time.tzset()


def _rewrite(call: dict) -> dict:
    return json.loads(asyncio.run(opencode_date.env_date_rewrite(json.dumps(call).encode())))


@pytest.mark.parametrize("date_out, today, year", [
    ("1801742400 000000000", "Thu Feb 04 2027", 2027),
    ("7258161600 250000000", "Wed Jan 01 2200", 2200),
    ("-631108800 000000000", "Sun Jan 01 1950", 1950),
    # Half a second before 1950: date prints the second below and a positive fraction.
    ("-631152001 500000000", "Sat Dec 31 1949", 1949),
])
@pytest.mark.parametrize("parts", [False, True])
def test_on_an_armed_clock_the_model_is_told_the_env_date_a_command_reads(env_clock, monkeypatch, date_out, today,
                                                                         year, parts):
    monkeypatch.setenv("FAKE_DATE_OUT", date_out)
    call = _rewrite(_call(parts=parts))

    system = call["messages"][0]["content"][1]["text"] if parts else call["messages"][0]["content"]
    assert system == SYSTEM.replace("Today's date: Sat Nov 21 2026", f"Today's date: {today}")
    assert call["messages"][1]["content"] == SYSTEM
    assert call["tools"][0]["function"]["description"] == SEARCH.replace("2026", str(year))
    assert call["tools"][1]["function"]["description"] == "The current year is 2026."
    assert "AGENTENV_REAL_TIME" not in env_clock.read_text() and "TZ=UTC" in env_clock.read_text()


def test_every_env_blocks_date_line_takes_the_env_date_so_the_clis_does_wherever_it_comes(env_clock, monkeypatch):
    """A config's agent prompt, which comes before the CLI's block, or its instructions, after it, may hold an <env>
    block of their own."""
    monkeypatch.setenv("FAKE_DATE_OUT", "1801742400 000000000")
    prompt = "You are careful.\n<env>\n  Platform: linux\n  Today's date: Mon Jan 01 2001\n</env>\n"
    written = prompt + SYSTEM + "\n" + prompt

    system = _rewrite(_call(written))["messages"][0]["content"]

    assert system == written.replace("Mon Jan 01 2001", "Thu Feb 04 2027").replace("Sat Nov 21 2026", "Thu Feb 04 2027")


def test_a_lone_surrogate_in_a_tool_an_mcp_server_sent_goes_as_it_came(env_clock, monkeypatch):
    """The CLI sends it escaped, as a server's tool list had it; UTF-8 can't encode it unescaped."""
    monkeypatch.setenv("FAKE_DATE_OUT", "1801742400 000000000")
    call = _call()
    call["tools"].append({"type": "function", "function": {"name": "env_cut", "description": "cut \ud83d"}})

    assert _rewrite(call)["tools"][2]["function"]["description"] == "cut \ud83d"


@pytest.mark.usefixtures("socket_enabled")
def test_the_relay_sends_the_endpoint_the_call_with_the_env_date(env_clock, monkeypatch):
    monkeypatch.setenv("FAKE_DATE_OUT", "1801742400 000000000")
    received = []

    async def endpoint(request):
        received.append((request.headers["Content-Length"], await request.read()))
        return web.json_response({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    async def run():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", endpoint)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        upstream = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        try:
            async with model_relay.model_relay(upstream, model_relay.Calls(),
                                               rewrite=opencode_date.env_date_rewrite) as root:
                async with httpx.AsyncClient(timeout=10) as client:
                    return (await client.post(f"{root}/v1/chat/completions", content=json.dumps(_call()))).status_code
        finally:
            await runner.cleanup()

    assert asyncio.run(run()) == 200
    ((length, body),) = received
    assert int(length) == len(body) and json.loads(body)["stream"] is True
    assert json.loads(body)["messages"][0]["content"] == SYSTEM.replace("Sat Nov 21 2026", "Thu Feb 04 2027")


def test_the_date_is_written_in_the_zone_the_cli_writes_its_own_in(env_clock, monkeypatch):
    monkeypatch.setenv("FAKE_DATE_OUT", "1801785600 000000000")  # 2027-02-05T00:00Z, still the 4th in New York
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()

    assert "Today's date: Thu Feb 04 2027" in _rewrite(_call())["messages"][0]["content"]


@pytest.mark.parametrize("armed", [False, True])
def test_a_call_with_nothing_to_rewrite_or_off_the_clock_goes_as_written_and_reads_no_date(env_clock, monkeypatch,
                                                                                          armed):
    monkeypatch.setenv("FAKE_DATE_OUT", "1801742400 000000000")
    monkeypatch.setattr(opencode_date, "clock_armed", lambda: armed)
    calls = [json.dumps(_call(), indent=1).encode(), b"not json"] if not armed else []
    calls += [json.dumps(_call("A title for this conversation", tools=False, user="hi")).encode()]

    for body in calls:
        assert asyncio.run(opencode_date.env_date_rewrite(body)) == body
    assert not env_clock.exists()


def test_on_an_armed_clock_a_call_that_isnt_json_is_refused_rather_than_sent_with_the_real_date(env_clock):
    with pytest.raises(opencode_date.Refusal, match="isn't JSON"):
        asyncio.run(opencode_date.env_date_rewrite(json.dumps(_call()).encode()[:-1]))


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
    monkeypatch.setattr(opencode_date, "_DATE_TIMEOUT_S", 2)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    started = time.monotonic()

    with pytest.raises(opencode_date.Refusal) as refused:
        _rewrite(_call())

    assert str(refused.value).startswith(f"the env's date couldn't be read for the model: {reason}")
    assert time.monotonic() - started < 6


def test_a_date_that_exits_leaving_a_process_on_its_output_is_refused_and_that_process_killed(env_clock, monkeypatch,
                                                                                              tmp_path):
    """The date has exited, so only its session's group still reaches what it left holding the pipes. The timeout
    leaves a loaded machine time to start the stand-in and its child."""
    monkeypatch.setattr(opencode_date, "_DATE_TIMEOUT_S", 3)
    monkeypatch.setenv("FAKE_DATE_OUT", "1801742400 000000000")
    monkeypatch.setenv("FAKE_DATE_KID", str(tmp_path / "kid"))

    with pytest.raises(opencode_date.Refusal, match="date gave no answer within 3 s"):
        _rewrite(_call())

    kid, deadline = (tmp_path / "kid").read_text().strip(), time.monotonic() + 5
    while (state := subprocess.run(["ps", "-o", "stat=", "-p", kid], capture_output=True, text=True).stdout.strip()) \
            and not state.startswith("Z") and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not state or state.startswith("Z"), state


def test_an_env_block_without_a_date_line_this_knows_refuses_the_call(env_clock, monkeypatch):
    monkeypatch.setenv("FAKE_DATE_OUT", "1801742400 000000000")

    with pytest.raises(opencode_date.Refusal, match="no date line"):
        _rewrite(_call(SYSTEM.replace("  Today's date:", "  Date:")))
