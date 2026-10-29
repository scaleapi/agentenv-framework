"""The CLI's JSON events reach `event_callback` as it prints them, and are exactly the events of the
turn's trajectory; the framework serves them while the task runs."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from agentenv_protocol.a2a_agent import card_request_accepts

from tst.unit.agents.opencode.agent import a2a_server as server
from tst.unit.agents.opencode.agent import opencode_bridge
from tst.unit.agents.opencode.fake_cli import FakeCLI

_START = {"type": "step_start", "sessionID": "ses_1", "part": {"type": "step-start"}}
_TOOL = {"type": "tool_use", "sessionID": "ses_1", "part": {"type": "tool", "callID": "c-1", "tool": "bash"}}
_TEXT = {"type": "text", "sessionID": "ses_1", "part": {"type": "text", "text": "done"}}
_STOP = {"type": "step_finish", "sessionID": "ses_1", "part": {"type": "step-finish", "reason": "stop"}}


def _lines(*events, ensure_ascii=True):
    return "".join(json.dumps(e, ensure_ascii=ensure_ascii) + "\n" for e in events).encode()


@pytest.fixture
def cli(monkeypatch):
    """Queue the `opencode` processes run_opencode starts, each made when it starts."""
    queue: list = []

    async def _exec(*cmd, **kwargs):
        return queue.pop(0)()

    monkeypatch.setattr(opencode_bridge.asyncio, "create_subprocess_exec", _exec)
    monkeypatch.setattr(opencode_bridge, "_context_xdg", lambda _context: {})
    return queue


def _run(**kwargs):
    reported: list[dict] = []
    result = asyncio.run(
        opencode_bridge.run_opencode(prompt="go", context_id="ctx", event_callback=reported.append, **kwargs)
    )
    return result, reported


def test_a_turn_reports_its_events_in_order_and_they_are_its_trajectory(cli):
    cli.append(lambda: FakeCLI(_lines(_START, _TOOL, _TEXT, _STOP)))

    result, reported = _run()

    assert reported == result.trajectory == [_START, _TOOL, _TEXT, _STOP]
    assert (result.response, result.tool_call_count, result.session_id) == ("done", 1, "ses_1")


def test_a_failed_turn_reports_the_events_it_printed(cli):
    error = {"type": "error", "error": {"name": "APIError", "data": {"message": "quota"}}}
    cli.append(lambda: FakeCLI(_lines(_START, error), returncode=1))

    result, reported = _run()

    assert result.error == "APIError: quota"
    assert reported == result.trajectory == [_START, error]


def test_events_are_reported_while_the_cli_runs(cli):
    started: list[FakeCLI] = []
    cli.append(lambda: started.append(FakeCLI(_lines(_START), hang=True)) or started[0])

    async def turn():
        reported: list[dict] = []
        running = asyncio.create_task(
            opencode_bridge.run_opencode(prompt="go", context_id="ctx", event_callback=reported.append)
        )
        async with asyncio.timeout(5):
            while not reported:
                await asyncio.sleep(0.01)
        seen_while_running = list(reported)
        proc = started[0]
        proc.stdout.feed_data(_lines(_TEXT, _STOP))
        proc.stdout.feed_eof()
        proc.stderr.feed_eof()
        await running
        return seen_while_running, reported

    seen_while_running, reported = asyncio.run(turn())

    assert seen_while_running == [_START]
    assert reported == [_START, _TEXT, _STOP]


def test_lines_split_exactly_as_the_whole_output_did(cli):
    """`splitlines` also breaks on U+2028, which the CLI leaves raw in JSON strings: such an event
    never parsed, and must not start to, or the trajectory would change."""
    separated = {"type": "text", "part": {"type": "text", "text": "a b"}}
    stdout = _lines(_START, separated, _STOP, ensure_ascii=False) + b"\r\n"
    cli.append(lambda: FakeCLI(stdout))

    result, reported = _run()

    assert reported == result.trajectory == [_START, _STOP]


def test_a_line_longer_than_a_stream_buffer_is_one_event(cli):
    big = {"type": "tool_use", "part": {"type": "tool", "callID": "c-2", "state": {"output": "x" * 300_000}}}
    cli.append(lambda: FakeCLI(_lines(_START, big, _STOP)))

    result, reported = _run()

    assert reported == result.trajectory == [_START, big, _STOP]


def test_a_json_line_that_is_not_an_object_is_skipped(cli):
    cli.append(lambda: FakeCLI(_lines(_START) + b"[1, 2]\n" + _lines(_STOP)))

    result, reported = _run()

    assert result.error is None
    assert reported == result.trajectory == [_START, _STOP]


def test_a_callback_that_raises_kills_the_cli(cli):
    started: list[FakeCLI] = []
    cli.append(lambda: started.append(FakeCLI(_lines(_START), hang=True)) or started[0])

    def refuse(event):
        raise ValueError("rejected")

    with pytest.raises(ValueError, match="rejected"):
        asyncio.run(opencode_bridge.run_opencode(prompt="go", context_id="ctx", event_callback=refuse))

    assert started[0].killed


def test_the_card_offers_the_live_reads(oc):
    for item in oc.app.state.agentenv_a2a.card.capabilities.extensions:
        if item.uri.endswith("trajectory/v1"):
            request = item.params["methods"]["get"]["request"]
    assert card_request_accepts(request, {"task_id", "after"})
    assert card_request_accepts(request, {"context_id", "after", "limit"})


@pytest.mark.asyncio
async def test_the_finished_task_is_readable_by_cursor(monkeypatch, oc):
    events = [_START, _TEXT, _STOP]

    async def fake_run(**kwargs):
        for event in events:
            kwargs["event_callback"](event)
        return SimpleNamespace(response="done", error=None, error_code=None,
                               session_id="ses_1", trajectory=events, tool_call_count=0)

    monkeypatch.setattr(server, "run_opencode", fake_run)
    message = {
        "jsonrpc": "2.0", "id": "r1", "method": "message/send",
        "params": {"message": {"kind": "message", "messageId": "m-live", "role": "user",
                              "contextId": "live-context", "parts": [{"kind": "text", "text": "go"}]},
                   "configuration": {"blocking": True}},
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=oc.app), base_url="http://test") as client:
        task = (await client.post("/a2a", json=message)).json()["result"]
        read = (await client.post("/ext/trajectory", json={"task_id": task["id"], "after": 0})).json()

    assert (read["format"], read["state"], read["events"]) == (server.TRAJECTORY_FORMAT, "completed", events)


def _blocked_cli(cli) -> list[FakeCLI]:
    """The next `opencode` the bridge starts prints `_START`, then blocks until fed or killed."""
    started: list[FakeCLI] = []
    cli.append(lambda: started.append(FakeCLI(_lines(_START), hang=True)) or started[-1])
    return started


async def _send(client: httpx.AsyncClient, context_id: str) -> str:
    message = {
        "jsonrpc": "2.0", "id": "r1", "method": "message/send",
        "params": {"message": {"kind": "message", "messageId": f"m-{context_id}", "role": "user",
                              "contextId": context_id, "parts": [{"kind": "text", "text": "go"}]},
                   "configuration": {"blocking": False}},
    }
    return (await client.post("/a2a", json=message)).json()["result"]["id"]


async def _read_until(client: httpx.AsyncClient, task_id: str, after: int, done) -> dict:
    async with asyncio.timeout(5):
        while not done(read := (await client.post("/ext/trajectory", json={"task_id": task_id, "after": after})).json()):
            await asyncio.sleep(0.01)
    return read


@pytest.mark.asyncio
async def test_a_running_turn_is_readable_by_cursor_and_its_reads_join_to_its_trajectory(cli, oc):
    """Through the real bridge: the CLI prints one event, then blocks until released."""
    started = _blocked_cli(cli)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=oc.app), base_url="http://test") as client:
        task_id = await _send(client, "live-running")
        running = await _read_until(client, task_id, 0, lambda read: read["next"] >= 1)
        started[0].stdout.feed_data(_lines(_TOOL, _TEXT, _STOP))
        started[0].stdout.feed_eof()
        started[0].stderr.feed_eof()
        ended = await _read_until(client, task_id, running["next"], lambda read: read["state"] == "completed")
        final = (await client.post("/ext/trajectory", json={"task_id": task_id})).json()

    assert (running["state"], running["format"], running["events"]) == ("running", server.TRAJECTORY_FORMAT, [_START])
    assert running["events"] + ended["events"] == final["trajectory"] == [_START, _TOOL, _TEXT, _STOP]


@pytest.mark.asyncio
async def test_a_canceled_turn_keeps_the_events_it_printed(cli, oc):
    started = _blocked_cli(cli)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=oc.app), base_url="http://test") as client:
        task_id = await _send(client, "live-canceled")
        await _read_until(client, task_id, 0, lambda read: read["next"] >= 1)
        await client.post("/a2a", json={"jsonrpc": "2.0", "id": "c1", "method": "tasks/cancel", "params": {"id": task_id}})
        canceled = await _read_until(client, task_id, 0, lambda read: read["state"] == "canceled")
        final = (await client.post("/ext/trajectory", json={"task_id": task_id})).json()

    assert started[0].killed
    assert canceled["events"] == final["trajectory"] == [_START]


def test_a_timed_out_turn_keeps_the_events_it_printed_and_the_session_it_started(cli):
    """The callback feeds the framework's log, which is the turn's final trajectory."""
    started = _blocked_cli(cli)

    result, reported = _run(timeout_s=1)

    assert started[0].killed
    assert "timed out" in result.error
    assert reported == [_START]
    assert result.session_id == "ses_1"
