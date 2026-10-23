"""The CLI's stream-json events reach `event_callback` as it prints them, and are exactly the
events of the turn's trajectory.

Run from this directory: pytest test_gemini_bridge_live.py -q
"""
import asyncio
import json
import uuid

import httpx
import pytest

from tst.unit.agents.gemini.agent import a2a_server, gemini_bridge
from tst.unit.agents.gemini.fake_cli import FakeCLI
from tst.unit.agents.gemini.support import send

_INIT = {"type": "init", "session_id": "s-1", "model": "gemini-3"}
_TOOL = {"type": "tool_use", "tool_name": "list", "tool_id": "t-1", "parameters": {}}
_SAID = {"type": "message", "role": "assistant", "content": "done", "delta": True}
_DONE = {"type": "result", "status": "success", "stats": {"tool_calls": 1}}


def _lines(*events, ensure_ascii=True):
    return "".join(json.dumps(e, ensure_ascii=ensure_ascii) + "\n" for e in events).encode()


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """Queue the `gemini` processes run_gemini starts, each made when it starts."""
    queue: list = []

    async def _exec(*cmd, **kwargs):
        return queue.pop(0)()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gemini_bridge, "GEMINI_PROJECTS_JSON", tmp_path / "projects.json")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)
    return queue


def _run(**kwargs):
    reported: list[dict] = []
    result = asyncio.run(
        gemini_bridge.run_gemini(prompt="go", session_id="s-1", event_callback=reported.append, **kwargs)
    )
    return result, reported


def test_a_turn_reports_its_events_in_order_and_they_are_its_trajectory(cli):
    cli.append(lambda: FakeCLI(_lines(_INIT, _TOOL, _SAID, _DONE)))

    result, reported = _run()

    assert reported == result.trajectory == [_INIT, _TOOL, _SAID, _DONE]
    assert (result.response, result.tool_call_count) == ("done", 1)


def test_a_failed_turn_reports_the_events_it_printed(cli):
    failed = {"type": "result", "status": "error", "error": {"message": "quota"}}
    cli.append(lambda: FakeCLI(_lines(_INIT, failed), returncode=1))

    result, reported = _run()

    assert result.error is not None
    assert reported == result.trajectory == [_INIT, failed]


def test_events_are_reported_while_the_cli_runs(cli):
    started: list[FakeCLI] = []
    cli.append(lambda: started.append(FakeCLI(_lines(_INIT), hang=True)) or started[0])

    async def turn():
        reported: list[dict] = []
        running = asyncio.create_task(
            gemini_bridge.run_gemini(prompt="go", session_id="s-1", event_callback=reported.append)
        )
        async with asyncio.timeout(5):
            while not reported:
                await asyncio.sleep(0.01)
        seen_while_running = list(reported)
        proc = started[0]
        proc.stdout.feed_data(_lines(_SAID, _DONE))
        proc.stdout.feed_eof()
        proc.stderr.feed_eof()
        await running
        return seen_while_running, reported

    seen_while_running, reported = asyncio.run(turn())

    assert seen_while_running == [_INIT]
    assert reported == [_INIT, _SAID, _DONE]


def test_lines_split_exactly_as_the_whole_output_did(cli):
    """`splitlines` also breaks on U+2028, which the CLI leaves raw in JSON strings: such an event
    never parsed, and must not start to, or the trajectory would change."""
    separated = {"type": "message", "role": "assistant", "content": "a b"}
    stdout = _lines(_INIT, separated, _DONE, ensure_ascii=False) + b"\r\n"
    cli.append(lambda: FakeCLI(stdout))

    result, reported = _run()

    assert reported == result.trajectory == [_INIT, _DONE]


def test_a_line_longer_than_a_stream_buffer_is_one_event(cli):
    big = {"type": "tool_result", "tool_id": "t-1", "output": "x" * 300_000}
    cli.append(lambda: FakeCLI(_lines(_INIT, big, _DONE)))

    result, reported = _run()

    assert reported == result.trajectory == [_INIT, big, _DONE]


def test_a_json_line_that_is_not_an_object_is_skipped(cli):
    cli.append(lambda: FakeCLI(_lines(_INIT) + b'"progress"\n' + _lines(_DONE)))

    result, reported = _run()

    assert result.error is None
    assert reported == result.trajectory == [_INIT, _DONE]


def test_a_callback_that_raises_kills_the_cli(cli):
    started: list[FakeCLI] = []
    cli.append(lambda: started.append(FakeCLI(_lines(_INIT), hang=True)) or started[0])

    def refuse(event):
        raise ValueError("rejected")

    with pytest.raises(ValueError, match="rejected"):
        asyncio.run(gemini_bridge.run_gemini(prompt="go", session_id="s-1", event_callback=refuse))

    assert started[0].killed


def test_the_finished_task_is_readable_by_cursor(monkeypatch, client):
    events = [_INIT, _SAID, _DONE]

    async def _run_gemini(**kwargs):
        for event in events:
            kwargs["event_callback"](event)
        return gemini_bridge.GeminiResult(response="done", tool_call_count=0, session_id="s-1", trajectory=events)

    monkeypatch.setattr(a2a_server, "run_gemini", _run_gemini)
    task = send(client, "go")

    read = client.post(a2a_server.TRAJECTORY_V1.endpoint, json={"task_id": task["id"], "after": 0}).json()

    assert (read["format"], read["state"], read["events"]) == (a2a_server.TRAJECTORY_FORMAT, "completed", events)


def _blocked_cli(cli) -> list[FakeCLI]:
    """The next `gemini` the bridge starts prints `_INIT`, then blocks until fed or killed."""
    started: list[FakeCLI] = []
    cli.append(lambda: started.append(FakeCLI(_lines(_INIT), hang=True)) or started[-1])
    return started


async def _send(client: httpx.AsyncClient, context_id: str) -> str:
    message = {
        "kind": "message", "role": "user", "messageId": uuid.uuid4().hex, "contextId": context_id,
        "parts": [{"kind": "text", "text": "go"}],
    }
    sent = await client.post("/a2a", json={
        "jsonrpc": "2.0", "id": 1, "method": "message/send",
        "params": {"message": message, "configuration": {"blocking": False}},
    })
    return sent.json()["result"]["id"]


async def _read_until(client: httpx.AsyncClient, request: dict, done) -> dict:
    async with asyncio.timeout(5):
        while not done(read := (await client.post(a2a_server.TRAJECTORY_V1.endpoint, json=request)).json()):
            await asyncio.sleep(0.01)
    return read


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gemini")


def test_a_running_turn_is_readable_by_task_and_context_and_its_reads_join_to_its_trajectory(cli, app):
    """Through the real bridge: the CLI prints one event, then blocks until released."""
    started = _blocked_cli(cli)

    async def exercise():
        async with _client(app) as client:
            task_id = await _send(client, "ctx-running")
            running = await _read_until(client, {"task_id": task_id, "after": 0}, lambda read: read["next"] >= 1)
            by_context = await _read_until(client, {"context_id": "ctx-running", "after": 0}, lambda read: True)
            started[0].stdout.feed_data(_lines(_TOOL, _SAID, _DONE))
            started[0].stdout.feed_eof()
            started[0].stderr.feed_eof()
            ended = await _read_until(
                client, {"task_id": task_id, "after": running["next"]}, lambda read: read["state"] == "completed"
            )
            final = (await client.post(a2a_server.TRAJECTORY_V1.endpoint, json={"task_id": task_id})).json()
            return running, by_context, ended, final

    running, by_context, ended, final = asyncio.run(exercise())

    assert (running["state"], running["format"], running["events"]) == ("running", a2a_server.TRAJECTORY_FORMAT, [_INIT])
    assert (by_context["state"], by_context["events"]) == ("running", [_INIT])
    assert running["events"] + ended["events"] == final["trajectory"] == [_INIT, _TOOL, _SAID, _DONE]


def test_a_canceled_turn_keeps_the_events_it_printed(cli, app):
    started = _blocked_cli(cli)

    async def exercise():
        async with _client(app) as client:
            task_id = await _send(client, "ctx-canceled")
            await _read_until(client, {"task_id": task_id, "after": 0}, lambda read: read["next"] >= 1)
            await client.post("/a2a", json={"jsonrpc": "2.0", "id": 2, "method": "tasks/cancel", "params": {"id": task_id}})
            canceled = await _read_until(client, {"task_id": task_id, "after": 0}, lambda read: read["state"] == "canceled")
            final = (await client.post(a2a_server.TRAJECTORY_V1.endpoint, json={"task_id": task_id})).json()
            return canceled, final

    canceled, final = asyncio.run(exercise())

    assert started[0].killed
    assert canceled["events"] == final["trajectory"] == [_INIT]


def test_a_turn_whose_cli_fails_keeps_the_events_it_printed(cli, app):
    failed = {"type": "result", "status": "error", "error": {"message": "quota"}}
    cli.append(lambda: FakeCLI(_lines(_INIT, failed), returncode=1))

    async def exercise():
        async with _client(app) as client:
            task_id = await _send(client, "ctx-failed")
            ended = await _read_until(client, {"task_id": task_id, "after": 0}, lambda read: read["state"] == "failed")
            final = (await client.post(a2a_server.TRAJECTORY_V1.endpoint, json={"task_id": task_id})).json()
            return ended, final

    ended, final = asyncio.run(exercise())

    assert ended["events"] == final["trajectory"] == [_INIT, failed]


def test_a_timed_out_turn_keeps_the_events_it_printed(cli):
    """The callback feeds the framework's log, which is the turn's final trajectory."""
    started = _blocked_cli(cli)

    result, reported = _run(timeout_s=1)

    assert started[0].killed
    assert result.error is not None
    assert reported == [_INIT]
