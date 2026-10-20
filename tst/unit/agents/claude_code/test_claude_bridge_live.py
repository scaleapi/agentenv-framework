"""The CLI's stdout events reach `event_callback` as it prints them, and are exactly the events of the
turn's trajectory: an attempt that is resumed never reports its `result`.

"""
import asyncio
import json

import pytest

from tst.unit.agents.claude_code.support import FakeCLI
from tst.unit.agents.claude_code.agent import claude_bridge

_INIT = {"type": "system", "subtype": "init", "session_id": "sess-1"}
_TOOL = {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {}}]}}
_DONE = {"type": "result", "subtype": "success", "result": "done", "session_id": "sess-1"}
_STALLED = {
    "type": "result", "subtype": "success", "is_error": True, "session_id": "sess-1",
    "terminal_reason": "api_error", "result": "API Error: The operation timed out.",
}


def _lines(*events, ensure_ascii=True):
    return "".join(json.dumps(e, ensure_ascii=ensure_ascii) + "\n" for e in events).encode()


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """Queue the `claude -p` processes run_claude starts, in order, each made when it starts."""
    queue: list = []

    async def _exec(*cmd, **kwargs):
        return queue.pop(0)()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)
    monkeypatch.setattr(claude_bridge, "SESSION_JSONL_DIR", tmp_path)
    monkeypatch.setattr(claude_bridge, "_RETRY_BACKOFF_S", (0, 0))
    monkeypatch.delenv("CLAUDE_API_ERROR_RETRIES", raising=False)
    (tmp_path / "sess-1.jsonl").write_text('{"type":"assistant"}\n')
    return queue


def _run(**kwargs):
    reported: list[dict] = []
    result = asyncio.run(
        claude_bridge.run_claude(
            prompt="p", session_id="sess-1", resume=False, event_callback=reported.append, **kwargs
        )
    )
    return result, reported


def test_a_turn_reports_its_events_in_order_and_they_are_its_trajectory(cli):
    cli.append(lambda: FakeCLI(_lines(_INIT, _TOOL, _DONE)))

    result, reported = _run()

    assert reported == result.trajectory == [_INIT, _TOOL, _DONE]
    assert result.response == "done"


def test_a_resumed_attempt_never_reports_its_result(cli):
    cli.append(lambda: FakeCLI(_lines(_INIT, _TOOL, _STALLED), returncode=1))
    cli.append(lambda: FakeCLI(_lines(_INIT, _DONE)))

    result, reported = _run()

    assert result.attempts == 2
    assert reported == result.trajectory == [_INIT, _TOOL, _INIT, _DONE]


def test_an_attempt_that_is_not_resumed_reports_its_result(cli, monkeypatch):
    monkeypatch.setenv("CLAUDE_API_ERROR_RETRIES", "0")
    cli.append(lambda: FakeCLI(_lines(_INIT, _STALLED), returncode=1))

    result, reported = _run()

    assert result.error is not None
    assert reported == result.trajectory == [_INIT, _STALLED]


def test_events_are_reported_while_the_cli_runs(cli):
    started: list[FakeCLI] = []
    cli.append(lambda: started.append(FakeCLI(_lines(_INIT), hang=True)) or started[0])

    async def turn():
        reported: list[dict] = []
        running = asyncio.create_task(
            claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False, event_callback=reported.append)
        )
        async with asyncio.timeout(5):
            while not reported:
                await asyncio.sleep(0.01)
        seen_while_running = list(reported)
        proc = started[0]
        proc.stdout.feed_data(_lines(_TOOL, _DONE))
        proc.stdout.feed_eof()
        proc.stderr.feed_eof()
        await running
        return seen_while_running, reported

    seen_while_running, reported = asyncio.run(turn())

    assert seen_while_running == [_INIT]
    assert reported == [_INIT, _TOOL, _DONE]


def test_lines_split_exactly_as_the_whole_output_did(cli):
    """`splitlines` also breaks on U+2028, which the CLI leaves raw in JSON strings: such an event never
    parsed, and must not start to, or the trajectory would change."""
    separated = {"type": "assistant", "text": "a b"}
    stdout = _lines(_INIT, separated, _DONE, ensure_ascii=False) + b"\r\n"
    expected = [json.loads(line) for line in stdout.decode(errors="replace").splitlines() if _parses(line)]
    cli.append(lambda: FakeCLI(stdout))

    result, reported = _run()

    assert reported == result.trajectory == expected == [_INIT, _DONE]


def test_a_line_longer_than_a_stream_buffer_is_one_event(cli):
    big = {"type": "user", "message": {"content": [{"type": "tool_result", "content": "x" * 300_000}]}}
    cli.append(lambda: FakeCLI(_lines(_INIT, big, _DONE)))

    result, reported = _run()

    assert reported == result.trajectory == [_INIT, big, _DONE]


def test_a_json_line_that_is_not_an_object_is_skipped(cli):
    cli.append(lambda: FakeCLI(_lines(_INIT) + b"[1, 2]\n" + _lines(_DONE)))

    result, reported = _run()

    assert reported == result.trajectory == [_INIT, _DONE]


def _parses(line: str) -> bool:
    try:
        return isinstance(json.loads(line), dict)
    except json.JSONDecodeError:
        return False


def test_a_callback_that_raises_kills_the_cli(cli):
    started: list[FakeCLI] = []
    cli.append(lambda: started.append(FakeCLI(_lines(_INIT), hang=True)) or started[0])

    def refuse(event):
        raise ValueError("rejected")

    with pytest.raises(ValueError, match="rejected"):
        asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False, event_callback=refuse))

    assert started[0].killed
