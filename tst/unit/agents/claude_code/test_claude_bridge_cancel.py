"""A turn the A2A client cancels kills the claude CLI and every tool it started, not only the CLI.

"""
import asyncio
import contextlib
import sys
import re
import json
import os
import shutil
import signal
import subprocess
import time

import pytest
from tst.unit.agents.claude_code.agent import AGENT_DIR, claude_bridge



def _state(pid: int) -> str:
    """The process's state letter, from /proc where there is one, else ps; empty once it is gone."""
    if os.path.isdir("/proc"):
        try:
            with open(f"/proc/{pid}/stat") as stat:
                return stat.read().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            return ""
    return subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()[:1]


def _gone(pid: int, within: float = 5.0) -> bool:
    deadline = time.monotonic() + within
    while _state(pid) not in ("", "Z"):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


@pytest.mark.parametrize("tool", [
    "sleep 600",
    pytest.param("setsid sleep 600", marks=pytest.mark.skipif(
        not (shutil.which("setsid") and os.path.isdir("/proc")), reason="needs setsid and /proc (Linux)")),
], ids=["in-its-group", "in-a-session-of-its-own"])
def test_a_cancelled_turn_kills_the_cli_and_the_tools_it_started(monkeypatch, tmp_path, tool):
    pids = tmp_path / "pids"
    spawn = asyncio.create_subprocess_exec

    async def cli_that_starts_a_tool(*_cmd, **kwargs):
        return await spawn("sh", "-c", f'{tool} & echo "$$ $!" > {pids}; wait', **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", cli_that_starts_a_tool)

    async def cancel_mid_turn():
        turn = asyncio.create_task(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))
        deadline = time.monotonic() + 10
        while not (pids.exists() and len(pids.read_text().split()) == 2):
            if turn.done() or time.monotonic() > deadline:
                pytest.fail(f"the CLI never started: {turn.result() if turn.done() else 'still waiting after 10 s'}")
            await asyncio.sleep(0.01)
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):  # a TimeoutError means a tool held the CLI's pipes open
            await asyncio.wait_for(turn, 10)
        return [int(pid) for pid in pids.read_text().split()]

    cli, tool = asyncio.run(cancel_mid_turn())

    assert _gone(cli) and _gone(tool)


@pytest.mark.skipif(not (shutil.which("setsid") and os.path.isdir("/proc")), reason="needs setsid and /proc (Linux)")
def test_a_process_that_left_the_clis_tree_does_not_hold_up_the_cancel(monkeypatch, tmp_path):
    pids, escaped = tmp_path / "pids", tmp_path / "escaped"
    spawn = asyncio.create_subprocess_exec

    async def cli_whose_tool_left_its_tree(*_cmd, **kwargs):
        script = f"sh -c 'setsid sleep 600 & echo $! > {escaped}'; echo $$ > {pids}; exec sleep 600"
        return await spawn("sh", "-c", script, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", cli_whose_tool_left_its_tree)

    async def cancel_while_it_holds_the_pipes():
        turn = asyncio.create_task(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))
        while not (pids.exists() and pids.read_text().strip() and escaped.exists() and escaped.read_text().strip()):
            await asyncio.sleep(0.01)
        tool = int(escaped.read_text())
        try:
            turn.cancel()
            with pytest.raises(asyncio.CancelledError):  # still holding the tool's pipes would be a TimeoutError
                await asyncio.wait_for(turn, 10)
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.kill(tool, signal.SIGKILL)

    asyncio.run(cancel_while_it_holds_the_pipes())


def test_a_turn_cancelled_while_the_cli_starts_kills_it_once_it_has(monkeypatch, tmp_path):
    """A cancel can arrive before the CLI's pipes are set up, when the bridge has no process to kill yet."""
    pids = tmp_path / "pids"
    spawn = asyncio.create_subprocess_exec
    starting = asyncio.Event()

    async def slow_to_start(*_cmd, **kwargs):
        proc = await spawn("sh", "-c", f'sleep 600 & echo "$$ $!" > {pids}; wait', **kwargs)
        while not (pids.exists() and len(pids.read_text().split()) == 2):  # under load the shell is slow to write
            await asyncio.sleep(0.01)
        starting.set()
        await asyncio.sleep(0.5)  # still starting when the cancel arrives
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", slow_to_start)

    async def cancel_while_starting():
        turn = asyncio.create_task(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))
        await asyncio.wait_for(starting.wait(), 10)
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(turn, 10)
        deadline = time.monotonic() + 10
        while not (pids.exists() and len(pids.read_text().split()) == 2):
            assert time.monotonic() < deadline, "the CLI never wrote its pids"
            await asyncio.sleep(0.01)
        return [int(pid) for pid in pids.read_text().split()]

    cli, tool = asyncio.run(cancel_while_starting())

    assert _gone(cli) and _gone(tool)


def test_a_turn_cancelled_again_while_it_scrubs_its_transcript_ends_only_once_the_scrub_has(monkeypatch, tmp_path):
    scrubbed = []
    ticks: list[int] = []
    real_scrub = claude_bridge.scrub_session_transcript

    def slow_scrub(session_id):
        before = len(ticks)
        time.sleep(0.3)
        real_scrub(session_id)
        scrubbed.append((session_id, len(ticks) - before))

    spawn = asyncio.create_subprocess_exec

    async def cli_that_hangs(*_cmd, **kwargs):
        return await spawn("sleep", "600", **kwargs)

    monkeypatch.setattr(claude_bridge, "SESSION_JSONL_DIR", tmp_path)
    monkeypatch.setattr(claude_bridge, "scrub_session_transcript", slow_scrub)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", cli_that_hangs)

    async def cancel_twice():
        async def tick():
            while True:
                ticks.append(1)
                await asyncio.sleep(0.01)

        ticker = asyncio.create_task(tick())
        turn = asyncio.create_task(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))
        await asyncio.sleep(0.2)
        turn.cancel()
        await asyncio.sleep(0.1)
        turn.cancel()  # mid-scrub, as a caller giving up on a turn that is slow to end does
        with pytest.raises(asyncio.CancelledError):
            await turn
        ticker.cancel()
        # The turn is over, so its context can take another: no scrub may be left to replace that turn's transcript.
        return list(scrubbed)

    ((session, ticks_while_scrubbing),) = asyncio.run(cancel_twice())

    assert session == "sess-1" and ticks_while_scrubbing >= 5  # off the event loop, which ran meanwhile


@pytest.mark.parametrize("tool", [
    "sleep 600",
    pytest.param("setsid sleep 600", marks=pytest.mark.skipif(
        not (shutil.which("setsid") and os.path.isdir("/proc")), reason="needs setsid and /proc (Linux)")),
], ids=["in-its-group", "in-a-session-of-its-own"])
def test_stopping_the_agents_server_kills_the_clis_of_turns_still_running_and_their_tools(tmp_path, tool):
    """A reinstall stops the agent's server mid-turn: its CLI, in a session of its own, mustn't go on after it. Its
    server runs as an install runs it, under uvicorn, which an in-process client's shutdown doesn't stand for."""
    pids, bin_dir, log = tmp_path / "pids", tmp_path / "bin", tmp_path / "server.log"
    bin_dir.mkdir()
    (bin_dir / "claude").write_text(f'#!/bin/sh\n{tool} > /dev/null 2>&1 < /dev/null & echo "$$ $!" > {pids}\nwait\n')
    (bin_dir / "claude").chmod(0o755)
    (tmp_path / "app").mkdir()
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "PYTHONPATH": str(AGENT_DIR),
           "CLAUDE_CONFIG_DIR": str(tmp_path / "state"), "AGENT_WORKSPACE_DIR": str(tmp_path / "app")}
    with log.open("w") as out:
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "a2a_server:app", "--host", "127.0.0.1",
                                   "--port", "0"], cwd=tmp_path / "app", env=env, stdout=out, stderr=subprocess.STDOUT)
    started: list[int] = []
    try:
        deadline = time.monotonic() + 30
        while not (port := re.search(r"running on http://127\.0\.0\.1:(\d+)", log.read_text())):
            assert time.monotonic() < deadline and server.poll() is None, log.read_text()
            time.sleep(0.05)
        body = json.dumps({"jsonrpc": "2.0", "id": "1", "method": "message/send", "params": {
            "configuration": {"blocking": False},
            "message": {"messageId": "m-1", "role": "user", "parts": [{"kind": "text", "text": "go"}]}}})
        subprocess.run(["curl", "-fsS", "-o", "/dev/null", "-H", "content-type: application/json", "-d", body,
                        f"http://127.0.0.1:{port.group(1)}/a2a"], check=True, timeout=30)
        while not (pids.exists() and len(pids.read_text().split()) == 2):
            assert time.monotonic() < deadline, log.read_text()
            time.sleep(0.05)
        started = [int(pid) for pid in pids.read_text().split()]

        server.send_signal(signal.SIGTERM)
        server.wait(timeout=30)

        assert all(_gone(pid) for pid in started), log.read_text()
    finally:
        server.kill()
        for pid in started:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
