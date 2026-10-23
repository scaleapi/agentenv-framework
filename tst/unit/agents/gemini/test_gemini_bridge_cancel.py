"""A turn the A2A client cancels kills the gemini CLI and every tool it started, not only the CLI.

Run from this directory: pytest test_gemini_bridge_cancel.py -q
"""
import asyncio
import contextlib
import os
import shutil
import signal
import subprocess
import time

import pytest

from tst.unit.agents.gemini.agent import gemini_bridge


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
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gemini_bridge, "GEMINI_PROJECTS_JSON", tmp_path / "projects.json")

    async def cancel_mid_turn():
        turn = asyncio.create_task(gemini_bridge.run_gemini(prompt="p", session_id=None))
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
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gemini_bridge, "GEMINI_PROJECTS_JSON", tmp_path / "projects.json")

    async def cancel_while_it_holds_the_pipes():
        turn = asyncio.create_task(gemini_bridge.run_gemini(prompt="p", session_id=None))
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
