"""An app whose server sees it end, a website browser's, launched again when a signal or a failing code ended it, up to
``CRASH_RELAUNCHES`` times a boot; not when it exited cleanly, when its server stopped it, or when it doesn't ask."""
from __future__ import annotations

import asyncio
import signal
from types import SimpleNamespace

import pytest

from agentenv_protocol.computer import Launch, app_runtime, session
from agentenv_protocol.computer.app_runtime import CRASH_RELAUNCHES, AppRuntime


@pytest.fixture
def desktop(tmp_path, monkeypatch):
    """The app's folder as the current directory, and a session whose launches each get a new pid that runs until
    ``ended`` gives it its code; the pids launched, in order."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENVIRONMENT_NAME", "browser")
    monkeypatch.setattr(app_runtime, "LAUNCHED", tmp_path / "launched")
    state = SimpleNamespace(launched=[], running=set(), codes={}, stopped=[])

    async def launch(argv, cwd, env, fds=None):
        pid = 100 + len(state.launched)
        state.launched.append(pid)
        state.running.add(pid)
        return pid

    async def returncode(pid, timeout_s):
        return state.codes.get(pid)

    async def stop(pid, timeout_s):
        state.stopped.append(pid)
        state.running.discard(pid)

    monkeypatch.setattr(session, "launch", launch)
    monkeypatch.setattr(session, "stop", stop)
    monkeypatch.setattr(session, "alive", lambda pid: pid in state.running)
    monkeypatch.setattr(session, "returncode", returncode)
    return state


def _runtime(relaunch_on_crash: bool = True) -> AppRuntime:
    return AppRuntime(Launch(at="boot", command=["chrome"]), {}, 30, desktop=False, relaunch_on_crash=relaunch_on_crash)


async def _ended(runtime: AppRuntime, desktop, code: int) -> bool:
    """The app last launched ended with ``code``, as its server sees it end; whether it ended on its own."""
    pid = desktop.launched[-1]
    desktop.running.discard(pid)
    desktop.codes[pid] = code
    on_its_own = runtime.exited()
    await asyncio.gather(*runtime.exits)
    return on_its_own


@pytest.mark.asyncio
async def test_a_browser_a_signal_ends_is_launched_again_and_says_why(desktop):
    runtime = _runtime()
    await runtime.launch_app("boot")

    assert await _ended(runtime, desktop, -signal.SIGKILL)

    assert desktop.launched == [100, 101] and desktop.stopped == [100] and runtime.running()
    assert [(entry["event"], entry["pid"], entry.get("cause") or entry.get("signal")) for entry in runtime.launches()] == [
        ("launch", 100, "boot"), ("exit", 100, "SIGKILL"), ("launch", 101, "crash")]


@pytest.mark.asyncio
async def test_what_a_crash_left_running_is_stopped_so_the_app_is_launched_afresh_not_reached_through_it(desktop):
    runtime, reached = _runtime(), []

    async def open_window():
        reached.append("window")

    runtime.launch_running = open_window
    await runtime.launch_app("boot")
    desktop.codes[100] = -signal.SIGKILL
    assert runtime.exited()
    await asyncio.gather(*runtime.exits)

    assert (desktop.stopped, desktop.launched, reached) == ([100], [100, 101], [])


@pytest.mark.asyncio
async def test_a_failing_code_is_a_crash_and_a_clean_exit_is_a_close(desktop):
    runtime = _runtime()
    await runtime.launch_app("boot")

    await _ended(runtime, desktop, 1)
    await _ended(runtime, desktop, 0)

    assert desktop.launched == [100, 101] and not runtime.running()


@pytest.mark.asyncio
async def test_one_its_server_stopped_or_that_doesnt_ask_stays_down(desktop):
    stopped, unasked = _runtime(), _runtime(relaunch_on_crash=False)
    await stopped.launch_app("boot")
    await stopped.stop_app()

    assert not stopped.exited()
    await unasked.launch_app("boot")
    await _ended(unasked, desktop, -signal.SIGTRAP)

    assert desktop.launched == [100, 101] and not unasked.running()


@pytest.mark.asyncio
async def test_it_is_launched_again_at_most_crash_relaunches_times_a_boot(desktop):
    runtime = _runtime()
    await runtime.launch_app("boot")

    for _ in range(CRASH_RELAUNCHES + 2):
        await _ended(runtime, desktop, -signal.SIGSEGV)

    assert len(desktop.launched) == 1 + CRASH_RELAUNCHES and not runtime.running()
    assert runtime.launches()[-1]["event"] == "crash_relaunches_spent"


@pytest.mark.asyncio
async def test_a_relaunch_after_a_crash_waits_for_the_data_call_that_holds_the_turn(desktop):
    runtime = _runtime()
    await runtime.launch_app("boot")
    await runtime.turn.acquire()
    desktop.running.discard(100)
    desktop.codes[100] = -signal.SIGKILL

    assert runtime.exited()
    await asyncio.sleep(0.05)
    during = list(desktop.launched)
    runtime.turn.release()
    await asyncio.gather(*runtime.exits)

    assert (during, desktop.launched) == ([100], [100, 101])


@pytest.mark.asyncio
async def test_one_launched_again_since_its_crash_isnt_launched_once_more_for_it(desktop):
    runtime = _runtime()
    await runtime.launch_app("boot")
    await runtime.turn.acquire()
    desktop.running.discard(100)
    desktop.codes[100] = -signal.SIGKILL

    assert runtime.exited()
    await runtime.launch_app("data/reset")
    runtime.turn.release()
    await asyncio.gather(*runtime.exits)

    assert (desktop.launched, desktop.stopped) == ([100, 101], [])
