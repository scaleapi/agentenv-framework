"""The Windows session's own logic, its Win32 calls faked and its job helper stood in for: a launch goes through the
helper and returns its pid once the app has started, or fails saying why it couldn't, and hands no fds; stopping asks
the app's windows to close, then ends the job after the timeout, or at once with no window to ask, and, once they've
closed, waits for the helper to leave, ending the job should it not, until each process it held has exited and it
holds none; pausing suspends each of the app's processes but the helper's, and
any one starts as they're suspended, and resumes them, even when the block raises; a job that's gone is neither alive,
stopped nor paused; the windows listed are those shown, by program, class and title; and the menu launcher is a Start
Menu shortcut PowerShell writes, named for the app."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from agentenv_protocol.computer import _reaping, _session_posix, _session_windows, session
from agentenv_protocol.computer._session_windows import Window

HELPER = 900
FAKE_HELPER = '''import sys, time
command = sys.argv[sys.argv.index("--") + 1:]
if command[0] == "missing":
    sys.stderr.write("Traceback (most recent call last):\\nFileNotFoundError: [WinError 2] missing\\n")
    sys.exit(2)
print("started", flush=True)
time.sleep(0.1)
sys.exit(int(command[0]))
'''


class FakeWin32:
    """Jobs by their helper's pid, each a list of the pids in it, the helper's first; the session's windows; and every
    call that acts, recorded. An app whose window is asked to close ``closes`` or doesn't, a process ``spawns`` one
    into its job just before it's suspended, a job ended by force holds its processes for ``lingers`` more looks, then
    none, though it's there until its handles close, and a helper alone in its job ``leaves`` after that many looks, as
    the real one does, or, for None, stays."""

    def __init__(self, jobs: dict[int, list[int]], windows: list[Window], closes: bool = True,
                 spawns: dict[int, int] | None = None, lingers: int = 0, leaves: int | None = 5) -> None:
        self.jobs, self._windows, self.closes, self.spawns, self.lingers = jobs, windows, closes, spawns or {}, lingers
        self.leaves = leaves
        self.ending: dict[int, int] = {}
        self.posted, self.terminated, self.suspended, self.resumed, self.waited, self.open = [], [], [], [], [], 0

    def open_job(self, helper):
        if helper not in self.jobs:
            return None
        self.open += 1
        return helper

    def job_pids(self, job):
        if job in self.ending:
            self.ending[job] -= 1
            if self.ending[job] < 0:
                self.jobs[job] = []
        elif self.jobs[job] == [job] and self.leaves is not None:
            self.leaves -= 1
            if self.leaves < 0:
                self.jobs[job] = []
        return list(self.jobs[job])

    def open_processes(self, pids):
        self.open += len(pids)
        return list(pids)

    def wait_exited(self, processes, timeout_s):
        self.waited.append(list(processes))

    def terminate_job(self, job):
        self.terminated.append(job)
        self.ending[job] = self.lingers

    def close(self, handle):
        self.open -= 1

    def suspend(self, pid):
        if pid in self.spawns:
            next(pids for pids in self.jobs.values() if pid in pids).append(self.spawns[pid])
        self.suspended.append(pid)
        return True

    def resume(self, pid):
        self.resumed.append(pid)

    def post_close(self, handle):
        self.posted.append(handle)
        if self.closes:
            window = next(window for window in self._windows if window.handle == handle)
            for pids in self.jobs.values():
                if window.pid in pids:
                    pids.remove(window.pid)

    def windows(self):
        return list(self._windows)


def _window(handle: int, pid: int, shown: bool = True, title: str = "Notes") -> Window:
    return Window(handle=handle, pid=pid, program="python.exe", class_name="TkTopLevel", title=title, shown=shown)


@pytest.fixture
def win32(monkeypatch):
    def install(*args, **kwargs) -> FakeWin32:
        fake = FakeWin32(*args, **kwargs)
        monkeypatch.setattr(_session_windows, "_win32", fake)
        monkeypatch.setattr(_session_windows, "POLL_S", 0.01)
        return fake
    return install


@pytest.mark.asyncio
async def test_stopping_asks_the_apps_windows_to_close_and_ends_the_job_only_if_they_dont(win32):
    obliging = win32({HELPER: [HELPER, 901, 902]}, [_window(1, 901), _window(2, 777), _window(3, 902, shown=False)])
    await _session_windows.stop(HELPER, timeout_s=5)
    obliging_left = not _session_windows.alive(HELPER)
    stubborn = win32({HELPER: [HELPER, 901]}, [_window(1, 901)], closes=False, lingers=3)
    started = time.monotonic()
    await _session_windows.stop(HELPER, timeout_s=0.2)

    assert obliging.posted == [1, 3] and obliging.terminated == obliging.waited == [] and obliging.open == 0
    assert obliging_left
    assert stubborn.posted == [1] and stubborn.terminated == [HELPER] and time.monotonic() - started >= 0.2
    assert stubborn.waited == [[HELPER, 901]] and stubborn.open == 0
    assert stubborn.jobs[HELPER] == [] and not _session_windows.alive(HELPER)


@pytest.mark.asyncio
async def test_a_helper_that_stays_once_its_apps_window_has_closed_is_ended_with_its_job(win32, monkeypatch):
    monkeypatch.setattr(_session_windows, "ENDED_TIMEOUT_S", 0.1)
    stuck = win32({HELPER: [HELPER, 901]}, [_window(1, 901)], leaves=None)

    await _session_windows.stop(HELPER, timeout_s=5)

    assert stuck.posted == [1] and stuck.terminated == [HELPER] and stuck.waited == [[HELPER]] and stuck.open == 0
    assert not _session_windows.alive(HELPER)


@pytest.mark.asyncio
async def test_an_app_with_no_window_to_ask_is_ended_at_once_and_a_gone_job_is_left_be(win32):
    windowless = win32({HELPER: [HELPER, 901]}, [_window(1, 777)])
    started = time.monotonic()
    await _session_windows.stop(HELPER, timeout_s=10)
    gone = win32({}, [])
    await _session_windows.stop(HELPER, timeout_s=10)

    assert windowless.terminated == [HELPER] and windowless.waited == [[HELPER, 901]] and windowless.open == 0
    assert time.monotonic() - started < 1
    assert gone.terminated == [] and gone.open == 0 and not _session_windows.alive(HELPER)


def test_pausing_suspends_every_process_but_the_helper_and_those_they_start_and_resumes_them_even_on_a_raise(win32):
    fake = win32({HELPER: [HELPER, 901, 902]}, [], spawns={902: 903, 903: 904})
    with pytest.raises(RuntimeError):
        with _session_windows.paused(HELPER):
            paused_then = (list(fake.suspended), list(fake.resumed))
            raise RuntimeError("the copy failed")
    gone = win32({}, [])
    with _session_windows.paused(HELPER):
        pass

    assert paused_then == ([901, 902, 903, 904], []) and fake.resumed == [901, 902, 903, 904] and fake.open == 0
    assert gone.suspended == [] and _session_windows.alive(HELPER) is False


@pytest.mark.asyncio
async def test_the_sessions_windows_are_the_shown_ones_by_program_class_and_title(win32):
    win32({}, [_window(1, 901, title="Inbox"), _window(2, 902, shown=False, title="Cloaked")])

    assert await _session_windows.windows() == [{"instance": "python.exe", "class": "TkTopLevel", "title": "Inbox"}]


def test_the_menu_launcher_is_a_start_menu_shortcut_powershell_writes_named_for_the_app(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(subprocess, "run", lambda argv, **kwargs: ran.append((argv, kwargs)))
    folder = Path("C:/agentenv/apps/notes")

    entry = _session_windows.add_launcher('Notes: "draft"', folder)

    ((argv, kwargs),) = ran
    assert entry == tmp_path / "Microsoft/Windows/Start Menu/Programs" / "Notes_ _draft_.lnk" and entry.parent.is_dir()
    assert argv[:3] == ["powershell", "-NoProfile", "-NonInteractive"] and argv[-1] == _session_windows.SHORTCUT
    assert {key: kwargs["env"][key] for key in ("AGENTENV_SHORTCUT", "AGENTENV_TARGET", "AGENTENV_ARGUMENTS")} == {
        "AGENTENV_SHORTCUT": str(entry), "AGENTENV_TARGET": str(folder / "venv" / "Scripts" / "pythonw.exe"),
        "AGENTENV_ARGUMENTS": f'-m agentenv_protocol.computer.launch "{folder}"'}
    assert kwargs["check"] is True and kwargs["creationflags"] == 0x08000000


@pytest.fixture
def helper(tmp_path, monkeypatch):
    """The job helper stood in for by a script, on this interpreter, that starts the app named by its exit code, or
    can't start ``missing``; process flags only Windows takes, as none."""
    script = tmp_path / "_job.py"
    script.write_text(FAKE_HELPER)
    monkeypatch.setattr(_session_windows, "JOB", script)
    monkeypatch.setattr(_session_windows, "PYTHON", Path(sys.executable))
    for flag in ("CREATE_NEW_PROCESS_GROUP", "CREATE_NO_WINDOW"):
        monkeypatch.setattr(subprocess, flag, 0, raising=False)
    return script


@pytest.mark.asyncio
async def test_a_launch_returns_the_helpers_pid_once_the_app_has_started_and_its_exit_is_the_apps(helper, tmp_path):
    pid = await _session_windows.launch(["7"], tmp_path, {})

    assert await _reaping.returncode(pid, timeout_s=10) == 7


@pytest.mark.asyncio
async def test_a_launch_that_cant_start_the_app_says_why_and_one_with_fds_is_refused(helper, tmp_path):
    with pytest.raises(OSError, match=r"couldn't start missing: FileNotFoundError: \[WinError 2\] missing"):
        await _session_windows.launch(["missing"], tmp_path, {})
    with pytest.raises(NotImplementedError, match="fds"):
        await _session_windows.launch(["0"], tmp_path, {}, fds={3: 7})


def test_the_session_is_its_oss_own():
    own = _session_windows if sys.platform == "win32" else _session_posix

    assert (session.launch, session.stop, session.paused, session.windows) == (own.launch, own.stop, own.paused,
                                                                                own.windows)
