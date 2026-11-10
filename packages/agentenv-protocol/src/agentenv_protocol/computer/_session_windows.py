"""What a computer app's own server does in the desktop session on Windows (``session``). agentenv-session starts the
server in the session user's sign-in, so the session is up and its desktop is the user's. An app launches through
``_job`` into a Job object that holds it and everything it starts, found by the helper's pid: stopping it asks its
windows to close, then ends the job after the timeout, and pausing it suspends each of its processes. Its menu launcher
is a Start Menu shortcut, and the session's windows are listed with ``EnumWindows``, each by its class, its program's
file name and its title."""
from __future__ import annotations

import asyncio
import contextlib
import ctypes
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, NamedTuple, Optional

from . import _job, _reaping

#: The helper that runs an app in its job, run on its own so it loads nothing but the standard library.
JOB = Path(__file__).with_name("_job.py")
#: The interpreter the helper runs on: the base one, as a venv's python.exe is a launcher that runs it as a child, and
#: the helper names its job for its own pid, which must be the pid a launch returns.
PYTHON = Path(sys.base_prefix) / "python.exe"
#: How long the helper may take to start the app and say so.
START_TIMEOUT_S = 30.0
#: How often a stopping app is looked at for having ended.
POLL_S = 0.2
#: How long a job ended by force is waited on for each of its processes to have exited, so none still writes once a
#: stop returns and whatever they had open is closed, as a reset then puts the app's files back.
ENDED_TIMEOUT_S = 5.0
WM_CLOSE = 0x0010
#: A Start Menu shortcut, from its path, program, arguments and folder in the environment, as PowerShell writes one.
SHORTCUT = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:AGENTENV_SHORTCUT); "
            "$s.TargetPath = $env:AGENTENV_TARGET; $s.Arguments = $env:AGENTENV_ARGUMENTS; "
            "$s.WorkingDirectory = $env:AGENTENV_FOLDER; $s.Save()")
#: What a file name can't hold on Windows.
UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


async def wait_until_up() -> None:
    """At once: the server runs in the session user's sign-in, so the session is up."""


async def launch(argv: list[str], cwd: Path, env: dict[str, str], fds: Optional[dict[int, int]] = None) -> int:
    """Start ``argv`` in ``cwd`` in a job of its own (``_job``), which outlives the server, with the server's
    environment and ``env``; the helper's pid, by which the job is found. A Windows app can't be handed fds."""
    if fds:
        raise NotImplementedError("a Windows app can't be handed the server's fds")
    process = await asyncio.create_subprocess_exec(
        str(PYTHON), "-I", "-S", str(JOB), "--", *argv, cwd=cwd, env={**os.environ, **env},
        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
    try:
        said = await asyncio.wait_for(process.stdout.readline(), START_TIMEOUT_S)
    except BaseException:
        # Its job's only handle closes with it, which ends whatever it started.
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        _reaping.reap(process)
        raise
    if said.strip() != _job.STARTED:
        why = ((await process.stderr.read()).decode(errors="replace").strip().splitlines() or [""])[-1]
        await process.wait()
        raise OSError(f"couldn't start {argv[0]}: {why or f'its job helper exited {process.returncode}'}")
    _reaping.reap(process)
    return process.pid


async def stop(pid: int, timeout_s: float) -> None:
    """End the job the helper ``pid`` holds, every process in it: its windows asked to close, then the job ended after
    ``timeout_s``, or at once when it has no window to close, and waited on until each process it held has exited and
    it holds none. No one asks how it exited once it's stopped."""
    _reaping.forget(pid)
    job = _win32.open_job(pid)
    if job is None:
        return
    try:
        windows = [window for window in await asyncio.to_thread(_win32.windows) if window.pid in _app_pids(job, pid)]
        for window in windows:
            _win32.post_close(window.handle)
        deadline = time.monotonic() + (timeout_s if windows else 0)
        while _app_pids(job, pid) and time.monotonic() < deadline:
            await asyncio.sleep(POLL_S)
        if _app_pids(job, pid):
            ending = _win32.open_processes(_win32.job_pids(job))
            try:
                _win32.terminate_job(job)
                deadline = time.monotonic() + ENDED_TIMEOUT_S
                await asyncio.to_thread(_win32.wait_exited, ending, ENDED_TIMEOUT_S)
                while _win32.job_pids(job) and time.monotonic() < deadline:
                    await asyncio.sleep(POLL_S)
            finally:
                for process in ending:
                    _win32.close(process)
    finally:
        _win32.close(job)


@contextlib.contextmanager
def paused(pid: int) -> Iterator[None]:
    """Every process of the job the helper ``pid`` holds suspended while the block runs, and resumed after. One not yet
    suspended can start another, so the job is listed again until it holds none that's new."""
    job = _win32.open_job(pid)
    if job is None:
        yield
        return
    suspended, seen = [], {pid}
    try:
        while fresh := [app_pid for app_pid in _win32.job_pids(job) if app_pid not in seen]:
            seen.update(fresh)
            suspended += [app_pid for app_pid in fresh if _win32.suspend(app_pid)]
        yield
    finally:
        for app_pid in suspended:
            _win32.resume(app_pid)
        _win32.close(job)


def alive(pid: int) -> bool:
    """Whether a process of the job the helper ``pid`` holds runs, the helper or one of the app's; a job a stop just
    ended is there, empty, until its last handle closes."""
    job = _win32.open_job(pid)
    if job is None:
        return False
    try:
        return bool(_win32.job_pids(job))
    finally:
        _win32.close(job)


def add_launcher(name: str, folder: Path) -> Path:
    """The Start Menu shortcut, named ``name``, that launches the app in ``folder`` through its own server, written
    where the session user's Start Menu finds it."""
    entry = Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs" / f"{UNSAFE.sub('_', name)}.lnk"
    entry.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", SHORTCUT], check=True,
                   capture_output=True, timeout=60, creationflags=subprocess.CREATE_NO_WINDOW,
                   env={**os.environ, "AGENTENV_SHORTCUT": str(entry), "AGENTENV_FOLDER": str(folder),
                        "AGENTENV_TARGET": str(folder / "venv" / "Scripts" / "pythonw.exe"),
                        "AGENTENV_ARGUMENTS": f'-m agentenv_protocol.computer.launch "{folder}"'})
    return entry


async def windows() -> list[dict[str, str]]:
    """The session's shown top-level windows, each as its program's file name, lowercased, its ``instance``, its window
    class its ``class``, and its ``title``; a window that's hidden or cloaked, as a suspended Store app's frame is, is
    left out."""
    return await asyncio.to_thread(lambda: [
        {"instance": window.program, "class": window.class_name, "title": window.title}
        for window in _win32.windows() if window.shown])


def _app_pids(job: int, helper: int) -> list[int]:
    """The processes of the job ``job`` but the helper ``helper`` that holds it."""
    return [pid for pid in _win32.job_pids(job) if pid != helper]


@dataclass(frozen=True)
class Window:
    """A top-level window as ``EnumWindows`` lists it: its ``handle``, the ``pid`` of its process, that process's
    ``program`` file name, lowercased, its ``class_name`` and ``title``, and whether it's ``shown``."""

    handle: int
    pid: int
    program: str
    class_name: str
    title: str
    shown: bool


class _Win32:
    """The Win32 calls this module makes, each a method, so its logic is tested apart from them."""

    PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SUSPEND_RESUME, SYNCHRONIZE = 0x1000, 0x0800, 0x00100000
    #: The most handles one wait takes (``MAXIMUM_WAIT_OBJECTS``).
    WAIT_HANDLES = 64
    DWMWA_CLOAKED = 14
    #: How many process ids a job's list is read with room for, before it's read again with room for all.
    PIDS = 64
    #: The characters a window class's name and a program's full path take at most, with the closing NUL.
    CLASS_NAME_CHARS, PATH_CHARS = 256, 32768

    def __init__(self) -> None:
        self._dlls: Optional[_Dlls] = None

    def open_job(self, helper: int) -> Optional[int]:
        """A handle to the job the helper ``helper`` holds, to query and end it, or None when it isn't there."""
        return self._api().kernel32.OpenJobObjectW(_job.JOB_OBJECT_QUERY | _job.JOB_OBJECT_TERMINATE, False,
                                                   _job.job_name(helper)) or None

    def job_pids(self, job: int) -> list[int]:
        kernel32 = self._api().kernel32
        room = self.PIDS
        while True:
            listed = _pid_list(room)()
            if kernel32.QueryInformationJobObject(job, _job.BASIC_PROCESS_ID_LIST, ctypes.byref(listed),
                                                  ctypes.sizeof(listed), None):
                return list(listed.ProcessIdList[:listed.NumberOfProcessIdsInList])
            if listed.NumberOfAssignedProcesses <= room:
                raise ctypes.WinError(ctypes.get_last_error())
            room = listed.NumberOfAssignedProcesses

    def terminate_job(self, job: int) -> None:
        if not self._api().kernel32.TerminateJobObject(job, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self, handle: int) -> None:
        self._api().kernel32.CloseHandle(handle)

    def open_processes(self, pids: list[int]) -> list[int]:
        """Handles to wait on each process of ``pids`` that's there; one that's gone has none."""
        kernel32 = self._api().kernel32
        return [process for pid in pids if (process := kernel32.OpenProcess(self.SYNCHRONIZE, False, pid))]

    def wait_exited(self, processes: list[int], timeout_s: float) -> None:
        """Wait until each of ``processes``, handles, has exited, every thread of it ended, or ``timeout_s`` passes: a
        process's own handle says it has, where its job's list says only that it's no longer in the job."""
        kernel32, deadline = self._api().kernel32, time.monotonic() + timeout_s
        for first in range(0, len(processes), self.WAIT_HANDLES):
            some = processes[first:first + self.WAIT_HANDLES]
            kernel32.WaitForMultipleObjects(len(some), (ctypes.c_void_p * len(some))(*some), True,
                                            max(0, int((deadline - time.monotonic()) * 1000)))

    def suspend(self, pid: int) -> bool:
        """Whether the process ``pid`` was there to suspend, and was."""
        return bool(self._on_process(pid, self.PROCESS_SUSPEND_RESUME,
                                     lambda process: self._api().ntdll.NtSuspendProcess(process) == 0))

    def resume(self, pid: int) -> None:
        self._on_process(pid, self.PROCESS_SUSPEND_RESUME, lambda process: self._api().ntdll.NtResumeProcess(process))

    def post_close(self, window: int) -> None:
        self._api().user32.PostMessageW(window, WM_CLOSE, 0, 0)

    def windows(self) -> list[Window]:
        dlls = self._api()
        user32, dwmapi = dlls.user32, dlls.dwmapi
        handles: list[int] = []
        listing = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)(
            lambda window, _: handles.append(window) or True)
        user32.EnumWindows(listing, None)
        found = []
        for handle in handles:
            pid, cloaked = ctypes.c_uint32(), ctypes.c_uint32()
            user32.GetWindowThreadProcessId(handle, ctypes.byref(pid))
            class_name, title = ctypes.create_unicode_buffer(self.CLASS_NAME_CHARS), ctypes.create_unicode_buffer(
                user32.GetWindowTextLengthW(handle) + 1)
            user32.GetClassNameW(handle, class_name, len(class_name))
            user32.GetWindowTextW(handle, title, len(title))
            dwmapi.DwmGetWindowAttribute(handle, self.DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
            found.append(Window(handle, pid.value, self._program(pid.value), class_name.value, title.value,
                                bool(user32.IsWindowVisible(handle)) and not cloaked.value))
        return found

    def _program(self, pid: int) -> str:
        """The file name of the program running as ``pid``, lowercased, or empty where it can't be read."""
        def read(process: int) -> str:
            path, size = ctypes.create_unicode_buffer(self.PATH_CHARS), ctypes.c_uint32(self.PATH_CHARS)
            if not self._api().kernel32.QueryFullProcessImageNameW(process, 0, path, ctypes.byref(size)):
                return ""
            return path.value.rpartition("\\")[2].lower()
        return self._on_process(pid, self.PROCESS_QUERY_LIMITED_INFORMATION, read) or ""

    def _on_process(self, pid: int, access: int, act):
        """``act`` on a handle to the process ``pid`` opened for ``access``, closed after; None where it can't be
        opened, as for a process that has exited."""
        kernel32 = self._api().kernel32
        process = kernel32.OpenProcess(access, False, pid)
        if not process:
            return None
        try:
            return act(process)
        finally:
            kernel32.CloseHandle(process)

    def _api(self) -> _Dlls:
        """The DLLs this class calls, bound on first use, so this module imports anywhere."""
        if self._dlls is None:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.EnumWindows.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            user32.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            user32.GetWindowTextLengthW.argtypes = user32.IsWindowVisible.argtypes = [ctypes.c_void_p]
            user32.GetClassNameW.argtypes = user32.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p,
                                                                              ctypes.c_int]
            user32.PostMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t]
            dwmapi = ctypes.WinDLL("dwmapi")
            dwmapi.DwmGetWindowAttribute.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
                                                     ctypes.c_uint32]
            ntdll = ctypes.WinDLL("ntdll")
            ntdll.NtSuspendProcess.argtypes = ntdll.NtResumeProcess.argtypes = [ctypes.c_void_p]
            self._dlls = _Dlls(_job.kernel32(), user32, dwmapi, ntdll)
        return self._dlls


class _Dlls(NamedTuple):
    """The DLLs ``_Win32`` calls, each with the calls it makes typed: ``kernel32``'s in ``_job``, as the helper makes
    some of them too."""

    kernel32: ctypes.WinDLL
    user32: ctypes.WinDLL
    dwmapi: ctypes.WinDLL
    ntdll: ctypes.WinDLL


def _pid_list(room: int) -> type[ctypes.Structure]:
    """``JOBOBJECT_BASIC_PROCESS_ID_LIST`` with room for ``room`` process ids."""
    class Listed(ctypes.Structure):
        _fields_ = [("NumberOfAssignedProcesses", ctypes.c_uint32), ("NumberOfProcessIdsInList", ctypes.c_uint32),
                    ("ProcessIdList", ctypes.c_size_t * room)]
    return Listed


_win32 = _Win32()
