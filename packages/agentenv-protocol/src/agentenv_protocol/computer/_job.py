"""Run a Windows app in a Job object that holds it and everything it starts: ``python -I -S _job.py -- <argv>``. The
Windows ``session`` launches every app through it, by path, so it loads nothing but the standard library.

It names the job for its own pid (``job_name``), joins the job, then starts the app, so the app and whatever it starts
are in the job; it says ``started`` once the app has, or why it couldn't on stderr, and stays until the job holds no
process but itself, then exits with the app's exit code. It holds the job's only handle, which ends the job when it
closes: the app lives as long as this does, outliving the server that launched it, as a POSIX app's session does, and a
server, restarted or not, finds the job by name to pause or stop it."""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time

#: How often the job is looked at for processes besides this one, once the app has exited.
POLL_S = 0.2
#: ``JOBOBJECT_INFORMATION_CLASS`` values and the limit and access flags used.
BASIC_ACCOUNTING_INFORMATION, BASIC_PROCESS_ID_LIST, EXTENDED_LIMIT_INFORMATION = 1, 3, 9
KILL_ON_JOB_CLOSE = 0x2000
JOB_OBJECT_QUERY, JOB_OBJECT_TERMINATE = 0x0004, 0x0008
STARTED = b"started"


class BasicLimits(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32), ("SchedulingClass", ctypes.c_uint32)]


class IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                    "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class Accounting(ctypes.Structure):
    _fields_ = [("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
                ("ThisPeriodTotalUserTime", ctypes.c_int64), ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                ("TotalPageFaultCount", ctypes.c_uint32), ("TotalProcesses", ctypes.c_uint32),
                ("ActiveProcesses", ctypes.c_uint32), ("TotalTerminatedProcesses", ctypes.c_uint32)]


def job_name(pid: int) -> str:
    """The name of the job the helper running as ``pid`` holds, in the session's own namespace."""
    return f"Local\\agentenv-app-{pid}"


def kernel32() -> ctypes.WinDLL:
    """``kernel32``, its calls typed as this module and ``session`` use them."""
    dll = ctypes.WinDLL("kernel32", use_last_error=True)
    handle, dword, wide = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_wchar_p
    dll.CreateJobObjectW.argtypes, dll.CreateJobObjectW.restype = [ctypes.c_void_p, wide], handle
    dll.OpenJobObjectW.argtypes, dll.OpenJobObjectW.restype = [dword, ctypes.c_int, wide], handle
    dll.SetInformationJobObject.argtypes = [handle, ctypes.c_int, ctypes.c_void_p, dword]
    dll.QueryInformationJobObject.argtypes = [handle, ctypes.c_int, ctypes.c_void_p, dword, ctypes.c_void_p]
    dll.AssignProcessToJobObject.argtypes = [handle, handle]
    dll.TerminateJobObject.argtypes = [handle, ctypes.c_uint]
    dll.GetCurrentProcess.restype = handle
    dll.CloseHandle.argtypes = [handle]
    dll.OpenProcess.argtypes, dll.OpenProcess.restype = [dword, ctypes.c_int, dword], handle
    dll.QueryFullProcessImageNameW.argtypes = [handle, dword, wide, ctypes.c_void_p]
    dll.WaitForMultipleObjects.argtypes, dll.WaitForMultipleObjects.restype = [dword, ctypes.c_void_p, ctypes.c_int,
                                                                               dword], dword
    return dll


def active_processes(dll: ctypes.WinDLL, job: int) -> int:
    """How many processes the job ``job`` holds now."""
    accounting = Accounting()
    if not dll.QueryInformationJobObject(job, BASIC_ACCOUNTING_INFORMATION, ctypes.byref(accounting),
                                         ctypes.sizeof(accounting), None):
        raise ctypes.WinError(ctypes.get_last_error())
    return accounting.ActiveProcesses


def main(argv: list[str]) -> None:
    command = argv[argv.index("--") + 1:]
    dll = kernel32()
    job = dll.CreateJobObjectW(None, job_name(os.getpid()))
    limits = ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = KILL_ON_JOB_CLOSE
    if not job or not dll.SetInformationJobObject(job, EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits),
                                                  ctypes.sizeof(limits)) \
            or not dll.AssignProcessToJobObject(job, dll.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        app = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    except OSError as e:
        sys.stderr.write(f"{type(e).__name__}: {e}\n")
        sys.exit(2)
    sys.stdout.buffer.write(STARTED + b"\n")
    sys.stdout.flush()
    code = app.wait()
    while active_processes(dll, job) > 1:
        time.sleep(POLL_S)
    # An exit code is 32 bits, which Python's exit takes as a signed number.
    sys.exit(code - (1 << 32) if code >= 1 << 31 else code)


if __name__ == "__main__":
    main(sys.argv[1:])
