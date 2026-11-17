"""A Python program's timerfd through subprocess, which starts its child with vfork: a realtime timerfd armed through libc
for a clock time an hour ahead, then subprocess.run(["true"]), then READY; the driver jumps 2 h, and the timerfd must
fire at the jump. Prints tfdalias's fields."""
import ctypes
import json
import os
import select
import subprocess
import time

libc = ctypes.CDLL(None, use_errno=True)


class Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


class Itimerspec(ctypes.Structure):
    _fields_ = [("it_interval", Timespec), ("it_value", Timespec)]


CLOCK_REALTIME, TFD_TIMER_ABSTIME = 0, 1
fd = libc.timerfd_create(CLOCK_REALTIME, 0)
due = time.time_ns() + 3600 * 10**9
assert libc.timerfd_settime(fd, TFD_TIMER_ABSTIME, ctypes.byref(Itimerspec(Timespec(0, 0), Timespec(*divmod(due, 10**9)))),
                            None) == 0
subprocess.run(["true"], check=True)
print("READY", flush=True)
fired, count = 0, 0
if select.select([fd], [], [], 5)[0]:
    fired, count = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW), int.from_bytes(os.read(fd, 8), "little")
print(json.dumps({"same_number": 0, "failed_calls": 0, "a2_fired_raw": fired, "a2_count": count, "o_fired_early": 0}))
