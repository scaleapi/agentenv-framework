"""A periodic realtime timerfd armed through libc for a clock time a second before the end of the clock's range, every
0.3 s, on a clock running into the end with no page written after: read 3 s later, it has expired at the four ticks
before the end and none after, as reads stand at the end. Prints its expirations and the time read then (ns)."""
import ctypes
import json
import os
import select
import time

libc = ctypes.CDLL(None, use_errno=True)


class Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


class Itimerspec(ctypes.Structure):
    _fields_ = [("it_interval", Timespec), ("it_value", Timespec)]


CLOCK_REALTIME, TFD_TIMER_ABSTIME, END = 0, 1, 9223372036

fd = libc.timerfd_create(CLOCK_REALTIME, 0)
every = Itimerspec(Timespec(0, 300_000_000), Timespec(END - 1, 0))
assert libc.timerfd_settime(fd, TFD_TIMER_ABSTIME, ctypes.byref(every), None) == 0
time.sleep(3)
count = int.from_bytes(os.read(fd, 8), "little") if select.select([fd], [], [], 0)[0] else 0
print(json.dumps({"expirations": count, "now": time.time_ns()}))
