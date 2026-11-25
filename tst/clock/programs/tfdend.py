"""A periodic realtime timerfd armed through libc for a clock time a second before the end of the clock's range, every
0.3 s, on a clock running into the end with no page written after: read as it ticks for 3 s, as a program's loop reads
it (a re-arm, at the end as at any clock change, drops what isn't read yet, as the kernel's timerfd_settime does), it
has expired at the four ticks before the end and none after, as reads stand at the end. Prints its expirations and the
time read then (ns)."""
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
deadline, count = time.monotonic() + 3, 0
while (left := deadline - time.monotonic()) > 0:
    if select.select([fd], [], [], left)[0]:
        count += int.from_bytes(os.read(fd, 8), "little")
print(json.dumps({"expirations": count, "now": time.time_ns()}))
