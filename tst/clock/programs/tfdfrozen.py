"""Two realtime timerfds armed through libc for clock times on a frozen clock: a periodic one at the universe's now,
every second, read 1.5 s later; and a one-shot at the realtime the argument names (epoch seconds), whose time left
timerfd_gettime gives. Prints the periodic one's expirations and the other's seconds left, then, once a line on stdin
says the clock runs again, the periodic one's expirations over the next 2.5 s."""
import ctypes
import json
import os
import select
import sys
import time

libc = ctypes.CDLL(None, use_errno=True)


class Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


class Itimerspec(ctypes.Structure):
    _fields_ = [("it_interval", Timespec), ("it_value", Timespec)]


CLOCK_REALTIME, TFD_TIMER_ABSTIME = 0, 1


def armed(value: Timespec, interval: Timespec) -> int:
    fd = libc.timerfd_create(CLOCK_REALTIME, 0)
    assert libc.timerfd_settime(fd, TFD_TIMER_ABSTIME, ctypes.byref(Itimerspec(interval, value)), None) == 0
    return fd


periodic = armed(Timespec(*divmod(time.time_ns(), 10**9)), Timespec(1, 0))
far = armed(Timespec(int(sys.argv[1]), 0), Timespec(0, 0))
left = Itimerspec()
assert libc.timerfd_gettime(far, ctypes.byref(left)) == 0
time.sleep(1.5)


def expirations() -> int:
    return int.from_bytes(os.read(periodic, 8), "little") if select.select([periodic], [], [], 0)[0] else 0


print(json.dumps({"periodic_count": expirations(),
                  "far_left_s": left.it_value.tv_sec + left.it_value.tv_nsec / 1e9}), flush=True)
sys.stdin.readline()
expirations()
time.sleep(2.5)
print(json.dumps({"resumed_count": expirations()}), flush=True)
