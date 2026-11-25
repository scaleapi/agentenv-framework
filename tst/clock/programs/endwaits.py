"""At the end of the clock's range: absolute realtime waits for a second past the time a read shows, through
clock_nanosleep and a timerfd, each given 2 s of real time. Prints the read and whether each ended."""
import ctypes
import json
import select
import threading
import time

libc = ctypes.CDLL(None, use_errno=True)


class Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


class Itimerspec(ctypes.Structure):
    _fields_ = [("it_interval", Timespec), ("it_value", Timespec)]


CLOCK_REALTIME, TIMER_ABSTIME, TFD_TIMER_ABSTIME = 0, 1, 1
now = time.time_ns()
due = Timespec(now // 10**9 + 1, 0)
sleeper = threading.Thread(target=libc.clock_nanosleep, args=(CLOCK_REALTIME, TIMER_ABSTIME, ctypes.byref(due), None),
                           daemon=True)
sleeper.start()
fd = libc.timerfd_create(CLOCK_REALTIME, 0)
assert libc.timerfd_settime(fd, TFD_TIMER_ABSTIME, ctypes.byref(Itimerspec(Timespec(0, 0), due)), None) == 0
fired = bool(select.select([fd], [], [], 2.0)[0])
sleeper.join(0.1)
print(json.dumps({"now": now, "clock_nanosleep_ended": not sleeper.is_alive(), "timerfd_fired": fired}))
