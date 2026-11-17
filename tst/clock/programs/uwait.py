"""One wait of SECONDS of universe time through a Python API:  uwait.py API SECONDS. Prints one JSON line (real
seconds from CLOCK_MONOTONIC_RAW, which the library leaves real)."""
import asyncio
import json
import queue
import select
import selectors
import sys
import threading
import time

RAW = time.CLOCK_MONOTONIC_RAW


def swallow(fn):
    try:
        fn()
    except queue.Empty:
        pass


def poll_monotonic(s):
    end = time.monotonic() + s
    while time.monotonic() < end:
        time.sleep(0.005)


def until_datetime(s):
    import datetime
    end = datetime.datetime.now() + datetime.timedelta(seconds=s)
    while (left := (end - datetime.datetime.now()).total_seconds()) > 0:
        time.sleep(min(left, 0.1))


TESTS = {
    "py_until_datetime": until_datetime,
    "py_time_sleep": lambda s: time.sleep(s),
    "py_event_wait": lambda s: threading.Event().wait(s),
    "py_condition_wait": lambda s: (lambda c: (c.acquire(), c.wait(s), c.release()))(threading.Condition()),
    "py_lock_acquire_timeout": lambda s: (lambda l: (l.acquire(), l.acquire(timeout=s)))(threading.Lock()),
    "py_queue_get_timeout": lambda s: swallow(lambda: queue.Queue().get(timeout=s)),
    "py_select": lambda s: select.select([], [], [], s),
    "py_selectors_epoll": lambda s: selectors.DefaultSelector().select(s),
    "py_asyncio_sleep": lambda s: asyncio.run(asyncio.sleep(s)),
    "py_poll_monotonic_deadline": poll_monotonic,
}
api, s = sys.argv[1], float(sys.argv[2])
r0, u0, m0 = time.clock_gettime(RAW), time.time(), time.monotonic()
TESTS[api](s)
r1, u1, m1 = time.clock_gettime(RAW), time.time(), time.monotonic()
print(json.dumps({"api": api, "lang": "python", "real": r1 - r0, "u_rt": u1 - u0, "u_mono": m1 - m0, "rc": 0,
                  "start_raw": r0}), flush=True)
