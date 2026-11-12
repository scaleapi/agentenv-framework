"""The clock gates' driver, run as root inside the gate image (tst/clock/Dockerfile). The library leaves root's
processes real, so this driver is on real time while it writes the clock page as the follower would, and every program
under test runs as the session user, whom the library fakes. Each gate prints one JSON document on stdout.

  harness.py waits | ratechange | jumps [ROUNDS] | cancel | timerfds | condstress [SECONDS] | semstress [SECONDS]
             | stall | torn [SECONDS] | deadwriter | marker | identity | envrace [SECONDS] | forklock [FORKS]
             | chromium [agentenv|stock LAUNCHES PARALLEL LIMIT] | probe
"""
import ctypes
import json
import os
import random
import select
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/opt/clock")
import clockpage  # noqa: E402

LIB = "/opt/clock/lib/libfaketimeMT.so.1"
STOCK = "/opt/clock/stock/libfaketimeMT.so.1"
PROGRAMS = "/opt/clock/programs"
CHROME = "/opt/chrome/chrome"
ENV = {"PATH": "/opt/clock/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "HOME": "/home/user",
       "LANG": "C.UTF-8"}
DAY = 86400
RAW = time.CLOCK_MONOTONIC_RAW

C_APIS = ["nanosleep", "clock_nanosleep_rt_rel", "clock_nanosleep_mono_rel", "clock_nanosleep_mono_abs",
          "clock_nanosleep_rt_abs", "clock_nanosleep_boot_abs", "usleep", "sleep", "select", "pselect", "poll", "ppoll",
          "epoll_wait", "epoll_pwait", "epoll_pwait2", "timerfd_mono_rel", "timerfd_mono_abs", "timerfd_rt_abs",
          "timerfd_mono_rel_read", "cond_timedwait_rt", "cond_timedwait_mono", "cond_clockwait_mono",
          "cond_clockwait_rt", "cond_timedwait_rt_loop", "cond_timedwait_mono_loop", "sem_timedwait",
          "sem_clockwait_mono", "mutex_timedlock", "mutex_clocklock_mono", "futex_wait_rel", "futex_bitset_mono_abs",
          "alarm", "setitimer_real", "timerfd_rt_abs_read", "timerfd_rt_abs_epoll", "ptimer_rt_abs_signal",
          "ptimer_mono_rel_signal", "ptimer_rt_abs_thread", "futex_bitset_rt_abs", "sem_clockwait_rt",
          "mutex_clocklock_rt", "cond_timedwait_gtod", "gtod_poll_loop"]
PY_APIS = ["py_time_sleep", "py_event_wait", "py_condition_wait", "py_lock_acquire_timeout", "py_queue_get_timeout",
           "py_select", "py_selectors_epoll", "py_asyncio_sleep", "py_poll_monotonic_deadline", "py_until_datetime"]
JAVA_APIS = ["java_thread_sleep", "java_object_wait", "java_park_nanos", "java_queue_poll", "java_scheduled_executor",
             "java_condition_await", "java_park_until", "java_await_until"]
NODE_APIS = ["node_settimeout", "node_setinterval", "node_promises_settimeout"]
APIS = C_APIS + PY_APIS + JAVA_APIS + NODE_APIS
#: Waits for a set clock time, which follow the universe; every other wait is elapsed time, which stays real.
CLOCK_TIME = {"clock_nanosleep_rt_abs", "timerfd_rt_abs", "timerfd_rt_abs_read", "timerfd_rt_abs_epoll",
              "ptimer_rt_abs_signal", "ptimer_rt_abs_thread", "futex_bitset_rt_abs", "cond_timedwait_rt",
              "cond_clockwait_rt", "cond_timedwait_rt_loop", "cond_timedwait_gtod", "sem_timedwait",
              "sem_clockwait_rt", "mutex_timedlock", "mutex_clocklock_rt", "gtod_poll_loop", "py_until_datetime",
              "java_park_until", "java_await_until"}


def raw() -> float:
    return time.clock_gettime(RAW)


def preload(lib=LIB) -> None:
    with open("/etc/ld.so.preload", "w") as f:
        f.write(lib + "\n" if lib else "")


def user_cmd(argv, env=None):
    """``argv`` as the session user, in a clean environment of ``ENV`` and ``env``."""
    e = dict(ENV, **(env or {}))
    return ["setpriv", "--reuid=1000", "--regid=1000", "--clear-groups", "env", "-i",
            *(f"{k}={v}" for k, v in e.items()), *argv]


def page():
    with open(clockpage.PAGE, "rb") as f:
        return clockpage.LAYOUT.unpack(f.read(clockpage.LAYOUT.size))


def universe_now() -> tuple[float, float]:
    _, _, _, u, a, rate, armed, _, _, _, _, _ = page()
    now = time.time()
    return (now, 1.0) if not armed else (u / 1e9 + (now - a / 1e9) * rate, rate)


def set_clock(universe: float, rate: float) -> dict:
    now = time.time()
    written = clockpage.write(1, universe, rate, now)
    return {**written, "universe": universe, "rate": rate, "anchor": now, "raw": raw()}


def advance(delta: float) -> dict:
    u, rate = universe_now()
    return set_clock(u + delta, rate)


def set_rate(rate: float) -> dict:
    u, _ = universe_now()
    return set_clock(u, rate)


def fresh_page() -> None:
    if os.path.exists(clockpage.PAGE):
        os.remove(clockpage.PAGE)
    clockpage.open_page().close()
    clockpage.write(0, 0.0, 1.0, time.time())


def spawn(api: str, seconds: float) -> dict:
    if api in C_APIS:
        argv = ["uwait", api, str(seconds)]
    elif api in PY_APIS:
        argv = ["python3", f"{PROGRAMS}/uwait.py", api, str(seconds)]
    elif api in JAVA_APIS:
        argv = ["java", "-Xshare:auto", "-XX:TieredStopAtLevel=1", "-cp", "/opt/clock/java", "UWait", api, str(seconds)]
    else:
        argv = ["node", f"{PROGRAMS}/uwait.js", api, str(seconds)]
    p = subprocess.Popen(user_cmd(argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd="/tmp")
    os.set_blocking(p.stdout.fileno(), False)
    return {"api": api, "p": p, "buf": "", "ready": None, "done": None, "line": None}


def collect(procs, deadline: float, actions=()):
    """Every process's output until all are done or ``deadline`` (raw), running ``actions`` [(raw_time, fn, label)] on
    the way; when each started and ended."""
    actions = sorted(actions, key=lambda a: a[0])
    ran = []
    while True:
        now = raw()
        while actions and now >= actions[0][0]:
            _, fn, label = actions.pop(0)
            started = raw()
            info = fn()
            ran.append({"label": label, "start_raw": started, "raw": raw(),
                        "info": {k: v for k, v in info.items() if k != "spec"}})
        live = [p for p in procs if p["done"] is None]
        if not live or now >= deadline:
            break
        ready, _, _ = select.select([p["p"].stdout for p in live], [], [], 0.002)
        for p in live:
            if p["p"].stdout in ready:
                p["buf"] += p["p"].stdout.read() or ""
                t = raw()
                for line in p["buf"].split("\n")[:-1]:
                    if line == "READY" and p["ready"] is None:
                        p["ready"] = t
                    elif (line.startswith("DONE") or line.startswith("{")) and p["done"] is None:
                        p["done"], p["line"] = t, line
            if p["p"].poll() is not None and p["done"] is None:
                p["buf"] += p["p"].stdout.read() or ""
                for line in p["buf"].splitlines():
                    if line.startswith("{") or line.startswith("DONE"):
                        p["done"], p["line"] = raw(), line
                if p["done"] is None:
                    p["done"], p["line"] = raw(), "EXIT " + (p["p"].stderr.read() or "")[-300:]
    for p in procs:
        if p["p"].poll() is None:
            p["p"].kill()
    return ran


def result(p, change=None) -> dict:
    """``p``'s outcome; with ``change`` (collect's record of it), when it ended after the change's write ended and
    after it started (ms)."""
    out = {"api": p["api"], "class": "clock_time" if p["api"] in CLOCK_TIME else "elapsed"}
    line = p["line"]
    if line is None:
        return {**out, "status": "still waiting"}
    if line.startswith("{"):
        j = json.loads(line)
        out.update(real=round(j["real"], 4), u_mono=round(j["u_mono"], 4), u_rt=round(j["u_rt"], 4), rc=j["rc"])
        ended = j["start_raw"] + j["real"]
    elif line.startswith("DONE"):
        _, mono, wall = line.split()
        out.update(real=round(p["done"] - p["ready"], 4), u_mono=round(float(mono), 4), u_rt=round(float(wall), 4))
        ended = p["done"]
    else:
        out["status"] = line[:300]
        return out
    if change is not None:
        out["end_after_change_ms"] = round((ended - change["raw"]) * 1000, 1)
        out["end_after_change_start_ms"] = round((ended - change["start_raw"]) * 1000, 1)
    return out


def scenario(name, start_rate, seconds, change=None, change_at=None, limit=8.0, start_universe=None, apis=APIS):
    set_clock(start_universe if start_universe is not None else time.time() - 180 * DAY, start_rate)
    t0 = raw()
    procs = [spawn(a, seconds) for a in apis]
    ran = collect(procs, t0 + limit, [(t0 + change_at, change, "change")] if change else [])
    return {"scenario": name, "start_rate": start_rate, "seconds": seconds, "changes": ran,
            "results": [result(p, ran[0] if ran else None) for p in procs]}


def waits():
    """Every API's 1 s wait at rates 1, 2 and 60, and frozen with +1 h at 2.5 s."""
    preload()
    fresh_page()
    out = [scenario(f"rate {rate}", rate, 1.0, limit=4.0) for rate in (1, 2, 60)]
    out.append(scenario("frozen, +1 h at 2.5 s", 0, 1.0, lambda: advance(3600), 2.5, limit=6.0))
    preload(None)
    return out


def ratechange():
    """Waits for a clock time 10 s ahead at rate 1, the rate going to 60 at 1.0 s (so they end at 1 + 9 / 60 = 1.15 s);
    and waits 2 s ahead when a set_time 182 days back comes at 0.5 s (they wait on, for their new deadline)."""
    preload()
    fresh_page()
    clock = [a for a in APIS if a in CLOCK_TIME]
    out = [scenario("rate 1, rate 60 at 1.0 s", 1, 10.0, lambda: set_rate(60), 1.0, limit=4.0, apis=clock),
           scenario("rate 1, set_time -182 d at 0.5 s", 1, 2.0, lambda: set_clock(universe_now()[0] - 182 * DAY, 1),
                    0.5, limit=4.0, start_universe=time.time(), apis=clock)]
    preload(None)
    return out


def jumps(rounds=30, seed=5420):
    """Forward jumps of +8 h, +1 d or +1 week at rate 1, one a round at a random moment; before each, waits for a clock
    time inside the jumped window ("in"), one past it ("after"), and relative waits ("rel"), from jumpwaits."""
    rng = random.Random(seed)
    preload()
    fresh_page()
    set_clock(time.time() - 182 * DAY, 1)
    p = subprocess.Popen(user_cmd(["jumpwaits"]), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                         cwd="/tmp")
    out = []
    for i in range(rounds):
        j = rng.choice([8 * 3600, DAY, 7 * DAY])
        delay = rng.uniform(0.2, 2.0)
        p.stdin.write(f"arm {j / 2} {j + 5}\n")
        p.stdin.flush()
        armed = p.stdout.readline().split()
        assert armed[0] == "ARMED", armed
        time.sleep(delay)
        w0 = raw()
        advance(j)
        w1 = raw()
        d = json.loads(p.stdout.readline())
        t0 = d["t0"] / 1e9
        row = {"round": i, "jump_s": j, "delay_s": round(delay, 3), "write_ms": round((w1 - w0) * 1e3, 3), "waiters": {}}
        for name, (kind, fired, rc) in d["w"].items():
            f = fired / 1e9
            if kind == 0:
                row["waiters"][name] = {"kind": "in", "after_write_start_ms": round((f - w0) * 1e3, 3),
                                        "after_write_end_ms": round((f - w1) * 1e3, 3), "rc": rc}
            elif kind == 1:
                row["waiters"][name] = {"kind": "after", "vs_expected_ms": round((f - (t0 + 5)) * 1e3, 3), "rc": rc}
            else:
                row["waiters"][name] = {"kind": "rel", "real_s": round(f - t0, 4), "rc": rc}
        row["mono_back"], row["mono_rate"] = d["mono_back"], d["mono_rate"]
        out.append(row)
    p.stdin.close()
    p.wait(timeout=30)
    preload(None)
    return {"rounds": out, "mono_back_total": out[-1]["mono_back"], "mono_rate_final": out[-1]["mono_rate"]}


def _jumped(argv, jump: float) -> dict:
    """``argv`` as the session user, jumped ``jump`` seconds 0.3 s after it prints READY: its two JSON lines, before
    READY and after the jump, and when the jump's write started and ended (raw)."""
    p = subprocess.Popen(user_cmd(argv), stdout=subprocess.PIPE, text=True, cwd="/tmp")
    first = None
    for line in p.stdout:
        if line.strip() == "READY":
            break
        first = json.loads(line)
    time.sleep(0.3)
    started = raw()
    advance(jump)
    written = raw()
    out, _ = p.communicate(timeout=30)
    return {"before": first, "after": json.loads(out), "write_started_raw": started, "written_raw": written}


def _after_write(fired_ns: int, r: dict) -> dict:
    """When a wait the jump ended fired, from the write's start and from its end (ms); None if it never did."""
    if not fired_ns:
        return {"after_write_start_ms": None, "after_write_end_ms": None}
    return {"after_write_start_ms": round((fired_ns / 1e9 - r["write_started_raw"]) * 1e3, 3),
            "after_write_end_ms": round((fired_ns / 1e9 - r["written_raw"]) * 1e3, 3)}


def cancel():
    """Waits for a clock time cancelled (cancel.c): sem_timedwait and clock_nanosleep threads go at once, the semaphore
    keeps no waiter for one, and after 600 cancelled condvar waits a condvar wait still ends at a jump past its deadline."""
    preload()
    fresh_page()
    set_clock(time.time() - 182 * DAY, 1)
    r = _jumped(["cancel"], 2 * 3600)
    preload(None)
    return {**r["before"], "rc": r["after"]["rc"], **_after_write(r["after"]["fired_raw"], r)}


def timerfds():
    """A timerfd's tracking through its duplicates (tfdalias.c), one run per way of making one, one of calls that fail, and
    ones where a vfork child (Python's subprocess among them, tfdspawn.py) closes it, each jumped 2 h."""
    preload()
    fresh_page()
    set_clock(time.time() - 182 * DAY, 1)
    out = {}
    for how in ("dup", "dup2", "dup3", "fcntl", "close_range", "failed", "vfork_close", "vfork_close_range",
                "python_subprocess"):
        argv = ["python3", f"{PROGRAMS}/tfdspawn.py"] if how == "python_subprocess" else ["tfdalias", how]
        r = _jumped(argv, 2 * 3600)
        a = r["after"]
        out[how] = {"same_number": a["same_number"], "failed_calls": a["failed_calls"], "a2_count": a["a2_count"],
                    "o_fired_early": a["o_fired_early"], **_after_write(a["a2_fired_raw"], r)}
    preload(None)
    return out


def _with_page_changes(p) -> int:
    """+10 min every 0.5 s while ``p`` runs; how many."""
    n = 0
    while p.poll() is None:
        time.sleep(0.5)
        advance(600)
        n += 1
    return n


def condstress(seconds=10.0):
    """Realtime condvar handoffs at 1 ms and 100 ms gaps, with and without page changes; then single timed waits: 3 s
    unchanged, 3 s under same-clock rewrites every 0.3 s, and 30 s with +1 h at 1 s, which passes them."""
    preload()
    out = []
    for gap in (1000, 100000):
        for changes in (False, True):
            fresh_page()
            set_clock(time.time() - 182 * DAY, 1)
            p = subprocess.Popen(user_cmd(["condstress2", "stress", str(seconds), str(gap)]), stdout=subprocess.PIPE,
                                 text=True)
            n = _with_page_changes(p) if changes else 0
            o, _ = p.communicate(timeout=seconds + 60)
            out.append({"test": "stress", "gap_us": gap, "page_changes": n, **json.loads(o)})

    def single(wait_s, label, during=None):
        fresh_page()
        set_clock(time.time() - 182 * DAY, 1)
        p = subprocess.Popen(user_cmd(["condstress2", "single", "20", str(wait_s)]), stdout=subprocess.PIPE, text=True)
        writes = during(p) if during else 0
        o, _ = p.communicate(timeout=120)
        return {"test": "single", "label": label, "writes": writes, **json.loads(o)}

    def rewrite(p):
        n = 0
        while p.poll() is None:
            time.sleep(0.3)
            set_clock(*universe_now())
            n += 1
        return n

    def jump(p):
        time.sleep(1.0)
        advance(3600)
        return 1

    out += [single(3, "unchanged"), single(3, "rewrites", rewrite), single(30, "jump", jump)]
    preload(None)
    return out


def semstress(seconds=10.0):
    """sem_timedwait with a realtime deadline against sem_post at 1 ms and 100 ms gaps, with and without page changes."""
    preload()
    out = []
    for gap in (1000, 100000):
        for changes in (False, True):
            fresh_page()
            set_clock(time.time() - 182 * DAY, 1)
            p = subprocess.Popen(user_cmd(["semstress", str(seconds), str(gap)]), stdout=subprocess.PIPE, text=True)
            n = _with_page_changes(p) if changes else 0
            o, _ = p.communicate(timeout=seconds + 60)
            out.append({"gap_us": gap, "page_changes": n, **json.loads(o)})
    preload(None)
    return out


def stall(writes=2000):
    """A writer stalled 5 ms after reading its real time, as a preempted follower is, making ``writes`` writes that
    alternate rate 60 and frozen, while 4 tight readers read CLOCK_REALTIME and CLOCK_MONOTONIC: neither may step
    back, which a page copied before a write whose anchor is earlier than the read's real time would make it do."""

    class Stalled:
        def __getattr__(self, name):
            return getattr(time, name)

        def clock_gettime_ns(self, clock):
            value = time.clock_gettime_ns(clock)
            time.sleep(0.005)
            return value

    preload()
    fresh_page()
    set_clock(time.time() - 182 * DAY, 1)
    seconds = writes * 0.0055 + 2
    readers = [subprocess.Popen(user_cmd(["rtread", str(seconds), "1", f"/tmp/rtread_{k}.json"]), cwd="/tmp")
               for k in range(4)]
    time.sleep(1.0)
    clockpage.time = Stalled()
    started = time.time()
    try:
        for k in range(writes):
            set_rate(60 if k % 2 == 0 else 0)
    finally:
        clockpage.time = time
    wrote = time.time() - started
    for p in readers:
        p.wait()
    threads = []
    for k in range(4):
        with open(f"/tmp/rtread_{k}.json") as f:
            threads += json.load(f)["threads"]
    preload(None)
    return {"writes": writes, "write_seconds": round(wrote, 3), "reads": sum(t["reads"] for t in threads),
            "rt_back": sum(t["rt_back"] for t in threads), "rt_max_back_ns": max(t["rt_max_back_ns"] for t in threads),
            "mono_back": sum(t["mono_back"] for t in threads), "mono_rate": [t["mono_rate"] for t in threads]}


def torn(seconds=10.0):
    """The page rewritten as fast as possible, alternating universes 182 days back and 21 days ahead, while 4 threads
    read CLOCK_REALTIME as fast as they can: every read must be one of the two (torn.c)."""
    preload()
    fresh_page()
    off_a, off_b = -182 * DAY, 21 * DAY
    set_clock(time.time() + off_a, 1)
    p = subprocess.Popen(user_cmd(["torn", str(seconds), "4", str(off_a), str(off_b)]), stdout=subprocess.PIPE,
                         text=True, cwd="/tmp")
    writes = 0
    while p.poll() is None:
        now = time.time()
        clockpage.write(1, now + (off_b if writes % 2 == 0 else off_a), 1.0, now)
        writes += 1
    o, _ = p.communicate(timeout=30)
    preload(None)
    return {"writes": writes, **json.loads(o)}


def deadwriter():
    """A writer that died mid-write leaves seq odd; the next write takes the page up, as given, and readers follow it."""
    preload()
    fresh_page()
    set_clock(time.time() - 182 * DAY, 1)
    m = clockpage.open_page()
    seq = struct.unpack("<Q", m[8:16])[0] + 1
    m[8:16] = struct.pack("<Q", seq)
    m.close()
    before = time.time()
    written = set_clock(time.time() + 21 * DAY, 1)
    after = float(subprocess.run(user_cmd(["date", "+%s.%N"]), capture_output=True, text=True).stdout)
    preload(None)
    return {"dead_seq": seq, "seq": page()[2], "continued": written["continued"],
            "read": _classify(after, before, 21 * DAY)}


def _classify(value: float, before: float, universe_offset: float) -> object:
    if abs(value - before) < 30:
        return "real"
    if abs(value - (before + universe_offset)) < 30:
        return "universe"
    return value


def marker():
    """A program started with AGENTENV_REAL_TIME=1 is real, the marker is gone from its environment, and its children,
    a marked bash's included, are faked (marker.sh)."""
    preload()
    fresh_page()
    set_clock(time.time() - 182 * DAY, 1)
    before = time.time()
    r = subprocess.run(user_cmd([f"{PROGRAMS}/marker.sh"]), capture_output=True, text=True, cwd="/tmp")
    lines = {}
    for line in r.stdout.splitlines():
        key, _, value = line.partition(" ")
        try:
            lines[key] = _classify(float(value), before, -182 * DAY)
        except ValueError:
            lines[key] = value
    preload(None)
    return {"lines": lines, "rc": r.returncode, "stderr": r.stderr[-300:]}


def identity():
    """Who is faked: the session user is, root and another user aren't, and an unarmed page is real for everyone."""
    preload()
    fresh_page()
    before = time.time()
    unarmed = float(subprocess.run(user_cmd(["date", "+%s.%N"]), capture_output=True, text=True).stdout)
    set_clock(time.time() - 182 * DAY, 1)
    date = ["date", "+%s.%N"]
    session = float(subprocess.run(user_cmd(date), capture_output=True, text=True).stdout)
    root = float(subprocess.run(date, capture_output=True, text=True).stdout)
    other = float(subprocess.run(["setpriv", "--reuid=1001", "--regid=1001", "--clear-groups", *date],
                                 capture_output=True, text=True).stdout)
    preload(None)
    return {name: _classify(value, before, -182 * DAY)
            for name, value in (("unarmed_session", unarmed), ("session", session), ("root", root), ("other_user", other))}


def envrace(seconds=10.0):
    """envrace three times: setenv and unsetenv on one thread while others read the clock and make timed waits, which
    stock libfaketime follows with getenv(), the race behind its Chromium crashes. This library reads its settings once,
    at load, so no run may crash."""
    preload()
    fresh_page()
    set_clock(time.time() - 182 * DAY, 1)
    runs = []
    for _ in range(3):
        r = subprocess.run(user_cmd(["envrace", str(seconds)]), capture_output=True, text=True, timeout=seconds + 60)
        runs.append({"rc": r.returncode, "out": r.stdout.strip()[-200:]})
    preload(None)
    return runs


def _stock_env() -> dict:
    return {"LD_PRELOAD": STOCK, "FAKETIME_TIMESTAMP_FILE": "/tmp/stockclock", "FAKETIME_DONT_FAKE_MONOTONIC": "1",
            "FAKETIME_FMT": "%s"}


def chromium(variant="agentenv", launches=40, parallel=8, limit=20.0):
    """Chrome launched directly ``launches`` times, ``parallel`` at a time: with this library in /etc/ld.so.preload
    (``agentenv``), where each launch must load about:blank headless, print it and exit 0 within ``limit`` seconds; or
    with stock 0.9.12 by LD_PRELOAD (``stock``), the positive control, whose launches fail (crash, or hang past
    ``limit``). The test runs the two variants in containers side by side."""
    with open("/tmp/stockclock", "w") as f:
        f.write(f"@{int(time.time() - 182 * DAY)}\n")
    os.chmod("/tmp/stockclock", 0o644)
    fresh_page()
    set_clock(time.time() - 182 * DAY, 1)
    preload(LIB if variant == "agentenv" else None)

    def launch(k: int) -> dict:
        profile = f"/tmp/chrome-{k:04d}"  # fixed width, so no profile is a prefix of another
        argv = [CHROME, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                f"--user-data-dir={profile}", "--dump-dom", "about:blank"]
        started = raw()
        try:
            r = subprocess.run(user_cmd(argv, {} if variant == "agentenv" else _stock_env()), capture_output=True,
                               text=True, timeout=limit, cwd="/tmp")
            rc, out, err = r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired as e:
            rc, out, err = "hung", "", "".join(part for part in (e.stdout, e.stderr) if isinstance(part, str))
        subprocess.run(["pkill", "-9", "-f", profile])
        subprocess.run(["rm", "-rf", profile])
        signalled = "Received signal" in err
        ok = rc == 0 and not signalled and "<html" in out
        return {"rc": rc, "signal": signalled, "ok": ok, "seconds": round(raw() - started, 3),
                "tail": "" if ok else err[-200:]}

    started = raw()
    with ThreadPoolExecutor(parallel) as pool:
        results = list(pool.map(launch, range(launches)))
    took = raw() - started
    preload(None)
    return {"variant": variant, "seconds": round(took, 2), "parallel": parallel, "launches": len(results),
            "failed": sum(not r["ok"] for r in results), "signalled": sum(r["signal"] for r in results),
            "hung": sum(r["rc"] == "hung" for r in results),
            "launch_s_max": max((r["seconds"] for r in results), default=0),
            "failures": [r for r in results if not r["ok"]][:5]}


def forklock(forks=1000):
    """fork while another thread duplicates and closes a timerfd (forklock.c): no child may be left stuck."""
    preload()
    fresh_page()
    r = subprocess.run(user_cmd(["forklock", str(forks)]), capture_output=True, text=True, timeout=forks * 2.5 + 60)
    preload(None)
    return json.loads(r.stdout)


def probe():
    """This container's answer to futex_waitv: the errno of an empty call (EINVAL where the kernel has it)."""
    libc = ctypes.CDLL(None, use_errno=True)
    r = libc.syscall(ctypes.c_long(449), None, ctypes.c_uint(0), ctypes.c_uint(0), None, ctypes.c_int(0))
    return {"futex_waitv_errno": ctypes.get_errno() if r == -1 else 0}


GATES = {f.__name__: f for f in (waits, ratechange, jumps, cancel, timerfds, condstress, semstress, stall, torn,
                                 deadwriter, marker, identity, envrace, forklock, chromium, probe)}

if __name__ == "__main__":
    gate = GATES[sys.argv[1]]
    args = [a if a.isalpha() else float(a) if "." in a else int(a) for a in sys.argv[2:]]
    print(json.dumps(gate(*args), indent=1))
