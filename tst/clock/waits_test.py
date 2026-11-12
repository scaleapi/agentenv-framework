"""Waits: every wait API in C, Python, Java and Node is either elapsed time, real at any rate, or a wait for a set clock
time, which follows the universe through rates, a freeze, a rate change, a set_time back and forward jumps, with
futex_waitv and without it; cancelled, it leaves nothing behind."""
import pytest

from tst.clock.bounds import LATE_MS, LATE_S, clock_time_bound

#: Single pthread_cond_timedwait/clockwait calls return 0 once when a change moves their deadline without passing it
#: (a rate change, a set_time back), as POSIX allows of a timed wait; every caller's predicate loop absorbs it.
SINGLE_CALL_CONDVARS = {"cond_timedwait_rt", "cond_clockwait_rt", "cond_timedwait_gtod"}
ETIMEDOUT, EINVAL = 110, 22  # Linux's, where the programs run
#: What each jumpwaits waiter returns: a timed wait's error, a timerfd's expirations, a loop's calls (two when the jump
#: moves its deadline without passing it, one spurious return), 0 for sleeps and POSIX timers.
JUMP_RC = {"cns_in": {0}, "tfd_poll_in": {1}, "tfd_read_in": {1}, "tfd_epoll_in": {1}, "ptimer_thread_in": {0},
           "ptimer_signal_in": {0}, "cond_rt_in": {ETIMEDOUT}, "cond_clock_rt_in": {ETIMEDOUT}, "cond_loop_in": {1},
           "sem_rt_in": {ETIMEDOUT}, "futex_rt_in": {ETIMEDOUT}, "mutex_rt_in": {ETIMEDOUT}, "cns_after": {0},
           "tfd_poll_after": {1}, "cond_loop_after": {1, 2}, "ptimer_thread_after": {0}, "nanosleep_rel": {0},
           "poll_rel": {0}, "cond_mono_rel": {ETIMEDOUT}, "tfd_mono_rel": {0}, "cns_mono_rel": {0}}


def within(r: dict, low: float, high: float) -> bool:
    """Ended ``low`` ms or more after the change's write started and ``high`` ms or less after it ended: the page
    changes, and waits wake, before the write's last step, which a slow machine can take tens of ms over."""
    return "end_after_change_ms" in r and r["end_after_change_start_ms"] >= low and r["end_after_change_ms"] <= high


def at_jump(w: dict) -> bool:
    """A wait the jump ended: not before its write started (the page changes, and waits wake, before the write's last
    step), and within LATE_MS of the write's end."""
    return (w["after_write_start_ms"] is not None and w["after_write_start_ms"] >= -1
            and w["after_write_end_ms"] <= LATE_MS)


def failed(results) -> dict:
    """Results with an error: C and Python report one (a timeout is 0); Java and Node end without one."""
    return {r["api"]: r.get("rc") for r in results
            if "status" not in r and not r["api"].startswith(("java_", "node_")) and r.get("rc") != 0}


@pytest.fixture(scope="module")
def waits(waits_gate):
    return {sc["scenario"]: sc["results"] for sc in waits_gate("waits")}


@pytest.mark.parametrize("rate", [1, 2, 60])
def test_a_wait_is_real_elapsed_time_or_a_clock_time_reached_at_the_rate(waits, rate):
    results = waits[f"rate {rate}"]
    assert len(results) == 64 and not [r for r in results if "status" in r]
    elapsed = {r["api"]: r["real"] for r in results if r["class"] == "elapsed"}
    clock = {r["api"]: r["real"] for r in results if r["class"] == "clock_time"}
    assert len(clock) == 19 and failed(results) == {}
    assert {api: s for api, s in elapsed.items() if not 0.99 <= s <= 1 + LATE_S} == {}
    assert {api: s for api, s in clock.items() if not 1 / rate - 0.01 <= s <= clock_time_bound(rate)} == {}


def test_frozen_a_clock_time_wait_ends_at_the_advance_and_elapsed_time_runs_on(waits):
    results = waits["frozen, +1 h at 2.5 s"]
    assert len(results) == 64 and not [r for r in results if "status" in r] and failed(results) == {}
    assert {r["api"]: r["real"] for r in results
            if r["class"] == "elapsed" and not 0.99 <= r["real"] <= 1 + LATE_S} == {}
    assert {r["api"]: r["end_after_change_ms"] for r in results
            if r["class"] == "clock_time" and not within(r, -1, LATE_MS)} == {}


@pytest.fixture(scope="module")
def changes(waits_gate):
    return {sc["scenario"]: sc["results"] for sc in waits_gate("ratechange")}


def test_a_rate_change_moves_a_clock_time_wait_to_the_new_rate(changes):
    """10 s of clock time at rate 1, rate 60 at 1.0 s: the 9 s left take 0.15 s."""
    results = {r["api"]: r for r in changes["rate 1, rate 60 at 1.0 s"]}
    assert len(results) == 19 and not [r for r in results.values() if "status" in r]
    assert failed(results.values()) == {}
    assert {api: r["end_after_change_ms"] for api, r in results.items()
            if api in SINGLE_CALL_CONDVARS and not (within(r, -1, LATE_MS) and r["rc"] == 0)} == {}
    assert {api: r["end_after_change_ms"] for api, r in results.items()
            if api not in SINGLE_CALL_CONDVARS and not within(r, 140, LATE_MS)} == {}


def test_a_set_time_back_makes_a_clock_time_wait_wait_on_for_its_deadline(changes):
    results = {r["api"]: r for r in changes["rate 1, set_time -182 d at 0.5 s"]}
    assert len(results) == 19
    assert {api: r.get("status") for api, r in results.items()
            if api not in SINGLE_CALL_CONDVARS and r.get("status") != "still waiting"} == {}
    assert {api: r.get("end_after_change_ms") for api, r in results.items()
            if api in SINGLE_CALL_CONDVARS and not within(r, -1, LATE_MS)} == {}
    assert failed(results.values()) == {}


def test_a_jump_ends_the_waits_it_passes_and_none_other_and_monotonic_time_runs_on(waits_gate):
    """30 forward jumps of 8 h, a day or a week at rate 1: a wait for a clock time inside the jump ends at it (once the
    write has started, and within LATE_MS of its end), one past it ends when its clock time comes and never early, and a
    relative wait takes its 3 s of real time; each ends as it should (a timeout, an expiration, a sleep done)."""
    jumps = waits_gate("jumps")
    assert len(jumps["rounds"]) == 30 and jumps["mono_back_total"] == 0
    assert abs(jumps["mono_rate_final"] - 1) < 1e-3
    wrong = []
    for row in jumps["rounds"]:
        assert row["waiters"].keys() == JUMP_RC.keys()
        for name, w in row["waiters"].items():
            if w["rc"] not in JUMP_RC[name]:
                wrong.append((row["round"], name, w))
            elif w["kind"] == "in" and not at_jump(w):
                wrong.append((row["round"], name, w))
            elif w["kind"] == "after" and not -1 <= w["vs_expected_ms"] <= LATE_MS:
                wrong.append((row["round"], name, w))
            elif w["kind"] == "rel" and not 2.999 <= w["real_s"] <= 3 + LATE_S:
                wrong.append((row["round"], name, w))
    assert wrong == []


def test_a_cancelled_wait_goes_at_once_and_leaves_nothing_behind(waits_gate):
    """sem_timedwait and clock_nanosleep are cancellation points; 600 cancelled condvar waits, more than the library's
    table holds, leave no entry, so the condvar wait after them still ends at a jump past its deadline. A realtime futex
    wait on an unaligned word fails at once, as the kernel's does."""
    r = waits_gate("cancel")
    assert (r["sem_cancelled"], r["sem_waiters"], r["cns_cancelled"], r["cond_cancelled"]) == (1, 0, 1, 600), r
    assert r["misaligned_errno"] == EINVAL and r["misaligned_ms"] < LATE_MS, r
    assert r["rc"] == ETIMEDOUT and at_jump(r), r


def test_a_timerfd_is_followed_through_its_duplicates_not_its_number(gate):
    """A timerfd armed for a clock time, duplicated and closed, fires at a jump past it through its duplicate, and the
    monotonic timerfd given its old number doesn't fire with it; dup2, dup3 and close_range calls on it that fail, and a
    vfork child that closes it (as Python's subprocess makes one), leave it followed."""
    runs = gate("timerfds")
    assert runs.keys() == {"dup", "dup2", "dup3", "fcntl", "close_range", "failed", "vfork_close", "vfork_close_range",
                           "python_subprocess"}
    for how, r in runs.items():
        expected = (0, 3) if how == "failed" else (0, 0) if how.startswith(("vfork_", "python_")) else (1, 0)
        assert (r["same_number"], r["failed_calls"]) == expected, (how, r)
        assert (r["a2_count"], r["o_fired_early"]) == (1, 0), (how, r)
        assert at_jump(r), (how, r)
