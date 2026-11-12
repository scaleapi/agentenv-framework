"""Stress: realtime condvar handoffs and sem_timedwait posts (with futex_waitv and without it), with the page changing
under them, lose nothing."""
import pytest

from tst.clock.bounds import LATE_MS, LATE_S

SECONDS = 10


@pytest.fixture(scope="module")
def condvars(gate):
    return gate("condstress", float(SECONDS))


def test_a_realtime_condvar_handoff_is_never_lost_or_late(condvars):
    stress = [r for r in condvars if r["test"] == "stress"]
    assert len(stress) == 4
    for r in stress:
        assert r["timeouts"] == 0 and r["max_ms"] < LATE_MS, r
        assert r["handoffs"] >= 0.25 * SECONDS * 1e6 / (r["gap_us"] + 30), r
        if r["page_changes"] == 0:
            assert r["spurious"] == 0, r


def test_a_single_timed_wait_times_out_at_its_clock_time(condvars):
    singles = {r["label"]: r for r in condvars if r["test"] == "single"}
    for label, (low, high) in {"unchanged": (2.99, 3 + LATE_S), "rewrites": (2.99, 3 + LATE_S),
                               "jump": (0.99, 1 + LATE_S)}.items():
        results = singles[label]["results"]
        assert len(results) == 20 and all(rc == "ETIMEDOUT" and low <= real <= high for rc, real in results), (label, results)


def test_a_sem_post_is_never_lost_under_a_realtime_timed_wait(waits_gate):
    rows = waits_gate("semstress", float(SECONDS))
    assert len(rows) == 4
    for r in rows:
        assert r["posts"] == r["taken"] > 0, r
        assert (r["errors"], r["timeouts"], r["stuck_2s"]) == (0, 0, 0) and r["max_ms"] < LATE_MS, r
