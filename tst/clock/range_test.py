"""The clock's range, 1677-09-21T00:12:44Z to 2262-04-11T23:47:16Z, the times 64 bits of nanoseconds since 1970 hold
(as the page and Python keep time): clock reads and every kind of wait follow the universe near either end, before 1970
and in the last hour the gateway arms at rate 1. Each gate runs as its own gate does at the date CLOCK_BASE names, held
to the same bounds."""
from datetime import datetime, timezone

import pytest

from tst.clock.bounds import LATE_MS, LATE_S, clock_time_bound
from tst.clock.waits_test import JUMP_RC, at_jump, failed, within

BASES = {
    "1677": datetime(1677, 9, 22, tzinfo=timezone.utc),
    "1950": datetime(1950, 1, 1, 12, tzinfo=timezone.utc),
    # Far enough before the end for the gates that move the universe on: the jumps' 8 weeks, the torn reads' 203 days.
    "2261": datetime(2261, 8, 1, tzinfo=timezone.utc),
}
# The latest the gateway arms at rate 1: an hour before the end of the range.
LAST_HOUR = datetime(2262, 4, 11, 22, 47, 16, tzinfo=timezone.utc)
JUMP_ROUNDS = 8


def _base(when: datetime) -> list[str]:
    return ["-e", f"CLOCK_BASE={int(when.timestamp())}"]


@pytest.fixture(scope="module", params=list(BASES))
def at(request, gate):
    """``gate``, with the universe starting at one of BASES."""
    def run(name: str, *args, timeout: float = 600):
        return gate(name, *args, timeout=timeout, docker_args=_base(BASES[request.param]))

    return run


@pytest.fixture(scope="module", params=[*BASES, "last hour"])
def waits(request, gate):
    when = LAST_HOUR if request.param == "last hour" else BASES[request.param]
    return {sc["scenario"]: sc["results"] for sc in gate("waits", docker_args=_base(when))}


@pytest.mark.parametrize("rate", [1, 2, 60])
def test_a_wait_is_real_elapsed_time_or_a_clock_time_reached_at_the_rate_near_either_end(waits, rate):
    results = waits[f"rate {rate}"]
    assert len(results) == 64 and not [r for r in results if "status" in r]
    elapsed = {r["api"]: r["real"] for r in results if r["class"] == "elapsed"}
    clock = {r["api"]: r["real"] for r in results if r["class"] == "clock_time"}
    assert len(clock) == 19 and failed(results) == {}
    assert {api: s for api, s in elapsed.items() if not 0.99 <= s <= 1 + LATE_S} == {}
    assert {api: s for api, s in clock.items() if not 1 / rate - 0.01 <= s <= clock_time_bound(rate)} == {}


def test_frozen_near_either_end_a_clock_time_wait_ends_at_the_advance(waits):
    results = waits["frozen, +1 h at 2.5 s"]
    assert len(results) == 64 and not [r for r in results if "status" in r] and failed(results) == {}
    assert {r["api"]: r["end_after_change_ms"] for r in results
            if r["class"] == "clock_time" and not within(r, -1, LATE_MS)} == {}


def test_a_jump_near_either_end_ends_the_waits_it_passes_and_none_other(at):
    jumps = at("jumps", JUMP_ROUNDS, 5420)
    assert len(jumps["rounds"]) == JUMP_ROUNDS and jumps["mono_back_total"] == 0
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


def test_a_timerfd_near_either_end_is_followed_through_its_duplicates(at):
    for how, r in at("timerfds").items():
        expected = (0, 3) if how == "failed" else (0, 0) if how.startswith(("vfork_", "python_")) else (1, 0)
        assert (r["same_number"], r["failed_calls"]) == expected, (how, r)
        assert (r["a2_count"], r["o_fired_early"]) == (1, 0), (how, r)
        assert at_jump(r), (how, r)


def test_a_cancelled_wait_near_either_end_goes_at_once(at):
    r = at("cancel")
    assert (r["sem_cancelled"], r["sem_waiters"], r["cns_cancelled"], r["cond_cancelled"]) == (1, 0, 1, 600), r
    assert r["rc"] == 110 and at_jump(r), r


def test_realtime_never_steps_back_under_a_stalled_writer_near_either_end(at):
    r = at("stall", 500)
    assert r["writes"] == 500 and r["reads"] > 100_000
    assert (r["rt_back"], r["mono_back"]) == (0, 0), r


def test_a_read_near_either_end_is_never_torn_between_two_pages(at):
    r = at("torn", 5.0)
    assert r["writes"] > 500 and r["universe_a"] > 0 and r["universe_b"] > 0 and r["torn"] == 0, r


@pytest.mark.parametrize("when", [BASES["1677"], LAST_HOUR], ids=["1677", "last hour"])
def test_chrome_runs_every_launch_near_either_end(gate, when):
    r = gate("chromium", "agentenv", 16, 8, 20.0, docker_args=_base(when))
    assert (r["launches"], r["failed"]) == (16, 0), r
