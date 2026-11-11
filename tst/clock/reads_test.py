"""Clock reads: never back, whatever the writer's timing, and never a mix of two pages."""


def test_a_stalled_writer_never_steps_realtime_or_monotonic_back(gate):
    """2,000 writes alternating rate 60 and frozen, each stalled 5 ms after reading its real time, under 4 tight readers."""
    r = gate("stall")
    assert r["writes"] == 2000 and r["reads"] > 1_000_000
    assert (r["rt_back"], r["mono_back"]) == (0, 0), r
    assert all(abs(rate - 1) < 1e-3 for rate in r["mono_rate"]), r


def test_a_read_is_never_torn_between_two_pages(gate):
    r = gate("torn", 10.0)
    assert r["writes"] > 1000 and r["reads"] > 1_000_000 and r["universe_a"] > 0 and r["universe_b"] > 0, r
    assert r["torn"] == 0, r


def test_a_page_a_writer_died_writing_is_taken_up_by_the_next_write(gate):
    """seq left odd: the next write goes on from the even seq after it, writes the universe as given and is read."""
    r = gate("deadwriter")
    assert r["seq"] == r["dead_seq"] + 3 and r["seq"] % 2 == 0 and not r["continued"] and r["read"] == "universe", r
