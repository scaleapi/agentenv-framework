"""Chromium, the client stock libfaketime crashed on about one launch in five: launched directly, it runs every time with
this library, and stock 0.9.12, the positive control, fails it. The whole gate runs in under 30 s."""


def test_chrome_runs_every_launch_with_this_library_and_stock_fails_it(gate):
    r = gate("chromium")
    assert r["seconds"] <= 30, r
    assert (r["agentenv"]["launches"], r["agentenv"]["failed"]) == (60, 0), r["agentenv"]
    assert r["stock"]["launches"] == 10 and r["stock"]["failed"] >= 1, r["stock"]
