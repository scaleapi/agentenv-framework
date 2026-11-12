"""Chromium, the client stock libfaketime crashed on about one launch in five: launched directly, it runs every time with
this library (40 launches, which that rate would leave clean about once in 7,500 runs), and stock 0.9.12, the positive
control, fails it. The two run side by side, in containers of their own, and the gate takes under 30 s."""
from concurrent.futures import ThreadPoolExecutor


def test_chrome_runs_every_launch_with_this_library_and_stock_fails_it(gate):
    with ThreadPoolExecutor(2) as pool:
        ours, stock = pool.map(lambda args: gate("chromium", *args), [("agentenv", 40, 8, 20.0), ("stock", 10, 10, 3.0)])
    assert max(ours["seconds"], stock["seconds"]) <= 30, (ours, stock)
    assert (ours["launches"], ours["failed"]) == (40, 0), ours
    assert stock["launches"] == 10 and stock["failed"] >= 1, stock
