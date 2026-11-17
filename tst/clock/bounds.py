"""How late the gates let a wait end. A broken library misses its mark by seconds or hours (a jump it never saw, a rate
or a deadline it ignored, a wakeup it lost); a slow or loaded machine, a CI runner say, by tens of milliseconds. The
bounds sit between the two, so a gate fails on the first and not the second. Early is never allowed."""

#: The most a wait may end after its mark.
LATE_S = 1.0
LATE_MS = LATE_S * 1000


def clock_time_bound(rate: float) -> float:
    """The latest a 1 s clock-time wait may end at ``rate``: its 1 / rate, plus LATE_S, or half the gap to 1 s when that
    is less, so a wait that ignored the rate (and took 1 s) still fails."""
    return 1 / rate + (LATE_S if rate == 1 else min(LATE_S, (1 - 1 / rate) / 2))
