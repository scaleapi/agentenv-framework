"""What the drivers on an OS's own input share: the pacing of clicks and drags, how many presses and points they take,
a drag's path, and a capture of every monitor through mss, whose class each driver passes from its own import."""
from __future__ import annotations

import math
from typing import Any

from .errors import DeviceUnavailable
from .operations import Image

MAX_CLICK_COUNT = 3
MAX_DRAG_POINTS = 50
#: The pause between the presses of a double or triple click, in seconds.
CLICK_GAP_S = 0.04
#: How far apart a drag's pointer moves are, in pixels.
DRAG_STEP_PX = 20


def drag_steps(path: list) -> list[tuple[float, float]]:
    """The points a drag moves through after its start, about every DRAG_STEP_PX pixels."""
    steps = []
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / DRAG_STEP_PX))
        steps += [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(1, n + 1)]
    return steps


def mss_capture(mss_class: Any) -> Image:
    """Every monitor, as mss's first monitor spans them, in BGRA."""
    try:
        with mss_class() as m:
            shot = m.grab(m.monitors[0])
            return Image(shot.width, shot.height, bytes(shot.bgra), "BGRA")
    except Exception as e:
        raise DeviceUnavailable(f"capture failed: {type(e).__name__}: {e}") from None
