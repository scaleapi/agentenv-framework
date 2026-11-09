"""The Fixture Desk's layout and what it records, shared by the Tk desk (fixture.py) and the GTK desk
(wayland_fixture.py), so the two can't drift apart. The probe stages it beside whichever desk it starts. It imports no
toolkit, since a Wayland computer may have no Tk."""
from __future__ import annotations

import json
import os
from pathlib import Path

#: The colours a grid cell cycles through, one more on each click.
COLOURS = ("#ffffff", "#ff0000", "#00aa00", "#0000ff")
CELL = 50
GRID = 8
#: Single-pixel markers on a black field, at odd and even pixels: (x, y) within the field, then the colour.
MARKERS = ((11, 11, "#ff0000"), (12, 11, "#00ff00"), (11, 12, "#0000ff"), (12, 12, "#ffff00"), (37, 20, "#ff00ff"),
           (20, 37, "#00ffff"))
CLIPBOARD = "agentenv-probe clipboard"
MIN_SIZE = (1280, 800)
#: Where each widget's top left goes, the name box's label at "label" and the list's frame at "list".
AT = {"label": (40, 40), "name": (140, 40), "save": (560, 34), "subscribe": (40, 100), "notes": (760, 40),
      "list": (40, 160), "grid": (400, 160), "pad": (840, 160), "wheel": (840, 500), "field": (1080, 500)}
#: The widgets sized in pixels: the list's frame, the drawing pad, the wheel and the marker field.
SIZE = {"list": {"width": 300, "height": 400}, "pad": {"width": 400, "height": 300},
        "wheel": {"width": 200, "height": 120}, "field": {"width": 48, "height": 48}}
#: The list's rows, Item 1 to Item ITEMS.
ITEMS = 100
#: How many of the latest keys and drags state.json keeps, and about how many points of a drag's path.
KEYS_KEPT, DRAGS_KEPT, PATH_POINTS = 100, 20, 50
#: How often the desk reads inbox.txt.
INBOX_EVERY_MS = 200


def new_state(width: int, height: int, windowing: str) -> dict:
    """What state.json holds before anything reaches the desk."""
    return {"name": "", "notes": "", "keys": [], "saved_count": 0, "subscribed": False, "selected": None,
            "scroll_top": 0.0, "cells": {}, "right_clicks": 0, "double_clicks": 0, "drags": [],
            "wheel": {"up": 0, "down": 0, "left": 0, "right": 0}, "clipboard": None, "loaded": None,
            "screen": [width, height], "windowing": windowing}


def write_state(state_dir: Path, state: dict) -> None:
    """state.json replaced whole, so the probe never reads half of one."""
    part = state_dir / ".state.json.part"
    part.write_text(json.dumps(state))
    os.replace(part, state_dir / "state.json")


def write_layout(state_dir: Path, boxes: dict, row_height: int, field: tuple[int, int]) -> None:
    """layout.json: each widget's box on the screen, a list row's height, the markers' pixels from the field's top left,
    and the cell size and clipboard text the probe checks against."""
    layout = {**boxes, "list_row_height": row_height,
              "markers": [[field[0] + x, field[1] + y, colour] for x, y, colour in MARKERS], "cell": CELL,
              "clipboard": CLIPBOARD}
    (state_dir / "layout.json").write_text(json.dumps(layout))


def record_key(state: dict, key: str) -> None:
    state["keys"] = (state["keys"] + [key])[-KEYS_KEPT:]


def record_drag(state: dict, stroke: list) -> None:
    """A finished drag: its ends, its number of points and its path thinned to about PATH_POINTS points."""
    path = stroke[::max(1, len(stroke) // PATH_POINTS)] + stroke[-1:]
    drag = {"start": stroke[0], "end": stroke[-1], "points": len(stroke), "path": path}
    state["drags"] = (state["drags"] + [drag])[-DRAGS_KEPT:]
