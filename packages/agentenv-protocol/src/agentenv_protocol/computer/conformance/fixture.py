"""The fixture the conformance probe stages with data/add and launches with computer-exec: a Tk window, the Fixture
Desk, whose widgets' screen boxes are written to <state-dir>/layout.json and every effect that reaches it, with
whatever lands in <state-dir>/inbox.txt, to <state-dir>/state.json, so the probe can aim at it and read back what
arrived. It records the same effects under X11, macOS's Aqua and Windows, whose Tk names keys, binds select-all and
reports the wheel differently. Run as ``python3 fixture.py <state-dir>``; it needs a 1280x800 screen or larger."""
from __future__ import annotations

import ctypes
import json
import os
import sys
import tkinter as tk
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
#: X11's keysyms for the keys Aqua's Tk names by their character.
KEYSYMS = {",": "comma", " ": "space"}
#: A wheel notch's <MouseWheel> delta where Tk scales the wheel: Windows', and Tk 9's everywhere.
NOTCH_DELTA = 120
#: Tk 9 packs a <TouchpadScroll>'s distances into its delta: y, a signed 16-bit number, in the low bits, x above.
Y_BITS = 16
#: The pixels of a scrolled line, macOS's for a scroll posted in lines.
LINE_PX = 10


def main(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    root.title("Fixture Desk")
    width, height = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{width}x{height}+0+0")
    root.attributes("-fullscreen", True)
    root.configure(bg="#f0f0f0")
    state: dict = {"name": "", "notes": "", "keys": [], "saved_count": 0, "subscribed": False, "selected": None,
                   "scroll_top": 0.0, "cells": {}, "right_clicks": 0, "double_clicks": 0, "drags": [],
                   "wheel": {"up": 0, "down": 0, "left": 0, "right": 0}, "clipboard": None, "loaded": None,
                   "screen": [width, height], "windowing": root.tk.call("tk", "windowingsystem")}

    def save() -> None:
        part = state_dir / ".state.json.part"
        part.write_text(json.dumps(state))
        os.replace(part, state_dir / "state.json")

    if width < MIN_SIZE[0] or height < MIN_SIZE[1]:
        state["error"] = f"the screen is {width}x{height}; the fixture needs {MIN_SIZE[0]}x{MIN_SIZE[1]}"
        save()
        root.destroy()
        return

    font = ("DejaVu Sans", 16)
    tk.Label(root, text="Name:", font=font, bg="#f0f0f0").place(x=40, y=40)
    name = tk.Entry(root, font=font, width=30)
    name.place(x=140, y=40)

    def key(event) -> None:
        state["name"] = name.get()
        state["keys"] = (state["keys"] + [KEYSYMS.get(event.keysym, event.keysym)])[-100:]
        state["clipboard"] = _clipboard(root)
        save()

    name.bind("<KeyRelease>", key)

    def select_all(event) -> str:
        name.selection_range(0, "end")
        name.icursor("end")
        return "break"

    # Tk binds Ctrl+/ to select-all on X11 alone; Aqua's is Cmd+A.
    name.bind("<Control-slash>", select_all)

    def on_save() -> None:
        state["saved_count"] += 1
        save()

    button = tk.Button(root, text="Save", font=font, command=on_save, width=8)
    button.place(x=560, y=34)

    subscribed = tk.BooleanVar(value=False)

    def on_subscribe() -> None:
        state["subscribed"] = subscribed.get()
        save()

    subscribe = tk.Checkbutton(root, text="Subscribe", font=font, variable=subscribed, command=on_subscribe,
                               bg="#f0f0f0")
    subscribe.place(x=40, y=100)

    notes = tk.Text(root, font=("DejaVu Sans", 14), width=28, height=3)
    notes.place(x=760, y=40)

    def note(event) -> None:
        state["notes"] = notes.get("1.0", "end-1c")
        save()

    notes.bind("<KeyRelease>", note)

    frame = tk.Frame(root)
    frame.place(x=40, y=160, width=300, height=400)
    listbox = tk.Listbox(frame, font=("DejaVu Sans", 14), exportselection=False)
    listbox.pack(side="left", fill="both", expand=True)
    bar = tk.Scrollbar(frame, command=listbox.yview)
    bar.pack(side="right", fill="y")

    def scrolled(first, last) -> None:
        bar.set(first, last)
        state["scroll_top"] = float(first)
        save()

    listbox.configure(yscrollcommand=scrolled)
    for i in range(1, 101):
        listbox.insert("end", f"Item {i}")

    def select(event) -> None:
        chosen = listbox.curselection()
        state["selected"] = listbox.get(chosen[0]) if chosen else None
        save()

    listbox.bind("<<ListboxSelect>>", select)

    grid = tk.Canvas(root, width=GRID * CELL, height=GRID * CELL, bg="#ffffff", highlightthickness=0)
    grid.place(x=400, y=160)
    for row in range(GRID):
        for col in range(GRID):
            grid.create_rectangle(col * CELL, row * CELL, (col + 1) * CELL, (row + 1) * CELL, fill=COLOURS[0],
                                  outline="#999999", tags=(f"cell-{row}-{col}",))

    def cell(event, right: bool) -> None:
        row, col = event.y // CELL, event.x // CELL
        count = (state["cells"].get(f"{row},{col}", 0) + 1) % len(COLOURS)
        state["cells"][f"{row},{col}"] = count
        grid.itemconfigure(f"cell-{row}-{col}", fill=COLOURS[count])
        state["right_clicks"] += right
        save()

    def double(event) -> None:
        state["double_clicks"] += 1
        save()

    grid.bind("<Button-1>", lambda e: cell(e, False))
    grid.bind("<Button-3>", lambda e: cell(e, True))
    grid.bind("<Double-Button-1>", double)

    pad = tk.Canvas(root, width=400, height=300, bg="#fffbe6", highlightthickness=0)
    pad.place(x=840, y=160)
    stroke: list = []

    def press(event) -> None:
        stroke[:] = [(event.x, event.y)]

    def motion(event) -> None:
        if stroke:
            pad.create_line(*stroke[-1], event.x, event.y, width=3)
            stroke.append((event.x, event.y))

    def release(event) -> None:
        if stroke:
            stroke.append((event.x, event.y))
            state["drags"] = (state["drags"] + [{"start": stroke[0], "end": stroke[-1], "points": len(stroke),
                                                 "path": stroke[::max(1, len(stroke) // 50)] + stroke[-1:]}])[-20:]
            save()
            stroke.clear()

    pad.bind("<ButtonPress-1>", press)
    pad.bind("<B1-Motion>", motion)
    pad.bind("<ButtonRelease-1>", release)

    wheel = tk.Canvas(root, width=200, height=120, bg="#e6f0ff", highlightthickness=0)
    wheel.place(x=840, y=500)

    def wheeled(horizontal: bool, steps: int) -> None:
        way = ("left" if horizontal else "up") if steps > 0 else ("right" if horizontal else "down")
        state["wheel"][way] += abs(steps)
        save()

    def scrolled(delta: int) -> None:
        if steps := touchpad_steps(delta):
            wheeled(*steps)

    # Tk 8.6 on X11 reports the wheel as buttons 4 and 5, Aqua, Windows and Tk 9 as <MouseWheel>, a horizontal scroll
    # (X's buttons 6 and 7) as a vertical one with Shift held. Tk 9 reports a scroll in pixels, a trackpad's or one a
    # driver posts as Aqua's do, as <TouchpadScroll>.
    if root.tk.call("tk", "windowingsystem") == "x11" and tk.TkVersion < 8.7:
        wheel.bind("<Button-4>", lambda e: wheeled(bool(e.state & 1), 1))
        wheel.bind("<Button-5>", lambda e: wheeled(bool(e.state & 1), -1))
    else:
        wheel.bind("<MouseWheel>", lambda e: wheeled(bool(e.state & 1), wheel_steps(e.delta)))
    if tk.TkVersion >= 8.7:
        wheel.bind("<TouchpadScroll>", lambda e: scrolled(e.delta))

    field = tk.Canvas(root, width=48, height=48, bg="#000000", highlightthickness=0, bd=0)
    field.place(x=1080, y=500)
    for x, y, colour in MARKERS:
        field.create_rectangle(x, y, x + 1, y + 1, fill=colour, outline="", width=0)

    def poll_inbox() -> None:
        message = state_dir / "inbox.txt"
        if message.exists() and (text := message.read_text()) != state["loaded"]:
            state["loaded"] = text
            save()
        root.after(200, poll_inbox)

    def write_layout() -> None:
        root.update_idletasks()
        boxes = {key: [w.winfo_rootx(), w.winfo_rooty(), w.winfo_width(), w.winfo_height()]
                 for key, w in (("name", name), ("save", button), ("subscribe", subscribe), ("notes", notes),
                                ("list", listbox), ("grid", grid), ("pad", pad), ("wheel", wheel), ("field", field))}
        boxes["list_row_height"] = listbox.bbox(0)[3] if listbox.bbox(0) else 0
        boxes["markers"] = [[field.winfo_rootx() + x, field.winfo_rooty() + y, colour] for x, y, colour in MARKERS]
        boxes["cell"] = CELL
        boxes["clipboard"] = CLIPBOARD
        save()
        (state_dir / "layout.json").write_text(json.dumps(boxes))

    root.clipboard_clear()
    root.clipboard_append(CLIPBOARD)
    state["clipboard"] = CLIPBOARD
    root.after(300, write_layout)
    root.after(500, poll_inbox)
    root.mainloop()


def wheel_steps(delta: int) -> int:
    """The wheel steps a <MouseWheel> event's ``delta`` holds, up positive: whole notches of 120 where Tk scales them
    so (Windows, and Tk 9 everywhere), else one a unit (Aqua's Tk 8.6)."""
    return int(delta / NOTCH_DELTA) if abs(delta) >= NOTCH_DELTA else delta


def touchpad_steps(delta: int) -> tuple[bool, int] | None:
    """The steps a <TouchpadScroll> event's ``delta`` holds, as horizontal or not and up or left positive, or None
    when it moves nothing. A step is a line, LINE_PX, and macOS may coalesce several into one event."""
    dx, dy = delta >> Y_BITS, ctypes.c_int16(delta).value
    if not (dx or dy):
        return None
    distance = dy or dx
    return not dy, (1 if distance > 0 else -1) * max(1, round(abs(distance) / LINE_PX))


def _clipboard(root: tk.Tk) -> str | None:
    try:
        return root.clipboard_get()
    except tk.TclError:
        return None


if __name__ == "__main__":
    main(Path(sys.argv[1]))
