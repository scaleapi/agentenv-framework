"""The fixture the conformance probe stages with data/add and launches with computer-exec: a Tk window, the Fixture
Desk, whose widgets' screen boxes are written to <state-dir>/layout.json and every effect that reaches it, with
whatever lands in <state-dir>/inbox.txt, to <state-dir>/state.json, so the probe can aim at it and read back what
arrived. It records the same effects under X11, macOS's Aqua and Windows, whose Tk names keys, binds select-all and
reports the wheel differently. On X11 it stays above an app's windows (``keep_on_top``); elsewhere it's full screen.
Its layout and what it records are desk.py's, which the GTK desk shares. Run as ``python3 fixture.py <state-dir>``
beside desk.py; it needs a 1280x800 screen or larger."""
from __future__ import annotations

import ctypes
import sys
import time
import tkinter as tk
from pathlib import Path

from desk import (AT, CELL, CLIPBOARD, COLOURS, GRID, INBOX_EVERY_MS, ITEMS, MARKERS, MIN_SIZE, SIZE, new_state,
                  record_drag, record_key, write_layout, write_state)

#: X11's keysyms for the keys Aqua's Tk names by their character.
KEYSYMS = {",": "comma", " ": "space"}
#: A wheel notch's <MouseWheel> delta where Tk scales the wheel: Windows', and Tk 9's everywhere.
NOTCH_DELTA = 120
#: Tk 9 packs a <TouchpadScroll>'s distances into its delta: y, a signed 16-bit number, in the low bits, x above.
Y_BITS = 16
#: The pixels of a scrolled line, macOS's for a scroll posted in lines.
LINE_PX = 10
#: The least time between two of the fixture's raises over other windows, so it can't fight one that raises itself.
LIFT_GAP_S = 0.5


def main(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    root.title("Fixture Desk")
    width, height = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{width}x{height}+0+0")
    if root.tk.call("tk", "windowingsystem") == "x11":
        keep_on_top(root)
    else:
        root.attributes("-fullscreen", True)
    root.configure(bg="#f0f0f0")
    state = new_state(width, height, root.tk.call("tk", "windowingsystem"))

    def save() -> None:
        write_state(state_dir, state)

    if width < MIN_SIZE[0] or height < MIN_SIZE[1]:
        state["error"] = f"the screen is {width}x{height}; the fixture needs {MIN_SIZE[0]}x{MIN_SIZE[1]}"
        save()
        root.destroy()
        return

    font = ("DejaVu Sans", 16)
    _place(tk.Label(root, text="Name:", font=font, bg="#f0f0f0"), "label")
    name = tk.Entry(root, font=font, width=30)
    _place(name, "name")

    def key(event) -> None:
        state["name"] = name.get()
        record_key(state, KEYSYMS.get(event.keysym, event.keysym))
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
    _place(button, "save")

    subscribed = tk.BooleanVar(value=False)

    def on_subscribe() -> None:
        state["subscribed"] = subscribed.get()
        save()

    subscribe = tk.Checkbutton(root, text="Subscribe", font=font, variable=subscribed, command=on_subscribe,
                               bg="#f0f0f0")
    _place(subscribe, "subscribe")

    notes = tk.Text(root, font=("DejaVu Sans", 14), width=28, height=3)
    _place(notes, "notes")

    def note(event) -> None:
        state["notes"] = notes.get("1.0", "end-1c")
        save()

    notes.bind("<KeyRelease>", note)

    frame = tk.Frame(root)
    _place(frame, "list", **SIZE["list"])
    listbox = tk.Listbox(frame, font=("DejaVu Sans", 14), exportselection=False)
    listbox.pack(side="left", fill="both", expand=True)
    bar = tk.Scrollbar(frame, command=listbox.yview)
    bar.pack(side="right", fill="y")

    def scrolled(first, last) -> None:
        bar.set(first, last)
        state["scroll_top"] = float(first)
        save()

    listbox.configure(yscrollcommand=scrolled)
    for i in range(1, ITEMS + 1):
        listbox.insert("end", f"Item {i}")

    def select(event) -> None:
        chosen = listbox.curselection()
        state["selected"] = listbox.get(chosen[0]) if chosen else None
        save()

    listbox.bind("<<ListboxSelect>>", select)

    grid = tk.Canvas(root, width=GRID * CELL, height=GRID * CELL, bg="#ffffff", highlightthickness=0)
    _place(grid, "grid")
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

    pad = tk.Canvas(root, **SIZE["pad"], bg="#fffbe6", highlightthickness=0)
    _place(pad, "pad")
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
            record_drag(state, stroke)
            save()
            stroke.clear()

    pad.bind("<ButtonPress-1>", press)
    pad.bind("<B1-Motion>", motion)
    pad.bind("<ButtonRelease-1>", release)

    wheel = tk.Canvas(root, **SIZE["wheel"], bg="#e6f0ff", highlightthickness=0)
    _place(wheel, "wheel")

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

    field = tk.Canvas(root, **SIZE["field"], bg="#000000", highlightthickness=0, bd=0)
    _place(field, "field")
    for x, y, colour in MARKERS:
        field.create_rectangle(x, y, x + 1, y + 1, fill=colour, outline="", width=0)

    def poll_inbox() -> None:
        message = state_dir / "inbox.txt"
        if message.exists() and (text := message.read_text()) != state["loaded"]:
            state["loaded"] = text
            save()
        root.after(INBOX_EVERY_MS, poll_inbox)

    def layout() -> None:
        root.update_idletasks()
        boxes = {key: [w.winfo_rootx(), w.winfo_rooty(), w.winfo_width(), w.winfo_height()]
                 for key, w in (("name", name), ("save", button), ("subscribe", subscribe), ("notes", notes),
                                ("list", listbox), ("grid", grid), ("pad", pad), ("wheel", wheel), ("field", field))}
        save()
        write_layout(state_dir, boxes, listbox.bbox(0)[3] if listbox.bbox(0) else 0,
                     (field.winfo_rootx(), field.winfo_rooty()))

    root.clipboard_clear()
    root.clipboard_append(CLIPBOARD)
    state["clipboard"] = CLIPBOARD
    root.after(300, layout)
    root.after(500, poll_inbox)
    root.mainloop()


def _place(widget: tk.Widget, at: str, **size: int) -> None:
    widget.place(x=AT[at][0], y=AT[at][1], **size)


def keep_on_top(root: tk.Tk) -> None:
    """Keep the fixture over the windows an app opens, which would take the probe's clicks: maximized and above other
    windows rather than full screen, since xfwm4 keeps a window an app puts above over a full-screen one, and raised
    again, giving the focus back to the widget that last had it, whenever another window covers some of it, at most once
    each LIFT_GAP_S."""
    root.attributes("-zoomed", True)
    root.attributes("-topmost", True)
    seen: dict = {"state": "VisibilityUnobscured", "lifted": 0.0, "pending": None}

    def lift() -> None:
        seen["pending"] = None
        if seen["state"] == "VisibilityUnobscured":
            return
        seen["lifted"] = time.monotonic()
        # Taking the layer above again restacks it on top, which a raise request alone doesn't get from xfwm4.
        root.attributes("-topmost", False)
        root.attributes("-topmost", True)
        root.lift()
        # The widget that last had the focus gets it back, so keys the probe types into a text box still reach it.
        (root.focus_lastfor() or root).focus_force()

    def visibility(event: tk.Event) -> None:
        if event.widget is not root:
            return
        seen["state"] = event.state
        if event.state != "VisibilityUnobscured" and seen["pending"] is None:
            wait = max(0.0, seen["lifted"] + LIFT_GAP_S - time.monotonic())
            seen["pending"] = root.after(round(wait * 1000), lift)

    root.bind("<Visibility>", visibility, add="+")


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
