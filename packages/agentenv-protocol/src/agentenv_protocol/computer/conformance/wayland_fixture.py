"""The probe's fixture on a Wayland desktop: fixture.py's Fixture Desk drawn with GTK 3 as a native Wayland client,
since Tk runs only under XWayland, whose windows a Wayland driver may not reach. Same widgets at the same places, and
the same layout.json and state.json, all from desk.py, which is staged beside it. Wayland tells a client
nothing of where its window is, so it goes full screen and takes its origin from Hyprland's IPC when it answers; the
clipboard goes through wl-clipboard, which needs no focus. Run as ``python3 wayland_fixture.py <state-dir>``; it needs
a 1280x800 screen or larger and PyGObject."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import gi

gi.require_version("Gdk", "3.0")
gi.require_version("Gtk", "3.0")
from gi.repository import Gdk, GLib, Gtk  # noqa: E402

from desk import (AT, CELL, CLIPBOARD, COLOURS, GRID, INBOX_EVERY_MS, ITEMS, MARKERS, MIN_SIZE, SIZE,  # noqa: E402
                  new_state, record_drag, record_key, write_layout, write_state)

#: How often the fixture asks Hyprland where its window is until it's there, and for how long.
PLACE_EVERY_MS, PLACE_TRIES = 100, 50
#: How long the fixture waits for hyprctl to answer, for wl-copy to take the clipboard and for wl-paste to read it.
HYPRCTL_TIMEOUT_S, COPY_TIMEOUT_S, PASTE_TIMEOUT_S = 2, 5, 2
FONT = "DejaVu Sans"


def main(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    GLib.set_prgname("FixtureDesk")
    monitor = Gdk.Display.get_default().get_monitor(0).get_geometry()
    width, height = monitor.width, monitor.height
    state = new_state(width, height, "wayland")

    def save() -> None:
        write_state(state_dir, state)

    if width < MIN_SIZE[0] or height < MIN_SIZE[1]:
        state["error"] = f"the screen is {width}x{height}; the fixture needs {MIN_SIZE[0]}x{MIN_SIZE[1]}"
        save()
        return

    css = Gtk.CssProvider()
    css.load_from_data(f"* {{ font-family: '{FONT}'; font-size: 16px; }} textview, treeview {{ font-size: 14px; }}"
                       f" window {{ background-color: #f0f0f0; }}".encode())
    Gtk.StyleContext.add_provider_for_screen(Gdk.Screen.get_default(), css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
    window = Gtk.Window(title="Fixture Desk")
    window.set_decorated(False)
    window.set_default_size(width, height)
    desk = Gtk.Fixed()
    window.add(desk)

    desk.put(Gtk.Label(label="Name:"), *AT["label"])
    name = Gtk.Entry(width_chars=30)
    desk.put(name, *AT["name"])

    def key(_widget, event) -> bool:
        state["name"] = name.get_text()
        record_key(state, Gdk.keyval_name(event.keyval))
        state["clipboard"] = _clipboard()
        save()
        return False

    name.connect("key-release-event", key)

    def on_save(_button) -> None:
        state["saved_count"] += 1
        save()

    button = Gtk.Button(label="Save")
    button.set_size_request(120, -1)
    button.connect("clicked", on_save)
    desk.put(button, *AT["save"])

    def on_subscribe(check) -> None:
        state["subscribed"] = check.get_active()
        save()

    subscribe = Gtk.CheckButton(label="Subscribe")
    subscribe.connect("toggled", on_subscribe)
    desk.put(subscribe, *AT["subscribe"])

    notes = Gtk.TextView()
    notes.set_size_request(340, 80)
    notes.set_accepts_tab(True)

    def note(_widget, _event) -> bool:
        buffer = notes.get_buffer()
        state["notes"] = buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True)
        save()
        return False

    notes.connect("key-release-event", note)
    desk.put(notes, *AT["notes"])

    items = Gtk.ListStore(str)
    for i in range(1, ITEMS + 1):
        items.append([f"Item {i}"])
    listing = Gtk.TreeView(model=items, headers_visible=False)
    listing.append_column(Gtk.TreeViewColumn("Item", Gtk.CellRendererText(), text=0))
    scroller = Gtk.ScrolledWindow()
    scroller.set_size_request(SIZE["list"]["width"], SIZE["list"]["height"])
    scroller.add(listing)
    desk.put(scroller, *AT["list"])

    def scrolled(adjustment) -> None:
        upper = adjustment.get_upper()
        state["scroll_top"] = adjustment.get_value() / upper if upper else 0.0
        save()

    scroller.get_vadjustment().connect("value-changed", scrolled)

    def select(selection) -> None:
        model, row = selection.get_selected()
        state["selected"] = model[row][0] if row else None
        save()

    listing.get_selection().connect("changed", select)

    grid = _area(GRID * CELL, GRID * CELL)
    desk.put(grid, *AT["grid"])

    def draw_grid(_area_, cr) -> None:
        for row in range(GRID):
            for col in range(GRID):
                cr.set_source_rgb(*_rgb(COLOURS[state["cells"].get(f"{row},{col}", 0)]))
                cr.rectangle(col * CELL, row * CELL, CELL, CELL)
                cr.fill()
                cr.set_source_rgb(*_rgb("#999999"))
                cr.set_line_width(1)
                cr.rectangle(col * CELL + 0.5, row * CELL + 0.5, CELL - 1, CELL - 1)
                cr.stroke()

    def click(_area_, event) -> bool:
        if event.type == Gdk.EventType._2BUTTON_PRESS:
            state["double_clicks"] += 1
        elif event.type == Gdk.EventType.BUTTON_PRESS and event.button in (1, 3):
            row, col = int(event.y) // CELL, int(event.x) // CELL
            state["cells"][f"{row},{col}"] = (state["cells"].get(f"{row},{col}", 0) + 1) % len(COLOURS)
            state["right_clicks"] += event.button == 3
            grid.queue_draw()
        save()
        return True

    grid.connect("draw", draw_grid)
    grid.connect("button-press-event", click)

    pad = _area(**SIZE["pad"])
    desk.put(pad, *AT["pad"])
    stroke: list = []
    drawn: list = []

    def draw_pad(_area_, cr) -> None:
        cr.set_source_rgb(*_rgb("#fffbe6"))
        cr.paint()
        cr.set_source_rgb(0, 0, 0)
        cr.set_line_width(3)
        for line in drawn + [stroke]:
            for (x0, y0), (x1, y1) in zip(line, line[1:]):
                cr.move_to(x0, y0)
                cr.line_to(x1, y1)
                cr.stroke()

    def press(_area_, event) -> bool:
        if event.button == 1:
            stroke[:] = [(round(event.x), round(event.y))]
        return True

    def motion(_area_, event) -> bool:
        if stroke:
            stroke.append((round(event.x), round(event.y)))
            pad.queue_draw()
        return True

    def release(_area_, event) -> bool:
        if stroke and event.button == 1:
            stroke.append((round(event.x), round(event.y)))
            record_drag(state, stroke)
            drawn.append(list(stroke))
            stroke.clear()
            save()
        return True

    pad.connect("draw", draw_pad)
    pad.connect("button-press-event", press)
    pad.connect("motion-notify-event", motion)
    pad.connect("button-release-event", release)

    wheel = _area(**SIZE["wheel"])
    wheel.add_events(Gdk.EventMask.SMOOTH_SCROLL_MASK)
    desk.put(wheel, *AT["wheel"])

    def wheeled(_area_, event) -> bool:
        """A step a scroll event: by its smooth deltas, since GTK 3 on Wayland names a horizontal wheel's discrete
        direction the wrong way round, a step left RIGHT, where its deltas and GTK 4 have it left."""
        _, dx, dy = event.get_scroll_deltas()
        for way, delta in (("left" if dx < 0 else "right", dx), ("up" if dy < 0 else "down", dy)):
            if delta:
                state["wheel"][way] += 1
        save()
        return True

    wheel.connect("draw", lambda _a, cr: (cr.set_source_rgb(*_rgb("#e6f0ff")), cr.paint()))
    wheel.connect("scroll-event", wheeled)

    field = _area(**SIZE["field"])
    desk.put(field, *AT["field"])

    def draw_field(_area_, cr) -> None:
        cr.set_source_rgb(0, 0, 0)
        cr.paint()
        for x, y, colour in MARKERS:
            cr.set_source_rgb(*_rgb(colour))
            cr.rectangle(x, y, 1, 1)
            cr.fill()

    field.connect("draw", draw_field)

    def poll_inbox() -> bool:
        message = state_dir / "inbox.txt"
        if message.exists() and (text := message.read_text()) != state["loaded"]:
            state["loaded"] = text
            save()
        return True

    tries = [0]

    def layout() -> bool:
        tries[0] += 1
        origin = _origin()
        if origin is None and tries[0] < PLACE_TRIES:
            return True
        ox, oy = origin or (0, 0)
        boxes = {}
        for key_, widget in (("name", name), ("save", button), ("subscribe", subscribe), ("notes", notes),
                             ("list", listing), ("grid", grid), ("pad", pad), ("wheel", wheel), ("field", field)):
            x, y = widget.translate_coordinates(window, 0, 0)
            allocation = widget.get_allocation()
            boxes[key_] = [ox + x, oy + y, allocation.width, allocation.height]
        first = listing.get_cell_area(Gtk.TreePath.new_first(), listing.get_column(0))
        fx, fy = field.translate_coordinates(window, 0, 0)
        save()
        write_layout(state_dir, boxes, first.height, (ox + fx, oy + fy))
        return False

    _set_clipboard(CLIPBOARD)
    state["clipboard"] = CLIPBOARD
    window.connect("destroy", Gtk.main_quit)
    window.show_all()
    window.fullscreen()
    GLib.timeout_add(PLACE_EVERY_MS, layout)
    GLib.timeout_add(INBOX_EVERY_MS, poll_inbox)
    Gtk.main()


def _area(width: int, height: int) -> Gtk.DrawingArea:
    area = Gtk.DrawingArea()
    area.set_size_request(width, height)
    area.add_events(Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK
                    | Gdk.EventMask.BUTTON1_MOTION_MASK | Gdk.EventMask.SCROLL_MASK)
    return area


def _origin() -> tuple[int, int] | None:
    """Where Hyprland put this process's full-screen window, once it's full screen; None until then, or without
    Hyprland."""
    try:
        clients = json.loads(subprocess.run(["hyprctl", "-j", "clients"], capture_output=True, text=True,
                                            timeout=HYPRCTL_TIMEOUT_S).stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    mine = [c for c in clients if c.get("pid") == os.getpid() and c.get("title") == "Fixture Desk"]
    return tuple(mine[0]["at"]) if mine and mine[0].get("fullscreen") else None


def _rgb(colour: str) -> tuple[float, float, float]:
    return tuple(int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5))


def _set_clipboard(text: str) -> None:
    try:
        subprocess.run(["wl-copy", "--", text], check=True, timeout=COPY_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD).set_text(text, -1)


def _clipboard() -> str | None:
    try:
        return subprocess.run(["wl-paste", "--no-newline"], capture_output=True, text=True,
                              timeout=PASTE_TIMEOUT_S).stdout
    except (OSError, subprocess.SubprocessError):
        return Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD).wait_for_text()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
