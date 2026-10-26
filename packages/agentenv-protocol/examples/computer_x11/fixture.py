"""The Fixture Desk: a Tk window whose every effect is recorded in ~/fixture/state.json, and whose widgets' screen
boxes are written to ~/fixture/layout.json, so a test can aim at them and read back what reached them."""
import json
import os
import tkinter as tk
from pathlib import Path

HOME = Path.home() / "fixture"
HOME.mkdir(exist_ok=True)
COLOURS = ("#ffffff", "#ff0000", "#00aa00", "#0000ff")

root = tk.Tk()
root.title("Fixture Desk")
root.geometry("1280x800+0+0")
root.configure(bg="#f0f0f0")
state = {"name": "", "saved_count": 0, "subscribed": False, "selected": None, "cells": {}, "drags": [],
         "keys": [], "scroll_top": 0.0, "right_clicks": 0, "double_clicks": 0, "loaded": None}


def save_state():
    tmp = HOME / ".state.json.part"
    tmp.write_text(json.dumps(state))
    os.replace(tmp, HOME / "state.json")


tk.Label(root, text="Name:", font=("DejaVu Sans", 16), bg="#f0f0f0").place(x=40, y=40)
name = tk.Entry(root, font=("DejaVu Sans", 16), width=30)
name.place(x=140, y=40)


def on_key(event):
    state["name"] = name.get()
    state["keys"] = (state["keys"] + [event.keysym])[-50:]
    save_state()


name.bind("<KeyRelease>", on_key)


def on_save():
    state["name"] = name.get()
    state["saved_count"] += 1
    save_state()


save = tk.Button(root, text="Save", font=("DejaVu Sans", 16), command=on_save, width=8)
save.place(x=560, y=34)

subscribed = tk.BooleanVar(value=False)


def on_subscribe():
    state["subscribed"] = subscribed.get()
    save_state()


subscribe = tk.Checkbutton(root, text="Subscribe", font=("DejaVu Sans", 16), variable=subscribed,
                           command=on_subscribe, bg="#f0f0f0")
subscribe.place(x=40, y=100)

frame = tk.Frame(root)
frame.place(x=40, y=160, width=300, height=400)
listbox = tk.Listbox(frame, font=("DejaVu Sans", 14), exportselection=False)
listbox.pack(side="left", fill="both", expand=True)
bar = tk.Scrollbar(frame, command=listbox.yview)
bar.pack(side="right", fill="y")


def on_scroll(first, last):
    bar.set(first, last)
    state["scroll_top"] = float(first)
    save_state()


listbox.configure(yscrollcommand=on_scroll)
for i in range(1, 101):
    listbox.insert("end", f"Item {i}")


def on_select(event):
    chosen = listbox.curselection()
    state["selected"] = listbox.get(chosen[0]) if chosen else None
    save_state()


listbox.bind("<<ListboxSelect>>", on_select)

grid = tk.Canvas(root, width=400, height=400, bg="#ffffff", highlightthickness=0)
grid.place(x=400, y=160)
CELL = 50
for row in range(8):
    for col in range(8):
        grid.create_rectangle(col * CELL, row * CELL, (col + 1) * CELL, (row + 1) * CELL, fill=COLOURS[0],
                              outline="#999999", tags=(f"cell-{row}-{col}",))


def on_cell(event, button):
    row, col = event.y // CELL, event.x // CELL
    key = f"{row},{col}"
    count = (state["cells"].get(key, 0) + 1) % len(COLOURS)
    state["cells"][key] = count
    grid.itemconfigure(f"cell-{row}-{col}", fill=COLOURS[count])
    if button == 3:
        state["right_clicks"] += 1
    save_state()


def on_double(event):
    state["double_clicks"] += 1
    save_state()


grid.bind("<Button-1>", lambda e: on_cell(e, 1))
grid.bind("<Button-3>", lambda e: on_cell(e, 3))
grid.bind("<Double-Button-1>", on_double)

pad = tk.Canvas(root, width=400, height=400, bg="#fffbe6", highlightthickness=1)
pad.place(x=840, y=160)
stroke = []


def on_press(event):
    stroke.clear()
    stroke.append((event.x, event.y))


def on_motion(event):
    if stroke:
        pad.create_line(stroke[-1][0], stroke[-1][1], event.x, event.y, width=3)
        stroke.append((event.x, event.y))


def on_release(event):
    if stroke:
        stroke.append((event.x, event.y))
        state["drags"] = (state["drags"] + [{"start": stroke[0], "end": stroke[-1], "points": len(stroke)}])[-20:]
        save_state()
        stroke.clear()


pad.bind("<ButtonPress-1>", on_press)
pad.bind("<B1-Motion>", on_motion)
pad.bind("<ButtonRelease-1>", on_release)

loaded = tk.Label(root, text="", font=("DejaVu Sans", 14), bg="#f0f0f0")
loaded.place(x=40, y=600)


def poll_loaded():
    path = Path.home() / "inbox" / "message.txt"
    if path.exists():
        text = path.read_text()
        if text != state["loaded"]:
            state["loaded"] = text
            loaded.configure(text=f"Loaded: {text}")
            save_state()
    root.after(200, poll_loaded)


def write_layout():
    root.update_idletasks()
    boxes = {}
    for key, widget in (("name", name), ("save", save), ("subscribe", subscribe), ("list", listbox),
                        ("grid", grid), ("pad", pad)):
        boxes[key] = [widget.winfo_rootx(), widget.winfo_rooty(), widget.winfo_width(), widget.winfo_height()]
    boxes["list_row_height"] = listbox.bbox(0)[3] if listbox.bbox(0) else 0
    (HOME / "layout.json").write_text(json.dumps(boxes))
    save_state()


root.after(300, write_layout)
root.after(500, poll_loaded)
root.mainloop()
