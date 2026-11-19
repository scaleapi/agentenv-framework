"""Record the startup accessibility setting, then persist an AX button activation."""
import json
import os
from pathlib import Path

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gio, Gtk

folder = Path.home() / ".hcu-accessibility"
folder.mkdir(exist_ok=True)
started = {"pid": os.getpid(),
           "toolkit_accessibility": Gio.Settings.new("org.gnome.desktop.interface").get_boolean("toolkit-accessibility")}
(folder / "startup.json").write_text(json.dumps(started))
window = Gtk.Window(title="HCU accessibility fixture")
window.set_wmclass("hcu-accessibility", "HcuAccessibility")
window.set_default_size(420, 180)
window.connect("destroy", Gtk.main_quit)
box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
box.set_border_width(24)
label = Gtk.Label(label="Waiting for HCU")
button = Gtk.Button(label="Confirm HCU action")


def confirmed(_button):
    (folder / "result.json").write_text(json.dumps({"clicked": True, "pid": os.getpid()}))
    label.set_text("HCU action confirmed")


button.connect("clicked", confirmed)
box.pack_start(label, True, True, 0)
box.pack_start(button, True, True, 0)
window.add(box)
window.show_all()
Gtk.main()
