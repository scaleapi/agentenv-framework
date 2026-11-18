"""The notes app's window, titled Notes: GTK 3 on Omarchy's own Python."""
import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk

window = Gtk.Window(title="Notes")
window.add(Gtk.Label(label="Notes"))
window.connect("destroy", Gtk.main_quit)
window.show_all()
Gtk.main()
