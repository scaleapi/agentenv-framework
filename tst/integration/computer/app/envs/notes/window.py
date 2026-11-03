"""The notes app's window, of class Notes, showing the signature its settings give."""
import tkinter as tk

from app import Notes

root = tk.Tk(className="notes")
root.title("Notes")
tk.Label(root, text=Notes().settings.signature).pack(padx=40, pady=40)
root.mainloop()
