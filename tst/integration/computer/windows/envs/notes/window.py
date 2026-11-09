"""The notes app's window, titled Notes, showing the signature its settings give."""
import tkinter as tk

from app import Notes

root = tk.Tk()
root.title("Notes")
tk.Label(root, text=Notes().settings.signature).pack(padx=40, pady=40)
root.mainloop()
