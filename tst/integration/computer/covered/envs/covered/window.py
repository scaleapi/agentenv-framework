"""The covered app's window, of class Covered: half the screen's size over its middle, kept above other windows."""
import tkinter as tk

root = tk.Tk(className="covered")
root.title("Covered")
width, height = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry(f"{width // 2}x{height // 2}+{width // 4}+{height // 4}")
root.attributes("-topmost", True)
tk.Label(root, text="Kept above").pack(expand=True)
root.mainloop()
