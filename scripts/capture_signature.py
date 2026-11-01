"""Sign with the mouse or a trackpad; Enter saves the strokes, C clears, Esc quits.

python scripts/capture_signature.py --out examples/my_signature.json
"""

import argparse
import json
import tkinter as tk
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("examples/my_signature.json"))
    args = ap.parse_args()

    strokes, current = [], []
    root = tk.Tk()
    root.title("Sign, then press Enter (C clears, Esc quits)")
    canvas = tk.Canvas(root, width=900, height=350, bg="white")
    canvas.pack()

    def press(ev):
        current.clear()
        current.append((ev.x, ev.y))

    def drag(ev):
        x0, y0 = current[-1]
        if (ev.x - x0) ** 2 + (ev.y - y0) ** 2 >= 4:
            canvas.create_line(x0, y0, ev.x, ev.y, width=3, capstyle=tk.ROUND, smooth=True)
            current.append((ev.x, ev.y))

    def release(_):
        if len(current) >= 2:
            strokes.append(list(current))

    def clear(_):
        strokes.clear()
        canvas.delete("all")

    def save(_):
        if not strokes:
            return
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"strokes": strokes}))
        print(f"saved {len(strokes)} strokes to {args.out}")
        root.destroy()

    canvas.bind("<ButtonPress-1>", press)
    canvas.bind("<B1-Motion>", drag)
    canvas.bind("<ButtonRelease-1>", release)
    root.bind("<Return>", save)
    root.bind("<KeyPress-c>", clear)
    root.bind("<Escape>", lambda _: root.destroy())
    root.mainloop()


if __name__ == "__main__":
    main()
