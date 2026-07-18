"""
Built-in steering test window (like a mini Notepad).

Opens automatically so you can see the text cursor move when you steer,
without fighting window focus or opening Windows Notepad.
"""

import threading
import time
import tkinter as tk
from typing import Optional

from steering import SteeringDirection


class SteeringTestWindow:
    """
    Small text window that moves its cursor left/right from steering input.

    Runs Tkinter on a background thread so OpenCV can stay on the main thread.
    """

    def __init__(self) -> None:
        self._root: Optional[tk.Tk] = None
        self._text: Optional[tk.Text] = None
        self._ready = threading.Event()
        self._last_move = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _run(self) -> None:
        """Create the Tkinter window and start its event loop."""
        self._root = tk.Tk()
        self._root.title("Steering Test Window")
        self._root.geometry("620x160")
        self._root.resizable(False, False)

        label = tk.Label(
            self._root,
            text="Tilt your hands — this cursor moves automatically (built-in test)",
            font=("Segoe UI", 10),
        )
        label.pack(pady=(8, 0))

        self._text = tk.Text(self._root, font=("Consolas", 14), height=4, wrap="word")
        self._text.pack(fill="both", expand=True, padx=10, pady=10)
        self._text.insert(
            "1.0",
            "Steering test line: move me with your hands ->|\n"
            "Left tilt = cursor left. Right tilt = cursor right.\n",
        )
        self._text.mark_set("insert", "1.32")

        self._ready.set()
        self._root.mainloop()

    def update(self, direction: SteeringDirection) -> None:
        """
        Move the text cursor when steering left or right.

        Called every frame from the main loop; movement is throttled for readability.
        """
        if not self._ready.is_set() or self._root is None or self._text is None:
            return

        if direction not in (SteeringDirection.LEFT, SteeringDirection.RIGHT):
            return

        now = time.perf_counter()
        if now - self._last_move < 0.06:
            return
        self._last_move = now

        if direction == SteeringDirection.LEFT:
            self._root.after(0, self._move_left)
        else:
            self._root.after(0, self._move_right)

    def _move_left(self) -> None:
        """Shift the insert cursor one character to the left."""
        if self._text is None:
            return
        self._text.mark_set("insert", self._text.index("insert-1c"))
        self._text.see("insert")

    def _move_right(self) -> None:
        """Shift the insert cursor one character to the right."""
        if self._text is None:
            return
        self._text.mark_set("insert", self._text.index("insert+1c"))
        self._text.see("insert")

    def close(self) -> None:
        """Close the test window when the app exits."""
        if self._root is not None:
            self._root.after(0, self._root.destroy)
