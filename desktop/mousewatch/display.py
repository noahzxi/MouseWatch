"""Small always-on-top window for the live you / not-you verdict."""

from __future__ import annotations

import queue
import threading

from mousewatch.identity import IdentityStatus

_COLORS = {
    "you": "#1f8a4c",
    "not_you": "#b42318",
    "waiting": "#3f3f46",
}


class StatusWindow:
    """Show the latest identity verdict without blocking the collector."""

    def __init__(self, on_close=None) -> None:
        self._on_close = on_close
        self._updates: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        try:
            import tkinter  # noqa: F401
        except ImportError:
            return False
        self._thread = threading.Thread(target=self._run, name="mousewatch-display", daemon=True)
        self._thread.start()
        return True

    def update(self, status: IdentityStatus) -> None:
        self._updates.put(("update", status))

    def close(self) -> None:
        self._updates.put(("close",))
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        import tkinter as tk

        root = tk.Tk()
        root.title("MouseWatch")
        root.geometry("420x150+48+48")
        root.resizable(False, False)
        root.attributes("-topmost", True)
        root.configure(bg="white")
        headline = tk.Label(
            root,
            text="MouseWatch",
            font=("Segoe UI", 22, "bold"),
            bg="white",
            fg=_COLORS["waiting"],
        )
        headline.pack(fill="x", padx=18, pady=(22, 4))
        detail = tk.Label(
            root,
            text="",
            font=("Segoe UI", 10),
            bg="white",
            fg="#3f3f46",
            wraplength=380,
            justify="left",
        )
        detail.pack(fill="x", padx=18)

        def finish() -> None:
            if self._on_close is not None:
                self._on_close()
            root.destroy()

        def poll() -> None:
            if not root.winfo_exists():
                return
            try:
                while True:
                    item = self._updates.get_nowait()
                    if item[0] == "close":
                        root.destroy()
                        return
                    status = item[1]
                    headline.config(text=status.headline, fg=_COLORS.get(status.tone, _COLORS["waiting"]))
                    detail.config(text=status.detail)
            except queue.Empty:
                pass
            root.after(150, poll)

        root.protocol("WM_DELETE_WINDOW", finish)
        poll()
        root.mainloop()
