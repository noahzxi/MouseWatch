"""Finer motion beside the dataset trace: path, jitter, and reversals.

The dataset file keeps the original seven columns. This sidecar keeps the
wobble inside each slice, including slices shorter than the dataset minimum,
so the personal profile can learn a user's small jitter.
"""

from __future__ import annotations

from pathlib import Path

from mousewatch.collector.buffer import Record

HEADER = (
    "user_id;when_ns;saved;duration_ms;angle;distance;velocity;path_px;jitter_px;"
    "reversals;samples;accel;drag;button;approach_px_s;slowdown;click_gap_ms;wheel_notches"
)


def motion_path_for(trace_path: Path) -> Path:
    return trace_path.with_name(trace_path.stem + ".motion.csv")


def format_motion(record: Record) -> str:
    duration = "" if record.duration_ms is None else str(record.duration_ms)
    angle = "" if record.angle is None else str(record.angle)
    distance = "" if record.distance_px is None else str(record.distance_px)
    velocity = "" if record.velocity_px_s is None else str(record.velocity_px_s)
    return (
        f"{record.user_id};{record.when_ns};{int(record.saved)};{duration};{angle};"
        f"{distance};{velocity};{record.path_px:.3f};{record.jitter_px:.3f};"
        f"{record.reversals};{record.samples};{record.accel:.3f};{int(record.drag)};"
        f"{record.button};{record.approach_px_s:.3f};{record.slowdown:.3f};"
        f"{record.click_gap_ms:.3f};{record.wheel_notches:.3f}\n"
    )


class MotionWriter:
    """Append-only sidecar. One row for every slice, including tiny jitter."""

    def __init__(self, path: Path, flush_every: int = 50) -> None:
        if flush_every < 1:
            raise ValueError("flush_every must be positive")
        self._flush_every = flush_every
        self._pending: list[str] = []
        path.parent.mkdir(parents=True, exist_ok=True)
        is_empty = not path.exists() or path.stat().st_size == 0
        self._file = path.open("a", encoding="utf-8", newline="\n")
        if is_empty:
            self._file.write(HEADER + "\n")
            self._file.flush()

    def write(self, record: Record) -> None:
        self._pending.append(format_motion(record))
        if len(self._pending) >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        if self._file is None or not self._pending:
            return
        self._file.writelines(self._pending)
        self._pending.clear()
        self._file.flush()

    def close(self) -> None:
        if self._file is None:
            return
        self.flush()
        self._file.close()
        self._file = None


def load_motion_actions(path: Path, user_id: str):
    """Rows for one user, in collector order, including slices the dataset dropped."""
    import numpy as np

    from pointerprint.dataset import TICKS_PER_SECOND
    from pointerprint.personal import MOTION_DTYPE

    rows: list[tuple] = []
    with path.open(encoding="utf-8", newline="") as handle:
        header = handle.readline().strip().split(";")
        if header != HEADER.split(";"):
            raise ValueError(f"unexpected motion columns: {header}")
        for line in handle:
            parts = line.rstrip("\n").split(";")
            if len(parts) != 18 or parts[0] != user_id:
                continue
            rows.append(parts)
    actions = np.zeros(len(rows), dtype=MOTION_DTYPE)
    button_code = {"": 0, "left": 1, "right": 2, "middle": 3, "wheel": 4}
    for index, parts in enumerate(rows):
        (
            _user,
            when_ns,
            _saved,
            duration,
            angle,
            distance,
            velocity,
            path,
            jitter,
            reversals,
            samples,
            accel,
            drag,
            button,
            approach,
            slowdown,
            click_gap,
            wheel,
        ) = parts
        actions["t"][index] = int(when_ns) // (1_000_000_000 // TICKS_PER_SECOND)
        if duration == "" and button != "wheel":
            actions["stroke"][index] = 1
            actions["angle"][index] = float(angle or 0)
            actions["distance"][index] = float(distance or 0)
            actions["velocity"][index] = float(velocity or 0)
            actions["duration"][index] = np.nan
        elif duration != "":
            actions["duration"][index] = float(duration)
        actions["path"][index] = float(path or 0)
        actions["jitter"][index] = float(jitter or 0)
        actions["reversals"][index] = float(reversals or 0)
        actions["samples"][index] = float(samples or 0)
        actions["accel"][index] = float(accel or 0)
        actions["drag"][index] = float(drag or 0)
        actions["button"][index] = button_code.get(button, 0)
        actions["approach"][index] = float(approach or 0)
        actions["slowdown"][index] = float(slowdown or 0)
        actions["click_gap"][index] = float(click_gap or 0)
        actions["wheel"][index] = float(wheel or 0)
    return actions
