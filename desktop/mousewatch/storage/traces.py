"""Append collector rows in the dataset.csv layout."""

from __future__ import annotations

import time
from pathlib import Path

from mousewatch.collector.buffer import Record

HEADER = "idman;duration;angle;distance;velocity;local_date;local_time"


def format_local_timestamp(when_ns: int) -> tuple[str, str]:
    """Local calendar date and time with 7 fractional digits, as in the dataset."""
    seconds, nanos = divmod(int(when_ns), 1_000_000_000)
    local = time.localtime(seconds)
    date = time.strftime("%Y-%m-%d", local)
    clock = time.strftime("%H:%M:%S", local)
    return date, f"{clock}.{nanos // 100:07d}"


def format_record(record: Record) -> str:
    date, clock = format_local_timestamp(record.when_ns)
    duration = "NULL" if record.duration_ms is None else str(record.duration_ms)
    angle = "NULL" if record.angle is None else str(record.angle)
    distance = "NULL" if record.distance_px is None else str(record.distance_px)
    velocity = "NULL" if record.velocity_px_s is None else str(record.velocity_px_s)
    return (
        f"{record.user_id};{duration};{angle};{distance};{velocity};{date};{clock}\n"
    )


class TraceWriter:
    """Buffered append-only writer. Creates the header when the file is empty."""

    def __init__(self, path: Path, flush_every: int = 50) -> None:
        if flush_every < 1:
            raise ValueError("flush_every must be positive")
        self._path = path
        self._flush_every = flush_every
        self._pending: list[str] = []
        self.rows_written = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        is_empty = not path.exists() or path.stat().st_size == 0
        self._file = path.open("a", encoding="utf-8", newline="\n")
        if is_empty:
            self._file.write(HEADER + "\n")
            self._file.flush()

    def write(self, record: Record) -> None:
        if self._file is None:
            raise RuntimeError("writer is closed")
        self._pending.append(format_record(record))
        self.rows_written += 1
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

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
