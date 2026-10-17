"""Load pointer actions and cut each user into an early train slice and a later test slice."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# 100-nanosecond ticks since the Unix epoch. The source file stores 7 fractional digits.
TICKS_PER_SECOND = 10_000_000

ACTION_DTYPE = np.dtype(
    [
        ("t", np.int64),
        ("stroke", np.uint8),
        ("angle", np.float32),
        ("distance", np.float32),
        ("velocity", np.float32),
        ("duration", np.float32),
    ]
)

COLUMNS = (
    "idman",
    "duration",
    "angle",
    "distance",
    "velocity",
    "local_date",
    "local_time",
)


def civil_days(year: int, month: int, day: int) -> int:
    """Days since 1970-01-01."""
    year -= month <= 2
    era = (year if year >= 0 else year - 399) // 400
    yoe = year - era * 400
    doy = (153 * (month + (-3 if month > 2 else 9)) + 2) // 5 + day - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def parse_timestamp(date: str, time: str) -> int:
    """Parse `YYYY-MM-DD` and `HH:MM:SS.fffffff` into 100-nanosecond ticks."""
    year = int(date[0:4])
    month = int(date[5:7])
    day = int(date[8:10])
    hour = int(time[0:2])
    minute = int(time[3:5])
    second = int(time[6:8])
    if len(time) >= 16 and time[8] == ".":
        subsecond = int(time[9:16])
    else:
        fraction = time[9:] if len(time) > 8 and time[8] == "." else ""
        subsecond = int((fraction + "0000000")[:7] or "0")
    days = civil_days(year, month, day)
    clock = hour * 3600 + minute * 60 + second
    return days * 86400 * TICKS_PER_SECOND + clock * TICKS_PER_SECOND + subsecond


def _is_null(value: str) -> bool:
    return value == "" or value.upper() == "NULL"


def sort_actions(actions: np.ndarray) -> np.ndarray:
    """Return actions in timestamp order. Already-ordered arrays are not copied."""
    if actions.shape[0] < 2 or np.all(np.diff(actions["t"]) >= 0):
        return actions
    return actions[np.argsort(actions["t"], kind="stable")]


def split_actions(actions: np.ndarray, train_fraction: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """Split one user into the earliest fraction and the later remainder.

    `actions` is sorted by time. The cut is by count, so a fraction of 0.4 keeps
    the first 40% of that user's actions for training and the later 60% for testing.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if actions.shape[0] > 1 and np.any(np.diff(actions["t"]) < 0):
        raise ValueError("actions must be sorted by time before they are split")
    cut = int(actions.shape[0] * train_fraction)
    return actions[:cut], actions[cut:]


def load_actions(path: str | Path, progress_every: int = 5_000_000) -> dict[str, np.ndarray]:
    """Load the semicolon-separated action file, grouped and sorted by user time.

    Each row is one action. A stroke has angle, distance, and velocity, with
    duration NULL. A pause has duration in milliseconds, with the kinematics NULL.
    """
    path = Path(path)
    counts: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle, delimiter=";")
        header = next(reader)
        normalized = tuple(column.strip().lower() for column in header)
        if normalized != COLUMNS:
            raise ValueError(f"unexpected columns: {header}")
        for row in reader:
            if len(row) != 7:
                raise ValueError(f"expected 7 fields, found {len(row)}")
            counts[row[0]] = counts.get(row[0], 0) + 1

    grouped = {
        user_id: np.empty(count, dtype=ACTION_DTYPE) for user_id, count in counts.items()
    }
    cursor = dict.fromkeys(counts, 0)
    seen = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle, delimiter=";")
        next(reader)
        for row in reader:
            user_id, duration, angle, distance, velocity, date, time = row
            slot = grouped[user_id]
            index = cursor[user_id]
            slot["t"][index] = parse_timestamp(date, time)
            if _is_null(duration):
                slot["stroke"][index] = 1
                slot["angle"][index] = float(angle)
                slot["distance"][index] = float(distance)
                slot["velocity"][index] = float(velocity)
                slot["duration"][index] = np.nan
            else:
                slot["stroke"][index] = 0
                slot["angle"][index] = np.nan
                slot["distance"][index] = np.nan
                slot["velocity"][index] = np.nan
                slot["duration"][index] = float(duration)
            cursor[user_id] = index + 1
            seen += 1
            if progress_every and seen % progress_every == 0:
                print(f"loaded {seen:,} rows", flush=True)

    print(f"loaded {seen:,} rows for {len(grouped):,} users", flush=True)
    return {user_id: sort_actions(actions) for user_id, actions in grouped.items()}
