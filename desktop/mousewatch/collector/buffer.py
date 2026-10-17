"""Turn cursor deltas and button presses into dataset rows.

dataset.csv is a stream of raw events. A movement event has its own length,
speed, and direction. A click event has its own duration. The angle stored
for a movement is atan(dy/dx) in degrees, limited to [-90, 90], which is the
slope of that movement from its start to its end. A later research step
averages directions across windows of events. That average is not collected.

A continuous stroke is written as a new row about every 130 milliseconds,
which is the usual spacing of movement rows in the dataset. The angle of
each row is the slope of that slice. Clicks are separate events, so a fast
series of clicks is recorded one by one. A pause or a button press also
ends the open slice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# A single raw sample has no measured duration. Keep velocity finite.
_MIN_DT_NS = 1_000_000


@dataclass(frozen=True)
class Record:
    user_id: int
    duration_ms: int | None
    angle: int | None
    distance_px: int | None
    velocity_px_s: int | None
    when_ns: int


@dataclass
class _Movement:
    t_first: int
    t_last: int
    net_dx: float
    net_dy: float


def direction_angle(dx: float, dy: float) -> int:
    """Slope of a screen-pixel vector, y positive downward, in [-90, 90]."""
    if dx == 0 and dy == 0:
        return 0
    if dx == 0:
        angle = 90 if dy > 0 else -90
    else:
        angle = int(round(math.degrees(math.atan(dy / dx))))
    if angle > 90:
        return 90
    if angle < -90:
        return -90
    return angle


class MovementAssembler:
    """Emit one movement row per stroke slice, and one row per click."""

    def __init__(
        self,
        user_id: int,
        pause_ns: int,
        min_distance_px: int = 4,
        segment_ns: int = 130_000_000,
    ) -> None:
        if pause_ns <= 0:
            raise ValueError("pause_ns must be positive")
        if min_distance_px < 1:
            raise ValueError("min_distance_px must be positive")
        if segment_ns <= 0:
            raise ValueError("segment_ns must be positive")
        self._user_id = user_id
        self._pause_ns = pause_ns
        self._min_distance = min_distance_px
        self._segment_ns = segment_ns
        self._movement: _Movement | None = None
        self._pressed: dict[str, int] = {}

    @property
    def has_open_movement(self) -> bool:
        return self._movement is not None

    def add_move(self, dx: float, dy: float, when_ns: int) -> list[Record]:
        if dx == 0 and dy == 0:
            return []
        records: list[Record] = []
        if self._movement is not None and when_ns - self._movement.t_last >= self._pause_ns:
            records.extend(self._close_movement())
        elif (
            self._movement is not None
            and when_ns - self._movement.t_first >= self._segment_ns
        ):
            records.extend(self._close_movement())
        if self._movement is None:
            self._movement = _Movement(when_ns, when_ns, 0.0, 0.0)
        movement = self._movement
        movement.t_last = when_ns
        movement.net_dx += dx
        movement.net_dy += dy
        return records

    def add_button(self, button: str, pressed: bool, when_ns: int) -> list[Record]:
        records: list[Record] = []
        if pressed:
            if self._movement is not None:
                records.extend(self._close_movement())
            self._pressed[button] = when_ns
            return records
        started = self._pressed.pop(button, None)
        if started is None:
            return records
        duration_ms = int(round((when_ns - started) / 1_000_000))
        if duration_ms < 0:
            duration_ms = 0
        records.append(
            Record(
                user_id=self._user_id,
                duration_ms=duration_ms,
                angle=None,
                distance_px=None,
                velocity_px_s=None,
                when_ns=when_ns,
            )
        )
        return records

    def flush(self, now_ns: int, *, force: bool = False) -> list[Record]:
        if self._movement is None:
            return []
        if not force and now_ns - self._movement.t_last < self._pause_ns:
            return []
        return self._close_movement()

    def _close_movement(self) -> list[Record]:
        movement = self._movement
        self._movement = None
        if movement is None:
            return []
        distance = int(round(math.hypot(movement.net_dx, movement.net_dy)))
        if distance < self._min_distance:
            return []
        dt_ns = movement.t_last - movement.t_first
        if dt_ns < _MIN_DT_NS:
            dt_ns = _MIN_DT_NS
        velocity = int(round(distance / (dt_ns / 1_000_000_000)))
        return [
            Record(
                user_id=self._user_id,
                duration_ms=None,
                angle=direction_angle(movement.net_dx, movement.net_dy),
                distance_px=distance,
                velocity_px_s=velocity,
                when_ns=movement.t_last,
            )
        ]
