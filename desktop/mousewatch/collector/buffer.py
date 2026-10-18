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
    path_px: float = 0.0
    jitter_px: float = 0.0
    reversals: int = 0
    samples: int = 0
    saved: bool = True
    accel: float = 0.0
    drag: bool = False
    button: str = ""
    approach_px_s: float = 0.0
    slowdown: float = 0.0
    click_gap_ms: float = 0.0
    wheel_notches: float = 0.0


@dataclass
class _Movement:
    t_first: int
    t_last: int
    net_dx: float
    net_dy: float
    path_px: float = 0.0
    samples: int = 0
    reversals: int = 0
    prev_dx: float = 0.0
    prev_dy: float = 0.0
    has_prev: bool = False
    prev_speed: float = 0.0
    has_speed: bool = False
    accel_sum: float = 0.0
    accel_count: int = 0


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
        self._recent: list[tuple[int, float, float]] = []
        self._last_down: dict[str, int] = {}
        self._press_context: dict[str, tuple[float, float, float]] = {}

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
        dt_s = (when_ns - movement.t_last) / 1_000_000_000 if movement.samples else 0.0
        movement.t_last = when_ns
        movement.net_dx += dx
        movement.net_dy += dy
        step = math.hypot(dx, dy)
        movement.path_px += step
        movement.samples += 1
        if movement.has_prev and dx * movement.prev_dx + dy * movement.prev_dy < 0:
            movement.reversals += 1
        if movement.has_speed and dt_s > 0:
            speed = step / dt_s
            movement.accel_sum += abs(speed - movement.prev_speed) / dt_s
            movement.accel_count += 1
            movement.prev_speed = speed
        elif dt_s > 0:
            movement.prev_speed = step / dt_s
            movement.has_speed = True
        movement.prev_dx = dx
        movement.prev_dy = dy
        movement.has_prev = True
        self._remember(dx, dy, when_ns)
        return records

    def add_button(self, button: str, pressed: bool, when_ns: int) -> list[Record]:
        records: list[Record] = []
        if pressed:
            if self._movement is not None:
                records.extend(self._close_movement())
            gap_ms = 0.0
            if button in self._last_down:
                gap_ms = (when_ns - self._last_down[button]) / 1_000_000
            self._last_down[button] = when_ns
            approach, slowdown = self._approach(when_ns)
            self._press_context[button] = (approach, slowdown, gap_ms)
            self._pressed[button] = when_ns
            return records
        started = self._pressed.pop(button, None)
        approach, slowdown, gap_ms = self._press_context.pop(button, (0.0, 0.0, 0.0))
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
                button=button,
                approach_px_s=approach,
                slowdown=slowdown,
                click_gap_ms=gap_ms,
            )
        )
        return records

    def add_wheel(self, notches: float, when_ns: int) -> list[Record]:
        """One scroll step. It stays out of the dataset file and in the motion sidecar."""
        return [
            Record(
                user_id=self._user_id,
                duration_ms=None,
                angle=None,
                distance_px=None,
                velocity_px_s=None,
                when_ns=when_ns,
                button="wheel",
                wheel_notches=notches,
                saved=False,
            )
        ]

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
        net = math.hypot(movement.net_dx, movement.net_dy)
        jitter = movement.path_px - net
        if jitter < 0:
            jitter = 0.0
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
                path_px=movement.path_px,
                jitter_px=jitter,
                reversals=movement.reversals,
                samples=movement.samples,
                saved=distance >= self._min_distance,
                accel=(movement.accel_sum / movement.accel_count) if movement.accel_count else 0.0,
                drag=bool(self._pressed),
            )
        ]

    def _remember(self, dx: float, dy: float, when_ns: int) -> None:
        self._recent.append((when_ns, dx, dy))
        cutoff = when_ns - 200_000_000
        while self._recent and self._recent[0][0] < cutoff:
            self._recent.pop(0)

    def _approach(self, when_ns: int) -> tuple[float, float]:
        """Speed over the last 200 ms, and how much that speed fell in the second half."""
        samples = [sample for sample in self._recent if sample[0] >= when_ns - 200_000_000]
        if len(samples) < 2:
            return 0.0, 0.0

        def speed(part: list[tuple[int, float, float]]) -> float:
            if len(part) < 2:
                return 0.0
            path = sum(math.hypot(dx, dy) for _when, dx, dy in part)
            dt_s = (part[-1][0] - part[0][0]) / 1_000_000_000
            if dt_s <= 0.01:
                return 0.0
            return path / dt_s

        mid = len(samples) // 2
        early = speed(samples[: mid + 1])
        late = speed(samples[mid:])
        return speed(samples), early - late
