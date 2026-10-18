"""Personal distance template for the live you / not-you check.

Each score uses 40 actions and compares flick speed, flick distance, and
how long the click is held. The number on screen is the middle of the last
21 scores, so one odd patch of targets does not pull a 70% match down into
the 40s. A shorter click on the same drill pulls the match down sharply.
"""

from __future__ import annotations

import numpy as np

from pointerprint.dataset import TICKS_PER_SECOND
from pointerprint.features import window_spans

WINDOW = 40
STRIDE = 40
SLIDE = 5
SESSION_GAP_SECONDS = 60.0
LIMIT_PERCENTILE = 90.0
SMOOTH_WINDOWS = 21
CONFIRM_WINDOWS = 2
YOU_MATCH = 70.0
NOT_YOU_MATCH = 50.0
FEATURE_PENALTY_CAP = 12.0
VELOCITY_CLIP = (0.0, 5000.0)

LEGACY_FEATURE_NAMES = (
    "log_velocity_median",
    "log_velocity_p10",
    "log_velocity_p90",
    "log_distance_median",
    "log_distance_p10",
    "log_distance_p90",
    "fast_stroke_frac",
    "long_stroke_frac",
    "angle_std",
    "angle_jump_median",
    "direction_flip_frac",
)

FEATURE_NAMES = (
    "log_velocity_median",
    "log_distance_median",
    "fast_stroke_frac",
    "log_duration_median",
)

MOTION_DTYPE = np.dtype(
    [
        ("t", np.int64),
        ("stroke", np.uint8),
        ("angle", np.float32),
        ("distance", np.float32),
        ("velocity", np.float32),
        ("duration", np.float32),
        ("path", np.float32),
        ("jitter", np.float32),
        ("reversals", np.float32),
        ("samples", np.float32),
        ("accel", np.float32),
        ("drag", np.float32),
        ("button", np.float32),
        ("approach", np.float32),
        ("slowdown", np.float32),
        ("click_gap", np.float32),
        ("wheel", np.float32),
    ]
)


def _feature_map(actions: np.ndarray) -> dict[str, float] | None:
    strokes = actions[actions["stroke"] == 1]
    if strokes.shape[0] < max(5, WINDOW // 5):
        return None
    angle = strokes["angle"].astype(np.float64, copy=False)
    distance = np.clip(strokes["distance"].astype(np.float64, copy=False), 0.0, None)
    velocity = np.clip(strokes["velocity"].astype(np.float64, copy=False), *VELOCITY_CLIP)
    log_distance = np.log1p(distance)
    clicks = _clicks(actions)
    if clicks.shape[0]:
        held = np.clip(clicks["duration"].astype(np.float64, copy=False), 0.0, 10_000.0)
        log_held = np.log1p(held)
        log_duration_median = float(np.median(log_held))
        log_duration_std = float(np.std(log_held)) if log_held.size > 1 else 0.0
    else:
        log_duration_median = 0.0
        log_duration_std = 0.0
    jumps = np.abs(np.diff(angle)) if angle.size > 1 else np.array([0.0])
    flips = np.diff(np.signbit(angle)).astype(np.float64) if angle.size > 2 else np.array([0.0])
    names = actions.dtype.names or ()
    if "jitter" in names and "path" in names:
        path = np.maximum(strokes["path"].astype(np.float64, copy=False), 1e-3)
        jitter_ratio = strokes["jitter"].astype(np.float64, copy=False) / path
        samples = np.maximum(strokes["samples"].astype(np.float64, copy=False), 1.0)
        reversal_rate = strokes["reversals"].astype(np.float64, copy=False) / samples
    else:
        jitter_ratio = np.zeros(strokes.shape[0])
        reversal_rate = np.zeros(strokes.shape[0])
    return {
        "log_duration_median": log_duration_median,
        "log_duration_std": log_duration_std,
        "log_velocity_median": float(np.median(np.log1p(velocity))),
        "log_velocity_p10": float(np.percentile(np.log1p(velocity), 10)),
        "log_velocity_p90": float(np.percentile(np.log1p(velocity), 90)),
        "log_distance_median": float(np.median(log_distance)),
        "log_distance_p10": float(np.percentile(log_distance, 10)),
        "log_distance_p90": float(np.percentile(log_distance, 90)),
        "log_distance_std": float(np.std(log_distance)),
        "abs_angle_mean": float(np.mean(np.abs(angle))),
        "fast_stroke_frac": float(np.mean(velocity > 2000.0)),
        "long_stroke_frac": float(np.mean(distance > 300.0)),
        "angle_std": float(np.std(angle)),
        "angle_jump_median": float(np.median(jumps)),
        "direction_flip_frac": float(np.mean(flips)),
        "jitter_ratio_median": float(np.median(jitter_ratio)),
        "reversal_rate_median": float(np.median(reversal_rate)),
        "micro_frac": float(np.mean(distance < 4.0)),
        "accel_median": _median_field(strokes, "accel"),
        "drag_frac": _mean_field(strokes, "drag"),
        "approach_speed_median": _click_stat(actions, "approach"),
        "slowdown_median": _click_stat(actions, "slowdown"),
        "click_gap_median_s": _click_gap_seconds(actions),
        "right_click_frac": _right_click_frac(actions),
        "wheel_rate": _wheel_rate(actions),
        "hold_jitter_median": _hold_jitter(strokes),
    }


def _named(actions: np.ndarray, field: str) -> np.ndarray | None:
    if field not in (actions.dtype.names or ()):
        return None
    return actions[field].astype(np.float64, copy=False)


def _median_field(actions: np.ndarray, field: str) -> float:
    values = _named(actions, field)
    if values is None or values.size == 0:
        return float("nan")
    return float(np.median(values))


def _mean_field(actions: np.ndarray, field: str) -> float:
    values = _named(actions, field)
    if values is None or values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _clicks(actions: np.ndarray) -> np.ndarray:
    if "duration" not in (actions.dtype.names or ()):
        return actions[:0]
    clicks = actions[np.isfinite(actions["duration"].astype(np.float64, copy=False))]
    wheel = _named(clicks, "wheel")
    if wheel is None:
        return clicks
    return clicks[wheel == 0]


def _click_stat(actions: np.ndarray, field: str) -> float:
    clicks = _clicks(actions)
    values = _named(clicks, field)
    if values is None or values.size == 0 or np.all(values == 0):
        return float("nan")
    return float(np.median(values))


def _click_gap_seconds(actions: np.ndarray) -> float:
    clicks = _clicks(actions)
    values = _named(clicks, "click_gap")
    if values is None or values.size == 0:
        return float("nan")
    positive = values[values > 0]
    if positive.size == 0:
        return float("nan")
    return float(np.median(positive) / 1000.0)


def _right_click_frac(actions: np.ndarray) -> float:
    clicks = _clicks(actions)
    buttons = _named(clicks, "button")
    if buttons is None or buttons.size == 0:
        return float("nan")
    return float(np.mean(buttons == 2))


def _wheel_rate(actions: np.ndarray) -> float:
    wheel = _named(actions, "wheel")
    if wheel is None or actions.shape[0] < 2 or np.all(wheel == 0):
        return float("nan")
    seconds = float(actions["t"][-1] - actions["t"][0]) / TICKS_PER_SECOND
    if seconds <= 0:
        return float("nan")
    return float(np.sum(np.abs(wheel)) / seconds)


def _hold_jitter(strokes: np.ndarray) -> float:
    if strokes.shape[0] == 0 or "jitter" not in (strokes.dtype.names or ()):
        return float("nan")
    held = strokes[strokes["distance"].astype(np.float64, copy=False) < 4.0]
    if held.shape[0] == 0:
        return float("nan")
    return float(np.median(held["jitter"].astype(np.float64, copy=False)))


def _vector(actions: np.ndarray, names: tuple[str, ...]) -> np.ndarray | None:
    values = _feature_map(actions)
    if values is None:
        return None
    return np.array([values[name] for name in names], dtype=np.float64)


def _window_rows(actions: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    spans = window_spans(actions, WINDOW, STRIDE, SESSION_GAP_SECONDS)
    rows = [row for start, end in spans if (row := _vector(actions[start:end], names)) is not None]
    if not rows:
        return np.empty((0, len(names)))
    return np.vstack(rows)


def session_start(actions: np.ndarray) -> int:
    """Index where the current session begins. A minute of stillness starts a new one."""
    if actions.shape[0] < 2:
        return 0
    gaps = np.diff(actions["t"].astype(np.int64, copy=False))
    cuts = np.flatnonzero(gaps > int(SESSION_GAP_SECONDS * TICKS_PER_SECOND))
    if cuts.size == 0:
        return 0
    return int(cuts[-1]) + 1


class PersonalTemplate:
    """Robust center of one user's windows, and how far is too far."""

    window = WINDOW
    stride = STRIDE
    slide = SLIDE
    session_gap_seconds = SESSION_GAP_SECONDS

    def __init__(
        self,
        center: np.ndarray,
        scale: np.ndarray,
        limit: float,
        feature_names: tuple[str, ...] = FEATURE_NAMES,
        typical: float | None = None,
    ) -> None:
        self.center = np.asarray(center, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.limit = float(limit)
        self.feature_names = tuple(feature_names)
        self.typical = None if typical is None else float(typical)

    @classmethod
    def fit(cls, actions: np.ndarray) -> PersonalTemplate:
        rows = _window_rows(actions, FEATURE_NAMES)
        if rows.shape[0] < 3:
            raise ValueError("a personal template needs at least 3 movement windows")
        center = np.zeros(rows.shape[1], dtype=np.float64)
        for index in range(rows.shape[1]):
            finite = rows[:, index]
            finite = finite[np.isfinite(finite)]
            if finite.size:
                center[index] = float(np.median(finite))
        filled = np.where(np.isfinite(rows), rows, center)
        mad = np.median(np.abs(filled - center), axis=0)
        scale = np.maximum(1.4826 * mad, 1e-2)
        penalties = _penalties(filled, center, scale)
        typical = float(np.median(penalties))
        limit = float(np.percentile(penalties, LIMIT_PERCENTILE))
        if limit <= 0:
            limit = float(np.max(penalties)) or 1.0
        if limit <= typical:
            limit = float(np.max(penalties)) or typical + 1.0
        return cls(center, scale, limit, FEATURE_NAMES, typical)

    def penalty(self, actions: np.ndarray) -> float | None:
        """Distance of one window from the enrolled user. Higher is less alike."""
        names = getattr(self, "feature_names", LEGACY_FEATURE_NAMES)
        row = _vector(actions, names)
        if row is None:
            return None
        row = np.where(np.isfinite(row), row, self.center)
        return float(_penalties(row.reshape(1, -1), self.center, self.scale)[0])


def _penalties(rows: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (rows - center) / scale
    return np.sum(np.minimum(0.5 * z * z, FEATURE_PENALTY_CAP), axis=1)


def match_percent(penalty: float, limit: float, typical: float | None = None) -> float:
    """100 on the enrolled user.

    With no typical penalty, 50 is the accept limit. With one, the curve stays
    high around the enrolled user's usual distance and falls steeply after it.
    """
    if limit <= 0:
        return 100.0 if penalty <= 0 else 0.0
    if typical is None or not 0 < typical < limit:
        return float(np.clip(100.0 / (1.0 + penalty / limit), 0.0, 100.0))
    midpoint = 0.5 * ((typical / limit) + 1.55)
    scaled = penalty / limit
    return float(np.clip(100.0 / (1.0 + np.exp(4.5 * (scaled - midpoint))), 0.0, 100.0))


def smoothed_penalty(penalties: list[float]) -> float:
    recent = penalties[-SMOOTH_WINDOWS:]
    return float(np.median(recent))


def verdict_for(penalties: list[float], limit: float, typical: float | None = None) -> str:
    """you, unsure, or not_you from the smoothed match level."""
    if not penalties:
        return "waiting"
    match = match_percent(smoothed_penalty(penalties), limit, typical)
    if len(penalties) < CONFIRM_WINDOWS:
        return "you" if match >= YOU_MATCH else "unsure"
    if match >= YOU_MATCH:
        return "you"
    if match < NOT_YOU_MATCH:
        return "not_you"
    return "unsure"
