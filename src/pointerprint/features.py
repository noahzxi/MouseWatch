"""Turn a short run of pointer actions into one feature vector."""

from __future__ import annotations

import numpy as np

from pointerprint.dataset import TICKS_PER_SECOND

WINDOW = 50
STRIDE = 50
SESSION_GAP_SECONDS = 60.0
VELOCITY_CLIP = (0.0, 5000.0)
DURATION_CLIP = (0.0, 10_000.0)
SHORT_PAUSE_MS = 200.0
LONG_PAUSE_MS = 2_000.0
# Windows with fewer strokes are idle time, not a signature. They are dropped
# from training and do not move the live score.
MIN_ACTIVE_STROKES = 10

FEATURE_NAMES = (
    "n_strokes",
    "n_pauses",
    "pause_stroke_ratio",
    "angle_mean",
    "angle_std",
    "abs_angle_mean",
    "angle_frac_below_n45",
    "angle_frac_n45_0",
    "angle_frac_0_45",
    "angle_frac_above_45",
    "log_distance_median",
    "log_distance_std",
    "log_velocity_median",
    "log_velocity_std",
    "log_duration_median",
    "log_duration_std",
    "pause_frac_short",
    "pause_frac_long",
    "gap_median_s",
)


def _std(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.std(values))


def _median(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.median(values))


def window_features(actions: np.ndarray) -> np.ndarray:
    """Summarize one window of strokes and pauses.

    Velocity is clipped to 0..5000 and pause duration to 0..10000 ms before the
    logs are taken. The source file contains rare corrupt spikes far past those
    ranges, and a single spike would otherwise dominate the window.
    """
    strokes = actions[actions["stroke"] == 1]
    pauses = actions[actions["stroke"] == 0]
    features = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    n_strokes = int(strokes.shape[0])
    n_pauses = int(pauses.shape[0])
    features[0] = n_strokes
    features[1] = n_pauses
    features[2] = n_pauses / n_strokes if n_strokes else float(n_pauses)

    if n_strokes:
        angle = strokes["angle"].astype(np.float64, copy=False)
        distance = np.log1p(np.clip(strokes["distance"].astype(np.float64, copy=False), 0.0, None))
        velocity = np.log1p(
            np.clip(strokes["velocity"].astype(np.float64, copy=False), *VELOCITY_CLIP)
        )
        features[3] = float(angle.mean())
        features[4] = _std(angle)
        features[5] = float(np.abs(angle).mean())
        # Bins: < -45, [-45, 0), [0, 45), >= 45.
        bins = np.digitize(angle, (-45.0, 0.0, 45.0))
        features[6:10] = np.bincount(bins, minlength=4) / n_strokes
        features[10] = _median(distance)
        features[11] = _std(distance)
        features[12] = _median(velocity)
        features[13] = _std(velocity)

    if n_pauses:
        raw_duration = np.clip(pauses["duration"].astype(np.float64, copy=False), *DURATION_CLIP)
        log_duration = np.log1p(raw_duration)
        features[14] = _median(log_duration)
        features[15] = _std(log_duration)
        features[16] = float(np.mean(raw_duration < SHORT_PAUSE_MS))
        features[17] = float(np.mean(raw_duration > LONG_PAUSE_MS))

    if actions.shape[0] >= 2:
        gaps = np.diff(actions["t"].astype(np.float64, copy=False)) / TICKS_PER_SECOND
        gaps = gaps[gaps >= 0]
        features[18] = _median(gaps)

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def window_spans(
    actions: np.ndarray,
    window: int = WINDOW,
    stride: int = STRIDE,
    session_gap_seconds: float = SESSION_GAP_SECONDS,
) -> list[tuple[int, int]]:
    """Fixed windows that stay inside one session.

    A session breaks when successive actions are more than `session_gap_seconds`
    apart. Windows never cross that break, and a short session is left unused.
    """
    count = int(actions.shape[0])
    if window < 2 or stride < 1:
        raise ValueError("window must be at least 2 and stride at least 1")
    if count < window:
        return []

    gap_ticks = int(session_gap_seconds * TICKS_PER_SECOND)
    cuts = np.flatnonzero(np.diff(actions["t"]) > gap_ticks) + 1
    boundaries = [0, *[int(cut) for cut in cuts], count]
    spans: list[tuple[int, int]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        pos = start
        while pos + window <= end:
            spans.append((pos, pos + window))
            pos += stride
    return spans


def window_frames(
    actions: np.ndarray,
    window: int = WINDOW,
    stride: int = STRIDE,
    session_gap_seconds: float = SESSION_GAP_SECONDS,
    max_windows: int | None = None,
    rng: np.random.Generator | None = None,
    min_strokes: int = MIN_ACTIVE_STROKES,
) -> tuple[np.ndarray, np.ndarray]:
    """Active windows and how many seconds of movement each one spans.

    A window with fewer than `min_strokes` strokes is idle. It is left out so
    an untouched pointer is not scored as a different person.
    """
    spans = window_spans(actions, window, stride, session_gap_seconds)
    if max_windows is not None and len(spans) > max_windows:
        if rng is None:
            raise ValueError("rng is required when windows are capped")
        picked = np.sort(rng.choice(len(spans), size=max_windows, replace=False))
        spans = [spans[int(index)] for index in picked]
    features: list[np.ndarray] = []
    durations: list[float] = []
    for start, end in spans:
        chunk = actions[start:end]
        row = window_features(chunk)
        if row[0] < min_strokes:
            continue
        features.append(row)
        durations.append(float(chunk["t"][-1] - chunk["t"][0]) / TICKS_PER_SECOND)
    if not features:
        empty = np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
        return empty, np.empty(0, dtype=np.float64)
    return np.vstack(features), np.asarray(durations, dtype=np.float64)


def feature_matrix(
    actions: np.ndarray,
    window: int = WINDOW,
    stride: int = STRIDE,
    session_gap_seconds: float = SESSION_GAP_SECONDS,
    max_windows: int | None = None,
    rng: np.random.Generator | None = None,
    min_strokes: int = MIN_ACTIVE_STROKES,
) -> np.ndarray:
    """Stack window vectors for one already-split slice of a user."""
    features, _durations = window_frames(
        actions,
        window=window,
        stride=stride,
        session_gap_seconds=session_gap_seconds,
        max_windows=max_windows,
        rng=rng,
        min_strokes=min_strokes,
    )
    return features
