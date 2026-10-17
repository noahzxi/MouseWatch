"""Checks for window features and session boundaries."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pointerprint.dataset import ACTION_DTYPE, TICKS_PER_SECOND
from pointerprint.features import FEATURE_NAMES, feature_matrix, window_features, window_frames, window_spans


def _empty(count: int) -> np.ndarray:
    return np.zeros(count, dtype=ACTION_DTYPE)


def test_window_features_match_hand_computed_values():
    actions = _empty(5)
    actions["t"] = np.arange(5, dtype=np.int64) * TICKS_PER_SECOND
    actions["stroke"] = np.array([1, 1, 1, 0, 0], dtype=np.uint8)
    actions["angle"] = np.array([0, 0, 90, np.nan, np.nan], dtype=np.float32)
    actions["distance"] = np.array([100, 100, 100, np.nan, np.nan], dtype=np.float32)
    actions["velocity"] = np.array([200, 200, 200, np.nan, np.nan], dtype=np.float32)
    actions["duration"] = np.array([np.nan, np.nan, np.nan, 100, 5000], dtype=np.float32)

    features = window_features(actions)
    by_name = dict(zip(FEATURE_NAMES, features))

    angles = (0.0, 0.0, 90.0)
    angle_mean = sum(angles) / 3
    angle_std = math.sqrt(sum((value - angle_mean) ** 2 for value in angles) / 3)
    log_distance = math.log1p(100)
    log_velocity = math.log1p(200)
    log_durations = (math.log1p(100), math.log1p(5000))
    duration_mean = sum(log_durations) / 2
    duration_std = math.sqrt(sum((value - duration_mean) ** 2 for value in log_durations) / 2)

    assert by_name["n_strokes"] == 3
    assert by_name["n_pauses"] == 2
    assert by_name["pause_stroke_ratio"] == pytest.approx(2 / 3)
    assert by_name["angle_mean"] == pytest.approx(angle_mean)
    assert by_name["angle_std"] == pytest.approx(angle_std)
    assert by_name["abs_angle_mean"] == pytest.approx(angle_mean)
    assert by_name["angle_frac_below_n45"] == 0
    assert by_name["angle_frac_n45_0"] == 0
    assert by_name["angle_frac_0_45"] == pytest.approx(2 / 3)
    assert by_name["angle_frac_above_45"] == pytest.approx(1 / 3)
    assert by_name["log_distance_median"] == pytest.approx(log_distance)
    assert by_name["log_distance_std"] == 0
    assert by_name["log_velocity_median"] == pytest.approx(log_velocity)
    assert by_name["log_velocity_std"] == 0
    assert by_name["log_duration_median"] == pytest.approx(sum(log_durations) / 2)
    assert by_name["log_duration_std"] == pytest.approx(duration_std)
    assert by_name["pause_frac_short"] == pytest.approx(0.5)
    assert by_name["pause_frac_long"] == pytest.approx(0.5)
    assert by_name["gap_median_s"] == pytest.approx(1.0)


def test_corrupt_velocity_and_duration_are_clipped():
    actions = _empty(2)
    actions["t"] = np.array([0, TICKS_PER_SECOND], dtype=np.int64)
    actions["stroke"] = np.array([1, 0], dtype=np.uint8)
    actions["angle"] = np.array([0, np.nan], dtype=np.float32)
    actions["distance"] = np.array([10, np.nan], dtype=np.float32)
    actions["velocity"] = np.array([-50, np.nan], dtype=np.float32)
    actions["duration"] = np.array([np.nan, 18_000_000], dtype=np.float32)

    clipped_velocity = window_features(actions)
    assert clipped_velocity[FEATURE_NAMES.index("log_velocity_median")] == pytest.approx(math.log1p(0))
    assert clipped_velocity[FEATURE_NAMES.index("log_duration_median")] == pytest.approx(math.log1p(10_000))

    actions["velocity"][0] = np.float32(9_000_000)
    spiked = window_features(actions)
    assert spiked[FEATURE_NAMES.index("log_velocity_median")] == pytest.approx(math.log1p(5000))


def test_windows_do_not_cross_a_session_gap():
    actions = _empty(100)
    actions["t"] = np.arange(100, dtype=np.int64) * TICKS_PER_SECOND
    actions["t"][50:] += 120 * TICKS_PER_SECOND
    actions["stroke"] = 1
    spans = window_spans(actions, window=50, stride=50, session_gap_seconds=60)
    assert spans == [(0, 50), (50, 100)]
    matrix = feature_matrix(actions, window=50, stride=50, session_gap_seconds=60)
    assert matrix.shape == (2, len(FEATURE_NAMES))


def test_short_sessions_produce_no_partial_window():
    actions = _empty(49)
    actions["t"] = np.arange(49, dtype=np.int64)
    assert window_spans(actions, window=50, stride=50) == []
    assert feature_matrix(actions, window=50, stride=50).shape == (0, len(FEATURE_NAMES))


def test_window_cap_is_deterministic():
    actions = _empty(200)
    actions["t"] = np.arange(200, dtype=np.int64) * TICKS_PER_SECOND
    actions["stroke"] = 1
    first = feature_matrix(actions, window=20, stride=20, max_windows=3, rng=np.random.default_rng(0))
    second = feature_matrix(actions, window=20, stride=20, max_windows=3, rng=np.random.default_rng(0))
    assert first.shape == (3, len(FEATURE_NAMES))
    assert np.array_equal(first, second)


def test_idle_windows_are_left_out_of_the_matrix():
    actions = _empty(40)
    actions["t"] = np.arange(40, dtype=np.int64) * TICKS_PER_SECOND
    actions["stroke"] = 0
    actions["duration"] = 100
    features, durations = window_frames(actions, window=20, stride=20, min_strokes=10)
    assert features.shape == (0, len(FEATURE_NAMES))
    assert durations.shape == (0,)
