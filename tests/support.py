"""Synthetic pointer rhythms for the verification tests."""

from __future__ import annotations

import numpy as np

from pointerprint.dataset import ACTION_DTYPE, TICKS_PER_SECOND


def rhythm(
    count: int,
    angle: float,
    distance: float,
    velocity: float,
    pause_ms: float,
    gap_seconds: float,
) -> np.ndarray:
    actions = np.zeros(count, dtype=ACTION_DTYPE)
    actions["t"] = (np.arange(count) * gap_seconds * TICKS_PER_SECOND).astype(np.int64)
    stroke = np.arange(count) % 2 == 0
    actions["stroke"] = stroke.astype(np.uint8)
    actions["angle"] = np.where(stroke, angle, np.nan)
    actions["distance"] = np.where(stroke, distance, np.nan)
    actions["velocity"] = np.where(stroke, velocity, np.nan)
    actions["duration"] = np.where(stroke, np.nan, pause_ms)
    return actions


def two_users(count: int = 200) -> dict[str, np.ndarray]:
    return {
        "owner": rhythm(count, angle=0, distance=40, velocity=150, pause_ms=80, gap_seconds=0.1),
        "other": rhythm(count, angle=70, distance=1400, velocity=1800, pause_ms=1400, gap_seconds=0.5),
    }
