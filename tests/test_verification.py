"""Verification that a fitted model accepts the enrolled user and rejects someone else."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pointerprint.dataset import split_actions
from pointerprint.features import window_frames
from pointerprint.model import PointerPrintModel, decide_access, should_lock, smooth_scores
from pointerprint.train import train_model
from support import two_users

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "models" / "metrics.json"
MODEL_PATH = ROOT / "models" / "pointerprint.joblib"


def test_held_out_owner_scores_above_an_impostor_and_only_the_impostor_locks():
    groups = two_users()
    for actions in groups.values():
        train, test = split_actions(actions, 0.4)
        assert train.shape[0] == int(0.4 * actions.shape[0])
        assert train["t"][-1] < test["t"][0]

    model, metrics = train_model(
        groups,
        train_fraction=0.4,
        window=20,
        stride=20,
        max_train_windows=None,
        max_test_windows=None,
        min_train_windows=1,
        patience=3,
        seed=0,
    )

    assert metrics["rows_loaded"] == 400
    assert metrics["users_trained"] == 2
    assert metrics["genuine_median_score"] > metrics["impostor_median_score"]
    assert metrics["genuine_accept_rate"] == 1.0
    assert metrics["impostor_accept_rate"] == 0.0

    owner_test = groups["owner"][int(0.4 * len(groups["owner"])) :]
    other_test = groups["other"][int(0.4 * len(groups["other"])) :]

    owner_windows, owner_durations = window_frames(owner_test, window=20, stride=20)
    other_windows, other_durations = window_frames(other_test, window=20, stride=20)
    owner_scores = smooth_scores(model.score(owner_windows, "owner"))
    impostor_scores = smooth_scores(model.score(other_windows, "owner"))
    owner_threshold = model.thresholds["owner"]

    assert not should_lock(owner_scores, owner_threshold, model.patience)
    assert should_lock(impostor_scores, owner_threshold, model.patience)
    assert owner_scores.min() > impostor_scores.max()
    assert decide_access(owner_scores, owner_durations, owner_threshold) == "allow"
    assert decide_access(impostor_scores[:1], other_durations[:1], owner_threshold) == "recheck"
    assert decide_access(impostor_scores, other_durations, owner_threshold) == "lock"


def test_saved_dataset_model_separates_the_later_60_percent():
    """The artifact produced from dataset.csv, trained on each user's earliest 40%."""
    if not METRICS_PATH.is_file() or not MODEL_PATH.is_file():
        pytest.fail("models/pointerprint.joblib is missing; train on dataset.csv first")

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    model = PointerPrintModel.load(MODEL_PATH)

    assert metrics["rows_loaded"] == 31_990_875
    assert metrics["users_seen"] == 1755
    assert metrics["train_fraction"] == 0.4
    assert model.train_fraction == 0.4
    assert metrics["users_trained"] == len(model.classes_)
    assert metrics["users_trained"] >= 1000
    assert metrics["genuine_median_score"] > metrics["impostor_median_score"] + 8
    assert metrics["genuine_accept_rate"] >= 0.65
    assert metrics["impostor_accept_rate"] <= 0.35
    assert metrics["genuine_accept_rate"] >= metrics["impostor_accept_rate"] + 0.30
    assert metrics["false_lock_percentile"] == 5.0
    assert metrics["recheck_seconds"] == 8.0
    assert metrics["lock_seconds"] == 25.0
    assert model.false_lock_percentile == 5.0
    assert model.recheck_seconds == 8.0
    assert model.lock_seconds == 25.0
    assert set(model.thresholds) == set(model.classes_.tolist())
    assert model.pipeline.named_steps["clf"].baseline_theta_.shape == model.pipeline.named_steps["clf"].theta_.shape
