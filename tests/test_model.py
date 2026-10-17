"""Checks for the lock rule and for training that stays inside the earliest 40%."""

from __future__ import annotations

import numpy as np
import pytest

from pointerprint.dataset import split_actions
from pointerprint.features import FEATURE_NAMES, feature_matrix
from pointerprint.train import build_user_windows, split_enrollment, train_model
from pointerprint.model import (
    FEATURE_PENALTY_CAP,
    MAX_MEAN_SHIFT,
    PointerPrintModel,
    accept_rate,
    decide_access,
    paired_scores,
    should_lock,
    smooth_scores,
)
from support import two_users


def test_smooth_scores_weights_the_newest_window():
    smoothed = smooth_scores(np.array([0.0, 10.0]), alpha=0.5)
    assert list(smoothed) == pytest.approx([0.0, 5.0])


def test_should_lock_requires_a_consecutive_run_below_the_threshold():
    assert should_lock(np.array([0.1, 0.1, 0.1]), threshold=0.5, patience=3)
    assert not should_lock(np.array([0.1, 0.9, 0.1, 0.1]), threshold=0.5, patience=3)
    assert not should_lock(np.array([0.5, 0.5, 0.5]), threshold=0.5, patience=3)
    assert not should_lock(np.array([]), threshold=0.5, patience=3)


def test_training_windows_come_only_from_the_earliest_40_percent():
    groups = two_users()
    built = build_user_windows(
        groups,
        train_fraction=0.4,
        window=20,
        stride=20,
        max_train_windows=None,
        max_test_windows=None,
        min_train_windows=1,
        seed=0,
    )
    for user_id, actions in groups.items():
        train_actions, test_actions = split_actions(actions, 0.4)
        assert train_actions["t"][-1] < test_actions["t"][0]
        assert built.users[user_id].train_x.shape[0] == train_actions.shape[0] // 20
        assert built.users[user_id].test_x.shape[0] == test_actions.shape[0] // 20
        # 200 actions -> 80 train and 120 test. A window is 20 actions.
        assert built.users[user_id].train_x.shape == (4, len(FEATURE_NAMES))
        assert built.users[user_id].test_x.shape == (6, len(FEATURE_NAMES))


def test_model_round_trip(tmp_path):
    groups = two_users()
    model, _metrics = train_model(
        groups,
        train_fraction=0.4,
        window=20,
        stride=20,
        max_train_windows=None,
        max_test_windows=None,
        min_train_windows=1,
        seed=0,
    )
    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = PointerPrintModel.load(path)
    sample = next(iter(groups.values()))[:20]
    from pointerprint.features import feature_matrix

    windows = feature_matrix(sample, window=20, stride=20)
    assert loaded.score(windows, "owner") == pytest.approx(model.score(windows, "owner"))
    with pytest.raises(KeyError, match="unknown user"):
        loaded.score(windows, "missing")


def test_accept_rate_counts_scores_at_the_threshold_as_matches():
    assert accept_rate(np.array([-2.0, -1.0, 0.0]), threshold=-1.0) == pytest.approx(2 / 3)


def test_enrollment_holdout_keeps_the_latest_training_windows():
    train_x = np.arange(10, dtype=np.float64).reshape(-1, 1)
    fit_x, calibration_x = split_enrollment(train_x)
    assert fit_x.shape[0] == 8
    assert list(calibration_x[:, 0]) == [8, 9]

    short = np.arange(4, dtype=np.float64).reshape(-1, 1)
    fit_short, calibration_short = split_enrollment(short)
    assert fit_short.shape[0] == 4
    assert calibration_short.shape[0] == 4


def test_short_mismatch_asks_for_sign_in_and_a_long_one_locks():
    low = np.array([0.0, 0.0, 0.0, 0.0])
    durations = np.array([5.0, 5.0, 5.0, 15.0])
    assert decide_access(low[:1], durations[:1], threshold=1.0) == "allow"
    assert decide_access(low[:2], durations[:2], threshold=1.0) == "recheck"
    assert decide_access(low, durations, threshold=1.0) == "lock"
    recovered = np.array([0.0, 0.0, 5.0, 0.0])
    assert decide_access(recovered, np.array([5.0, 5.0, 5.0, 4.0]), threshold=1.0) == "allow"


def test_idle_windows_do_not_count_toward_a_lock():
    groups = two_users()
    model, _metrics = train_model(
        groups,
        train_fraction=0.4,
        window=20,
        stride=20,
        max_train_windows=None,
        max_test_windows=None,
        min_train_windows=1,
        seed=0,
    )
    owner = feature_matrix(groups["owner"][:20], window=20, stride=20)
    idle = owner.copy()
    idle[0, FEATURE_NAMES.index("n_strokes")] = 0
    stream = np.vstack([owner, idle, owner])
    scores, active = model.score_stream(stream, "owner")
    assert list(active) == [True, False, True]
    assert scores[1] == pytest.approx(scores[0])
    durations = np.array([30.0, 600.0, 30.0])
    assert decide_access(scores, durations, model.thresholds["owner"], active=active) == "allow"


def test_one_wild_feature_cannot_collapse_the_score():
    groups = two_users()
    model, _metrics = train_model(
        groups,
        train_fraction=0.4,
        window=20,
        stride=20,
        max_train_windows=None,
        max_test_windows=None,
        min_train_windows=1,
        seed=0,
    )
    window = feature_matrix(groups["owner"][:20], window=20, stride=20)
    spiked = window.copy()
    spiked[0, FEATURE_NAMES.index("angle_mean")] = 10_000
    drop = model.score(window, "owner")[0] - model.score(spiked, "owner")[0]
    assert abs(drop) <= 2 * FEATURE_PENALTY_CAP + 1e-6


def test_adaptation_ignores_failures_and_stays_near_enrollment():
    groups = two_users()
    model, _metrics = train_model(
        groups,
        train_fraction=0.4,
        window=20,
        stride=20,
        max_train_windows=None,
        max_test_windows=None,
        min_train_windows=1,
        seed=0,
    )
    classifier = model.pipeline.named_steps["clf"]
    index = model._user_index("owner")
    baseline = classifier.baseline_theta_[index].copy()
    impostor = feature_matrix(groups["other"][:20], window=20, stride=20)
    assert model.adapt("owner", impostor) == 0
    assert classifier.theta_[index] == pytest.approx(baseline)

    model.thresholds["owner"] = -1e18
    shifted = feature_matrix(groups["owner"][:20], window=20, stride=20)
    shifted = np.repeat(shifted, 30, axis=0)
    shifted[:, FEATURE_NAMES.index("angle_mean")] += 1_000
    assert model.adapt("owner", shifted, alpha=1.0) == shifted.shape[0]
    assert np.max(np.abs(classifier.theta_[index] - baseline)) == pytest.approx(MAX_MEAN_SHIFT)


def test_paired_scores_reject_a_single_user():
    groups = two_users()
    model, _metrics = train_model(
        groups,
        train_fraction=0.4,
        window=20,
        stride=20,
        max_train_windows=None,
        max_test_windows=None,
        min_train_windows=1,
    )
    with pytest.raises(ValueError, match="two users"):
        paired_scores(model, {"owner": np.zeros((2, len(FEATURE_NAMES)))})


def _trained() -> PointerPrintModel:
    model, _metrics = train_model(
        two_users(),
        train_fraction=0.4,
        window=20,
        stride=20,
        max_train_windows=None,
        max_test_windows=None,
        min_train_windows=1,
        seed=0,
    )
    return model


def test_fine_tune_adds_a_new_user_from_their_windows():
    model = _trained()
    windows = build_user_windows(
        two_users(),
        train_fraction=0.4,
        window=20,
        stride=20,
        max_train_windows=None,
        max_test_windows=None,
        min_train_windows=1,
        seed=0,
    ).users["owner"].train_x
    summary = model.fine_tune("7", windows)
    assert summary["added"] is True
    assert "7" in set(map(str, model.classes_.tolist()))
    assert "owner" in set(map(str, model.classes_.tolist()))
    scores = model.score(windows, "7")
    assert np.mean(scores >= model.thresholds["7"]) >= 0.9


def test_fine_tune_keeps_an_existing_user_near_enrollment():
    model = _trained()
    classifier = model.pipeline.named_steps["clf"]
    index = model._user_index("owner")
    baseline = classifier.baseline_theta_[index].copy()
    scaler_mean = model.pipeline.named_steps["scale"].mean_.copy()
    shifted = np.zeros((12, len(FEATURE_NAMES)))
    shifted[:, FEATURE_NAMES.index("angle_mean")] = 50
    summary = model.fine_tune("owner", shifted, max_shift=2.0)
    assert summary["added"] is False
    assert np.max(np.abs(classifier.theta_[index] - baseline)) <= 2.0 + 1e-6
    assert np.max(np.abs(classifier.theta_[index] - baseline)) > 0
    assert model.pipeline.named_steps["scale"].mean_ == pytest.approx(scaler_mean)
