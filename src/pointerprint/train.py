"""Train PointerPrint on the earliest 40% of each user's actions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pointerprint.dataset import load_actions, split_actions
from pointerprint.features import SESSION_GAP_SECONDS, STRIDE, WINDOW, feature_matrix
from pointerprint.model import PointerPrintModel, enroll_thresholds, held_out_acceptance

MIN_TRAIN_WINDOWS = 3
MIN_FIT_WINDOWS = 3
MIN_CALIBRATION_WINDOWS = 2
CALIBRATION_FRACTION = 0.2
DEFAULT_DATASET = Path(r"C:\Users\Noah\Downloads\dataset.csv")


@dataclass
class UserWindows:
    train_x: np.ndarray
    test_x: np.ndarray


@dataclass
class BuiltWindows:
    users: dict[str, UserWindows]
    users_skipped: int


def build_user_windows(
    groups: dict[str, np.ndarray],
    train_fraction: float = 0.4,
    window: int = WINDOW,
    stride: int = STRIDE,
    session_gap_seconds: float = SESSION_GAP_SECONDS,
    max_train_windows: int | None = 80,
    max_test_windows: int | None = 20,
    min_train_windows: int = MIN_TRAIN_WINDOWS,
    seed: int = 0,
) -> BuiltWindows:
    """Window each user after cutting off the earliest `train_fraction` of actions."""
    rng = np.random.default_rng(seed)
    built: dict[str, UserWindows] = {}
    skipped = 0
    for seen, user_id in enumerate(sorted(groups), start=1):
        train_actions, test_actions = split_actions(groups[user_id], train_fraction)
        train_x = feature_matrix(
            train_actions,
            window=window,
            stride=stride,
            session_gap_seconds=session_gap_seconds,
            max_windows=max_train_windows,
            rng=rng,
        )
        if train_x.shape[0] < min_train_windows:
            skipped += 1
            continue
        test_x = feature_matrix(
            test_actions,
            window=window,
            stride=stride,
            session_gap_seconds=session_gap_seconds,
            max_windows=max_test_windows,
            rng=rng,
        )
        built[user_id] = UserWindows(train_x=train_x, test_x=test_x)
        if seen % 250 == 0:
            print(f"windowed {seen:,} users", flush=True)
    return BuiltWindows(users=built, users_skipped=skipped)


def split_enrollment(
    train_x: np.ndarray,
    calibration_fraction: float = CALIBRATION_FRACTION,
    min_fit: int = MIN_FIT_WINDOWS,
    min_calibration: int = MIN_CALIBRATION_WINDOWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold the latest enrollment windows out of the Gaussian fit.

    The earlier windows estimate the user. The latest windows set the lock
    threshold. A user with too few windows calibrates on the same windows
    they were fit on, because there is nothing left to hold out.
    """
    count = int(train_x.shape[0])
    held_out = int(round(count * calibration_fraction))
    if held_out < min_calibration:
        held_out = min_calibration
    if count - held_out < min_fit:
        return train_x, train_x
    return train_x[:-held_out], train_x[-held_out:]


def fit_model(built: BuiltWindows, **model_kwargs) -> PointerPrintModel:
    """Fit on the early enrollment windows, then threshold on the held-out remainder."""
    fit_blocks = []
    fit_labels = []
    calibration: dict[str, np.ndarray] = {}
    for user_id, windows in built.users.items():
        fit_x, calibration_x = split_enrollment(windows.train_x)
        fit_blocks.append(fit_x)
        fit_labels.append(np.full(fit_x.shape[0], user_id))
        calibration[user_id] = calibration_x
    model = PointerPrintModel(**model_kwargs)
    model.fit(np.vstack(fit_blocks), np.concatenate(fit_labels))
    enroll_thresholds(model, calibration, percentile=model.false_lock_percentile)
    return model


def evaluate_model(model: PointerPrintModel, built: BuiltWindows) -> dict[str, float | int]:
    """Score the later 60% of each enrolled user. The threshold came from training only."""
    test_windows = {
        user_id: windows.test_x
        for user_id, windows in built.users.items()
        if windows.test_x.shape[0] > 0
    }
    genuine, impostor, genuine_hits, impostor_hits = held_out_acceptance(model, test_windows)
    metrics: dict[str, float | int] = {
        "train_fraction": model.train_fraction,
        "window": model.window,
        "stride": model.stride,
        "users_trained": int(len(model.classes_)),
        "users_skipped": built.users_skipped,
        "train_windows": int(sum(windows.train_x.shape[0] for windows in built.users.values())),
        "eval_users": len(test_windows),
        "eval_genuine_windows": int(genuine.size),
        "eval_impostor_windows": int(impostor.size),
        "genuine_mean_score": float(genuine.mean()),
        "impostor_mean_score": float(impostor.mean()),
        "genuine_median_score": float(np.median(genuine)),
        "impostor_median_score": float(np.median(impostor)),
        "threshold": float(model.threshold),
        "false_lock_percentile": model.false_lock_percentile,
        "recheck_seconds": model.recheck_seconds,
        "lock_seconds": model.lock_seconds,
        "genuine_accept_rate": float(np.mean(genuine_hits)),
        "impostor_accept_rate": float(np.mean(impostor_hits)),
    }
    return metrics


def train_model(groups: dict[str, np.ndarray], **kwargs) -> tuple[PointerPrintModel, dict[str, float | int]]:
    model_keys = (
        "window",
        "stride",
        "train_fraction",
        "session_gap_seconds",
        "patience",
    )
    build_keys = (
        "train_fraction",
        "window",
        "stride",
        "session_gap_seconds",
        "max_train_windows",
        "max_test_windows",
        "min_train_windows",
        "seed",
    )
    built = build_user_windows(groups, **{key: kwargs[key] for key in build_keys if key in kwargs})
    if len(built.users) < 2:
        raise ValueError("fewer than two users produced enough training windows")
    model = fit_model(built, **{key: kwargs[key] for key in model_keys if key in kwargs})
    metrics = evaluate_model(model, built)
    metrics["users_seen"] = len(groups)
    metrics["rows_loaded"] = int(sum(actions.shape[0] for actions in groups.values()))
    return model, metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train PointerPrint on the earliest 40% of each user's pointer actions."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-out", type=Path, default=Path("models/pointerprint.joblib"))
    parser.add_argument("--metrics-out", type=Path, default=Path("models/metrics.json"))
    parser.add_argument("--train-fraction", type=float, default=0.4)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--max-train-windows", type=int, default=80)
    parser.add_argument("--max-test-windows", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if not args.dataset.is_file():
        raise SystemExit(f"dataset not found: {args.dataset}")

    groups = load_actions(args.dataset)
    model, metrics = train_model(
        groups,
        train_fraction=args.train_fraction,
        window=args.window,
        stride=args.stride,
        max_train_windows=args.max_train_windows,
        max_test_windows=args.max_test_windows,
        seed=args.seed,
    )
    model.save(args.model_out)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    if metrics["genuine_median_score"] <= metrics["impostor_median_score"]:
        raise SystemExit("held-out genuine scores did not beat impostor scores")


if __name__ == "__main__":
    main()
