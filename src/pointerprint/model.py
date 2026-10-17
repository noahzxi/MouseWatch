"""Per-user Gaussian model of pointer-window features, and the lock rule."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pointerprint.features import FEATURE_NAMES, MIN_ACTIVE_STROKES

# 0.5 * z^2 at three standard deviations. One odd feature cannot add more than this.
FEATURE_PENALTY_CAP = 4.5
FALSE_LOCK_PERCENTILE = 5.0
RECHECK_SECONDS = 8.0
LOCK_SECONDS = 25.0
ADAPT_ALPHA = 0.05
MAX_MEAN_SHIFT = 0.5


class PointerPrintModel:
    """Identify the enrolled user from window features.

    The score of a window is how much better it fits the enrolled user than a
    generic user. Each user's lock threshold is a low percentile of their own
    enrollment scores, so ordinary variation stays unlocked. A hard lock waits
    until the mismatch has lasted through real movement; a shorter mismatch
    asks for the normal sign-in first.
    """

    def __init__(
        self,
        window: int = 50,
        stride: int = 50,
        train_fraction: float = 0.4,
        session_gap_seconds: float = 60.0,
        threshold: float = 0.0,
        patience: int = 3,
        false_lock_percentile: float = FALSE_LOCK_PERCENTILE,
        recheck_seconds: float = RECHECK_SECONDS,
        lock_seconds: float = LOCK_SECONDS,
    ) -> None:
        self.window = window
        self.stride = stride
        self.train_fraction = train_fraction
        self.session_gap_seconds = session_gap_seconds
        self.threshold = threshold
        self.thresholds: dict[str, float] = {}
        self.patience = patience
        self.false_lock_percentile = false_lock_percentile
        self.recheck_seconds = recheck_seconds
        self.lock_seconds = lock_seconds
        self.owner_id: str | None = None
        self.pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", GaussianNB(var_smoothing=1e-6)),
            ]
        )

    def fit(self, features: np.ndarray, user_ids: np.ndarray) -> PointerPrintModel:
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"expected features with {len(FEATURE_NAMES)} columns")
        labels = np.asarray(user_ids)
        if len(np.unique(labels)) < 2:
            raise ValueError("training needs at least two users")
        self.pipeline.fit(features, labels)
        classifier = self.pipeline.named_steps["clf"]
        # Window caps already limit how much one user can contribute. A uniform
        # prior keeps a person with more saved windows from winning on volume.
        classifier.class_prior_ = np.full(len(classifier.classes_), 1.0 / len(classifier.classes_))
        classifier.baseline_theta_ = classifier.theta_.copy()
        return self

    @property
    def classes_(self) -> np.ndarray:
        return self.pipeline.named_steps["clf"].classes_

    def log_likelihood(self, features: np.ndarray, user_id: str) -> np.ndarray:
        """Log-likelihood of `user_id` for each window. Higher means a closer match."""
        if features.size == 0:
            return np.empty(0, dtype=np.float64)
        classifier = self.pipeline.named_steps["clf"]
        index = self._user_index(user_id)
        scaled = self.pipeline.named_steps["scale"].transform(features)
        variance = classifier.var_[index]
        centered = scaled - classifier.theta_[index]
        constant = -0.5 * np.sum(np.log(2.0 * np.pi * variance))
        return constant - 0.5 * np.sum(centered * centered / variance, axis=1)

    def _user_index(self, user_id: str) -> int:
        classifier = self.pipeline.named_steps["clf"]
        matches = np.flatnonzero(classifier.classes_ == user_id)
        if matches.size != 1:
            raise KeyError(f"unknown user {user_id}")
        return int(matches[0])

    def score(self, features: np.ndarray, user_id: str) -> np.ndarray:
        """Capped likelihood ratio of the enrolled user against a generic user.

        Each feature can contribute at most `FEATURE_PENALTY_CAP` nats of penalty,
        which is the penalty of a three-standard-deviation miss. A single odd
        stroke then cannot pull an enrolled user down to an impostor's score.
        """
        if features.size == 0:
            return np.empty(0, dtype=np.float64)
        classifier = self.pipeline.named_steps["clf"]
        index = self._user_index(user_id)
        scaled = self.pipeline.named_steps["scale"].transform(features)
        user_loglik = _capped_log_density(scaled, classifier.theta_[index], classifier.var_[index])
        background = _capped_log_density(scaled, np.zeros(scaled.shape[1]), np.ones(scaled.shape[1]))
        return user_loglik - background

    def score_stream(self, features: np.ndarray, user_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Score a live sequence, carrying the last real score across idle windows.

        Returns the scores and a mask that is true when the window had enough
        strokes to be evidence. Idle windows keep the previous score and are
        marked inactive so they do not count toward a lock.
        """
        count = features.shape[0]
        scores = np.full(count, np.nan, dtype=np.float64)
        active = features[:, FEATURE_NAMES.index("n_strokes")] >= MIN_ACTIVE_STROKES if count else np.empty(0, dtype=bool)
        if count and np.any(active):
            scores[active] = self.score(features[active], user_id)
        last = np.nan
        for index, score in enumerate(scores):
            if np.isnan(score):
                scores[index] = last
            else:
                last = score
        return scores, active

    def fine_tune(
        self,
        user_id: str,
        features: np.ndarray,
        *,
        max_shift: float = 2.0,
    ) -> dict[str, float | int | bool | str]:
        """Update the population model with one user's new windows.

        The shared scaler and every other person stay fixed. A user who is
        already enrolled moves toward the new windows, but no feature mean can
        leave the original enrollment mean by more than `max_shift` standardized
        units. A user who is not enrolled yet is added from these windows.
        The lock threshold is the low percentile of the latest held-out windows.
        """
        if max_shift <= 0:
            raise ValueError("max_shift must be positive")
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"expected features with {len(FEATURE_NAMES)} columns")
        if features.shape[0] < 3:
            raise ValueError("fine-tuning needs at least 3 movement windows")
        label = str(user_id)
        classifier = self.pipeline.named_steps["clf"]
        if not hasattr(classifier, "baseline_theta_"):
            classifier.baseline_theta_ = classifier.theta_.copy()
        fit_rows, calibration_rows = _split_fit_and_calibration(features)
        scaled = self.pipeline.named_steps["scale"].transform(fit_rows)
        mean = scaled.mean(axis=0)
        variance = np.var(scaled, axis=0)
        variance = np.maximum(variance, 1e-9) + float(getattr(classifier, "epsilon_", 0.0))
        added = label not in set(map(str, classifier.classes_.tolist()))
        if added:
            self._add_user(label, mean, variance, float(scaled.shape[0]))
        else:
            self._update_user(label, mean, variance, float(scaled.shape[0]), max_shift)
        self.thresholds[label] = owner_threshold(
            self.score(calibration_rows, label),
            self.false_lock_percentile,
        )
        self.threshold = float(np.median(list(self.thresholds.values())))
        return {
            "user_id": label,
            "windows": int(features.shape[0]),
            "added": added,
            "threshold": float(self.thresholds[label]),
        }

    def _add_user(self, user_id: str, mean: np.ndarray, variance: np.ndarray, count: float) -> None:
        classifier = self.pipeline.named_steps["clf"]
        labels = [str(label) for label in classifier.classes_.tolist()]
        classifier.classes_ = np.asarray(labels + [user_id])
        classifier.theta_ = np.vstack([classifier.theta_, mean])
        classifier.var_ = np.vstack([classifier.var_, variance])
        classifier.class_count_ = np.append(classifier.class_count_, count)
        classifier.baseline_theta_ = np.vstack([classifier.baseline_theta_, mean])
        classifier.class_prior_ = np.full(len(classifier.classes_), 1.0 / len(classifier.classes_))

    def _update_user(
        self,
        user_id: str,
        mean: np.ndarray,
        variance: np.ndarray,
        count: float,
        max_shift: float,
    ) -> None:
        classifier = self.pipeline.named_steps["clf"]
        index = self._user_index(user_id)
        previous = float(classifier.class_count_[index])
        total = previous + count
        blended = (previous * classifier.theta_[index] + count * mean) / total
        baseline = classifier.baseline_theta_[index]
        classifier.theta_[index] = np.clip(blended, baseline - max_shift, baseline + max_shift)
        classifier.var_[index] = (previous * classifier.var_[index] + count * variance) / total
        classifier.class_count_[index] = total

    def adapt(self, user_id: str, features: np.ndarray, alpha: float = ADAPT_ALPHA) -> int:
        """Move the enrolled user's mean slightly toward windows that clearly passed.

        Windows below the user's threshold are ignored. The mean cannot move
        more than `MAX_MEAN_SHIFT` standardized units from the enrollment mean
        on any feature, so a lingering impostor cannot walk the template over.
        """
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        classifier = self.pipeline.named_steps["clf"]
        if not hasattr(classifier, "baseline_theta_"):
            raise RuntimeError("the model has not been fit")
        if user_id not in self.thresholds:
            raise KeyError(f"no threshold for {user_id}")
        if features.size == 0:
            return 0
        passed = features[self.score(features, user_id) >= self.thresholds[user_id]]
        if passed.shape[0] == 0:
            return 0
        index = self._user_index(user_id)
        batch_mean = self.pipeline.named_steps["scale"].transform(passed).mean(axis=0)
        baseline = classifier.baseline_theta_[index]
        proposed = (1.0 - alpha) * classifier.theta_[index] + alpha * batch_mean
        classifier.theta_[index] = np.clip(proposed, baseline - MAX_MEAN_SHIFT, baseline + MAX_MEAN_SHIFT)
        return int(passed.shape[0])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> PointerPrintModel:
        model = joblib.load(path)
        if not isinstance(model, PointerPrintModel):
            raise TypeError(f"expected PointerPrintModel, found {type(model).__name__}")
        return model


def _capped_log_density(scaled: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
    """Sum of per-feature log densities, with each feature's penalty capped."""
    peak = -0.5 * np.log(2.0 * np.pi * variance)
    penalty = 0.5 * (scaled - mean) ** 2 / variance
    return np.sum(peak - np.minimum(penalty, FEATURE_PENALTY_CAP), axis=1)


def smooth_scores(scores: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Exponential moving average. `alpha` is the weight of the newest window."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return values
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for index in range(1, values.size):
        smoothed[index] = alpha * values[index] + (1.0 - alpha) * smoothed[index - 1]
    return smoothed


def decide_access(
    scores: np.ndarray,
    durations: np.ndarray,
    threshold: float,
    recheck_seconds: float = RECHECK_SECONDS,
    lock_seconds: float = LOCK_SECONDS,
    active: np.ndarray | None = None,
) -> str:
    """Choose allow, recheck, or lock from how long real movement has scored low.

    `durations` are seconds of movement in each window. Inactive windows, such
    as an idle pointer, neither add to that time nor clear it. A mismatch that
    lasts `recheck_seconds` asks for the normal sign-in. A mismatch that lasts
    `lock_seconds` locks the machine.
    """
    if recheck_seconds <= 0 or lock_seconds < recheck_seconds:
        raise ValueError("lock_seconds must be at least recheck_seconds, and both must be positive")
    values = np.asarray(scores, dtype=np.float64)
    seconds = np.asarray(durations, dtype=np.float64)
    if values.shape != seconds.shape:
        raise ValueError("scores and durations must have the same shape")
    if active is None:
        active = np.ones(values.shape, dtype=bool)
    elapsed = 0.0
    decision = "allow"
    for score, duration, is_active in zip(values, seconds, active):
        if not is_active or np.isnan(score):
            continue
        if score < threshold:
            elapsed += float(duration)
        else:
            elapsed = 0.0
        if elapsed >= lock_seconds:
            return "lock"
        decision = "recheck" if elapsed >= recheck_seconds else "allow"
    return decision


def should_lock(scores: np.ndarray, threshold: float, patience: int = 3) -> bool:
    """Lock after `patience` consecutive windows score strictly below `threshold`."""
    if patience < 1:
        raise ValueError("patience must be at least 1")
    run = 0
    for score in np.asarray(scores, dtype=np.float64):
        run = run + 1 if score < threshold else 0
        if run >= patience:
            return True
    return False


def paired_scores(
    model: PointerPrintModel,
    windows_by_user: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Genuine scores, and the same identities scored on the next user's windows.

    The impostor case is the lock case: the machine is enrolled to user U, and
    the windows were produced by someone else.
    """
    known = set(model.classes_.tolist())
    users = sorted(user_id for user_id, windows in windows_by_user.items() if windows.shape[0] and user_id in known)
    if len(users) < 2:
        raise ValueError("pairing impostors requires at least two users with windows")

    genuine: list[np.ndarray] = []
    impostor: list[np.ndarray] = []
    for index, user_id in enumerate(users):
        other_id = users[(index + 1) % len(users)]
        genuine.append(model.score(windows_by_user[user_id], user_id))
        impostor.append(model.score(windows_by_user[other_id], user_id))
    return np.concatenate(genuine), np.concatenate(impostor)


def _split_fit_and_calibration(
    features: np.ndarray,
    fraction: float = 0.2,
    min_fit: int = 3,
    min_calibration: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the latest windows out of the Gaussian update when there are enough."""
    count = int(features.shape[0])
    held_out = int(round(count * fraction))
    if held_out < min_calibration:
        held_out = min_calibration
    if count - held_out < min_fit:
        return features, features
    return features[:-held_out], features[-held_out:]


def owner_threshold(scores: np.ndarray, percentile: float = FALSE_LOCK_PERCENTILE) -> float:
    """Lock threshold that accepts all but the lowest percentile of the owner's scores."""
    if scores.size == 0:
        raise ValueError("owner threshold needs at least one enrollment score")
    if not 0.0 < percentile < 100.0:
        raise ValueError("percentile must be between 0 and 100")
    return float(np.percentile(scores, percentile))


def enroll_thresholds(
    model: PointerPrintModel,
    windows_by_user: dict[str, np.ndarray],
    percentile: float | None = None,
) -> dict[str, float]:
    """Set each user's lock threshold from that user's own enrollment windows.

    The windows must be held out from the Gaussian fit when enough enrollment
    data exists. The threshold is a low percentile of those scores, so the
    enrolled user's normal variation stays above it.
    """
    known = set(model.classes_.tolist())
    users = sorted(user_id for user_id, windows in windows_by_user.items() if windows.shape[0] and user_id in known)
    if not users:
        raise ValueError("enrollment thresholds require a user with calibration windows")
    chosen = model.false_lock_percentile if percentile is None else percentile
    thresholds = {
        user_id: owner_threshold(model.score(windows_by_user[user_id], user_id), chosen) for user_id in users
    }
    model.thresholds = thresholds
    model.threshold = float(np.median(list(thresholds.values())))
    return thresholds


def held_out_acceptance(
    model: PointerPrintModel,
    windows_by_user: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Genuine and impostor scores, plus whether each window clears that user's threshold.

    The impostor is another enrolled user. With three or more users, that person
    is not the neighbor used anywhere else in the evaluation loop.
    """
    users = sorted(
        user_id
        for user_id, windows in windows_by_user.items()
        if windows.shape[0] and user_id in model.thresholds
    )
    if len(users) < 2:
        raise ValueError("acceptance checks require at least two enrolled users")
    shift = 2 if len(users) > 2 else 1
    genuine_scores: list[np.ndarray] = []
    impostor_scores: list[np.ndarray] = []
    genuine_hits: list[np.ndarray] = []
    impostor_hits: list[np.ndarray] = []
    for index, user_id in enumerate(users):
        other_id = users[(index + shift) % len(users)]
        genuine = model.score(windows_by_user[user_id], user_id)
        impostor = model.score(windows_by_user[other_id], user_id)
        threshold = model.thresholds[user_id]
        genuine_scores.append(genuine)
        impostor_scores.append(impostor)
        genuine_hits.append(genuine >= threshold)
        impostor_hits.append(impostor >= threshold)
    return (
        np.concatenate(genuine_scores),
        np.concatenate(impostor_scores),
        np.concatenate(genuine_hits),
        np.concatenate(impostor_hits),
    )


def accept_rate(scores: np.ndarray, threshold: float) -> float:
    if scores.size == 0:
        return 0.0
    return float(np.mean(scores >= threshold))
