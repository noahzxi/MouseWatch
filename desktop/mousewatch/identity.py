"""Score live collector rows against the fine-tuned master profile."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mousewatch.collector.buffer import Record


def _import_pointerprint():
    source = Path(__file__).resolve().parents[2] / "src"
    if source.is_dir():
        entry = str(source)
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from pointerprint.dataset import ACTION_DTYPE, TICKS_PER_SECOND
    from pointerprint.features import MIN_ACTIVE_STROKES, window_features, window_spans
    from pointerprint.model import PointerPrintModel, decide_access

    return (
        ACTION_DTYPE,
        TICKS_PER_SECOND,
        MIN_ACTIVE_STROKES,
        window_features,
        window_spans,
        PointerPrintModel,
        decide_access,
    )


@dataclass(frozen=True)
class IdentityStatus:
    """What the display should say about the person currently moving the mouse."""

    verdict: str
    headline: str
    detail: str
    tone: str


def missing_profile_status(path: Path) -> IdentityStatus:
    return IdentityStatus(
        verdict="unavailable",
        headline="No master profile",
        detail=f"Fine-tune first. MouseWatch looked for {path}",
        tone="waiting",
    )


def record_to_action(record: Record, action_dtype, ticks_per_second: int) -> np.ndarray:
    """One collector row, in the same array layout the trainer scores."""
    action = np.zeros(1, dtype=action_dtype)
    action["t"][0] = int(record.when_ns // (1_000_000_000 // ticks_per_second))
    if record.duration_ms is None:
        action["stroke"][0] = 1
        action["angle"][0] = np.nan if record.angle is None else record.angle
        action["distance"][0] = np.nan if record.distance_px is None else record.distance_px
        action["velocity"][0] = np.nan if record.velocity_px_s is None else record.velocity_px_s
        action["duration"][0] = np.nan
    else:
        action["duration"][0] = record.duration_ms
    return action


class LiveIdentity:
    """Turn the actions collected in this run into a you / not-you verdict."""

    def __init__(self, model, owner_id: str) -> None:
        (
            self._dtype,
            self._ticks_per_second,
            self._min_strokes,
            self._window_features,
            self._window_spans,
            _model_type,
            self._decide_access,
        ) = _import_pointerprint()
        self.model = model
        self.owner_id = str(owner_id)
        if self.owner_id not in self.model.thresholds:
            raise KeyError(f"master profile has no threshold for user {self.owner_id}")
        self._actions = np.empty(0, dtype=self._dtype)
        self._scored_spans = 0
        self._scores: list[float] = []
        self._durations: list[float] = []
        self._latest_score: float | None = None

    @classmethod
    def load(cls, path: str | Path) -> LiveIdentity:
        _dtype, _ticks, _min, _features, _spans, PointerPrintModel, _decide = _import_pointerprint()
        model = PointerPrintModel.load(path)
        owner = getattr(model, "owner_id", None)
        if not owner:
            raise ValueError(f"{path} has no owner. Fine-tune it as the master profile.")
        return cls(model, str(owner))

    def add(self, record: Record) -> IdentityStatus:
        action = record_to_action(record, self._dtype, self._ticks_per_second)
        self._actions = np.concatenate([self._actions, action]) if self._actions.size else action
        spans = self._window_spans(
            self._actions,
            window=self.model.window,
            stride=self.model.stride,
            session_gap_seconds=self.model.session_gap_seconds,
        )
        if len(spans) > self._scored_spans:
            for start, end in spans[self._scored_spans :]:
                self._score_span(start, end)
            self._scored_spans = len(spans)
        return self.status()

    def status(self) -> IdentityStatus:
        threshold = float(self.model.thresholds[self.owner_id])
        if self._latest_score is None:
            return IdentityStatus(
                verdict="waiting",
                headline="Need more movement",
                detail=(
                    f"{len(self._actions)} of {self.model.window} actions collected "
                    f"for the next score."
                ),
                tone="waiting",
            )
        you = self._latest_score >= threshold
        sustained = self._decide_access(
            np.asarray(self._scores, dtype=np.float64),
            np.asarray(self._durations, dtype=np.float64),
            threshold,
            recheck_seconds=self.model.recheck_seconds,
            lock_seconds=self.model.lock_seconds,
        )
        held = {"allow": "still matches", "recheck": "checking", "lock": "mismatch held"}[sustained]
        return IdentityStatus(
            verdict="you" if you else "not_you",
            headline="This is you" if you else "This is not you",
            detail=f"Score {self._latest_score:.1f}    threshold {threshold:.1f}    {held}",
            tone="you" if you else "not_you",
        )

    def _score_span(self, start: int, end: int) -> None:
        chunk = self._actions[start:end]
        row = self._window_features(chunk)
        if row[0] < self._min_strokes:
            return
        score = float(self.model.score(row.reshape(1, -1), self.owner_id)[0])
        duration = float(chunk["t"][-1] - chunk["t"][0]) / self._ticks_per_second
        self._scores.append(score)
        self._durations.append(max(duration, 0.0))
        self._latest_score = score
