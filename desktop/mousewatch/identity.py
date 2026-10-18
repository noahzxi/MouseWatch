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
    from pointerprint.dataset import TICKS_PER_SECOND
    from pointerprint.features import MIN_ACTIVE_STROKES, window_features, window_spans
    from pointerprint.model import PointerPrintModel, decide_access
    from pointerprint.personal import MOTION_DTYPE, match_percent, session_start, verdict_for

    return (
        MOTION_DTYPE,
        TICKS_PER_SECOND,
        MIN_ACTIVE_STROKES,
        window_features,
        window_spans,
        PointerPrintModel,
        decide_access,
        match_percent,
        session_start,
        verdict_for,
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
    if record.button == "wheel":
        action["duration"][0] = np.nan
    elif record.duration_ms is None:
        action["stroke"][0] = 1
        action["angle"][0] = np.nan if record.angle is None else record.angle
        action["distance"][0] = np.nan if record.distance_px is None else record.distance_px
        action["velocity"][0] = np.nan if record.velocity_px_s is None else record.velocity_px_s
        action["duration"][0] = np.nan
    else:
        action["duration"][0] = record.duration_ms
    if "path" in action.dtype.names:
        action["path"][0] = record.path_px
        action["jitter"][0] = record.jitter_px
        action["reversals"][0] = record.reversals
        action["samples"][0] = record.samples
        action["accel"][0] = record.accel
        action["drag"][0] = float(record.drag)
        action["button"][0] = {"left": 1, "right": 2, "middle": 3, "wheel": 4}.get(record.button, 0)
        action["approach"][0] = record.approach_px_s
        action["slowdown"][0] = record.slowdown
        action["click_gap"][0] = record.click_gap_ms
        action["wheel"][0] = record.wheel_notches
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
            self._match_percent,
            self._session_start,
            self._verdict_for,
        ) = _import_pointerprint()
        self.model = model
        self.owner_id = str(owner_id)
        self._template = getattr(model, "personal_template", None)
        if self._template is None and self.owner_id not in self.model.thresholds:
            raise KeyError(f"master profile has no threshold for user {self.owner_id}")
        self._actions = np.empty(0, dtype=self._dtype)
        self._scored_spans = 0
        self._scores: list[float] = []
        self._durations: list[float] = []
        self._penalties: list[float] = []
        self._latest_score: float | None = None
        self._session_at = 0
        self._last_scored_length = -1

    @classmethod
    def load(cls, path: str | Path) -> LiveIdentity:
        _dtype, _ticks, _min, _features, _spans, PointerPrintModel, _decide, _match, _start, _verdict = (
            _import_pointerprint()
        )
        model = PointerPrintModel.load(path)
        owner = getattr(model, "owner_id", None)
        if not owner:
            raise ValueError(f"{path} has no owner. Fine-tune it as the master profile.")
        return cls(model, str(owner))

    def add(self, record: Record) -> IdentityStatus:
        action = record_to_action(record, self._dtype, self._ticks_per_second)
        self._actions = np.concatenate([self._actions, action]) if self._actions.size else action
        if self._template is not None:
            self._score_slide()
            return self.status()
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
        needed = self._template.window if self._template is not None else self.model.window
        if self._template is not None and self._penalties:
            return self._personal_status()
        if self._latest_score is None:
            return IdentityStatus(
                verdict="waiting",
                headline="Need more movement",
                detail=f"{len(self._actions)} of {needed} actions collected for the next score.",
                tone="waiting",
            )
        threshold = float(self.model.thresholds[self.owner_id])
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

    def _personal_status(self) -> IdentityStatus:
        limit = self._template.limit
        match = self._match_percent(self._smoothed_penalty(), limit)
        verdict = self._verdict_for(self._penalties, limit)
        headlines = {
            "you": "This is you",
            "not_you": "This is not you",
            "unsure": "Not sure",
        }
        return IdentityStatus(
            verdict=verdict,
            headline=headlines[verdict],
            detail=f"Match {match:.0f}%",
            tone=verdict,
        )

    def _smoothed_penalty(self) -> float:
        recent = self._penalties[-5:]
        return float(np.median(recent))

    def _score_slide(self) -> None:
        start = self._session_start(self._actions)
        if start != self._session_at:
            self._session_at = start
            self._last_scored_length = -1
            self._penalties.clear()
        tail = self._actions[start:]
        window = self._template.window
        if tail.shape[0] < window:
            return
        if (tail.shape[0] - window) % self._template.slide != 0:
            return
        if tail.shape[0] == self._last_scored_length:
            return
        self._last_scored_length = int(tail.shape[0])
        penalty = self._template.penalty(tail[-window:])
        if penalty is not None:
            self._penalties.append(penalty)

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
