"""The desktop trace can fine-tune a trained PointerPrint model."""

from __future__ import annotations

from mousewatch.collector.buffer import Record
from mousewatch.finetune import fine_tune_trace
from mousewatch.storage.traces import TraceWriter
from pointerprint.model import PointerPrintModel
from pointerprint.train import train_model
from support import two_users


def _base_model(path):
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
    model.save(path)
    return path


def _trace(path, user_id: int = 7, rows: int = 160) -> None:
    start = 1_700_000_000 * 1_000_000_000
    with TraceWriter(path, flush_every=rows) as writer:
        for index in range(rows):
            writer.write(
                Record(
                    user_id=user_id,
                    duration_ms=None,
                    angle=12,
                    distance_px=48,
                    velocity_px_s=240,
                    when_ns=start + index * 200_000_000,
                )
            )


def test_fine_tune_trace_writes_a_personal_model(tmp_path):
    base = _base_model(tmp_path / "base.joblib")
    trace = tmp_path / "traces.csv"
    _trace(trace)
    summary = fine_tune_trace(trace, 7, base)
    tuned = PointerPrintModel.load(summary["model_path"])
    assert summary["added"] is True
    assert summary["windows"] >= 3
    assert summary["model_path"].endswith("master.joblib")
    assert tuned.owner_id == "7"
    assert tuned.personal_template is not None
    assert tuned.personal_template.limit >= 0
    assert "7" in set(map(str, tuned.classes_.tolist()))
    assert "owner" in set(map(str, tuned.classes_.tolist()))
    assert "7" in tuned.thresholds
