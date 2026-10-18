"""Live rows from the collector can be scored against a master profile."""

from __future__ import annotations

from mousewatch.collector.buffer import Record
from mousewatch.identity import LiveIdentity
from pointerprint.personal import PersonalTemplate
from pointerprint.train import train_model
from support import two_users


def _model(path):
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
    model.owner_id = "owner"
    model.save(path)
    return path


def _feed(identity: LiveIdentity, *, angle: int, distance: int, velocity: int, pause_ms: int, gap_ns: int, count: int):
    status = identity.status()
    start = 1_700_000_000 * 1_000_000_000
    for index in range(count):
        when = start + index * gap_ns
        if index % 2 == 0:
            record = Record(1, None, angle, distance, velocity, when)
        else:
            record = Record(1, pause_ms, None, None, None, when)
        status = identity.add(record)
    return status


def test_live_identity_recognizes_the_enrolled_rhythm(tmp_path):
    identity = LiveIdentity.load(_model(tmp_path / "master.joblib"))
    assert identity.status().verdict == "waiting"
    status = _feed(
        identity,
        angle=0,
        distance=40,
        velocity=150,
        pause_ms=80,
        gap_ns=100_000_000,
        count=20,
    )
    assert status.verdict == "you"
    assert status.headline == "This is you"


def test_live_identity_rejects_a_different_rhythm(tmp_path):
    identity = LiveIdentity.load(_model(tmp_path / "master.joblib"))
    status = _feed(
        identity,
        angle=70,
        distance=1400,
        velocity=1800,
        pause_ms=1400,
        gap_ns=500_000_000,
        count=20,
    )
    assert status.verdict == "not_you"
    assert status.headline == "This is not you"


def test_personal_template_rejects_after_two_different_windows(tmp_path):
    path = _model(tmp_path / "master.joblib")
    identity = LiveIdentity.load(path)
    identity._template = PersonalTemplate.fit(two_users()["owner"])
    status = _feed(
        identity,
        angle=70,
        distance=1400,
        velocity=1800,
        pause_ms=1400,
        gap_ns=500_000_000,
        count=55,
    )
    assert status.verdict == "not_you"
