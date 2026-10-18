"""The live check measures distance from the enrolled user, not a generic user."""

from __future__ import annotations

import numpy as np

from pointerprint.personal import PersonalTemplate, match_percent, verdict_for
from support import two_users


def test_personal_template_rejects_a_different_rhythm():
    groups = two_users()
    template = PersonalTemplate.fit(groups["owner"])
    owner_penalty = template.penalty(groups["owner"][:20])
    other_penalty = template.penalty(groups["other"][:20])
    assert owner_penalty is not None and other_penalty is not None
    assert owner_penalty <= template.limit
    assert other_penalty > template.limit
    assert match_percent(owner_penalty, template.limit) > match_percent(other_penalty, template.limit)
    assert verdict_for([other_penalty, other_penalty], template.limit) == "not_you"


def test_one_odd_window_does_not_count_as_someone_else():
    assert verdict_for([100.0], limit=1.0) == "unsure"
    assert verdict_for([1.0, 100.0, 1.0], limit=10.0) == "you"
    assert verdict_for([1.0, 100.0, 100.0], limit=10.0) == "not_you"
    assert np.isfinite(PersonalTemplate.fit(two_users()["owner"]).limit)


def test_jitter_raises_the_wobble_measured_for_a_window():
    from pointerprint.personal import MOTION_DTYPE, _feature_map

    def window(jitter: float) -> np.ndarray:
        actions = np.zeros(20, dtype=MOTION_DTYPE)
        actions["t"] = np.arange(20) * 1_000_000
        actions["stroke"] = 1
        actions["angle"] = 0
        actions["distance"] = 40
        actions["velocity"] = 400
        actions["path"] = 40 + jitter
        actions["jitter"] = jitter
        actions["samples"] = 4
        actions["reversals"] = 1 if jitter else 0
        return actions

    straight = _feature_map(window(0))
    shaky = _feature_map(window(30))
    assert straight is not None and shaky is not None
    assert shaky["jitter_ratio_median"] > straight["jitter_ratio_median"]
    assert shaky["reversal_rate_median"] > straight["reversal_rate_median"]


def test_click_hold_and_vertical_angle_are_scored():
    from pointerprint.personal import _feature_map

    short = two_users()["owner"][:50]
    long = short.copy()
    long["duration"] = np.where(long["stroke"] == 0, 1400, np.nan)
    long["angle"] = np.where(long["stroke"] == 1, 70, np.nan)
    short_features = _feature_map(short)
    long_features = _feature_map(long)
    assert short_features is not None and long_features is not None
    assert long_features["log_duration_median"] > short_features["log_duration_median"]
    assert long_features["abs_angle_mean"] > short_features["abs_angle_mean"]


def test_match_percent_is_50_at_the_limit():
    assert match_percent(0, 10) == 100
    assert match_percent(10, 10) == 50
    assert match_percent(40, 10) < 50


def test_match_stays_high_near_the_user_and_falls_past_the_limit():
    assert match_percent(2.5, limit=6.0, typical=2.5) > 80
    assert match_percent(12.0, limit=6.0, typical=2.5) < 20
