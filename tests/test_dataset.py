"""Checks for loading and the earliest-40% split."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from pointerprint.dataset import (
    ACTION_DTYPE,
    civil_days,
    load_actions,
    parse_timestamp,
    sort_actions,
    split_actions,
)


def test_civil_days_matches_unix_epoch():
    assert civil_days(1970, 1, 1) == 0
    assert civil_days(2018, 3, 29) == date(2018, 3, 29).toordinal() - date(1970, 1, 1).toordinal()


def test_timestamps_order_by_calendar_time_and_fraction():
    earlier = parse_timestamp("2016-04-20", "15:19:37.3305058")
    later = parse_timestamp("2018-03-29", "17:40:30.5821653")
    just_after = parse_timestamp("2018-03-29", "17:40:30.5821654")
    assert earlier < later < just_after


def test_short_fractional_seconds_keep_their_magnitude():
    six_digits = parse_timestamp("2018-01-01", "00:00:00.500000")
    seven_digits = parse_timestamp("2018-01-01", "00:00:00.5000000")
    assert six_digits == seven_digits


def test_split_keeps_the_earliest_40_percent():
    actions = np.zeros(10, dtype=ACTION_DTYPE)
    actions["t"] = np.arange(10, dtype=np.int64)
    train, test = split_actions(actions, 0.4)
    assert train.shape[0] == 4
    assert test.shape[0] == 6
    assert train["t"][-1] < test["t"][0]
    assert np.array_equal(train["t"], np.arange(4))
    assert np.array_equal(test["t"], np.arange(4, 10))


def test_split_rejects_an_unsorted_user():
    actions = np.zeros(4, dtype=ACTION_DTYPE)
    actions["t"] = np.array([3, 1, 2, 4])
    with pytest.raises(ValueError, match="sorted"):
        split_actions(actions, 0.4)


def test_sort_actions_is_stable_for_tied_timestamps():
    actions = np.zeros(4, dtype=ACTION_DTYPE)
    actions["t"] = np.array([2, 1, 1, 3])
    actions["distance"] = np.array([10, 20, 30, 40], dtype=np.float32)
    ordered = sort_actions(actions)
    assert list(ordered["t"]) == [1, 1, 2, 3]
    assert list(ordered["distance"]) == [20, 30, 10, 40]


def _write_sample(path: Path) -> None:
    # User 42's first file row is their latest action. User 7 is interleaved.
    path.write_text(
        "\n".join(
            [
                "idman;duration;angle;distance;velocity;local_date;local_time",
                "42;NULL;10;100;200;2018-05-01;00:00:05.0000000",
                "7;NULL;80;50;900;2018-05-01;00:00:01.0000000",
                "42;150;NULL;NULL;NULL;2018-05-01;00:00:01.5000000",
                "42;NULL;11;110;210;2018-05-01;00:00:02.0000000",
                "42;NULL;12;120;220;2018-05-01;00:00:03.0000000",
                "42;NULL;13;130;230;2018-05-01;00:00:04.0000000",
                "7;40;NULL;NULL;NULL;2018-05-01;00:00:01.2500000",
            ]
        ),
        encoding="utf-8",
    )


def test_load_sorts_each_user_and_keeps_both_action_types(tmp_path: Path):
    path = tmp_path / "sample.csv"
    _write_sample(path)
    grouped = load_actions(path, progress_every=0)
    user = grouped["42"]
    assert list(user["t"]) == [
        parse_timestamp("2018-05-01", "00:00:01.5000000"),
        parse_timestamp("2018-05-01", "00:00:02.0000000"),
        parse_timestamp("2018-05-01", "00:00:03.0000000"),
        parse_timestamp("2018-05-01", "00:00:04.0000000"),
        parse_timestamp("2018-05-01", "00:00:05.0000000"),
    ]
    assert list(user["stroke"]) == [0, 1, 1, 1, 1]
    assert user["duration"][0] == pytest.approx(150)
    assert np.isnan(user["angle"][0])
    assert user["angle"][1] == pytest.approx(11)
    assert np.isnan(user["duration"][1])

    train, test = split_actions(user, 0.4)
    assert train.shape[0] == 2
    assert train["t"][-1] < test["t"][0]
    # The latest action was first in the file, so file order would have trained on it.
    assert train["t"][-1] == parse_timestamp("2018-05-01", "00:00:02.0000000")


def test_load_rejects_an_unexpected_header(tmp_path: Path):
    path = tmp_path / "bad.csv"
    path.write_text("id,x,y\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected columns"):
        load_actions(path, progress_every=0)
