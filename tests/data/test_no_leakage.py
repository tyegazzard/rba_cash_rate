"""Leakage + correctness tests for ``rba.data.align``.

The headline guard is **synthetic future-data injection**: rows published on or
after a meeting date carry deliberately absurd values; the point-in-time join
must never surface them. No network — every frame is built inline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rba.data.align import (
    HISTORY_START,
    as_of_levels,
    build_master,
    build_meeting_frame,
)

# A value no legitimate observation would ever take — if it appears in the
# master frame, a future row leaked.
_FUTURE_SENTINEL = 9_999_999.0


# -----------------------------------------------------------------------------
# Fixtures / builders.
# -----------------------------------------------------------------------------
def _meetings(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"meeting_date": pd.to_datetime(dates)})


def _long(rows: list[tuple[str, str, float]], series_id: str = "x") -> pd.DataFrame:
    """Build a long source frame from ``(observation_date, publication_date, value)``."""
    return pd.DataFrame(
        {
            "observation_date": pd.to_datetime([r[0] for r in rows]),
            "publication_date": pd.to_datetime([r[1] for r in rows]),
            "series_id": series_id,
            "value": [r[2] for r in rows],
        }
    )


# -----------------------------------------------------------------------------
# The leakage guard.
# -----------------------------------------------------------------------------
def test_same_day_publication_is_not_visible() -> None:
    """A value published *on* the meeting day is treated as not-yet-known."""
    meetings = _meetings(["2020-01-01"])
    long_df = _long(
        [
            ("2019-12-10", "2019-12-15", 1.0),  # known before → usable
            ("2020-01-01", "2020-01-01", 99.0),  # same day → excluded
        ]
    )
    out = as_of_levels(meetings, long_df)
    assert out.loc[0, "x"] == 1.0
    assert (out["x"] != 99.0).all()


def test_future_publication_never_leaks() -> None:
    """A row published after a meeting must never be picked for that meeting."""
    meetings = _meetings(["2020-01-01", "2020-02-01", "2020-03-01"])
    long_df = _long(
        [
            ("2019-12-10", "2019-12-15", 1.0),
            ("2020-01-15", "2020-01-20", 2.0),
            ("2020-03-04", "2020-03-05", _FUTURE_SENTINEL),  # after meeting 3
        ]
    )
    out = as_of_levels(meetings, long_df)

    assert out["x"].tolist() == [1.0, 2.0, 2.0]
    assert (out["x"] != _FUTURE_SENTINEL).all()


def test_injected_future_value_absent_from_master() -> None:
    """End-to-end: an injected future row never appears anywhere in the master."""
    f11 = _synthetic_f11()
    meeting_frame = build_meeting_frame(f11)
    # A future row for every meeting: published one day *after* the last meeting.
    future = _long(
        [("2099-01-01", "2099-01-02", _FUTURE_SENTINEL)],
        series_id="macro_series",
    )
    history = _long(
        [
            ("2018-12-01", "2018-12-20", 3.3),
            ("2019-01-01", "2019-01-20", 4.4),
        ],
        series_id="macro_series",
    )
    long_df = pd.concat([history, future], ignore_index=True)

    master = build_master(meeting_frame, {"macro": long_df})
    assert not (master["macro_series"] == _FUTURE_SENTINEL).any()


def test_is_missing_flag_never_reveals_future_series() -> None:
    """A series whose only reading is in the future is all-missing — no leak."""
    meeting_frame = build_meeting_frame(_synthetic_f11())
    future = _long(
        [("2099-01-01", "2099-01-02", _FUTURE_SENTINEL)], series_id="future_only"
    )
    master = build_master(meeting_frame, {"macro": future})

    # No meeting can see the 2099 reading → level all NaN, indicator all 1.
    assert master["future_only"].isna().all()
    assert (master["future_only_is_missing"] == 1).all()
    assert not (master["future_only"] == _FUTURE_SENTINEL).any()
    # The flag is exactly the level's NaN mask (Invariant #4), never the value.
    assert (
        master["future_only_is_missing"].tolist()
        == master["future_only"].isna().astype(int).tolist()
    )


def test_age_days_measures_staleness() -> None:
    """``<sid>_age_days`` is the calendar gap from publication to the meeting."""
    meetings = _meetings(["2020-01-01"])
    long_df = _long([("2019-12-10", "2019-12-15", 1.0)])
    out = as_of_levels(meetings, long_df)
    assert out.loc[0, "x_age_days"] == 17  # 2019-12-15 → 2020-01-01


def test_no_prior_observation_yields_nan() -> None:
    """A meeting before the series' first publication gets NaN level + age."""
    meetings = _meetings(["2019-01-01"])
    long_df = _long([("2020-01-01", "2020-01-20", 5.0)])
    out = as_of_levels(meetings, long_df)
    assert pd.isna(out.loc[0, "x"])
    assert pd.isna(out.loc[0, "x_age_days"])


# -----------------------------------------------------------------------------
# Meeting frame parsing.
# -----------------------------------------------------------------------------
def _synthetic_f11() -> pd.DataFrame:
    """A tiny F11-shaped frame spanning the history floor and a range row."""
    return pd.DataFrame(
        {
            "observation_date": pd.to_datetime(
                ["1990-01-23", "2019-02-06", "2019-06-05", "2024-02-07"]
            ),
            "publication_date": pd.to_datetime(
                ["1990-01-22", "2019-02-05", "2019-06-04", "2024-02-06"]
            ),
            "change_raw": ["15.00 to 15.50", "0.00", "-0.25", "+0.25"],
            "new_cash_rate_raw": ["15.00 to 15.50", "1.50", "1.25", "4.35"],
            "statement_url": [None, "/a", "/b", "/c"],
            "minutes_url": [None, "/m1", "/m2", "/m3"],
        }
    )


def test_meeting_frame_filters_pre_history_and_parses_targets() -> None:
    frame = build_meeting_frame(_synthetic_f11())

    # The 1990 range row is dropped by the history window.
    assert (frame["meeting_date"] >= HISTORY_START).all()
    assert len(frame) == 3

    first = frame.iloc[0]
    assert first["meeting_date"] == pd.Timestamp("2019-02-05")
    assert first["rate_change_bps"] == 0.0
    assert first["new_rate_pct"] == 1.50
    assert frame.iloc[1]["rate_change_bps"] == -25.0
    assert frame.iloc[2]["rate_change_bps"] == 25.0


def test_prior_rate_and_gap_use_full_history() -> None:
    """The first in-window meeting still sees its true predecessor + gap."""
    frame = build_meeting_frame(_synthetic_f11())
    # First kept meeting (2019-02-05) had a pre-window predecessor (1990 row),
    # so prior_rate_pct is that row's parsed rate (a range → NaN here) and the
    # gap is the real day count — not NaN from windowing.
    assert pd.notna(frame.iloc[0]["gap_days_since_last_meeting"])
    # Second meeting's prior rate is the first meeting's new rate.
    assert frame.iloc[1]["prior_rate_pct"] == 1.50
    assert frame.iloc[1]["gap_days_since_last_meeting"] == 119  # 2019-02-05 → 06-04


# -----------------------------------------------------------------------------
# build_master plumbing.
# -----------------------------------------------------------------------------
def test_build_master_joins_multiple_sources() -> None:
    meeting_frame = build_meeting_frame(_synthetic_f11())
    src_a = _long([("2019-01-01", "2019-01-20", 1.1)], series_id="a_series")
    src_b = _long([("2019-01-01", "2019-01-20", 2.2)], series_id="b_series")

    master = build_master(meeting_frame, {"a": src_a, "b": src_b})
    for col in ("a_series", "a_series_age_days", "b_series", "b_series_age_days"):
        assert col in master.columns
    assert len(master) == len(meeting_frame)


def test_build_master_raises_on_series_collision() -> None:
    meeting_frame = build_meeting_frame(_synthetic_f11())
    dup_a = _long([("2019-01-01", "2019-01-20", 1.0)], series_id="dup")
    dup_b = _long([("2019-01-01", "2019-01-20", 2.0)], series_id="dup")

    with pytest.raises(ValueError, match="globally unique"):
        build_master(meeting_frame, {"a": dup_a, "b": dup_b})


def test_as_of_levels_rejects_malformed_long_frame() -> None:
    meetings = _meetings(["2020-01-01"])
    bad = pd.DataFrame({"publication_date": pd.to_datetime(["2019-01-01"]), "value": [1.0]})
    with pytest.raises(ValueError, match="missing required column"):
        as_of_levels(meetings, bad)


def test_strict_rule_holds_across_random_history() -> None:
    """Property: every aligned value's publication is strictly before its meeting."""
    rng = np.random.default_rng(12)
    meetings = _meetings(["2021-01-01", "2021-04-01", "2021-07-01", "2021-10-01"])
    pubs = pd.date_range("2020-06-01", "2021-12-01", freq="7D")
    long_df = pd.DataFrame(
        {
            "observation_date": pubs,
            "publication_date": pubs,
            "series_id": "rnd",
            "value": rng.normal(size=len(pubs)),
        }
    )
    out = as_of_levels(meetings, long_df)
    # Reconstruct: for each meeting, the chosen value must equal the last pub < t.
    for i, t in enumerate(meetings["meeting_date"]):
        eligible = long_df[long_df["publication_date"] < t]
        expected = eligible.iloc[-1]["value"] if not eligible.empty else np.nan
        got = out.loc[i, "rnd"]
        assert (pd.isna(got) and pd.isna(expected)) or got == expected
