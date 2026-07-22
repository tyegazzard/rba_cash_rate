"""Tests for ``rba.data.align.append_future_meeting`` — the undecided future row.

The headline guard reuses the synthetic-future-injection pattern from
``test_no_leakage``: the appended future meeting must draw its features only from
observations published strictly before it. No network — frames are built inline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rba.data.align import append_future_meeting, build_master

# A value no legitimate observation would take — if it reaches the future row, a
# post-meeting publication leaked.
_FUTURE_SENTINEL = 9_999_999.0

_MEETING_COLUMNS = (
    "meeting_date",
    "effective_date",
    "rate_change_bps",
    "new_rate_pct",
    "prior_rate_pct",
    "gap_days_since_last_meeting",
    "statement_url",
    "minutes_url",
)


def _meeting_frame(rows: list[tuple[str, float]]) -> pd.DataFrame:
    """Build an 8-column meeting frame from ``(meeting_date, new_rate_pct)`` rows."""
    dates = pd.to_datetime([r[0] for r in rows])
    rates = [r[1] for r in rows]
    priors = [np.nan, *rates[:-1]]
    gaps = pd.Series(dates).diff().dt.days.astype("Int64")
    return pd.DataFrame(
        {
            "meeting_date": dates,
            "effective_date": dates,
            "rate_change_bps": [0.0] * len(rows),
            "new_rate_pct": rates,
            "prior_rate_pct": priors,
            "gap_days_since_last_meeting": gaps,
            "statement_url": pd.NA,
            "minutes_url": pd.NA,
        },
        columns=list(_MEETING_COLUMNS),
    )


def _long(rows: list[tuple[str, str, float]], series_id: str = "macro") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observation_date": pd.to_datetime([r[0] for r in rows]),
            "publication_date": pd.to_datetime([r[1] for r in rows]),
            "series_id": series_id,
            "value": [r[2] for r in rows],
        }
    )


# -----------------------------------------------------------------------------
# The append contract.
# -----------------------------------------------------------------------------
def test_append_adds_one_undecided_row() -> None:
    mf = _meeting_frame([("2021-01-05", 1.0), ("2021-02-02", 1.25)])
    out = append_future_meeting(mf, "2021-03-02")

    assert len(out) == len(mf) + 1
    fut = out.iloc[-1]
    assert pd.Timestamp(fut["meeting_date"]) == pd.Timestamp("2021-03-02")
    # Undecided outcome columns are NaN; effective date unknown.
    assert pd.isna(fut["rate_change_bps"])
    assert pd.isna(fut["new_rate_pct"])
    assert pd.isna(fut["effective_date"])
    # prior_rate is the last decided rate; gap is calendar days since last meeting.
    assert fut["prior_rate_pct"] == 1.25
    assert fut["gap_days_since_last_meeting"] == 28


def test_append_preserves_gap_dtype() -> None:
    mf = _meeting_frame([("2021-01-05", 1.0), ("2021-02-02", 1.25)])
    out = append_future_meeting(mf, "2021-03-02")
    assert str(out["gap_days_since_last_meeting"].dtype) == "Int64"


def test_append_rejects_non_future_date() -> None:
    mf = _meeting_frame([("2021-01-05", 1.0), ("2021-02-02", 1.25)])
    with pytest.raises(ValueError, match="strictly after"):
        append_future_meeting(mf, "2021-02-02")  # == last meeting
    with pytest.raises(ValueError, match="strictly after"):
        append_future_meeting(mf, "2020-06-01")  # before last meeting


# -----------------------------------------------------------------------------
# Leakage: the future row reads only past publications.
# -----------------------------------------------------------------------------
def test_future_row_reads_only_pre_meeting_publications() -> None:
    mf = _meeting_frame([("2021-01-05", 1.0), ("2021-02-02", 1.25)])
    ext = append_future_meeting(mf, "2021-03-02")
    long_df = _long(
        [
            ("2021-02-18", "2021-02-25", 7.0),  # published before the meeting → usable
            ("2021-03-03", "2021-03-05", _FUTURE_SENTINEL),  # published after → must not leak
        ]
    )
    master = build_master(ext, {"macro": long_df})

    fut = master.loc[master["meeting_date"] == pd.Timestamp("2021-03-02")]
    assert fut["macro"].iloc[0] == 7.0
    assert not (master["macro"] == _FUTURE_SENTINEL).any()


def test_same_day_publication_not_visible_to_future_row() -> None:
    mf = _meeting_frame([("2021-01-05", 1.0), ("2021-02-02", 1.25)])
    ext = append_future_meeting(mf, "2021-03-02")
    long_df = _long(
        [
            ("2021-02-18", "2021-02-25", 7.0),
            (
                "2021-03-02",
                "2021-03-02",
                99.0,
            ),  # published ON the meeting day → excluded (strict <)
        ]
    )
    master = build_master(ext, {"macro": long_df})
    fut = master.loc[master["meeting_date"] == pd.Timestamp("2021-03-02")]
    assert fut["macro"].iloc[0] == 7.0
