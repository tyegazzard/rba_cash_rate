"""Tests for ``rba.data.rba_i2_release_calendar``.

Exercises the two rules (override precedence, first-business-day algorithm)
plus the holiday-skip behaviour verified against Wayback I2 snapshots.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from rba.data.rba_i2_release_calendar import (
    _OVERRIDES,
    _au_public_holidays,
    _easter_sunday,
    _mondayise,
    build_rba_i2_release_calendar,
    first_business_day_of_month,
    rba_i2_publication_date,
)

# -----------------------------------------------------------------------------
# _easter_sunday + _mondayise primitives
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (2018, date(2018, 4, 1)),   # Easter Sunday 2018
        (2024, date(2024, 3, 31)),  # Easter Sunday 2024
        (2026, date(2026, 4, 5)),   # Easter Sunday 2026
        (2000, date(2000, 4, 23)),  # Easter Sunday 2000
    ],
)
def test_easter_sunday(year: int, expected: date) -> None:
    assert _easter_sunday(year) == expected


@pytest.mark.parametrize(
    ("d", "expected"),
    [
        (date(2026, 1, 1), date(2026, 1, 1)),    # Thu — unchanged
        (date(2022, 1, 1), date(2022, 1, 3)),    # Sat → Mon
        (date(2023, 1, 1), date(2023, 1, 2)),    # Sun → Mon
        (date(2025, 12, 25), date(2025, 12, 25)),  # Thu — unchanged
    ],
)
def test_mondayise(d: date, expected: date) -> None:
    assert _mondayise(d) == expected


def test_au_public_holidays_includes_expected_anchors_for_2026() -> None:
    holidays = _au_public_holidays(2026)
    # Anzac Day always 25 April.
    assert date(2026, 4, 25) in holidays
    # Good Friday 2026 = 3 April (Easter Sunday 5 April).
    assert date(2026, 4, 3) in holidays
    # Easter Monday 2026 = 6 April.
    assert date(2026, 4, 6) in holidays
    # New Year's Day 2026 = Thu Jan 1 (not Mondayised).
    assert date(2026, 1, 1) in holidays
    # Queen's / Sovereign's Birthday 2026 = 2nd Mon June = 8 June.
    assert date(2026, 6, 8) in holidays
    # NSW Labour Day 2026 = 1st Mon October = 5 October.
    assert date(2026, 10, 5) in holidays


# -----------------------------------------------------------------------------
# first_business_day_of_month
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("year", "month", "expected"),
    [
        # Jan 2018 — Mon Jan 1 = New Year's Day → first business day is Tue Jan 2.
        (2018, 1, date(2018, 1, 2)),
        # Apr 2018 — Sun Apr 1 = Easter Sunday, Mon Apr 2 = Easter Monday → Tue Apr 3.
        (2018, 4, date(2018, 4, 3)),
        # May 2018 — Tue May 1 first business day.
        (2018, 5, date(2018, 5, 1)),
        # Mar 2016 — Tue Mar 1 first business day.
        (2016, 3, date(2016, 3, 1)),
        # Jun 2020 — Mon Jun 1 first business day (Queen's Birthday is 2nd Mon).
        (2020, 6, date(2020, 6, 1)),
        # Jan 2023 — Sun Jan 1, in-lieu Mon Jan 2 → first business day Tue Jan 3.
        (2023, 1, date(2023, 1, 3)),
        # Jan 2022 — Sat Jan 1, in-lieu Mon Jan 3 → first business day Tue Jan 4.
        (2022, 1, date(2022, 1, 4)),
        # Apr 2026 — Wed Apr 1 first business day (Easter is Apr 5).
        (2026, 4, date(2026, 4, 1)),
    ],
)
def test_first_business_day_of_month(year: int, month: int, expected: date) -> None:
    assert first_business_day_of_month(year, month) == expected


# -----------------------------------------------------------------------------
# rba_i2_publication_date — algorithmic verification samples
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        # From the 14 verified Wayback snapshots (see calendar module docstring).
        (date(2015, 3, 31), date(2015, 4, 1)),    # 01-Apr-2015
        (date(2015, 6, 30), date(2015, 7, 1)),    # 01-Jul-2015
        (date(2016, 2, 29), date(2016, 3, 1)),    # 01-Mar-2016
        (date(2016, 10, 31), date(2016, 11, 1)),  # 01-Nov-2016
        (date(2017, 2, 28), date(2017, 3, 1)),    # 01-Mar-2017
        (date(2017, 5, 31), date(2017, 6, 1)),    # 01-Jun-2017
        (date(2017, 12, 31), date(2018, 1, 2)),   # 02-Jan-2018 (Jan 1 = NY Day)
        (date(2018, 3, 31), date(2018, 4, 3)),    # 03-Apr-2018 (Easter shift)
        (date(2018, 4, 30), date(2018, 5, 1)),    # 01-May-2018
        (date(2019, 2, 28), date(2019, 3, 1)),    # 01-Mar-2019
        (date(2020, 5, 31), date(2020, 6, 1)),    # 01-Jun-2020
        (date(2021, 2, 28), date(2021, 3, 1)),    # 01-Mar-2021
        (date(2023, 2, 28), date(2023, 3, 1)),    # 01-Mar-2023
        (date(2023, 5, 31), date(2023, 6, 1)),    # 01-Jun-2023
        # Current live header at the time of writing.
        (date(2026, 4, 30), date(2026, 5, 1)),    # 01-May-2026
    ],
)
def test_rba_i2_publication_date_matches_wayback_samples(ref: date, expected: date) -> None:
    assert rba_i2_publication_date(ref) == expected


# -----------------------------------------------------------------------------
# December reference month → January publication
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dec_ref", "expected"),
    [
        # Dec 2017 ref → Jan 2018 first business day = Tue Jan 2 (Mon NY Day).
        (date(2017, 12, 31), date(2018, 1, 2)),
        # Dec 2022 ref → Jan 2023 first business day = Tue Jan 3 (Sun + in-lieu Mon).
        (date(2022, 12, 31), date(2023, 1, 3)),
        # Dec 2025 ref → Jan 2026 first business day = Fri Jan 2 (Thu Jan 1 NY Day).
        (date(2025, 12, 31), date(2026, 1, 2)),
    ],
)
def test_december_reference_month_year_rollover(dec_ref: date, expected: date) -> None:
    assert rba_i2_publication_date(dec_ref) == expected


# -----------------------------------------------------------------------------
# _OVERRIDES precedence
# -----------------------------------------------------------------------------


def test_overrides_take_precedence_over_algorithm() -> None:
    override_key = date(2024, 6, 30)
    forced = date(2024, 7, 15)
    _OVERRIDES[override_key] = forced
    try:
        assert rba_i2_publication_date(override_key) == forced
    finally:
        del _OVERRIDES[override_key]


# -----------------------------------------------------------------------------
# build_rba_i2_release_calendar
# -----------------------------------------------------------------------------


def test_build_rba_i2_release_calendar_shape_and_dtypes() -> None:
    df = build_rba_i2_release_calendar(start_year=2023, end_year=2024)
    # 12 months * (end_year - start_year + 2) = 12 * 3 = 36 rows.
    assert len(df) == 36
    assert list(df.columns) == ["reference_month_end", "publication_date"]
    assert df["reference_month_end"].dtype == "datetime64[ns]"
    assert df["publication_date"].dtype == "datetime64[ns]"


def test_build_rba_i2_release_calendar_unique_reference_month() -> None:
    df = build_rba_i2_release_calendar(start_year=1993, end_year=2026)
    assert df["reference_month_end"].is_unique
    assert df["reference_month_end"].is_monotonic_increasing


def test_build_rba_i2_release_calendar_publication_strictly_after_reference() -> None:
    """Every publication date must be strictly after the reference-month end.
    (The RBA cannot release a month's data before that month has ended.)"""
    df = build_rba_i2_release_calendar(start_year=1993, end_year=2026)
    assert (df["publication_date"] > df["reference_month_end"]).all()


def test_build_rba_i2_release_calendar_publication_is_weekday() -> None:
    """Every publication date must be Mon-Fri."""
    df = build_rba_i2_release_calendar(start_year=1993, end_year=2026)
    weekdays = df["publication_date"].dt.weekday
    assert (weekdays < 5).all(), (
        "Found weekend publication date(s); algorithm should always shift forward "
        f"to a weekday. Sample violations: "
        f"{df.loc[weekdays >= 5, ['reference_month_end', 'publication_date']].head().to_dict('records')}"
    )
