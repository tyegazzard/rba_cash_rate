"""Tests for ``rba.data.rba_d_release_calendar``.

Exercises the two rules (override precedence, last-weekday algorithm) plus
edge cases:
  - Month-end falling on Sat/Sun (rolls back to Friday).
  - December reference month (publication in following January).
  - The three hand-verified override entries.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from rba.data.rba_d_release_calendar import (
    _OVERRIDES,
    build_rba_d_release_calendar,
    last_weekday_of_month,
    rba_d_publication_date,
)

# -----------------------------------------------------------------------------
# last_weekday_of_month
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("year", "month", "expected_day"),
    [
        # Apr 2026: 30 Apr is a Thursday — last weekday is Apr 30.
        (2026, 4, 30),
        # Aug 2025: 31 Aug is a Sunday; 30 = Sat; 29 = Fri — last weekday is Aug 29.
        (2025, 8, 29),
        # Nov 2025: 30 Nov is Sun, 29 Sat, 28 Fri — last weekday is Nov 28.
        (2025, 11, 28),
        # May 2010: 31 May is Mon — last weekday is May 31.
        (2010, 5, 31),
        # Jan 2006: 31 Jan is Tue — last weekday is Jan 31.
        (2006, 1, 31),
        # Mar 2024: 31 Mar is Sun, 30 Sat, 29 Fri (Good Friday) — algorithmically
        # the last weekday is Mar 29. The override entry handles the
        # holiday adjustment separately.
        (2024, 3, 29),
    ],
)
def test_last_weekday_of_month(year: int, month: int, expected_day: int) -> None:
    assert last_weekday_of_month(year, month) == date(year, month, expected_day)


# -----------------------------------------------------------------------------
# rba_d_publication_date — algorithmic rule
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        # 2026-03 ref → last weekday of Apr 2026 = Thu Apr 30
        (date(2026, 3, 31), date(2026, 4, 30)),
        # 2025-08 ref → last weekday of Sep 2025 = Tue Sep 30
        (date(2025, 8, 31), date(2025, 9, 30)),
        # 2025-07 ref → last weekday of Aug 2025 = Fri Aug 29 (31 = Sun)
        (date(2025, 7, 31), date(2025, 8, 29)),
        # 2010-04 ref → last weekday of May 2010 = Mon May 31
        (date(2010, 4, 30), date(2010, 5, 31)),
        # 2008-09 ref → last weekday of Oct 2008 = Fri Oct 31
        (date(2008, 9, 30), date(2008, 10, 31)),
        # 2001-07 ref → last weekday of Aug 2001 = Fri Aug 31
        (date(2001, 7, 31), date(2001, 8, 31)),
    ],
)
def test_algorithmic_last_weekday_rule(ref: date, expected: date) -> None:
    assert rba_d_publication_date(ref) == expected


# -----------------------------------------------------------------------------
# December reference month → January publication
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dec_ref", "expected"),
    [
        # 2005-12 → Jan 2006 last weekday = Tue Jan 31 2006
        (date(2005, 12, 31), date(2006, 1, 31)),
        # 2022-12 → Jan 2023 last weekday = Tue Jan 31 2023
        (date(2022, 12, 31), date(2023, 1, 31)),
        # 2025-12 → Jan 2026 last weekday = Fri Jan 30 2026 (31 = Sat)
        (date(2025, 12, 31), date(2026, 1, 30)),
    ],
)
def test_december_reference_month_year_rollover(dec_ref: date, expected: date) -> None:
    assert rba_d_publication_date(dec_ref) == expected


# -----------------------------------------------------------------------------
# _OVERRIDES precedence
# -----------------------------------------------------------------------------


def test_overrides_take_precedence_over_algorithm() -> None:
    override_key = date(2024, 6, 30)
    forced = date(2024, 7, 15)
    _OVERRIDES[override_key] = forced
    try:
        assert rba_d_publication_date(override_key) == forced
    finally:
        del _OVERRIDES[override_key]


def test_easter_collision_2013_uses_override() -> None:
    """2013-02 reference → 28 Mar 2013 (Thu, day before Good Friday). The
    algorithmic last weekday would have been Fri 29 Mar (Good Friday)."""
    assert rba_d_publication_date(date(2013, 2, 28)) == date(2013, 3, 28)


def test_easter_collision_2024_uses_override() -> None:
    """2024-02 reference → 28 Mar 2024 (Thu, day before Good Friday)."""
    assert rba_d_publication_date(date(2024, 2, 29)) == date(2024, 3, 28)


def test_d2a_decommissioning_override_2025_11() -> None:
    """2025-11 reference → 19 Dec 2025 (early release during D2A
    decommissioning; would normally have been Wed 31 Dec)."""
    assert rba_d_publication_date(date(2025, 11, 30)) == date(2025, 12, 19)


# -----------------------------------------------------------------------------
# build_rba_d_release_calendar
# -----------------------------------------------------------------------------


def test_build_rba_d_release_calendar_shape_and_dtypes() -> None:
    df = build_rba_d_release_calendar(start_year=2023, end_year=2024)
    # 12 months * (end_year - start_year + 2) = 12 * 3 = 36 rows
    assert len(df) == 36
    assert list(df.columns) == ["reference_month_end", "publication_date"]
    assert df["reference_month_end"].dtype == "datetime64[ns]"
    assert df["publication_date"].dtype == "datetime64[ns]"


def test_build_rba_d_release_calendar_includes_overrides() -> None:
    df = build_rba_d_release_calendar(start_year=2024, end_year=2025)
    lookup = dict(zip(df["reference_month_end"], df["publication_date"]))
    # Override entry for 2024-02 must propagate to the materialised calendar.
    assert lookup[pd.Timestamp("2024-02-29")] == pd.Timestamp("2024-03-28")
    # Override entry for 2025-11 (D2A delay)
    assert lookup[pd.Timestamp("2025-11-30")] == pd.Timestamp("2025-12-19")


def test_build_rba_d_release_calendar_unique_reference_month() -> None:
    df = build_rba_d_release_calendar(start_year=1993, end_year=2026)
    assert df["reference_month_end"].is_unique
    assert df["reference_month_end"].is_monotonic_increasing
