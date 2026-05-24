"""Tests for ``rba.data.nab_release_calendar``.

Exercises the 2nd-Tuesday-of-following-month rule, the December →
January rollover, ``_OVERRIDES`` precedence, and the DataFrame
builder. Pure-algorithmic — no scrape path to exercise.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from rba.data.nab_release_calendar import (
    _OVERRIDES,
    build_nab_release_calendar,
    nab_publication_date,
    nth_tuesday_of_month,
)

# -----------------------------------------------------------------------------
# nth_tuesday_of_month
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("year", "month", "n", "expected_day"),
    [
        # June 2025 — Tuesdays: 3, 10, 17, 24
        (2025, 6, 1, 3),
        (2025, 6, 2, 10),
        (2025, 6, 4, 24),
        # January 2025 — Tuesdays: 7, 14, 21, 28
        (2025, 1, 2, 14),
        # April 2026 — Tuesdays: 7, 14, 21, 28
        (2026, 4, 2, 14),
        # 5-Tuesday month: December 2025 — Tuesdays: 2, 9, 16, 23, 30
        (2025, 12, 5, 30),
    ],
)
def test_nth_tuesday_of_month(
    year: int, month: int, n: int, expected_day: int
) -> None:
    assert nth_tuesday_of_month(year, month, n) == date(year, month, expected_day)


def test_nth_tuesday_out_of_range_raises() -> None:
    # June 2025 has 4 Tuesdays; asking for the 5th must raise.
    with pytest.raises(ValueError, match="out of range"):
        nth_tuesday_of_month(2025, 6, 5)


# -----------------------------------------------------------------------------
# nab_publication_date — default rule (2nd Tue of following month)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        # Verified: May 2025 ref → released 2025-06-10 (2nd Tue of June 2025)
        (date(2025, 5, 31), date(2025, 6, 10)),
        # Apr 2025 ref → 2nd Tue of May 2025 (Tuesdays: 6, 13, 20, 27) = May 13
        (date(2025, 4, 30), date(2025, 5, 13)),
        # Mar 2026 ref → 2nd Tue of Apr 2026 (Tuesdays: 7, 14, 21, 28) = Apr 14
        (date(2026, 3, 31), date(2026, 4, 14)),
    ],
)
def test_publication_date_uses_second_tuesday_of_following_month(
    ref: date, expected: date
) -> None:
    assert nab_publication_date(ref) == expected


def test_publication_date_december_rolls_to_following_january() -> None:
    """December reference rolls into January of the following year."""
    # Dec 2024 → 2nd Tue of Jan 2025 (Tuesdays: 7, 14, 21, 28) = Jan 14
    assert nab_publication_date(date(2024, 12, 31)) == date(2025, 1, 14)
    # Dec 2023 → 2nd Tue of Jan 2024 (Tuesdays: 2, 9, 16, 23, 30) = Jan 9
    assert nab_publication_date(date(2023, 12, 31)) == date(2024, 1, 9)


def test_publication_date_always_after_observation() -> None:
    """For NAB, publication is in the month following the reference month —
    always strictly after the reference month end."""
    for me in [
        date(2020, 1, 31),
        date(2022, 6, 30),
        date(2024, 12, 31),
        date(2026, 4, 30),
    ]:
        assert nab_publication_date(me) > me


# -----------------------------------------------------------------------------
# _OVERRIDES precedence
# -----------------------------------------------------------------------------


def test_overrides_take_precedence_over_algorithm() -> None:
    override_key = date(2024, 12, 31)
    forced = date(2025, 1, 7)  # 1st Tue instead of 2nd
    _OVERRIDES[override_key] = forced
    try:
        assert nab_publication_date(override_key) == forced
    finally:
        del _OVERRIDES[override_key]


# -----------------------------------------------------------------------------
# build_nab_release_calendar
# -----------------------------------------------------------------------------


def test_build_release_calendar_shape_and_dtypes() -> None:
    df = build_nab_release_calendar(start_year=2023, end_year=2024)
    # 12 months * (end_year - start_year + 2) = 12 * 3 = 36 rows.
    assert len(df) == 36
    assert list(df.columns) == ["reference_month_end", "publication_date"]
    assert df["reference_month_end"].dtype == "datetime64[ns]"
    assert df["publication_date"].dtype == "datetime64[ns]"


def test_build_release_calendar_includes_known_dates() -> None:
    df = build_nab_release_calendar(start_year=2024, end_year=2026)
    lookup = dict(zip(df["reference_month_end"], df["publication_date"]))
    assert lookup[pd.Timestamp("2025-05-31")] == pd.Timestamp("2025-06-10")
    assert lookup[pd.Timestamp("2024-12-31")] == pd.Timestamp("2025-01-14")


def test_build_release_calendar_publication_after_observation() -> None:
    """Every NAB publication should follow its reference month end."""
    df = build_nab_release_calendar(start_year=2023, end_year=2024)
    assert (df["publication_date"] > df["reference_month_end"]).all()
