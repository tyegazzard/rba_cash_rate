"""Tests for ``rba.data.westpac_mi_release_calendar``.

Exercises the 2nd-Wednesday rule, ``_OVERRIDES`` precedence, and the
DataFrame builder. The algorithmic calendar has no scrape path, so all
tests run offline.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from rba.data.westpac_mi_release_calendar import (
    _OVERRIDES,
    build_westpac_mi_release_calendar,
    nth_wednesday_of_month,
    westpac_mi_publication_date,
)

# -----------------------------------------------------------------------------
# nth_wednesday_of_month
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("year", "month", "n", "expected_day"),
    [
        # May 2026 — Wednesdays: 6, 13, 20, 27
        (2026, 5, 1, 6),
        (2026, 5, 2, 13),
        (2026, 5, 4, 27),
        # April 2026 — Wednesdays: 1, 8, 15, 22, 29 (5 Wednesdays)
        (2026, 4, 1, 1),
        (2026, 4, 2, 8),
        (2026, 4, 5, 29),
        # January 2010 — Wednesdays: 6, 13, 20, 27
        (2010, 1, 2, 13),
    ],
)
def test_nth_wednesday_of_month(
    year: int, month: int, n: int, expected_day: int
) -> None:
    assert nth_wednesday_of_month(year, month, n) == date(year, month, expected_day)


def test_nth_wednesday_out_of_range_raises() -> None:
    # May 2026 has only 4 Wednesdays; asking for the 5th must raise.
    with pytest.raises(ValueError, match="out of range"):
        nth_wednesday_of_month(2026, 5, 5)


# -----------------------------------------------------------------------------
# westpac_mi_publication_date — default rule
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        # 2nd Wed of May 2026 = May 13
        (date(2026, 5, 31), date(2026, 5, 13)),
        # 2nd Wed of Apr 2026 = Apr 8
        (date(2026, 4, 30), date(2026, 4, 8)),
        # 2nd Wed of Jan 2010 = Jan 13
        (date(2010, 1, 31), date(2010, 1, 13)),
        # 2nd Wed of Dec 2025 = Dec 10 (December reference month uses
        # SAME-month rule, no following-month rollover for Westpac).
        (date(2025, 12, 31), date(2025, 12, 10)),
    ],
)
def test_publication_date_uses_second_wednesday_of_reference_month(
    ref: date, expected: date
) -> None:
    assert westpac_mi_publication_date(ref) == expected


def test_publication_date_precedes_month_end() -> None:
    """For Westpac-MI the 2nd Wed of the reference month is always before
    the last day of the month — a quirk of this series we document."""
    for me in [
        date(2020, 1, 31),
        date(2022, 6, 30),
        date(2024, 9, 30),
        date(2026, 5, 31),
    ]:
        assert westpac_mi_publication_date(me) < me


# -----------------------------------------------------------------------------
# _OVERRIDES precedence
# -----------------------------------------------------------------------------


def test_overrides_take_precedence_over_algorithm() -> None:
    override_key = date(2026, 5, 31)
    forced = date(2026, 5, 19)  # actual observed release day
    _OVERRIDES[override_key] = forced
    try:
        assert westpac_mi_publication_date(override_key) == forced
    finally:
        del _OVERRIDES[override_key]


# -----------------------------------------------------------------------------
# build_westpac_mi_release_calendar
# -----------------------------------------------------------------------------


def test_build_release_calendar_shape_and_dtypes() -> None:
    df = build_westpac_mi_release_calendar(start_year=2023, end_year=2024)
    # 12 months * (end_year - start_year + 2) = 12 * 3 = 36 rows.
    assert len(df) == 36
    assert list(df.columns) == ["reference_month_end", "publication_date"]
    assert df["reference_month_end"].dtype == "datetime64[ns]"
    assert df["publication_date"].dtype == "datetime64[ns]"


def test_build_release_calendar_includes_known_dates() -> None:
    df = build_westpac_mi_release_calendar(start_year=2023, end_year=2026)
    lookup = dict(zip(df["reference_month_end"], df["publication_date"]))
    assert lookup[pd.Timestamp("2026-05-31")] == pd.Timestamp("2026-05-13")
    assert lookup[pd.Timestamp("2026-04-30")] == pd.Timestamp("2026-04-08")


def test_build_release_calendar_publication_before_observation() -> None:
    """Every Westpac publication should precede its reference month end."""
    df = build_westpac_mi_release_calendar(start_year=2023, end_year=2024)
    assert (df["publication_date"] < df["reference_month_end"]).all()
