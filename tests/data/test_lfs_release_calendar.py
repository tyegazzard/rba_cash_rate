"""Tests for ``rba.data.lfs_release_calendar``.

Exercises the three rules (override precedence, December exception, era rule)
plus edge cases in ``nth_thursday_of_month``. Era boundary cases (Dec 2015 /
Jan 2016) get dedicated tests because that's where rule 2 (December
exception) interacts with rule 3 (era switch).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from rba.data.lfs_release_calendar import (
    _OVERRIDES,
    _THIRD_THURSDAY_ERA_START,
    build_lfs_release_calendar,
    lfs_publication_date,
    nth_thursday_of_month,
)

# -----------------------------------------------------------------------------
# nth_thursday_of_month
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("year", "month", "n", "expected_day"),
    [
        # March 2024 — Thursdays: 7, 14, 21, 28
        (2024, 3, 1, 7),
        (2024, 3, 2, 14),
        (2024, 3, 3, 21),
        (2024, 3, 4, 28),
        # January 2016 — Thursdays: 7, 14, 21, 28
        (2016, 1, 4, 28),
        # February 2016 (post-McCarthy first month) — Thursdays: 4, 11, 18, 25
        (2016, 2, 3, 18),
        # January 2010 — Thursdays: 7, 14, 21, 28
        (2010, 1, 4, 28),
        # Five-Thursday month: July 2026 — Thursdays: 2, 9, 16, 23, 30
        (2026, 7, 5, 30),
    ],
)
def test_nth_thursday_of_month(year: int, month: int, n: int, expected_day: int) -> None:
    assert nth_thursday_of_month(year, month, n) == date(year, month, expected_day)


def test_nth_thursday_out_of_range_raises() -> None:
    # March 2024 has 4 Thursdays; asking for the 5th must raise.
    with pytest.raises(ValueError, match="out of range"):
        nth_thursday_of_month(2024, 3, 5)


# -----------------------------------------------------------------------------
# Era boundary (Dec 2015 / Jan 2016)
# -----------------------------------------------------------------------------


def test_dec_2015_uses_december_exception_not_old_era_rule() -> None:
    """Dec 2015 is the last month of the pre-2016 era — but December always
    takes the 4th-Thursday-of-January exception, not the era rule."""
    # Jan 2016 Thursdays: 7, 14, 21, 28 → 4th = Jan 28.
    assert lfs_publication_date(date(2015, 12, 31)) == date(2016, 1, 28)


def test_jan_2016_is_first_third_thursday_case() -> None:
    """Jan 2016 is the first reference month of the new era."""
    # Feb 2016 Thursdays: 4, 11, 18, 25 → 3rd = Feb 18.
    assert lfs_publication_date(date(2016, 1, 31)) == date(2016, 2, 18)


def test_nov_2015_uses_old_era_second_thursday() -> None:
    """Nov 2015 (pre-2016, non-December) → 2nd Thursday of Dec 2015."""
    # Dec 2015 Thursdays: 3, 10, 17, 24, 31 → 2nd = Dec 10.
    assert lfs_publication_date(date(2015, 11, 30)) == date(2015, 12, 10)


def test_era_boundary_constant() -> None:
    """Era switch keyed on Jan 2016 reference month, not Jan 2016 publication."""
    assert _THIRD_THURSDAY_ERA_START == date(2016, 1, 1)


# -----------------------------------------------------------------------------
# December exception (both eras)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dec_ref", "expected"),
    [
        # Old-era December (1995): Jan 1996 Thursdays 4, 11, 18, 25 → 4th = Jan 25
        (date(1995, 12, 31), date(1996, 1, 25)),
        # Old-era December (2010): Jan 2011 Thursdays 6, 13, 20, 27 → 4th = Jan 27
        (date(2010, 12, 31), date(2011, 1, 27)),
        # New-era December (2020): Jan 2021 Thursdays 7, 14, 21, 28 → 4th = Jan 28
        (date(2020, 12, 31), date(2021, 1, 28)),
        # New-era December (2023): Jan 2024 Thursdays 4, 11, 18, 25 → 4th = Jan 25
        (date(2023, 12, 31), date(2024, 1, 25)),
    ],
)
def test_december_uses_fourth_thursday_of_january(
    dec_ref: date, expected: date
) -> None:
    assert lfs_publication_date(dec_ref) == expected


# -----------------------------------------------------------------------------
# Era rule (non-December reference months)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        # Pre-2016 era — 2nd Thursday of following month
        # Jan 2010 → 2nd Thu of Feb 2010 (Thursdays: 4, 11, 18, 25) → Feb 11
        (date(2010, 1, 31), date(2010, 2, 11)),
        # Jun 2013 → 2nd Thu of Jul 2013 (Thursdays: 4, 11, 18, 25) → Jul 11
        (date(2013, 6, 30), date(2013, 7, 11)),
        # Sep 2015 → 2nd Thu of Oct 2015 (Thursdays: 1, 8, 15, 22, 29) → Oct 8
        (date(2015, 9, 30), date(2015, 10, 8)),
    ],
)
def test_pre_2016_era_uses_second_thursday(ref: date, expected: date) -> None:
    assert lfs_publication_date(ref) == expected


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        # Post-McCarthy era — 3rd Thursday of following month
        # Feb 2024 → 3rd Thu of Mar 2024 (Thursdays: 7, 14, 21, 28) → Mar 21
        (date(2024, 2, 29), date(2024, 3, 21)),
        # Jul 2020 → 3rd Thu of Aug 2020 (Thursdays: 6, 13, 20, 27) → Aug 20
        (date(2020, 7, 31), date(2020, 8, 20)),
        # Mar 2026 → 3rd Thu of Apr 2026 (Thursdays: 2, 9, 16, 23, 30) → Apr 16
        (date(2026, 3, 31), date(2026, 4, 16)),
    ],
)
def test_post_2016_era_uses_third_thursday(ref: date, expected: date) -> None:
    assert lfs_publication_date(ref) == expected


# -----------------------------------------------------------------------------
# _OVERRIDES precedence
# -----------------------------------------------------------------------------


def test_overrides_take_precedence_over_algorithm() -> None:
    override_key = date(2024, 2, 29)
    forced = date(2024, 4, 1)
    _OVERRIDES[override_key] = forced
    try:
        assert lfs_publication_date(override_key) == forced
    finally:
        del _OVERRIDES[override_key]


def test_overrides_take_precedence_over_december_exception() -> None:
    """Overrides win even over the December exception (rule 1 > rule 2)."""
    override_key = date(2023, 12, 31)
    forced = date(2024, 1, 15)  # not the 4th Thursday of January
    _OVERRIDES[override_key] = forced
    try:
        assert lfs_publication_date(override_key) == forced
    finally:
        del _OVERRIDES[override_key]


# -----------------------------------------------------------------------------
# build_lfs_release_calendar
# -----------------------------------------------------------------------------


def test_build_lfs_release_calendar_shape_and_dtypes() -> None:
    df = build_lfs_release_calendar(start_year=2023, end_year=2024)
    # 12 months * (end_year - start_year + 2) = 12 * 3 = 36 rows
    assert len(df) == 36
    assert list(df.columns) == ["reference_month_end", "publication_date"]
    assert df["reference_month_end"].dtype == "datetime64[ns]"
    assert df["publication_date"].dtype == "datetime64[ns]"


def test_build_lfs_release_calendar_contains_era_boundary_dates() -> None:
    """The boundary cases must appear in a calendar spanning both eras."""
    df = build_lfs_release_calendar(start_year=2015, end_year=2016)
    lookup = dict(zip(df["reference_month_end"], df["publication_date"]))
    # Dec 2015 → Jan 28 2016 (December exception)
    assert lookup[pd.Timestamp("2015-12-31")] == pd.Timestamp("2016-01-28")
    # Jan 2016 → Feb 18 2016 (first 3rd-Thursday case)
    assert lookup[pd.Timestamp("2016-01-31")] == pd.Timestamp("2016-02-18")
    # Nov 2015 → Dec 10 2015 (last 2nd-Thursday case before December break)
    assert lookup[pd.Timestamp("2015-11-30")] == pd.Timestamp("2015-12-10")
