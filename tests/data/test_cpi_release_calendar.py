"""Tests for ``rba.data.cpi_release_calendar``.

Spot-checks the algorithmic publication-date computation against known
historical ABS release dates and exercises edge cases in
``last_wednesday_of_month``.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from rba.data.cpi_release_calendar import (
    _OVERRIDES,
    build_cpi_release_calendar,
    cpi_publication_date,
    last_wednesday_of_month,
)

# Verified against the ABS Cat. 6401.0 release archive: (quarter end, release date).
_KNOWN_RELEASE_DATES: list[tuple[date, date]] = [
    (date(2023, 3, 31), date(2023, 4, 26)),
    (date(2023, 6, 30), date(2023, 7, 26)),
    (date(2023, 9, 30), date(2023, 10, 25)),
    (date(2023, 12, 31), date(2024, 1, 31)),
    (date(2024, 3, 31), date(2024, 4, 24)),
    (date(2024, 6, 30), date(2024, 7, 31)),
]


@pytest.mark.parametrize(("quarter_end", "expected"), _KNOWN_RELEASE_DATES)
def test_cpi_publication_date_matches_known_release(
    quarter_end: date, expected: date
) -> None:
    actual = cpi_publication_date(quarter_end)
    assert actual == expected, (
        f"For quarter ending {quarter_end}: expected {expected}, "
        f"got {actual} (off by {(actual - expected).days} days)"
    )


def test_all_known_release_dates_match() -> None:
    """Accumulating check — reports every mismatch in one failure message.

    Complements the parametrized check above by giving a single, scannable
    diff if the algorithmic rule diverges from observed ABS releases.
    """
    mismatches: list[tuple[date, date, date]] = []
    for qe, expected in _KNOWN_RELEASE_DATES:
        actual = cpi_publication_date(qe)
        if actual != expected:
            mismatches.append((qe, expected, actual))

    if mismatches:
        lines = ["CPI release-date algorithmic mismatches:"]
        for qe, exp, act in mismatches:
            quarter = ((qe.month - 1) // 3) + 1
            diff = (act - exp).days
            lines.append(
                f"  Q{quarter} {qe.year} (ending {qe}): expected {exp}, "
                f"got {act} (diff {diff:+d} days)"
            )
        pytest.fail("\n".join(lines))


@pytest.mark.parametrize(
    ("year", "month", "expected_day"),
    [
        (2023, 4, 26),
        (2023, 7, 26),
        (2023, 10, 25),
        (2024, 1, 31),  # Wednesday IS the last day of the month
        (2024, 4, 24),
        (2024, 7, 31),  # Wednesday IS the last day of the month
        (2024, 2, 28),  # leap-year February, Wed lands on 28
        (2023, 2, 22),  # non-leap February, last Wed is 22
    ],
)
def test_last_wednesday_of_month(year: int, month: int, expected_day: int) -> None:
    assert last_wednesday_of_month(year, month) == date(year, month, expected_day)


def test_overrides_take_precedence() -> None:
    """If an override is registered, it wins over the algorithmic date."""
    override_qe = date(2024, 3, 31)
    forced = date(2024, 5, 1)
    _OVERRIDES[override_qe] = forced
    try:
        assert cpi_publication_date(override_qe) == forced
    finally:
        del _OVERRIDES[override_qe]


def test_build_cpi_release_calendar_shape_and_dtypes() -> None:
    df = build_cpi_release_calendar(start_year=2023, end_year=2024)
    # 4 quarters * (end_year - start_year + 2) = 4 * 3 = 12 rows
    assert len(df) == 12
    assert list(df.columns) == ["reference_quarter_end", "publication_date"]
    assert df["reference_quarter_end"].dtype == "datetime64[ns]"
    assert df["publication_date"].dtype == "datetime64[ns]"


def test_build_cpi_release_calendar_contains_known_dates() -> None:
    df = build_cpi_release_calendar(start_year=2023, end_year=2024)
    lookup = dict(zip(df["reference_quarter_end"], df["publication_date"]))
    for qe, expected in _KNOWN_RELEASE_DATES:
        assert lookup[pd.Timestamp(qe)] == pd.Timestamp(expected)
