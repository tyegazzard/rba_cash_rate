"""Tests for ``rba.data.agb_yields_release_calendar``.

Exercises the override-precedence rule, the +1 business-day algorithm
against the Wayback-verified samples (see calendar module docstring),
and the build-DataFrame helper.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from rba.data.agb_yields_release_calendar import (
    _OVERRIDES,
    _next_business_day,
    agb_yields_publication_date,
    build_agb_yields_release_calendar,
)

# -----------------------------------------------------------------------------
# _next_business_day primitive
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("trade", "expected"),
    [
        # Plain Mon → Tue.
        (date(2026, 5, 18), date(2026, 5, 19)),
        # Fri → Mon (skip weekend).
        (date(2026, 5, 22), date(2026, 5, 25)),
        # Sat / Sun observation (degenerate — no trade) still resolves to next BDay.
        (date(2026, 5, 23), date(2026, 5, 25)),
        (date(2026, 5, 24), date(2026, 5, 25)),
        # Thu 2026-04-02 (day before Good Fri 2026-04-03) → Tue 2026-04-07
        # (skip Fri Good Fri, Sat/Sun, Mon Easter Mon).
        (date(2026, 4, 2), date(2026, 4, 7)),
        # Wed Dec 24 2025 → Mon Dec 29 2025 (skip Thu Xmas, Fri Boxing, Sat/Sun).
        (date(2025, 12, 24), date(2025, 12, 29)),
        # Wed Dec 31 2025 → Fri Jan 2 2026 (skip Thu Jan 1 NY).
        (date(2025, 12, 31), date(2026, 1, 2)),
    ],
)
def test_next_business_day(trade: date, expected: date) -> None:
    assert _next_business_day(trade) == expected


# -----------------------------------------------------------------------------
# agb_yields_publication_date — Wayback-verified samples
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("observation", "expected"),
    [
        # 5 Wayback F2 snapshots: each row's date + 1 BDay = header date.
        (date(2018, 1, 18), date(2018, 1, 19)),   # Thu → Fri
        (date(2018, 3, 22), date(2018, 3, 23)),   # Thu → Fri
        (date(2018, 4, 23), date(2018, 4, 24)),   # Mon → Tue
        (date(2018, 5, 24), date(2018, 5, 25)),   # Thu → Fri
        (date(2018, 8, 8), date(2018, 8, 9)),     # Wed → Thu
    ],
)
def test_publication_date_matches_wayback_samples(
    observation: date, expected: date
) -> None:
    assert agb_yields_publication_date(observation) == expected


# -----------------------------------------------------------------------------
# Holiday-skip behaviour
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("observation", "expected"),
    [
        # Thu 2026-04-02 trade → Tue 2026-04-07 (Good Fri 3rd, Easter Mon 6th).
        (date(2026, 4, 2), date(2026, 4, 7)),
        # Thu 2025-04-24 → Mon 2025-04-28 (Fri 25 Apr = Anzac Day, weekend).
        (date(2025, 4, 24), date(2025, 4, 28)),
        # Fri 2026-06-05 → Tue 2026-06-09 (Mon 8 = Queen's Birthday, weekend).
        (date(2026, 6, 5), date(2026, 6, 9)),
        # Fri 2026-10-02 → Tue 2026-10-06 (Mon 5 = NSW Labour Day, weekend).
        (date(2026, 10, 2), date(2026, 10, 6)),
    ],
)
def test_publication_date_skips_holidays(observation: date, expected: date) -> None:
    assert agb_yields_publication_date(observation) == expected


# -----------------------------------------------------------------------------
# Year-boundary roll
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("observation", "expected"),
    [
        # Wed Dec 31 2025 → Fri Jan 2 2026 (Thu Jan 1 = NY Day).
        (date(2025, 12, 31), date(2026, 1, 2)),
        # Fri Dec 29 2017 → Tue Jan 2 2018 (Mon Jan 1 NY Day, weekend).
        (date(2017, 12, 29), date(2018, 1, 2)),
    ],
)
def test_year_boundary_publication(observation: date, expected: date) -> None:
    assert agb_yields_publication_date(observation) == expected


# -----------------------------------------------------------------------------
# _OVERRIDES precedence
# -----------------------------------------------------------------------------


def test_overrides_take_precedence_over_algorithm() -> None:
    override_key = date(2024, 6, 21)
    forced = date(2024, 7, 1)
    _OVERRIDES[override_key] = forced
    try:
        assert agb_yields_publication_date(override_key) == forced
    finally:
        del _OVERRIDES[override_key]


# -----------------------------------------------------------------------------
# build_agb_yields_release_calendar
# -----------------------------------------------------------------------------


def test_build_calendar_shape_and_dtypes() -> None:
    df = build_agb_yields_release_calendar(
        start_date=date(2026, 5, 18), end_date=date(2026, 5, 22)
    )
    # Mon-Fri inclusive = 5 weekdays.
    assert len(df) == 5
    assert list(df.columns) == ["observation_date", "publication_date"]
    assert df["observation_date"].dtype == "datetime64[ns]"
    assert df["publication_date"].dtype == "datetime64[ns]"


def test_build_calendar_skips_weekends_in_observation_index() -> None:
    df = build_agb_yields_release_calendar(
        start_date=date(2026, 5, 16),  # Sat
        end_date=date(2026, 5, 25),    # Mon
    )
    weekdays = df["observation_date"].dt.weekday
    assert (weekdays < 5).all()


def test_build_calendar_publication_strictly_after_observation() -> None:
    """Every publication date must be strictly after the observation date.
    (F2 publishes the next morning, never same-day.)"""
    df = build_agb_yields_release_calendar(
        start_date=date(2013, 5, 20), end_date=date(2026, 6, 30)
    )
    assert (df["publication_date"] > df["observation_date"]).all()


def test_build_calendar_publication_is_weekday() -> None:
    df = build_agb_yields_release_calendar(
        start_date=date(2013, 5, 20), end_date=date(2026, 6, 30)
    )
    weekdays = df["publication_date"].dt.weekday
    assert (weekdays < 5).all()


def test_build_calendar_unique_and_monotonic_observation_date() -> None:
    df = build_agb_yields_release_calendar(
        start_date=date(2020, 1, 1), end_date=date(2026, 6, 30)
    )
    assert df["observation_date"].is_unique
    assert df["observation_date"].is_monotonic_increasing


def test_build_calendar_max_lag_is_small() -> None:
    """The longest stretch of consecutive AU/NSW holidays is the
    Christmas/Boxing/Year-end block (~5-6 calendar days). The
    observation → publication lag should never exceed that."""
    df = build_agb_yields_release_calendar(
        start_date=date(2013, 5, 20), end_date=date(2026, 6, 30)
    )
    lag = (df["publication_date"] - df["observation_date"]).dt.days
    assert lag.min() == 1, "Minimum lag should always be exactly 1 day (Mon → Tue)"
    assert lag.max() <= 7, f"Unexpectedly large lag observed: {lag.max()} days"
