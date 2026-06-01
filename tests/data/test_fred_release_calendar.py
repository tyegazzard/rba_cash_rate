"""Tests for ``rba.data.fred_release_calendar``.

Exercises:

- the override-precedence rule keyed by ``(series_id, observation_date)``,
- the three publication-date rules (daily T+1 US business day, VIX
  same-day, monthly flat + 20 days — covering BLS CPI and IMF Primary
  Commodity Prices),
- the ``_next_us_business_day`` primitive against US-federal-holiday
  samples (New Year's, MLK, Juneteenth, July 4th, Thanksgiving,
  Christmas with Mondayisation),
- the diagnostic ``build_fred_release_calendar`` builder.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from rba.data.fred_release_calendar import (
    _MONTHLY_FLAT_OFFSET_DAYS,
    _OVERRIDES,
    _RULES,
    _next_us_business_day,
    _us_federal_holidays,
    attach_fred_publication_dates,
    build_fred_release_calendar,
    fred_publication_date,
)


# -----------------------------------------------------------------------------
# _us_federal_holidays — pandas USFederalHolidayCalendar dispatch
# -----------------------------------------------------------------------------


def test_us_federal_holidays_2025_contains_canonical_dates() -> None:
    holidays_2025 = _us_federal_holidays(2025)
    # 2025 observed dates per the USFederalHolidayCalendar:
    assert date(2025, 1, 1) in holidays_2025          # New Year's Day
    assert date(2025, 1, 20) in holidays_2025         # MLK Day (3rd Mon)
    assert date(2025, 2, 17) in holidays_2025         # Presidents' Day (3rd Mon)
    assert date(2025, 5, 26) in holidays_2025         # Memorial Day (last Mon)
    assert date(2025, 6, 19) in holidays_2025         # Juneteenth
    assert date(2025, 7, 4) in holidays_2025          # Independence Day
    assert date(2025, 9, 1) in holidays_2025          # Labor Day (1st Mon)
    assert date(2025, 11, 11) in holidays_2025        # Veterans Day
    assert date(2025, 11, 27) in holidays_2025        # Thanksgiving (4th Thu)
    assert date(2025, 12, 25) in holidays_2025        # Christmas


def test_us_federal_holidays_mondayisation_2022() -> None:
    """Christmas Day 2022 fell on Sunday → observed Mon 26-Dec. The
    Sunday itself is *not* in the observed-holidays set."""
    holidays_2022 = _us_federal_holidays(2022)
    assert date(2022, 12, 26) in holidays_2022
    assert date(2022, 12, 25) not in holidays_2022


# -----------------------------------------------------------------------------
# _next_us_business_day primitive
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("trade", "expected"),
    [
        # Plain Mon → Tue.
        (date(2025, 5, 19), date(2025, 5, 20)),
        # Fri → Mon (skip weekend).
        (date(2025, 5, 23), date(2025, 5, 27)),  # Mon 26 = Memorial Day, → Tue 27
        # Wed before July 4th 2025 (Fri) → Mon July 7 (skip Fri-July 4, weekend).
        (date(2025, 7, 3), date(2025, 7, 7)),
        # Wed 24-Dec-2025 → Fri 26-Dec-2025 (Thu 25 = Christmas).
        (date(2025, 12, 24), date(2025, 12, 26)),
        # Wed 31-Dec-2025 → Fri 02-Jan-2026 (Thu 1 Jan = New Year).
        (date(2025, 12, 31), date(2026, 1, 2)),
        # Wed 26-Nov-2025 → Fri 28-Nov-2025 (Thu 27 = Thanksgiving).
        (date(2025, 11, 26), date(2025, 11, 28)),
    ],
)
def test_next_us_business_day(trade: date, expected: date) -> None:
    assert _next_us_business_day(trade) == expected


# -----------------------------------------------------------------------------
# fred_publication_date — per-rule dispatch
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("series_id", "observation", "expected"),
    [
        # us_daily_t_plus_1: DFF / DGS10 / DTWEXBGS — next US BDay.
        ("us_fed_funds", date(2025, 5, 19), date(2025, 5, 20)),
        ("us_10y_treasury", date(2025, 7, 3), date(2025, 7, 7)),  # skip July 4 + weekend
        ("us_dxy_broad", date(2025, 12, 24), date(2025, 12, 26)),  # skip Christmas
        # us_same_day: VIX — same calendar date.
        ("us_vix", date(2025, 5, 19), date(2025, 5, 19)),
        ("us_vix", date(2025, 12, 31), date(2025, 12, 31)),
        # us_monthly_flat: CPIAUCSL / CPILFESL — +20 calendar days.
        ("us_headline_cpi", date(2024, 10, 31), date(2024, 11, 20)),
        ("us_core_cpi", date(2025, 1, 31), date(2025, 2, 20)),
        # us_monthly_flat: IMF commodity series — same +20 calendar days rule.
        ("iron_ore_spot", date(2024, 10, 31), date(2024, 11, 20)),
        ("copper_spot", date(2025, 1, 31), date(2025, 2, 20)),
        # us_daily_t_plus_1: Brent / WTI oil — same primitive as DGS10.
        ("brent_crude", date(2025, 5, 19), date(2025, 5, 20)),
        ("wti_crude", date(2025, 7, 3), date(2025, 7, 7)),  # skip July 4 + weekend
    ],
)
def test_publication_date_per_rule(
    series_id: str, observation: date, expected: date
) -> None:
    assert fred_publication_date(observation, series_id) == expected


def test_publication_date_raises_for_unknown_series() -> None:
    with pytest.raises(ValueError, match="Unknown FRED series_id"):
        fred_publication_date(date(2025, 5, 19), "not_a_real_series")


def test_overrides_take_precedence_over_rule() -> None:
    """A (series_id, observation_date) entry in _OVERRIDES must short-circuit
    the algorithmic dispatch."""
    key = ("us_headline_cpi", date(2099, 1, 31))
    forced = date(2099, 2, 28)
    _OVERRIDES[key] = forced
    try:
        assert fred_publication_date(key[1], key[0]) == forced
    finally:
        del _OVERRIDES[key]


def test_monthly_flat_offset_constant_is_20_days() -> None:
    """If the flat offset is ever retuned, the assumption "no RBA meeting
    after monthly release within the 20-day window risks leakage" needs
    to be revisited — pin the constant explicitly. Applies to BLS CPI
    and IMF Primary Commodity Prices (iron ore / copper)."""
    assert _MONTHLY_FLAT_OFFSET_DAYS == 20


# -----------------------------------------------------------------------------
# _RULES registry — must stay in sync with the source module's SERIES tuple
# -----------------------------------------------------------------------------


def test_rules_cover_every_series_in_source_modules() -> None:
    """The calendar's _RULES dict and *every* FRED-fed source module's
    SERIES tuple must agree, jointly — otherwise _attach_publication_dates
    would silently leave rows unmatched and the validator's raise-on-miss
    would never trigger until the validation-floor check downstream. Both
    fred_global_signals and commodity_prices contribute series; the union
    of their logical IDs must equal _RULES.keys() exactly."""
    from rba.data.sources.commodity_prices import SERIES as COMMODITY_SERIES
    from rba.data.sources.fred_global_signals import SERIES as FRED_GLOBAL_SERIES

    source_ids = {s.series_id for s in FRED_GLOBAL_SERIES} | {
        s.series_id for s in COMMODITY_SERIES
    }
    rule_ids = set(_RULES.keys())
    assert source_ids == rule_ids, (
        f"Drift between source SERIES and _RULES: "
        f"missing rules {source_ids - rule_ids}, orphan rules {rule_ids - source_ids}"
    )


def test_source_modules_do_not_share_series_ids() -> None:
    """fred_global_signals and commodity_prices must not register the
    same logical series_id (would make _RULES dispatch ambiguous and
    double-count in the joint union test above)."""
    from rba.data.sources.commodity_prices import SERIES as COMMODITY_SERIES
    from rba.data.sources.fred_global_signals import SERIES as FRED_GLOBAL_SERIES

    fred_ids = {s.series_id for s in FRED_GLOBAL_SERIES}
    commodity_ids = {s.series_id for s in COMMODITY_SERIES}
    assert fred_ids.isdisjoint(commodity_ids)


def test_rules_are_known_strings() -> None:
    """Every rule must be one of the three implemented dispatch keys —
    catches typos in _RULES additions before they raise at lookup time."""
    known_rules = {"us_daily_t_plus_1", "us_same_day", "us_monthly_flat"}
    assert set(_RULES.values()).issubset(known_rules)


# -----------------------------------------------------------------------------
# attach_fred_publication_dates — vectorised stamp
# -----------------------------------------------------------------------------


def test_attach_stamps_each_row_per_its_series_rule() -> None:
    long_df = pd.DataFrame(
        {
            "observation_date": pd.to_datetime(
                [
                    "2025-05-19",  # Mon — daily T+1 → Tue 20
                    "2024-10-31",  # CPI ref Oct 2024 → Nov 20
                    "2025-05-19",  # Mon — VIX same-day → Mon 19
                ]
            ),
            "series_id": ["us_fed_funds", "us_headline_cpi", "us_vix"],
            "value": [4.33, 315.0, 18.2],
        }
    )
    out = attach_fred_publication_dates(long_df)
    by_sid = dict(zip(out["series_id"], out["publication_date"]))
    assert by_sid["us_fed_funds"] == pd.Timestamp("2025-05-20")
    assert by_sid["us_headline_cpi"] == pd.Timestamp("2024-11-20")
    assert by_sid["us_vix"] == pd.Timestamp("2025-05-19")


def test_attach_drops_stale_publication_date_column() -> None:
    long_df = pd.DataFrame(
        {
            "observation_date": pd.to_datetime(["2025-05-19"]),
            "series_id": ["us_fed_funds"],
            "value": [4.33],
            "publication_date": pd.to_datetime(["2099-01-01"]),
        }
    )
    out = attach_fred_publication_dates(long_df)
    assert (out["publication_date"] != pd.Timestamp("2099-01-01")).all()


def test_attach_preserves_canonical_column_order() -> None:
    long_df = pd.DataFrame(
        {
            "value": [4.33],
            "series_id": ["us_fed_funds"],
            "observation_date": pd.to_datetime(["2025-05-19"]),
        }
    )
    out = attach_fred_publication_dates(long_df)
    assert list(out.columns) == [
        "observation_date",
        "publication_date",
        "series_id",
        "value",
    ]


def test_attach_handles_empty_frame() -> None:
    long_df = pd.DataFrame(
        {
            "observation_date": pd.Series(dtype="datetime64[ns]"),
            "series_id": pd.Series(dtype="object"),
            "value": pd.Series(dtype="float64"),
        }
    )
    out = attach_fred_publication_dates(long_df)
    assert out.empty
    assert "publication_date" in out.columns
    assert out["publication_date"].dtype == "datetime64[ns]"


# -----------------------------------------------------------------------------
# build_fred_release_calendar — diagnostic full-window builder
# -----------------------------------------------------------------------------


def test_build_calendar_includes_every_series() -> None:
    """Window must straddle a month-end so monthly-CPI rows materialise."""
    df = build_fred_release_calendar(
        start_date=date(2025, 5, 19), end_date=date(2025, 6, 2)
    )
    assert set(df["series_id"].unique()) == set(_RULES.keys())


def test_build_calendar_dtypes() -> None:
    df = build_fred_release_calendar(
        start_date=date(2025, 5, 19), end_date=date(2025, 5, 23)
    )
    assert df["observation_date"].dtype == "datetime64[ns]"
    assert df["publication_date"].dtype == "datetime64[ns]"


def test_build_calendar_daily_rows_skip_weekends() -> None:
    df = build_fred_release_calendar(
        start_date=date(2025, 5, 17),  # Sat
        end_date=date(2025, 5, 26),    # Mon (Memorial Day — but still in obs index)
    )
    daily = df[df["series_id"].isin({
        "us_fed_funds", "us_10y_treasury", "us_dxy_broad", "us_vix",
        "brent_crude", "wti_crude",
    })]
    weekdays = daily["observation_date"].dt.weekday
    assert (weekdays < 5).all()


def test_build_calendar_monthly_rows_are_month_ends() -> None:
    df = build_fred_release_calendar(
        start_date=date(2025, 1, 1), end_date=date(2025, 12, 31)
    )
    monthly = df[df["series_id"].isin(
        {"us_headline_cpi", "us_core_cpi", "iron_ore_spot", "copper_spot"}
    )]
    # Each unique observation_date among monthly rows must be a month-end.
    for ts in monthly["observation_date"].unique():
        ts = pd.Timestamp(ts)
        assert ts == ts + pd.offsets.MonthEnd(0), (
            f"Monthly calendar row has non-month-end observation_date {ts}"
        )


def test_build_calendar_publication_after_or_equal_observation() -> None:
    """Daily T+1 → strictly after; VIX → equal; CPI → +20 days. None
    before."""
    df = build_fred_release_calendar(
        start_date=date(2025, 5, 19), end_date=date(2025, 5, 23)
    )
    assert (df["publication_date"] >= df["observation_date"]).all()


def test_build_calendar_sorted_by_series_and_observation() -> None:
    df = build_fred_release_calendar(
        start_date=date(2025, 5, 19), end_date=date(2025, 5, 23)
    )
    # Sort key (series_id, observation_date) must be monotonic-increasing.
    sort_key = list(zip(df["series_id"], df["observation_date"]))
    assert sort_key == sorted(sort_key)
